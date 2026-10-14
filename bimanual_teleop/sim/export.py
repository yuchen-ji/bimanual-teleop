"""Independent sim exporter compatible with the existing 52-D DP dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from bimanual_teleop.sim._dependencies import np
from bimanual_teleop.sim._dependencies import zarr
from bimanual_teleop.sim._dependencies import Blosc
from bimanual_teleop.sim._dependencies import Rotation

from .recording import CAMERAS, atomic_json, discover, read_boundaries
from .segments import load_scheme


def _pose_vectors(poses):
    poses = np.asarray(poses)
    if poses.ndim != 3 or poses.shape[1:] != (2, 7) or not np.isfinite(poses).all():
        raise ValueError("Invalid recorded end-effector poses")
    norms = np.linalg.norm(poses[..., 3:], axis=-1)
    if not np.allclose(norms, 1, atol=.001):
        raise ValueError("Recorded quaternions must be unit XYZW")
    result = np.concatenate([poses[..., :3], Rotation.from_quat(poses[..., 3:].reshape(-1, 4)).as_rotvec().reshape(-1, 2, 3)], axis=-1)
    return result.reshape(-1, 12).astype(np.float32)


def _runs(frames, start, end, camera_stride=8):
    blocks = np.asarray(frames["block_id"][start:end])
    ticks = np.asarray(frames["tick"][start:end])
    if not len(ticks):
        return []
    # 30 Hz frames under 240 Hz physics. A real gap always forms a sampling boundary.
    split = np.flatnonzero((np.diff(blocks) != 0) | (np.diff(ticks) != camera_stride)) + 1
    edges = np.r_[0, split, len(ticks)]
    return [(start+int(a), start+int(b)) for a, b in zip(edges[:-1], edges[1:])]


def _create_array(group, key, values, image=False):
    values = np.asarray(values)
    if key not in group:
        group.create_dataset(key, shape=(0, *values.shape[1:]), dtype=values.dtype,
                             chunks=(1 if image else 256, *values.shape[1:]),
                             compressor=Blosc(cname="lz4" if image else "zstd", clevel=3,
                                              shuffle=Blosc.NOSHUFFLE if image else Blosc.BITSHUFFLE))
    if group[key].shape[1:] != values.shape[1:] or group[key].dtype != values.dtype:
        raise ValueError(f"Dataset shape/dtype changed for {key}")
    group[key].append(values)


def _copy_video(path, count, selections, destination, key):
    import av
    mapping = {}
    for start, end, output in selections:
        for index in range(start, end):
            mapping[index] = output + index-start
    total = max((output+end-start for start, end, output in selections), default=0)
    array = None; decoded = 0
    with av.open(str(path)) as video:
        for index, frame in enumerate(video.decode(video=0)):
            decoded += 1
            if index not in mapping:
                continue
            image = frame.to_ndarray(format="rgb24")
            if array is None:
                _create_array(destination, key, np.empty((0, *image.shape), dtype=np.uint8), image=True)
                array = destination[key]
                array.resize((max(len(array), total), *image.shape))
            array[mapping[index]] = image
    if decoded != count:
        raise ValueError(f"{path}: video has {decoded} frames, manifest expects {count}")


def export_dataset(records, destination, ranges=None, scheme_name=None, allow_test_data=False):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    episodes, segments, boundaries, offset = [], [], [], 0
    signatures = set()
    try:
        output = zarr.open_group(str(temporary), mode="w")
        data, meta = output.require_group("data"), output.require_group("meta")
        for path, document in records:
            if document["status"] != "complete":
                continue
            if (document["metadata"].get("backend") != "isaac" or document["metadata"].get("input_source") == "synthetic"
                    or document["metadata"].get("test_data", False)) and not allow_test_data:
                raise ValueError("Transport-test data is not physical demonstration data; explicit test override required")
            signatures.add(document["metadata"].get("asset_signature", "unknown"))
            if len(signatures) > 1:
                raise ValueError("Cannot mix different robot assets in one dataset")
            raw = zarr.open_group(str(path / "raw.zarr"), mode="r")
            if "frames" not in raw:
                continue
            frames = raw["frames"]
            count = document["frames"]
            if any(len(array) != count for _key, array in frames.arrays()):
                raise ValueError(f"Frame array lengths differ: {path}")
            source = document["source_demo_id"]
            spans = [(0, count)] if ranges is None else ranges.get(source, [])
            selections = []
            for span_index, (start, end) in enumerate(spans):
                if not 0 <= start < end <= count:
                    raise ValueError("Invalid export frame range")
                rows = {key: np.asarray(frames[key][start:end]) for key in (
                    "robot_joint", "hand_joint", "eef_pose_xyzw", "arm_command_pose_xyzw", "hand_command",
                    "arm_joint_command", "wrench", "time_ns", "tick", "command_tick", "block_id")}
                if np.any(np.diff(rows["time_ns"]) <= 0) or np.any(rows["command_tick"] > rows["tick"]):
                    raise ValueError("Invalid simulation capture/application timeline")
                arrays = {"robot_joint": rows["robot_joint"].astype(np.float32),
                          "hand_joint": rows["hand_joint"].astype(np.float32),
                          "robot_eef_pose": _pose_vectors(rows["eef_pose_xyzw"]),
                          "wrench": rows["wrench"].astype(np.float32),
                          "action": np.c_[_pose_vectors(rows["arm_command_pose_xyzw"]), rows["hand_command"]].astype(np.float32),
                          "timestamp": (rows["time_ns"]-int(frames["time_ns"][0])).astype(np.float64)/1e9}
                if any(not np.isfinite(value).all() for value in arrays.values()):
                    raise ValueError("Non-finite simulation observations/actions")
                for key, values in arrays.items():
                    _create_array(data, key, values)
                for key, values in {"source_frame": np.arange(start, end, dtype=np.int64),
                                    "source_time_ns": rows["time_ns"], "simulation_tick": rows["tick"],
                                    "command_tick": rows["command_tick"]}.items():
                    _create_array(meta, key, values)
                selections.append((start, end, offset))
                runtime = document["metadata"].get("config", {}).get("runtime", {})
                stride = runtime.get("physics_hz", 240) // runtime.get("camera_hz", 30)
                for low, high in _runs(frames, start, end, stride):
                    segments.append({"source_episode": source, "source_path": str(path.resolve()),
                                     "reference_frame_start": low, "reference_frame_end": high,
                                     "output_start": offset+low-start, "output_end": offset+high-start,
                                     "scheme": scheme_name, "range_index": span_index})
                for point in read_boundaries(path):
                    if start <= point["frame_index"] < end:
                        boundaries.append({**point, "output_frame": offset+point["frame_index"]-start})
                offset += end-start; episodes.append(offset)
            if selections:
                for camera in CAMERAS:
                    _copy_video(path / f"{camera}.mp4", count, selections, data, camera)
        if not offset:
            raise ValueError("No complete, nonempty simulation ranges to export")
        meta.create_dataset("episode_ends", data=np.asarray(episodes, dtype=np.int64))
        meta.create_dataset("segment_ends", data=np.asarray([s["output_end"] for s in segments], dtype=np.int64))
        meta.attrs.update(segments=segments, boundary_events=boundaries, scheme=scheme_name,
                          source_time_ns="simulation time; pauses are preserved as sampling boundaries")
        output.attrs.update(schema_version=2, format="diffusion_policy_replay_buffer", action_space="eef",
                            side_order=["left", "right"], eef_pose_format="xyz_m+rotvec_rad", joint_unit="rad",
                            action_layout="left_arm,right_arm,left_hand,right_hand", fps=30,
                            sampling_boundaries="meta/segment_ends", source="bimanual_teleop.sim",
                            asset_signatures=sorted(signatures), test_data=allow_test_data)
        temporary.rename(destination)
        return {"path": str(destination), "frames": offset, "episodes": len(episodes), "segments": len(segments)}
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def export(directory, destination, mode="full", scheme_path=None, allow_test_data=False):
    if mode not in ("full", "subtasks", "both"):
        raise ValueError("Export mode must be full, subtasks or both")
    records = discover(directory)
    scheme = load_scheme(scheme_path) if scheme_path else None
    if mode in ("subtasks", "both") and scheme is None:
        raise ValueError("Subtask export requires an explicit reviewed scheme; use sim.segments to create a template")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_counts = {d["source_demo_id"]: d["frames"] for _p, d in records if d["status"] == "complete"}
    tasks = {}
    if scheme:
        for item in scheme["episodes"]:
            if source_counts.get(item["source_demo_id"]) != item["frame_count"]:
                raise ValueError("Subtask scheme does not match the selected complete recording")
            for span in item["ranges"]:
                if span["task"] is None:
                    raise ValueError("Every selected range needs a task label; raw boundary order is not a task label")
                tasks.setdefault(span["task"], {}).setdefault(item["source_demo_id"], []).append((span["start"], span["end"]))
        if mode in ("subtasks", "both") and not tasks:
            raise ValueError("Subtask scheme selects no tasks")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    reports = {}
    try:
        if mode in ("full", "both"):
            reports["full"] = export_dataset(records, temporary / "full.zarr", allow_test_data=allow_test_data)
        if mode in ("subtasks", "both"):
            for task, ranges in tasks.items():
                reports[task] = export_dataset(records, temporary / "subtasks" / f"{task}.zarr", ranges,
                                               scheme["name"], allow_test_data)
        atomic_json(temporary / "export_report.json", reports)
        if scheme:
            shutil.copy2(scheme_path, temporary / "scheme.yaml")
        temporary.rename(destination)
        for report in reports.values():
            report["path"] = str(destination / Path(report["path"]).relative_to(temporary))
        atomic_json(destination / "export_report.json", reports)
        return reports
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description="Export full tasks and/or explicitly grouped subtasks")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("full", "subtasks", "both"), default="full")
    parser.add_argument("--scheme")
    parser.add_argument("--allow-test-data", action="store_true", help="Only for software transport tests")
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(export(args.input, args.output, args.mode, args.scheme, args.allow_test_data), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
