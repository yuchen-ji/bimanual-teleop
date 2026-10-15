"""Offline Tianji -> LeRobot v3.0 export, retaining native data in sidecars.

Alignment is shared with the DP exporter. Each uninterrupted training run is
a separate LeRobot episode, so fixed-rate timestamps never hide invalid gaps.
Depth stays in its original uint16 Zarr representation and native timebase.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from bimanual_teleop.common.console import EpisodeProgress

from .convert import _publish, convert_recordings
from .episodes import episode_label, find_episodes
from .lerobot_format import VERSION, write_dataset
from .plan import read_stream
from .policy import load_policy
from .schema import CAMERAS, DEPTH_STREAM, rgb_stream
from .storage import write_json
from .timeline import camera_pauses, nearest, recording_blocks


def _options(conversion_config, fps):
    policy = load_policy(conversion_config)
    if policy.mode != "repair":
        raise ValueError("LeRobot export requires mode: repair (a fixed-rate training grid)")
    if fps is not None:
        policy = load_policy({**policy.config, "target_fps": fps})
    if not float(policy.target_fps).is_integer():
        raise ValueError("LeRobot target_fps must be an integer")
    return policy


def _contexts(episodes, report, task):
    result = {}
    for item in report["episodes"]:
        if not item.get("output_frames"):
            continue
        name = item["source_episode"]
        episode = episodes[name]
        path = episode.parent / "dataset.json"
        dataset = None
        if path.exists():
            dataset = json.loads(path.read_text(encoding="utf-8"))
            if (not isinstance(dataset, dict) or dataset.get("format") != "tianji_raw_dataset"
                    or type(dataset.get("schema_version")) is not int or dataset["schema_version"] != 1):
                raise ValueError(f"Unsupported dataset.json: {path}")
            for camera in CAMERAS:
                recorded = item["metadata"].get("cameras", {}).get(camera, {}).get("serial")
                if not recorded or dataset.get("cameras", {}).get(camera, {}).get("serial") != recorded:
                    raise ValueError(f"{name}: dataset.json serial differs from episode metadata for {camera}")
        sentence = task if task is not None else (dataset or {}).get("task")
        if not isinstance(sentence, str) or not sentence.strip():
            raise ValueError(f"{name}: provide --task or a dataset.json containing the task sentence")
        result[name] = {"recording_index": len(result), "path": episode, "task": sentence,
                        "source_dataset_id": (dataset or {}).get("dataset_id", episode.parent.name),
                        "source_episode_id": episode.name, "source_episode": name,
                        "dataset_manifest": dataset, "metadata": item["metadata"]}
    return result


def _depth_rows(raw, descriptor, query, tolerance_ns, bounds):
    """Nearest native depth row, or -1 when no frame is within tolerance."""
    result = np.full(len(query), -1, np.int64)
    if DEPTH_STREAM not in raw:
        return result
    group = raw[DEPTH_STREAM]
    times = group["time_ns"][:]
    image = group["image"]
    if (times.ndim != 1 or times.dtype != np.dtype("int64")
            or (len(times) > 1 and np.any(np.diff(times) <= 0))
            or len(image.shape) != 3 or image.shape[0] != len(times) or image.dtype != np.dtype("uint16")):
        raise ValueError("Native depth requires increasing int64 timestamps and uint16 (N,H,W) images")
    depth = read_stream(raw, DEPTH_STREAM, {}, descriptor["start_ns"], descriptor["end_ns"])
    rows, valid = nearest(depth, query, tolerance_ns=tolerance_ns, bounds=bounds)
    result[valid] = rows[valid]
    return result


def _native_sources(output, contexts, *, include_raw):
    import zarr

    sources = []
    for name, context in contexts.items():
        source = context["path"]
        raw = zarr.open_group(str(source / "raw.zarr"), mode="r")
        record = {key: context[key] for key in ("recording_index", "task", "source_dataset_id",
                                               "source_episode_id", "source_episode", "dataset_manifest")}
        record["metadata"] = context["metadata"]
        record["raw_path"] = None
        record["episode_manifest_path"] = None
        if include_raw:
            directory = Path("raw") / f"recording-{context['recording_index']:06d}"
            # Snapshot names deliberately differ from episode.json/dataset.json:
            # recording discovery must not mistake these attachments for inputs.
            archive = output / directory
            archive.mkdir(parents=True)
            shutil.copytree(source / "raw.zarr", archive / "raw.zarr")
            shutil.copyfile(source / "episode.json", archive / "source_episode.json")
            record["raw_path"] = (directory / "raw.zarr").as_posix()
            record["episode_manifest_path"] = (directory / "source_episode.json").as_posix()
            if context["dataset_manifest"] is not None:
                shutil.copyfile(source.parent / "dataset.json", archive / "source_dataset.json")
        record["native_depth_frames"] = int(raw[f"{DEPTH_STREAM}/image"].shape[0]) if DEPTH_STREAM in raw else 0
        sources.append(record)
    return sources


def _card(output, repo_id, report):
    width = 54 if report["action_space"] == "joint" else 52
    text = f"""---
tags:
- robotics
- lerobot
- tianji
configs:
- config_name: default
  data_files:
  - split: train
    path: data/*/*.parquet
---
# Tianji recordings ({repo_id})

LeRobotDataset v3.0, {report['fps']} Hz, {report['lerobot_episodes']} continuous episodes,
{report['output_frames']} frames from {report['output_episodes']} source demonstrations.

`observation.state` is 54 measured joint positions; `action` is {width} recorded
controller targets ({report['action_space']}). Vectors concatenate left arm,
right arm, left hand, right hand. Joint positions are radians. Additional
observations contain flange poses (xyz metres + rotation vectors radians) and
wrenches (forces N + torques N m). Three RGB cameras use H.264, CRF 18.

Every valid continuous run becomes its own episode. Frame timestamps start at
zero and use the fixed training rate. `source.*` columns retain native timing,
selected camera rows, interpolation and command ages. See `meta/tianji.json`
for source identities, camera roles, calibration, devices and episode mapping;
see `meta/conversion_report.json` for dropped/repaired frames and quality.

Native raw Zarr attachments: {report['include_raw']}. When enabled, they retain
all high-frequency state/command streams and original uint16 depth frames,
without quantization or resampling. The standard LeRobot loader reads RGB and
table features; depth is accessed via `source.recording_index`, the corresponding
`raw_path` in `meta/tianji.json`, and `source.depth_row` (-1 means unmatched).
Depth availability does not remove otherwise valid RGB training samples.

RGB normalization statistics use up to 64 uniformly sampled frames and a
spatial grid. Scalar/vector mean, std, min and max use all exported rows;
quantiles are episode-weighted estimates.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def convert_lerobot(input_path, output_path=None, *, repo_id="local/tianji", action_space,
                    task=None, fps=None, conversion_config=None, quality_config=None,
                    include_raw=True, allow_mixed_metadata=False, dry_run=False,
                    max_frames_per_shard=10000, encoder_threads=2, progress=None):
    """Publish a new local LeRobot dataset atomically; never upload or overwrite.

The fixed-rate DP planner decides validity, repairs and segment boundaries.
An intermediate Zarr lives only in a hidden temporary directory. No torch or
LeRobot package is required to export. Native raw attachments are on by default.
"""
    import zarr
    from huggingface_hub.utils import validate_repo_id

    validate_repo_id(repo_id)
    if task is not None and (not isinstance(task, str) or not task.strip()):
        raise ValueError("task must be a nonempty sentence")
    if type(include_raw) is not bool:
        raise ValueError("include_raw must be a boolean")
    for key, value in (("max_frames_per_shard", max_frames_per_shard), ("encoder_threads", encoder_threads)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    policy = _options(conversion_config, fps)
    source = Path(input_path).expanduser().resolve()
    episodes = {episode_label(path, source): path for path in find_episodes(source)}
    destination = temporary = None
    if not dry_run:
        if output_path is None:
            raise ValueError("output_path is required unless dry_run")
        candidate = Path(output_path).expanduser().absolute()
        if candidate.exists() or candidate.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {candidate}")
        destination = candidate.parent.resolve() / candidate.name
        if source == destination or source in destination.parents:
            raise ValueError("Output must be outside the searched input directory")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.lerobot-", dir=destination.parent))
    progress = EpisodeProgress(enabled=False) if progress is None else progress
    try:
        report = convert_recordings(source, None if dry_run else temporary / "aligned.zarr",
                                    action_space=action_space, include_depth=False,
                                    conversion_config=policy.config, quality_config=quality_config,
                                    allow_mixed_metadata=allow_mixed_metadata, dry_run=dry_run, progress=progress)
        contexts = _contexts(episodes, report, task)
        report.update(lerobot_version=VERSION, repo_id=repo_id, fps=int(policy.target_fps),
                      include_raw=include_raw, lerobot_episodes=report["output_segments"],
                      max_frames_per_shard=max_frames_per_shard, encoder_threads=encoder_threads)
        if dry_run:
            return report
        output = temporary / "dataset"
        aligned = zarr.open_group(str(temporary / "aligned.zarr"), mode="r")
        mapping = write_dataset(output, aligned,
                                _with_depth_indices(aligned, contexts, include_raw, policy),
                                action_space=action_space, fps=report["fps"], max_frames=max_frames_per_shard,
                                encoder_threads=encoder_threads, progress=progress)
        sources = _native_sources(output, contexts, include_raw=include_raw)
        write_json(output / "meta/tianji.json", {
            "schema_version": 1, "format": "tianji_lerobot_export", "repo_id": repo_id,
            "sources": sources, "episodes": mapping,
            "state_layout": "left_arm_J1..J7,right_arm_J1..J7,left_hand_finger1..5_joint1..4,right_hand_finger1..5_joint1..4",
            "action_space": action_space, "joint_unit": "rad", "eef_pose_format": "xyz_m+rotvec_rad",
            "wrench_format": "Fx,Fy,Fz [N]; Tx,Ty,Tz [N*m]",
            "provenance": {key: value for key, value in aligned["meta"].attrs.items()
                           if key not in ("segments", "quality_report")},
            "depth": {"storage": "native uint16 Zarr; no quantization", "stream": DEPTH_STREAM,
                      "frame_mapping": "source.recording_index + source.depth_row; -1 is unmatched",
                      "match_tolerance_ns": policy.camera_tolerance_ns, "preserved": include_raw},
        })
        write_json(output / "meta/conversion_report.json", report)
        _card(output, repo_id, report)
        _publish(output, destination)
        return report
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def _with_depth_indices(aligned, contexts, include_raw, policy):
    """Prepare per-row references before the Parquet writer uses them."""
    import zarr

    segments = aligned["meta"].attrs["segments"]
    for name, context in contexts.items():
        context["depth_rows"] = np.full(aligned["data/action"].shape[0], -1, np.int64)
        if not include_raw:
            continue
        raw = zarr.open_group(str(context["path"] / "raw.zarr"), mode="r")
        if DEPTH_STREAM not in raw:
            continue
        descriptor = json.loads((context["path"] / "episode.json").read_text(encoding="utf-8"))
        start_ns, end_ns = descriptor["start_ns"], descriptor["end_ns"]
        cameras = [read_stream(raw, rgb_stream(camera), {}, start_ns, end_ns) for camera in CAMERAS]
        offsets, seams = camera_pauses(cameras, [raw[f"{rgb_stream(camera)}/source_time_ms"][:][stream.rows]
                                               for camera, stream in zip(CAMERAS, cameras)],
                                      policy.pause_offset_tolerance_ms)
        blocks = recording_blocks(cameras, offsets, seams, start_ns, end_ns, target_fps=policy.target_fps,
                                  tolerance_ms=policy.pause_offset_tolerance_ms)
        bounds_by_block = {int(block): bounds for block, bounds in zip(blocks.ids, blocks.bounds)}
        for segment in segments:
            if segment["source_episode"] == name:
                start, end = segment["output_start"], segment["output_end"]
                bounds = np.asarray([bounds_by_block[int(block)]
                                     for block in aligned["meta/recording_block"][start:end]])
                # raw time_ns excludes paused wall time. source_time_ns is a
                # reconstructed clock and cannot index the native depth table.
                context["depth_rows"][start:end] = _depth_rows(
                    raw, descriptor, aligned["meta/recording_time_ns"][start:end],
                    policy.camera_tolerance_ns, bounds)
    return contexts
