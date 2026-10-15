"""Write LeRobotDataset v3.0 files without importing the training stack.

The layout follows huggingface/lerobot's v3.0 metadata contract. Each shard
packs complete episodes into a Parquet file and three camera videos. Episode
rows carry both global table offsets and local video offsets.
"""

from fractions import Fraction
from pathlib import Path

import numpy as np

from .schema import CAMERAS, SIDES
from .storage import write_json

VERSION = "v3.0"
CHUNK_SIZE = 1000
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EPISODES_PATH = "meta/episodes/chunk-000/file-000.parquet"
PROVENANCE = {
    "source_time_ns": "source.time_ns",
    "recording_time_ns": "source.recording_time_ns",
    "recording_block": "source.recording_block",
    "camera_source_row": "source.camera_row",
    "camera_reused": "source.camera_reused",
    "camera_time_offset_ns": "source.camera_offset_ns",
    "camera_source_time_ms": "source.camera_timestamp_ms",
    "state_interpolated": "source.state_interpolated",
    "command_age_ns": "source.command_age_ns",
}


def joint_names():
    return ([f"{side}.arm.J{joint}" for side in SIDES for joint in range(1, 8)]
            + [f"{side}.hand.finger{finger}.joint{joint}" for side in SIDES
               for finger in range(1, 6) for joint in range(1, 5)])


def pose_names():
    return [f"{side}.{axis}" for side in SIDES
            for axis in ("x", "y", "z", "rotvec_x", "rotvec_y", "rotvec_z")]


def video_key(camera):
    return f"observation.images.{camera}"


def features_for(aligned, action_space):
    def vector(dtype, names):
        return {"dtype": dtype, "shape": (len(names),), "names": names}

    joints = joint_names()
    features = {
        "observation.state": vector("float32", joints),
        "action": vector("float32", joints if action_space == "joint" else pose_names() + joints[14:]),
        "observation.eef_pose": vector("float32", pose_names()),
        "observation.wrench": vector("float32", [f"{side}.{axis}" for side in SIDES
                                                 for axis in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")]),
    }
    for camera in CAMERAS:
        shape = aligned[f"data/{camera}"].shape[1:]
        if len(shape) != 3 or shape[-1] != 3 or any(size % 2 for size in shape[:2]):
            raise ValueError(f"{camera}: RGB video requires even height/width and three channels")
        features[video_key(camera)] = {"dtype": "video", "shape": shape,
                                       "names": ["height", "width", "channels"]}
    for source, target in PROVENANCE.items():
        array = aligned[f"meta/{source}"]
        features[target] = {"dtype": str(array.dtype), "shape": array.shape[1:] or (1,), "names": None}
    for key in ("source.recording_index", "source.depth_row", "frame_index", "episode_index", "index", "task_index"):
        features[key] = {"dtype": "int64", "shape": (1,), "names": None}
    features["timestamp"] = {"dtype": "float32", "shape": (1,), "names": None}
    return features


def _table(path, columns, features=None):
    import datasets
    import pyarrow.parquet as pq

    schema = None
    if features is not None:
        schema = datasets.Features({
            key: (datasets.Value(value["dtype"]) if value["shape"] == (1,) else
                  datasets.Sequence(datasets.Value(value["dtype"]), length=value["shape"][0]))
            for key, value in features.items() if value["dtype"] != "video"
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = datasets.Dataset.from_dict(columns, features=schema)
    pq.write_table(dataset.data.table, path, compression="snappy")


def _stats(values, *, image=False):
    values = np.asarray(values, dtype=np.float64)
    if not image and values.ndim == 1:
        values = values[:, None]
    result = {"min": values.min(axis=0), "max": values.max(axis=0),
              "mean": values.mean(axis=0), "std": values.std(axis=0)}
    for quantile in (.01, .1, .5, .9, .99):
        result[f"q{int(quantile * 100):02d}"] = np.quantile(values, quantile, axis=0)
    if image:
        result = {key: value.reshape(3, 1, 1) for key, value in result.items()}
    result["count"] = np.asarray([len(values)], dtype=np.int64)
    return {key: value.tolist() for key, value in result.items()}


def _image_stats(array, start, end):
    # Uniform time samples and a spatial grid bound RAM and avoid materializing
    # the videos. LeRobot also estimates image statistics from sampled images.
    rows = np.unique(np.linspace(start, end - 1, min(64, end - start), dtype=int))
    stride = max(1, max(array.shape[1:3]) // 128)
    pixels = np.concatenate([array[int(row)][::stride, ::stride].reshape(-1, 3) for row in rows])
    result = _stats(pixels / 255., image=True)
    result["count"] = [len(rows)]
    return result


def _merge_stats(previous, current):
    """Exact streaming mean/std/min/max; episode-weighted quantile estimates."""
    if previous is None:
        return current
    a, b = previous["count"][0], current["count"][0]
    total = a + b
    old_mean, new_mean = np.asarray(previous["mean"]), np.asarray(current["mean"])
    delta = new_mean - old_mean
    variance = (a * np.asarray(previous["std"]) ** 2 + b * np.asarray(current["std"]) ** 2
                + delta ** 2 * a * b / total) / total
    merged = {"count": [total], "mean": (old_mean + delta * b / total).tolist(),
              "std": np.sqrt(variance).tolist(),
              "min": np.minimum(previous["min"], current["min"]).tolist(),
              "max": np.maximum(previous["max"], current["max"]).tolist()}
    for key in current:
        if key.startswith("q"):
            merged[key] = ((np.asarray(previous[key]) * a + np.asarray(current[key]) * b) / total).tolist()
    return merged


def _encode_video(path, images, start, end, fps, encoder_threads):
    import av

    height, width, _ = images.shape[1:]
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.codec_context.thread_count = encoder_threads
        stream.options = {"crf": "18", "preset": "fast", "g": "2", "bf": "0"}
        for row in range(start, end):
            frame = av.VideoFrame.from_ndarray(images[row], format="rgb24")
            frame.pts, frame.time_base = row - start, Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.frames != end - start:
            raise ValueError(f"{path}: encoded frame count differs from table rows")
        return {"video.height": height, "video.width": width, "video.codec": stream.codec_context.name,
                "video.pix_fmt": stream.codec_context.format.name, "video.fps": fps,
                "video.channels": 3, "has_audio": False, "is_depth_map": False}


def _shards(segments, max_frames):
    current, frames = [], 0
    for segment in segments:
        length = segment["output_end"] - segment["output_start"]
        if current and frames + length > max_frames:
            yield current
            current, frames = [], 0
        current.append(segment)
        frames += length
    if current:
        yield current


def write_dataset(root, aligned, contexts, *, action_space, fps, max_frames, encoder_threads, progress):
    """Write standard v3 metadata, scalar/vector Parquet features and RGB MP4s."""
    root = Path(root)
    root.mkdir()
    features = features_for(aligned, action_space)
    tasks = list(dict.fromkeys(context["task"] for context in contexts.values()))
    _write_tasks(root, tasks)
    task_indices = {task: index for index, task in enumerate(tasks)}
    numeric_keys = [key for key, value in features.items() if value["dtype"] != "video"]
    stats = {}
    episode_rows, mapping = [], []
    segments = aligned["meta"].attrs["segments"]
    progress.start(len(aligned["meta"].attrs["quality_report"]["episodes"]) + len(segments))
    for shard_index, shard in enumerate(_shards(segments, max_frames)):
        chunk, file = divmod(shard_index, CHUNK_SIZE)
        first, last = shard[0]["output_start"], shard[-1]["output_end"]
        shard_columns = {key: [] for key in numeric_keys}
        for segment in shard:
            start, end = segment["output_start"], segment["output_end"]
            length, episode_index = end - start, len(episode_rows)
            context = contexts[segment["source_episode"]]
            if length > 1 and not np.allclose(np.diff(aligned["data/timestamp"][start:end]),
                                               1 / fps, rtol=0, atol=2e-8):
                raise ValueError("A LeRobot episode must be a continuous fixed-rate run")
            columns = {
                "observation.state": np.concatenate((aligned["data/robot_joint"][start:end],
                                                      aligned["data/hand_joint"][start:end]), axis=1),
                "action": aligned["data/action"][start:end],
                "observation.eef_pose": aligned["data/robot_eef_pose"][start:end],
                "observation.wrench": aligned["data/wrench"][start:end],
                "timestamp": (np.arange(length) / fps).astype(np.float32),
                "frame_index": np.arange(length, dtype=np.int64),
                "episode_index": np.full(length, episode_index, np.int64),
                "index": np.arange(start, end, dtype=np.int64),
                "task_index": np.full(length, task_indices[context["task"]], np.int64),
                "source.recording_index": np.full(length, context["recording_index"], np.int64),
                "source.depth_row": context["depth_rows"][start:end],
            }
            columns.update({target: aligned[f"meta/{source}"][start:end] for source, target in PROVENANCE.items()})
            row = {"episode_index": episode_index, "length": length, "tasks": [context["task"]],
                   "dataset_from_index": start, "dataset_to_index": end,
                   "data/chunk_index": chunk, "data/file_index": file,
                   "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0}
            for key, values in columns.items():
                shard_columns[key].append(values)
                episode_stats = _stats(values)
                stats[key] = _merge_stats(stats.get(key), episode_stats)
                for stat, value in episode_stats.items():
                    row[f"stats/{key}/{stat}"] = value
            for camera in CAMERAS:
                key = video_key(camera)
                row.update({f"videos/{key}/chunk_index": chunk, f"videos/{key}/file_index": file,
                            f"videos/{key}/from_timestamp": (start - first) / fps,
                            f"videos/{key}/to_timestamp": (end - first) / fps})
                for stat, value in _image_stats(aligned[f"data/{camera}"], start, end).items():
                    row[f"stats/{key}/{stat}"] = value
            episode_rows.append(row)
            mapping.append({"episode_index": episode_index, "recording_index": context["recording_index"],
                            **segment})
        _table(root / DATA_PATH.format(chunk_index=chunk, file_index=file),
               {key: np.concatenate(values) for key, values in shard_columns.items()}, features)
        progress.episode(f"LeRobot shard {shard_index}", total=(last - first) * len(CAMERAS))
        for camera in CAMERAS:
            key = video_key(camera)
            features[key]["info"] = _encode_video(
                root / VIDEO_PATH.format(video_key=key, chunk_index=chunk, file_index=file),
                aligned[f"data/{camera}"], first, last, fps, encoder_threads)
            progress.advance(last - first)
        for _ in shard:
            progress.finish_episode()
    _table(root / EPISODES_PATH, {key: [row[key] for row in episode_rows] for key in episode_rows[0]})
    for camera in CAMERAS:
        stats[video_key(camera)] = _image_stats(aligned[f"data/{camera}"], 0, len(aligned[f"data/{camera}"]))
    write_json(root / "meta/stats.json", stats)
    write_json(root / "meta/info.json", {
        "codebase_version": VERSION, "robot_type": "tianji", "total_episodes": len(episode_rows),
        "total_frames": aligned["data/action"].shape[0], "total_tasks": len(tasks),
        "chunks_size": CHUNK_SIZE, "data_files_size_in_mb": 100, "video_files_size_in_mb": 200,
        "fps": fps, "splits": {"train": f"0:{len(episode_rows)}"},
        "data_path": DATA_PATH, "video_path": VIDEO_PATH, "features": features,
    })
    return mapping


def _write_tasks(root, tasks):
    import pandas as pd

    (root / "meta").mkdir()
    pd.DataFrame({"task_index": range(len(tasks))}, index=tasks).to_parquet(root / "meta/tasks.parquet")
