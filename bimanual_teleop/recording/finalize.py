"""Turn a captured spool into the existing episode.json + MP4 + raw.zarr contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

from .sink import Record, STREAM_FIELDS
from .spool import CAMERA_META, NUMERIC_STRUCTS
from .storage import EpisodeWriter, write_json


def _records_from_numeric(path, stream):
    parser = NUMERIC_STRUCTS[stream]
    fields = STREAM_FIELDS[stream]
    with Path(path).open("rb") as source:
        while True:
            payload = source.read(parser.size)
            if not payload:
                return
            if len(payload) != parser.size:
                raise ValueError(f"低维分段尾部不完整：{path}")
            unpacked = parser.unpack(payload)
            values, cursor = {}, 2
            for name, size in fields:
                values[name] = tuple(unpacked[cursor:cursor + size])
                cursor += size
            yield Record(stream, unpacked[0], unpacked[1], values)


def _camera_records(path, stream):
    with Path(path).open("rb") as source:
        while True:
            payload = source.read(CAMERA_META.size)
            if not payload:
                return
            if len(payload) != CAMERA_META.size:
                raise ValueError(f"相机元数据尾部不完整：{path}")
            stamp, sequence, source_ms = CAMERA_META.unpack(payload)
            yield Record(stream, stamp, sequence, {"source_time_ms": source_ms})


def _video_frame_count(path):
    import av
    with av.open(str(path)) as container:
        return sum(1 for _frame in container.decode(video=0))


def _append_spool(writer, episode, document):
    raw = episode / "raw_spool"
    for stream in STREAM_FIELDS:
        path = raw / "streams" / (stream.replace("/", "__") + ".bin")
        if not path.is_file():
            continue
        previous = None
        for record in _records_from_numeric(path, stream):
            if previous is not None and record.sequence <= previous:
                raise ValueError(f"低维序号未严格递增：{stream}")
            previous = record.sequence
            writer.append(record)
    for index in range(3):
        camera = f"camera_{index}"
        stream = f"cameras/{camera}/rgb"
        metadata = raw / "cameras" / f"{camera}_rgb.bin"
        video = episode / f"{camera}.mp4"
        if not metadata.is_file() or not video.is_file():
            raise ValueError(f"缺少 {camera} 视频或元数据")
        records = list(_camera_records(metadata, stream))
        if _video_frame_count(video) != len(records):
            raise ValueError(f"{camera} 视频帧数与元数据不一致")
        previous = None
        for record in records:
            if previous is not None and record.sequence <= previous:
                raise ValueError(f"相机序号未严格递增：{stream}")
            previous = record.sequence
            writer.append(record)
    depth_meta = raw / "cameras" / "camera_0_depth.bin"
    depth_raw = raw / "cameras" / "camera_0_depth.raw"
    if depth_meta.exists() != depth_raw.exists():
        raise ValueError("深度图像与深度元数据必须同时存在")
    if depth_meta.is_file():
        import numpy as np
        frame_bytes = 480 * 640 * 2
        stream = "cameras/camera_0/depth"
        previous = None
        with depth_raw.open("rb") as images:
            for record in _camera_records(depth_meta, stream):
                if previous is not None and record.sequence <= previous:
                    raise ValueError(f"相机序号未严格递增：{stream}")
                previous = record.sequence
                payload = images.read(frame_bytes)
                if len(payload) != frame_bytes:
                    raise ValueError("深度图像分段尾部不完整")
                image = np.frombuffer(payload, dtype="<u2").reshape(480, 640).copy()
                writer.append(Record(stream, record.time_ns, record.sequence,
                    {**record.values, "image": image}))
            if images.read(1):
                raise ValueError("深度图像数量多于深度元数据")
    expected = document.get("counts", {})
    if writer.counts != expected:
        raise ValueError(f"原始计数不一致：清单={expected}，读取={writer.counts}")


def finalize_episode(path, *, sdk_root=None):
    """Finalize one captured episode and return its resulting status."""
    episode = Path(path).resolve()
    manifest = episode / "episode.json"
    if not manifest.is_file():
        raise ValueError(f"缺少 episode.json：{episode}")
    document = json.loads(manifest.read_text(encoding="utf-8"))
    if document.get("status") == "complete":
        return "complete"
    if document.get("status") not in ("captured", "finalizing"):
        raise ValueError(f"条目状态不可整理：{document.get('status')} ({episode})")
    document["status"] = "finalizing"
    document.pop("finalize_error", None)
    write_json(manifest, document)
    temporary = episode.parent / f".{episode.name}.finalizing-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    writer = None
    try:
        from bimanual_teleop.devices.tianji.model import TianjiKinematics
        kinematics = TianjiKinematics(sdk_root)
        expected_model = document.get("metadata", {}).get("model_sha256")
        if expected_model and kinematics.model.digest != expected_model:
            raise ValueError("离线整理使用的天机运动学模型与采集时不一致")
        writer = EpisodeWriter(temporary, document["start_ns"], document["metadata"], kinematics)
        _append_spool(writer, episode, document)
        writer.close(document["end_ns"], status="complete")
        destination = episode / "raw.zarr"
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary / "raw.zarr", destination)
        os.replace(temporary / "episode.json", manifest)
        temporary.rmdir()
        return "complete"
    except BaseException as error:
        if writer is not None and temporary.exists():
            try:
                writer.close(document.get("end_ns") or document["start_ns"],
                             status="failed", reason=str(error))
            except BaseException:
                pass
        if temporary.exists():
            shutil.rmtree(temporary)
        document["status"] = "captured"
        document["finalize_error"] = str(error)
        write_json(manifest, document)
        raise


def finalize_recordings(path, *, sdk_root=None):
    source = Path(path).resolve()
    manifests = [source / "episode.json"] if (source / "episode.json").is_file() else sorted(
        source.rglob("episode.json"))
    if not manifests:
        raise ValueError(f"未找到 episode.json：{source}")
    report = {"complete": 0, "skipped": 0, "episodes": []}
    for manifest in manifests:
        document = json.loads(manifest.read_text(encoding="utf-8"))
        if document.get("status") not in ("captured", "finalizing", "complete"):
            report["skipped"] += 1
            continue
        status = finalize_episode(manifest.parent, sdk_root=sdk_root)
        report["complete"] += status == "complete"
        report["episodes"].append(str(manifest.parent))
    return report
