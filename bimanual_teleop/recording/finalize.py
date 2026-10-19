"""Turn a captured spool into the existing episode.json + MP4 + raw.zarr contract."""

from __future__ import annotations

import errno
import math
import os
from pathlib import Path
import shutil

from bimanual_teleop.common.console import EpisodeProgress

from .episodes import (CAPTURED, COMPLETE, DISCARDED, FAILED, FINALIZING, MANIFEST,
                       episode_label, find_episodes, read_manifest, read_phase_markers)
from .schema import CAMERAS, DEPTH_SHAPE, DEPTH_STREAM, MAIN_CAMERA, RGB_STREAMS, STREAM_FIELDS, rgb_stream
from .sink import Record
from .spool import CAMERA_META, NUMERIC_STRUCTS, camera_meta_path, depth_image_path, numeric_path
from .storage import EpisodeWriter, write_json

SPOOL = "raw_spool"


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


def _video_frame_count(path, progress):
    import av
    count = 0
    with av.open(str(path)) as container:
        for _frame in container.decode(video=0):
            count += 1
            progress.advance()
    return count


def _append_spool(writer, episode, document, progress):
    raw = episode / SPOOL
    for stream in STREAM_FIELDS:
        path = numeric_path(raw, stream)
        if not path.is_file():
            continue
        previous = None
        for record in _records_from_numeric(path, stream):
            if previous is not None and record.sequence <= previous:
                raise ValueError(f"低维序号未严格递增：{stream}")
            previous = record.sequence
            writer.append(record)
    for camera in CAMERAS:
        stream = rgb_stream(camera)
        metadata = camera_meta_path(raw, camera)
        video = episode / f"{camera}.mp4"
        if not metadata.is_file() or not video.is_file():
            raise ValueError(f"缺少 {camera} 视频或元数据")
        records = list(_camera_records(metadata, stream))
        if _video_frame_count(video, progress) != len(records):
            raise ValueError(f"{camera} 视频帧数与元数据不一致")
        previous = None
        for record in records:
            if previous is not None and record.sequence <= previous:
                raise ValueError(f"相机序号未严格递增：{stream}")
            previous = record.sequence
            writer.append(record)
    depth_meta = camera_meta_path(raw, MAIN_CAMERA, "depth")
    depth_raw = depth_image_path(raw)
    if depth_meta.exists() != depth_raw.exists():
        raise ValueError("深度图像与深度元数据必须同时存在")
    if depth_meta.is_file():
        import numpy as np
        frame_bytes = math.prod(DEPTH_SHAPE) * 2
        previous = None
        with depth_raw.open("rb") as images:
            for record in _camera_records(depth_meta, DEPTH_STREAM):
                if previous is not None and record.sequence <= previous:
                    raise ValueError(f"相机序号未严格递增：{DEPTH_STREAM}")
                previous = record.sequence
                payload = images.read(frame_bytes)
                if len(payload) != frame_bytes:
                    raise ValueError("深度图像分段尾部不完整")
                image = np.frombuffer(payload, dtype="<u2").reshape(DEPTH_SHAPE).copy()
                writer.append(Record(DEPTH_STREAM, record.time_ns, record.sequence,
                    {**record.values, "image": image}))
                progress.advance()
            if images.read(1):
                raise ValueError("深度图像数量多于深度元数据")
    expected = document.get("counts", {})
    if writer.counts != expected:
        raise ValueError(f"原始计数不一致：清单={expected}，读取={writer.counts}")


def finalize_episode(path, *, sdk_root=None, progress=None):
    """Finalize one captured episode and return its resulting status.

    ``progress`` counts decoded RGB frames and depth frames, which dominate the
    run time; low-dimensional records are not counted.
    """
    progress = EpisodeProgress(enabled=False) if progress is None else progress
    episode = Path(path).resolve()
    manifest = episode / MANIFEST
    if not manifest.is_file():
        raise ValueError(f"缺少 episode.json：{episode}")
    document = read_manifest(episode)
    if document.get("status") == COMPLETE:
        return COMPLETE
    if document.get("status") not in (CAPTURED, FINALIZING):
        raise ValueError(f"条目状态不可整理：{document.get('status')} ({episode})")
    document["status"] = FINALIZING
    document.pop("finalize_error", None)
    write_json(manifest, document)
    counts = document.get("counts", {})
    progress.total(sum(counts.get(stream, 0) for stream in RGB_STREAMS + (DEPTH_STREAM,)))
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
        writer = EpisodeWriter(temporary, document["start_ns"], document["metadata"], kinematics,
                               phase_markers=read_phase_markers(document))
        _append_spool(writer, episode, document, progress)
        writer.close(document["end_ns"], status=COMPLETE)
        destination = episode / "raw.zarr"
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary / "raw.zarr", destination)
        os.replace(temporary / MANIFEST, manifest)
        temporary.rmdir()
        return COMPLETE
    except BaseException as error:
        if writer is not None and temporary.exists():
            try:
                writer.close(document.get("end_ns") or document["start_ns"],
                             status=FAILED, reason=str(error))
            except BaseException:
                pass
        if temporary.exists():
            shutil.rmtree(temporary)
        document["status"] = CAPTURED
        document["finalize_error"] = str(error)
        write_json(manifest, document)
        raise


def archived_spool(episode, archive):
    """Where ``archive_spool`` keeps an episode's spool: ``<archive>/<task>/<episode>``."""
    episode = Path(episode)
    return Path(archive).expanduser().resolve() / episode.parent.name / episode.name


def _file_sizes(root):
    return sorted((path.relative_to(root).as_posix(), path.stat().st_size)
                  for path in Path(root).rglob("*") if path.is_file())


def archive_spool(episode, archive):
    """Move a finalized raw_spool to the archive and leave a symlink in its place.

    Returns whether a new link was made. The spool leaves the episode only after
    the archive copy is complete, so an interrupted run is resumed by rerunning.
    """
    episode = Path(episode)
    link = episode / SPOOL
    target = archived_spool(episode, archive)
    archived = False
    if not link.is_symlink():
        if link.is_dir():
            if target.exists():
                if _file_sizes(target) != _file_sizes(link):
                    raise ValueError(f"归档位置已有不同内容：{target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.rename(link, target)
                except OSError as error:
                    if error.errno != errno.EXDEV:
                        raise
                    staging = target.with_name(f".{target.name}.archiving-{os.getpid()}")
                    shutil.rmtree(staging, ignore_errors=True)
                    shutil.copytree(link, staging)
                    os.replace(staging, target)
            if link.is_dir():
                os.replace(link, episode / f".{SPOOL}.archived-{os.getpid()}")
        elif not target.is_dir():
            return False
        link.symlink_to(target, target_is_directory=True)
        archived = True
    for leftover in episode.glob(f".{SPOOL}.archived-*"):
        shutil.rmtree(leftover)
    return archived


def _restore_spool(episode, archive):
    link = episode / SPOOL
    if link.is_dir():
        return
    target = None if archive is None else archived_spool(episode, archive)
    if target is None or not target.is_dir():
        where = "" if target is None else f"，归档目录中也没有 {target}"
        raise ValueError(f"找不到 {SPOOL}：{link}{where}")
    if link.is_symlink():
        link.unlink()
    link.symlink_to(target, target_is_directory=True)


def finalize_recordings(path, *, sdk_root=None, spool_archive=None, refinalize=False, progress=None):
    """Finalize every episode below ``path``.

    With ``spool_archive`` each complete episode's raw_spool moves out of the
    recording tree. With ``refinalize`` complete episodes are reset to captured
    and rebuilt from their spool, followed through the symlink or found again
    under ``spool_archive``; a missing spool leaves the episode complete.
    """
    progress = EpisodeProgress(enabled=False) if progress is None else progress
    source = Path(path).resolve()
    episodes = find_episodes(source)
    if not episodes:
        raise ValueError(f"未找到 episode.json：{source}")
    if spool_archive is not None:
        archive = Path(spool_archive).expanduser().resolve()
        if archive == source or source in archive.parents:
            raise ValueError(f"{SPOOL} 归档目录不能位于输入目录内：{archive}")
    report = {"complete": 0, "discarded": 0, "skipped": 0, "failed": 0,
              "episodes": [], "deleted": [], "archived": [], "errors": []}
    progress.start(len(episodes))
    for episode in episodes:
        label = episode_label(episode, source)
        progress.episode(label)
        try:
            document = read_manifest(episode)
            status = document.get("status")
            if status == DISCARDED:
                shutil.rmtree(episode)
                report["discarded"] += 1
                report["deleted"].append(str(episode))
                continue
            if status not in (CAPTURED, FINALIZING, COMPLETE):
                report["skipped"] += 1
                continue
            if status != COMPLETE or refinalize:
                _restore_spool(episode, spool_archive)
            if status == COMPLETE and refinalize:
                document["status"] = CAPTURED
                write_json(episode / MANIFEST, document)
            status = finalize_episode(episode, sdk_root=sdk_root, progress=progress)
            if status == COMPLETE and spool_archive is not None:
                progress.episode(f"{label}：归档 {SPOOL}")
                if archive_spool(episode, spool_archive):
                    report["archived"].append(str(episode))
        except (OSError, ValueError, KeyError, ImportError, RuntimeError) as error:
            report["failed"] += 1
            report["errors"].append((str(episode), str(error)))
            continue
        finally:
            progress.finish_episode()
        report["complete"] += status == COMPLETE
        report["episodes"].append(str(episode))
    return report
