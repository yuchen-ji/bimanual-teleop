"""Convert complete raw episodes to the official DP ReplayBuffer layout.

All vectors concatenate left before right. Poses are xyz (m) followed by
rotation vectors (rad); joints are radians and wrench is Fx,Fy,Fz (N),
Tx,Ty,Tz (N m). Actions concatenate both arms, then both hands. Cartesian
actions use the recorded controller input goals, never FK of joint commands.

Alignment and validity come from ``plan.plan_episode`` and reports from
``quality``; this module walks the episodes, writes what each plan selects
and publishes the dataset atomically.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile

from bimanual_teleop.common.console import EpisodeProgress

from .episodes import (COMPLETE, DISCARDED, FAILED, MANIFEST, episode_label, find_episodes,
                       read_manifest, read_phase_markers)
from .plan import STATE_FIELDS, plan_episode
from .policy import load_policy
from .quality import FAIL, SKIP, WARN, episode_report, load_thresholds, metadata_differences, problem, worst
from .review import REJECT, read_review
from .schema import CAMERAS, DEPTH_STREAM, MAIN_CAMERA, SIDES


def _append(data, key, values):
    from numcodecs import Blosc

    if key not in data:
        data.create_dataset(key, shape=(0,) + values.shape[1:], dtype=values.dtype,
                            chunks=(1024,) + values.shape[1:],
                            compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE))
    if data[key].shape[1:] != values.shape[1:]:
        raise ValueError(f"Output shape changed for {key}")
    data[key].append(values, axis=0)


def _phase_marker_records(source, markers, *, times=None, offset=0):
    """Map recording-time markers to insertion boundaries in retained frames."""
    import numpy as np

    return [{"source_episode": source, "index": marker["index"],
             "recording_time_ns": marker["time_ns"],
             "output_boundary": (offset + int(np.searchsorted(times, marker["time_ns"], side="left"))
                                 if times is not None and len(times) else None)}
            for marker in markers]


def _image_array(data, key, total, shape, dtype):
    from numcodecs import Blosc

    if key not in data:
        return data.create_dataset(key, shape=(total,) + tuple(shape), dtype=dtype,
                                   chunks=(1,) + tuple(shape),
                                   compressor=Blosc(cname="lz4", clevel=3, shuffle=Blosc.NOSHUFFLE))
    array = data[key]
    if array.shape[1:] != tuple(shape) or array.dtype != dtype:
        raise ValueError(f"Output image format changed for {key}")
    array.resize((total,) + tuple(shape))
    return array


def _copy_video(path, expected_count, source_rows, data, key, offset, progress):
    import av
    import numpy as np

    destinations = {}
    for i, source_row in enumerate(source_rows):
        destinations.setdefault(int(source_row), []).append(offset + i)
    count = 0
    array = None
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            count += 1
            progress.advance()
            if index not in destinations:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            if array is None:
                array = _image_array(data, key, offset + len(source_rows), rgb.shape, np.dtype("uint8"))
            if rgb.shape != array.shape[1:]:
                raise ValueError(f"Video resolution changes within {path}")
            for target in destinations[index]:
                array[target] = rgb
    if count != expected_count:
        raise ValueError(f"{path}: decoded {count} frames, timestamp table has {expected_count}")


def _write_episode(output, plan, raw, policy, progress):
    import numpy as np

    keep = plan.keep
    data = output["data"]
    offset = data["timestamp"].shape[0] if "timestamp" in data else 0
    for key, values in plan.state.items():
        _append(data, key, values[keep].astype(np.float32))
    _append(data, "action", plan.action[keep].astype(np.float32))
    _append(data, "timestamp", (plan.training_time[keep] - plan.start_ns).astype(np.float64) / 1e9)
    if policy.provenance:
        for key, values in plan.provenance().items():
            _append(output["meta"], key, values[keep])
    counts = [stream.raw_count for stream in plan.cameras]
    progress.total(sum(counts) + (len(keep) if plan.depth_rows is not None else 0))
    for camera, count, rows in zip(CAMERAS, counts, plan.camera_rows):
        _copy_video(plan.path / f"{camera}.mp4", count, rows[keep], data, camera, offset, progress)
    if plan.depth_rows is not None:
        source = raw[f"{DEPTH_STREAM}/image"]
        target = _image_array(data, f"{MAIN_CAMERA}_depth", offset + len(keep), source.shape[1:], source.dtype)
        for i, row in enumerate(plan.depth_rows[keep]):
            target[offset + i] = source[int(row)]
            progress.advance()


def _publish(temporary, destination):
    # Exclusively reserve the name, then atomically replace our empty
    # reservation with the complete directory on the same filesystem.
    destination.mkdir()
    try:
        os.replace(temporary, destination)
    except BaseException:
        try:
            destination.rmdir()  # Never remove contents written by someone else.
        except OSError:
            pass
        raise


def convert_recordings(input_path, output_path=None, *, action_space, include_depth=False, conversion_config=None,
                       quality_config=None, allow_mixed_metadata=False, dry_run=False, progress=None):
    """Create a new dataset; never overwrite an existing file or directory.

    Episode bounds are [start_ns,end_ns). ``conversion_config`` defaults to
    configs/recording_conversion.yaml: repair on a fixed grid with bounded
    image reuse and per-recording-block interpolation, splitting at pauses
    unless a seam was reviewed for merging. ``mode: strict`` instead samples
    real main-camera times with 50 ms state brackets and command age.
    No interpolation, held command or image match crosses an episode boundary.
    Depth is ignored unless explicitly included. episode_ends counts source
    demonstrations; segment_ends marks continuous runs for training sampling.

    Episodes rejected in review.json are skipped. Complete episodes must agree
    on data-defining metadata unless ``allow_mixed_metadata``. Each episode
    gets an OK, WARN, FAIL or SKIP level; planned episodes are checked against
    the ``quality_config`` thresholds (project defaults for None). Levels never
    change what is written.

    ``dry_run`` plans and reports without decoding video or writing, and
    records an episode's error instead of stopping, so one run lists every
    problem that would stop the conversion; ``output_path`` is then unused.
    Returns the quality report, also stored in meta.attrs['quality_report'].
    ``progress`` counts decoded RGB frames plus copied depth frames.
    """
    import numpy as np
    import zarr

    if action_space not in ("eef", "joint"):
        raise ValueError("action_space must be 'eef' or 'joint'")
    if type(include_depth) is not bool:
        raise ValueError("include_depth must be a boolean")
    policy = load_policy(conversion_config)
    thresholds = load_thresholds(quality_config)
    source = Path(input_path).expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"Input is not a directory: {source}")
    if not dry_run:
        if output_path is None:
            raise ValueError("output_path is required unless dry_run")
        destination = Path(output_path).expanduser().absolute()
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {destination}")
    episodes = find_episodes(source)
    if not episodes:
        raise ValueError(f"No episode.json found under {source}")
    progress = EpisodeProgress(enabled=False) if progress is None else progress
    progress.start(len(episodes))
    report = {"episodes": [], "action_space": action_space, "include_depth": include_depth,
              "conversion_config": policy.config, "quality_thresholds": thresholds, "dry_run": dry_run}
    temporary = output = None
    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.converting-", dir=destination.parent))
    try:
        if not dry_run:
            output = zarr.open_group(str(temporary), mode="w")
            output.create_group("data")
            output.create_group("meta")
        segments, episode_ends, phase_markers, offset = [], [], [], 0
        has_depth = reference = None
        for episode in episodes:
            descriptor = read_manifest(episode)
            name = episode_label(episode, source)
            item = {"source_episode": name, "status": descriptor.get("status", "unknown"), "phase_markers": []}
            report["episodes"].append(item)
            try:
                markers = read_phase_markers(descriptor)
                item["phase_markers"] = _phase_marker_records(name, markers)
                if item["status"] != COMPLETE:
                    item["level"] = SKIP if item["status"] in (FAILED, DISCARDED) else WARN
                    item["reason"] = descriptor.get("finalize_error") or descriptor.get("reason") or ""
                    progress.finish_episode()
                    continue
                review = read_review(episode)
                if review is not None:
                    item["review"] = {key: review[key] for key in ("decision", "note") if key in review}
                if (review or {}).get("decision") == REJECT:
                    item.update(skipped="review_reject", level=SKIP)
                    progress.finish_episode()
                    continue
                if descriptor.get("schema_version") != 1:
                    raise ValueError(f"Unsupported raw schema in {episode / MANIFEST}")
                raw = zarr.open_group(str(episode / "raw.zarr"), mode="r")
                this_depth = include_depth and DEPTH_STREAM in raw
                if has_depth is not None and this_depth != has_depth:
                    raise ValueError("Complete episodes must consistently include or omit camera_0 depth")
                has_depth = this_depth
                item["metadata"] = descriptor.get("metadata", raw.attrs.get("metadata", {}))
                if reference is None:
                    reference = name, item["metadata"]
                elif differences := metadata_differences(reference[1], item["metadata"], include_depth=this_depth):
                    item["metadata_differences"] = differences
                    if not allow_mixed_metadata:
                        mixed = (f"{name}: metadata differs from {reference[0]} in {differences}; "
                                 "pass allow_mixed_metadata to merge them anyway")
                        if not dry_run:
                            raise ValueError(mixed)
                        item["error"] = mixed
                progress.episode(name)
                plan = plan_episode(episode, descriptor, raw, policy, source=name, action_space=action_space,
                                    include_depth=include_depth,
                                    pause_reviews=policy.reviews_for(name, (review or {}).get("pause_reviews")))
            except (ValueError, KeyError, OSError) as error:
                if not dry_run:
                    raise
                item.update(error=str(error), level=FAIL)
                progress.finish_episode()
                continue
            item.update(episode_report(plan, policy, thresholds))
            item["phase_markers"] = _phase_marker_records(name, markers, times=plan.query[plan.keep], offset=offset)
            if item.get("metadata_differences") and allow_mixed_metadata:
                item["checks"].append(problem("metadata", f"元数据与 {reference[0]} 不同："
                                              f"{'，'.join(item['metadata_differences'])}", WARN))
            item["level"] = FAIL if "error" in item else worst(item["checks"])
            if plan.runs:
                phase_markers.extend(item["phase_markers"])
                if output is not None:
                    _write_episode(output, plan, raw, policy, progress)
                runs = plan.segment_records(offset)
                segments.extend(runs)
                offset = runs[-1]["output_end"]
                episode_ends.append(offset)
            progress.finish_episode()
        report.update(output_episodes=len(episode_ends), output_segments=len(segments), output_frames=offset)
        if dry_run:
            return report
        if not segments:
            raise ValueError("No valid frames in complete episodes")
        meta = output["meta"]
        meta.create_dataset("episode_ends", data=np.asarray(episode_ends, dtype=np.int64), compressor=None)
        meta.create_dataset("segment_ends", data=np.asarray([segment["output_end"] for segment in segments],
                                                            dtype=np.int64), compressor=None)
        meta.attrs.update(segments=segments, phase_markers=phase_markers, quality_report=report)
        output.attrs.update(schema_version=2, format="diffusion_policy_replay_buffer", action_space=action_space,
                            include_depth=include_depth, episode_ends_semantics="source_demonstrations",
                            sampling_boundaries="meta/segment_ends",
                            side_order=list(SIDES), eef_pose_format="xyz_m+rotvec_rad", joint_unit="rad",
                            wrench_format="Fx,Fy,Fz [N]; Tx,Ty,Tz [N*m]",
                            action_layout="left_arm,right_arm,left_hand,right_hand",
                            **policy.output_attrs())
        if policy.provenance:
            meta.attrs.update(source_time_ns="recording time_ns plus accumulated removed pause duration",
                              state_interpolated_fields=[f"{side}_{key}" for side in SIDES for key, _, _ in STATE_FIELDS],
                              command_age_fields=[f"{side}_{kind}" for side in SIDES for kind in ("arm", "hand")])
        _publish(temporary, destination)
        return report
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
