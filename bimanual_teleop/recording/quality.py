"""Per-episode quality reports shared by dry runs and conversion.

A report holds the frame accounting of a plan and the health of its raw
streams: camera frame rate, drops and gaps, cross-camera offsets, state and
command rates and gaps. Each health value is compared with the {warn, fail}
thresholds of configs/recording_quality.yaml. Levels only annotate a report;
they never change which frames are written.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import math
from pathlib import Path

from bimanual_teleop.common.config import load_yaml_config
from bimanual_teleop.paths import PROJECT_ROOT

from .policy import STRICT
from .schema import CAMERA_FPS, CAMERAS, MAIN_CAMERA, STREAMS
from .timeline import Stream, nearest

DEFAULT_THRESHOLDS = PROJECT_ROOT / "configs/recording_quality.yaml"
OK, WARN, FAIL, SKIP = "OK", "WARN", "FAIL", "SKIP"
LEVELS = (OK, WARN, FAIL, SKIP)
_SEVERITY = {SKIP: 0, OK: 0, WARN: 1, FAIL: 2}
MAIN_CAMERA_GAP_NS = 50_000_000

_LOWER, _UPPER, _VALUE = "lower", "upper", "value"
_SCHEMA = {
    "episode": {"min_duration_s": _LOWER},
    "camera": {"gap_periods": _VALUE, "dropped_frames": _UPPER, "gaps": _UPPER, "max_gap_ms": _UPPER},
    "cross_camera": {"p99_offset_ms": _UPPER},
    "streams": {kind: {"min_hz": _LOWER, "max_gap_ms": _UPPER}
                for kind in ("arms", "hands", "arm_commands", "hand_commands")},
}

# Metadata that changes what recorded values mean. Operator names, network
# addresses, keys and timeouts may differ between merged episodes.
CONSISTENT_METADATA = (
    "model_sha256", "pose_source", "arm_frames", "joint_order", "wrench",
    "state_time", "command_time", "camera_time", "recording.state_hz",
    "tianji_config.profile", "tianji_config.quest",
    "wuji_config.profile_id", "wuji_config.parameters", "wuji_config.control_hz",
)
CAMERA_METADATA = ("serial", "model", "resolution", "fps", "rgb_intrinsics", "image_rotation_deg")
DEPTH_METADATA = ("depth_intrinsics", "depth_to_rgb", "depth_scale")
_MISSING = object()


def _number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _validate(config, schema, where):
    if not isinstance(config, Mapping):
        raise ValueError(f"{where or 'quality thresholds'} must be a mapping")
    if set(config) != set(schema):
        raise ValueError(f"{where or 'quality thresholds'} must have exactly {sorted(schema)}")
    for key, kind in schema.items():
        name, value = f"{where}.{key}" if where else key, config[key]
        if isinstance(kind, dict):
            _validate(value, kind, name)
        elif kind == _VALUE:
            if not _number(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        elif (not isinstance(value, Mapping) or set(value) != {"warn", "fail"}
              or not all(_number(limit) and limit >= 0 for limit in value.values())
              or (value["warn"] > value["fail"] if kind == _UPPER else value["warn"] < value["fail"])):
            order = "warn <= fail" if kind == _UPPER else "warn >= fail"
            raise ValueError(f"{name} must be {{warn, fail}} nonnegative numbers with {order}")


def load_thresholds(config=None):
    """Validated thresholds from a mapping or YAML path; the project defaults for None."""
    if config is None or isinstance(config, (str, Path)):
        config = load_yaml_config(DEFAULT_THRESHOLDS if config is None else config)
    _validate(config, _SCHEMA, "")
    return config


def metadata_fields(include_depth=False):
    cameras = tuple(f"cameras.{camera}.{field}" for camera in CAMERAS for field in CAMERA_METADATA)
    depth = tuple(f"cameras.{MAIN_CAMERA}.{field}" for field in DEPTH_METADATA) if include_depth else ()
    return CONSISTENT_METADATA + cameras + depth


def _lookup(metadata, path):
    value = metadata
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            # Older recordings keep the native orientation and omit this field.
            if path.endswith(".image_rotation_deg"):
                return 0
            return _MISSING
        value = value[key]
    return value


def metadata_differences(reference, metadata, *, include_depth=False):
    """Dotted metadata fields whose values differ from ``reference``."""
    return [path for path in metadata_fields(include_depth)
            if _lookup(reference, path) != _lookup(metadata, path)]


def check(name, value, limits, message, *, lower=False):
    if lower:
        level = FAIL if value < limits["fail"] else WARN if value < limits["warn"] else OK
    else:
        level = FAIL if value >= limits["fail"] else WARN if value >= limits["warn"] else OK
    return {"name": name, "level": level, "value": value, "message": message}


def problem(name, message, level=FAIL):
    return {"name": name, "level": level, "message": message}


def worst(checks):
    return max((item["level"] for item in checks), key=_SEVERITY.__getitem__, default=OK)


def _camera(name, stream, seams, limits):
    """Frame numbers and intervals of one camera; pause seams count as neither drops nor gaps."""
    import numpy as np

    times = stream.times
    stats = {"frames": len(times), "pauses": len(seams),
             "pause_after_rows": [int(row) for row in stream.rows[seams + 1]]}
    if len(times) < 2:
        return stats, [problem(f"{name}/frames", f"{name} 窗口内只有 {len(times)} 帧")]
    continuous = np.ones(len(times) - 1, bool)
    continuous[seams] = False
    steps = np.diff(stream.sequence)[continuous]
    intervals = np.diff(times)[continuous] / 1e6
    stats.update(fps=round((len(times) - 1) / ((times[-1] - times[0]) / 1e9), 3),
                 dropped_frames=int(np.clip(steps - 1, 0, None).sum()),
                 gaps=int(np.count_nonzero(intervals > limits["gap_periods"] * 1e3 / CAMERA_FPS)),
                 max_gap_ms=round(float(intervals.max()) if len(intervals) else 0., 3))
    return stats, [
        check(f"{name}/dropped_frames", stats["dropped_frames"], limits["dropped_frames"],
              f"{name} 丢帧 {stats['dropped_frames']}"),
        check(f"{name}/gaps", stats["gaps"], limits["gaps"], f"{name} 断档 {stats['gaps']} 次"),
        check(f"{name}/max_gap_ms", stats["max_gap_ms"], limits["max_gap_ms"],
              f"{name} 最大帧间隔 {stats['max_gap_ms']:.0f} ms"),
    ]


def _cross_camera(name, stream, main, tolerance_ns, limits):
    """Offsets from each main-camera frame to this camera's nearest frame, independent of the policy grid."""
    import numpy as np

    times = stream.times
    if not len(main) or not len(times):
        return {}, []
    query = main[(main >= times[0]) & (main <= times[-1])]
    if not len(query):
        return {}, [problem(f"{name}/overlap", f"{name} 与 {MAIN_CAMERA} 没有重叠时段")]
    rows, matched = nearest(Stream(times, {}, np.arange(len(times)), len(times)), query, tolerance_ns=tolerance_ns)
    offsets = np.abs(times[rows] - query) / 1e6
    # Percentiles describe the phase of matched frames; unmatched frames are counted on their own.
    phase = offsets[matched] if matched.any() else offsets
    stats = {"p50_offset_ms": round(float(np.percentile(phase, 50)), 3),
             "p99_offset_ms": round(float(np.percentile(phase, 99)), 3),
             "max_offset_ms": round(float(offsets.max()), 3),
             "unmatched_frames": int(np.count_nonzero(~matched))}
    return stats, [check(f"{name}/p99_offset_ms", stats["p99_offset_ms"], limits["p99_offset_ms"],
                         f"{name} 与 {MAIN_CAMERA} 偏差 p99 {stats['p99_offset_ms']:.1f} ms")]


def _stream(name, times, duration_s, limits):
    import numpy as np

    if len(times) < 2:
        return {"samples": len(times)}, [problem(f"{name}/samples", f"{name} 窗口内只有 {len(times)} 个样本")]
    stats = {"samples": len(times), "hz": round(len(times) / duration_s, 3),
             "max_gap_ms": round(float(np.diff(times).max() / 1e6), 3)}
    return stats, [
        check(f"{name}/hz", stats["hz"], limits["min_hz"], f"{name} {stats['hz']:.0f} Hz", lower=True),
        check(f"{name}/max_gap_ms", stats["max_gap_ms"], limits["max_gap_ms"],
              f"{name} 最大间隔 {stats['max_gap_ms']:.0f} ms"),
    ]


def _health(plan, policy, thresholds):
    duration_s = (plan.end_ns - plan.start_ns) / 1e9
    result = {"duration_s": round(duration_s, 3), "cameras": {}, "cross_camera": {}, "streams": {}}
    checks = [check("episode/duration_s", result["duration_s"], thresholds["episode"]["min_duration_s"],
                    f"时长 {duration_s:.1f} s", lower=True)]
    for camera, stream, seams in zip(CAMERAS, plan.cameras, plan.camera_seams):
        result["cameras"][camera], found = _camera(camera, stream, seams, thresholds["camera"])
        checks += found
    pauses = {camera: len(seams) for camera, seams in zip(CAMERAS, plan.camera_seams)}
    if len(set(pauses.values())) > 1:
        checks.append(problem("cameras/pauses", f"三台相机识别出的暂停数不一致：{pauses}"))
    elif pauses[MAIN_CAMERA] and policy.target_fps is None:
        checks.append(problem("cameras/pauses", f"有 {pauses[MAIN_CAMERA]} 处暂停，严格模式不会在暂停处切开；"
                              "应改用默认的修复模式", WARN))
    for camera, stream in zip(CAMERAS[1:], plan.cameras[1:]):
        result["cross_camera"][camera], found = _cross_camera(
            camera, stream, plan.cameras[0].times, policy.camera_tolerance_ns, thresholds["cross_camera"])
        checks += found
    for name in STREAMS:
        limits = thresholds["streams"][name.split("/")[0]]
        result["streams"][name], found = _stream(name, plan.stream_times[name], duration_s, limits)
        checks += found
    result["checks"] = checks
    return result


def episode_report(plan, policy, thresholds):
    """Frame accounting and raw-stream health of one planned episode.

    Invalid frames before the first or after the last valid frame are edge
    trimming, such as the first frame before both state samples exist; the
    rest are interior rejections that split the episode.
    """
    import numpy as np

    valid = plan.valid
    good = np.flatnonzero(valid)
    interior = slice(int(good[0]), int(good[-1]) + 1) if len(good) else slice(0, 0)
    edge = np.ones(len(valid), bool)
    edge[interior] = False
    report = dict(
        reference_frames=len(plan.query), valid_frames=int(valid.sum()), segments=len(plan.runs),
        output_frames=len(plan.keep),
        main_camera_gaps=int(np.count_nonzero(np.diff(plan.cameras[0].times) > MAIN_CAMERA_GAP_NS)),
        invalid_reasons={name: count for name, mask in plan.masks.items()
                         if (count := int(np.count_nonzero(~mask)))},
        edge_trimmed_frames={"start": int(good[0]) if len(good) else len(valid),
                             "end": len(valid) - 1 - int(good[-1]) if len(good) else 0},
        edge_trimmed_reasons={name: count for name, mask in plan.masks.items()
                              if (count := int(np.count_nonzero(~mask & edge)))},
        interior_invalid_frames=int(np.count_nonzero(~valid[interior])),
        interior_invalid_reasons={name: count for name, mask in plan.masks.items()
                                  if (count := int(np.count_nonzero(~mask[interior])))},
        short_segments={"segments": len(plan.short_runs), "frames": sum(map(len, plan.short_runs))})
    if policy.provenance:
        report["pauses"] = plan.pauses
        keep = plan.keep
        if len(keep):
            report["camera_repairs"] = [
                dict(camera=camera, reused_frames=int(reused[keep].sum()),
                     reused_ratio=float(reused[keep].mean()),
                     max_image_age_ms=float(np.maximum(-offsets[keep], 0).max() / 1e6))
                for camera, reused, offsets in zip(CAMERAS, plan.camera_reused, plan.camera_offsets)]
            # Frames kept only because repair interpolates state across gaps strict would reject.
            gaps = plan.state_gap_ns[keep].max(axis=1)
            report["state_repairs"] = dict(frames=int(np.count_nonzero(gaps > STRICT.state_gap_ns)),
                                           beyond_ms=STRICT.state_gap_ns / 1e6, max_gap_ms=float(gaps.max() / 1e6))
    report.update(_health(plan, policy, thresholds))
    return report


def summarize(report):
    """Totals over a conversion report's episodes."""
    episodes = report["episodes"]
    planned = [item for item in episodes if "reference_frames" in item]
    total = lambda key: sum(item[key] for item in planned)
    reasons = Counter()
    for item in planned:
        reasons.update(item["interior_invalid_reasons"])
    reference = total("reference_frames")
    levels = Counter(item["level"] for item in episodes if "level" in item)
    problems = Counter(entry["name"] for item in episodes for entry in item.get("checks", ())
                       if entry["level"] in (WARN, FAIL))
    # Whole episodes left out: by status, review_reject or error.
    skipped = Counter(item.get("skipped") or ("error" if "error" in item else item["status"])
                      for item in episodes if "reference_frames" not in item)
    repairs = None
    if any("pauses" in item for item in planned):
        reused = Counter()
        for item in planned:
            for entry in item.get("camera_repairs", ()):
                reused[entry["camera"]] += entry["reused_frames"]
        pauses = [pause for item in planned for pause in item["pauses"]]
        repairs = {"camera_reused_frames": dict(reused),
                   "state_frames": sum(item.get("state_repairs", {}).get("frames", 0) for item in planned),
                   "pauses_split": sum(not pause["merged"] for pause in pauses),
                   "pauses_merged": sum(pause["merged"] for pause in pauses)}
    return {
        "episodes": len(episodes),
        "status": dict(Counter(item["status"] for item in episodes)),
        "skipped": dict(skipped),
        "repairs": repairs,
        "levels": {level: levels[level] for level in LEVELS if levels[level]},
        "problems": dict(problems.most_common()),
        "errors": {item["source_episode"]: item["error"] for item in episodes if "error" in item},
        "planned_episodes": len(planned),
        "review_rejected": [item["source_episode"] for item in episodes if item.get("skipped") == "review_reject"],
        "without_output": [item["source_episode"] for item in planned if not item["output_frames"]],
        "metadata_differences": {item["source_episode"]: item["metadata_differences"]
                                 for item in episodes if item.get("metadata_differences")},
        "reference_frames": reference,
        "valid_frames": total("valid_frames"),
        "output_frames": total("output_frames"),
        "output_segments": total("segments"),
        "utilization": total("output_frames") / reference if reference else 0.,
        "edge_trimmed_frames": sum(sum(item["edge_trimmed_frames"].values()) for item in planned),
        "interior_invalid_frames": total("interior_invalid_frames"),
        "interior_invalid_reasons": dict(reasons.most_common()),
        "short_segments": sum(item["short_segments"]["segments"] for item in planned),
        "short_segment_frames": sum(item["short_segments"]["frames"] for item in planned),
    }
