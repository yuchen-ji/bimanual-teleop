"""Validate capture provenance, solve offline, and export only passed results."""

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .board import CircleBoard, camera_matrix, detect_board, opencv_distortion, projected_points
from .external import Limits, View, solve_external, validation_group
from .geometry import checked_pose
from .robot import base_flange, load_model
from .storage import digest, write_json


def checked_record(row, previous=None):
    stamp, request, joint_stamp = (row[key] for key in
                                  ("external_time_ns", "capture_request_ns", "joint_time_ns"))
    sequence = row["external_frame_number"]
    if (any(type(v) is not int or v < 0 for v in (stamp, request, joint_stamp, sequence))
            or joint_stamp - stamp != row["time_error_ns"]):
        raise ValueError("Sample timestamp metadata is inconsistent")
    if previous is not None and (stamp < previous["external_time_ns"]
                                 or sequence < previous["external_frame_number"]):
        raise ValueError("Image time/frame number moved backwards")


def solve_session(source, output, limits=None, *, model_path=None, sdk_root=None):
    limits = limits or Limits()
    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "external_eye_to_hand_result_v2", "method": "eye_to_hand",
              "created_at": datetime.now().astimezone().isoformat(), "passed": False,
              "errors": [], "limits": asdict(limits)}
    try:
        session = json.loads((source / "session.json").read_text())
        if (session.get("schema") not in ("external_eye_to_hand_capture_v2", "external_eye_to_hand_capture_v3")
                or session.get("status") != "captured" or session.get("method") != "eye_to_hand"
                or session.get("board_mount") != "rigid_on_selected_flange"
                or session.get("capture_mode") != "manual"
                or session.get("images") != "raw RGB sensor orientation"):
            raise ValueError("Expected a finished manual eye-to-hand capture; old wrist-reference captures are unsupported")
        side = session["side"]
        if side not in ("left", "right") or session["reference_frame"] != f"tianji_{side}_base":
            raise ValueError("Capture must identify the selected arm's native base frame")
        model = load_model(model_path, session["robot_model_sha256"], sdk_root)
        board, camera = CircleBoard(**session["board"]), session["cameras"]["external"]
        records = [json.loads(line) for line in (source / "samples.jsonl").read_text().splitlines() if line]
        groups = [row["group"] for row in records]
        if any(type(g) is not int or g < 0 for g in groups) or len(set(groups)) != len(groups):
            raise ValueError("Capture group IDs must be unique nonnegative integers")
        if session["counts"]["groups"] != len(records):
            raise ValueError("Session count differs from the sample ledger")
        views, observations, images = [], [], {}
        for index, row in enumerate(records):
            observation = {"group": row["group"], "validation": validation_group(row["group"])}
            try:
                checked_record(row, records[index - 1] if index else None)
                flange = base_flange(model, side, row["joints_rad"])
                if not np.allclose(flange, checked_pose(row["T_base_flange"]), atol=1e-7, rtol=0):
                    raise ValueError("Saved flange pose disagrees with measured-joint FK")
                path = (source / row["external_image"]).resolve()
                path.relative_to(source)
                if digest(path) != row["external_sha256"]:
                    raise ValueError("External image hash changed")
                rgb = np.asarray(Image.open(path).convert("RGB"))
                if list(rgb.shape[1::-1]) != camera["resolution"]:
                    raise ValueError("Image resolution differs from its raw intrinsics")
                pixels = detect_board(rgb, board)
                if pixels is None:
                    raise ValueError("All 20 circles must be visible")
                images[row["group"]] = path
            except (ValueError, TypeError, KeyError, OSError, cv2.error) as error:
                observation["error"] = str(error)
                # Keep failed observations in their original split and quota.
                flange, pixels = np.eye(4), np.full((20, 2), np.nan)
            views.append(View(flange, pixels, row["group"]))
            observations.append(observation)
        report.update(synthetic=bool(session.get("synthetic", False)), side=side, camera=camera,
                      reference_frame=f"tianji_{side}_base", board=session["board"],
                      robot_model_sha256=model.model.digest,
                      session_sha256=digest(source / "session.json"),
                      samples_sha256=digest(source / "samples.jsonl"), observations=observations)
        camera_pose, mount, quality = solve_external(views, camera, board, limits)
        report.update(passed=quality["passed"], T_base_camera_opencv=camera_pose.tolist(),
                      T_flange_board=mount.tolist(), quality=quality)
        if not quality["passed"]:
            report["errors"].extend(quality["failure_reasons"])
        lens = (camera_matrix(camera), opencv_distortion(camera)[0])
        for view in views:
            if not validation_group(view.group):
                continue
            predicted = projected_points(board.points(), np.linalg.inv(camera_pose) @ view.base_flange @ mount, *lens)
            image = np.array(Image.open(images[view.group]).convert("RGB"))
            for seen, projected in zip(view.external_pixels, predicted):
                cv2.circle(image, tuple(np.round(seen).astype(int)), 4, (0, 255, 0), 1)
                cv2.drawMarker(image, tuple(np.round(projected).astype(int)), (255, 0, 255), cv2.MARKER_CROSS, 7, 1)
            Image.fromarray(image).save(output / f"validation_{view.group:04d}.png")
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, cv2.error) as error:
        report["passed"] = False
        report["errors"].append(str(error))
    write_json(output / "result.json", report)
    summary = ["# 主视角眼在手外标定报告", "", f"验收：{'通过' if report['passed'] else '未通过'}", ""]
    if "quality" in report:
        summary.extend(f"- {key}: {value:.6f}" for key, value in report["quality"]["validation"].items())
    summary.extend(["", "结果相对于所选臂原生基座，单位为米。像素误差及位姿一致性不代表绝对定位精度。",
                    *report["errors"]])
    (output / "report.md").write_text("\n".join(summary) + "\n")
    return report


def export_result(result_file, output, world_from_base=None):
    report = json.loads(Path(result_file).read_text())
    if (report.get("schema") != "external_eye_to_hand_result_v2" or report.get("passed") is not True
            or report.get("synthetic") or report.get("quality", {}).get("passed") is not True):
        raise ValueError("Only measured eye-to-hand results that passed validation can be exported")
    camera = checked_pose(report["T_base_camera_opencv"])
    checked_pose(report["T_flange_board"])
    payload = {key: report[key] for key in ("camera", "side", "reference_frame", "method", "board",
                                          "T_base_camera_opencv", "T_flange_board", "robot_model_sha256")}
    payload.update(schema="camera_extrinsics_v2", source_result_sha256=digest(result_file),
                   validation=report["quality"]["validation"], uncertainty=report["quality"]["uncertainty"])
    if world_from_base is not None:
        world = checked_pose(json.loads(Path(world_from_base).read_text()))
        payload.update(T_world_robot_base=world.tolist(), T_world_camera_opencv=(world @ camera).tolist())
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() in (".yaml", ".yml"):
        import yaml
        raw = yaml.safe_dump(payload, sort_keys=False)
    else:
        raw = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    with output.open("x") as stream:
        stream.write(raw)
    return output
