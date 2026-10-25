"""Offline native-FK wrist solve; no robot or camera connection is created."""

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .board import CircleBoard, camera_matrix, detect_board, opencv_distortion, projected_points
from .external import Limits, validation_group
from .external_workflow import checked_record
from .geometry import checked_pose, transform
from .robot import base_flange, load_model
from .storage import digest, write_json
from .wrist import WristView, solve_wrist


def solve_session(source, output, limits=None, *, model_path=None, sdk_root=None):
    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().resolve()
    limits = limits or Limits()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "wrist_handeye_result_v2", "method": "eye_in_hand", "passed": False,
              "created_at": datetime.now().astimezone().isoformat(), "calibrations": {},
              "observations": {}, "errors": {}, "limits": asdict(limits)}
    try:
        session = json.loads((source / "session.json").read_text())
        if (session.get("schema") != "wrist_handeye_capture_v2" or session.get("status") != "captured"
                or session.get("capture_mode") != "manual" or session.get("method") != "eye_in_hand"
                or session.get("board_mount") != "fixed_in_environment"
                or session.get("images") != "raw RGB sensor orientation"):
            raise ValueError("Expected a finished manual wrist capture with a fixed board and raw sensor images")
        model = load_model(model_path, session["robot_model_sha256"], sdk_root)
        board = CircleBoard(**session["board"])
        records = [json.loads(line) for line in (source / "samples.jsonl").read_text().splitlines() if line]
        report.update(robot_model_sha256=model.model.digest, board=session["board"],
                      synthetic=bool(session.get("synthetic", False)),
                      session_sha256=digest(source / "session.json"), samples_sha256=digest(source / "samples.jsonl"))
        if not session["cameras"] or any(side not in ("left", "right") for side in session["cameras"]):
            raise ValueError("Capture must identify left/right wrist cameras")
        for side, camera in session["cameras"].items():
            try:
                rows = [row for row in records if row["side"] == side]
                groups = [row["group"] for row in rows]
                if (any(type(g) is not int or g < 0 for g in groups) or len(set(groups)) != len(groups)
                        or session["counts"][side] != len(rows)):
                    raise ValueError("Wrist sample groups/counts do not match the session")
                views, observations, paths = [], [], {}
                for row in rows:
                    observation = {"group": row["group"], "validation": validation_group(row["group"])}
                    try:
                        checked_record(dict(row, external_time_ns=row["image_time_ns"],
                                            external_frame_number=row["image_frame_number"]))
                        flange = base_flange(model, side, row["joints_rad"])
                        if not np.allclose(flange, checked_pose(row["T_base_flange"]), atol=1e-7, rtol=0):
                            raise ValueError("Saved flange pose disagrees with measured-joint FK")
                        path = (source / row["image"]).resolve()
                        path.relative_to(source)
                        if digest(path) != row["image_sha256"]:
                            raise ValueError("Wrist image hash changed")
                        rgb = np.asarray(Image.open(path).convert("RGB"))
                        if list(rgb.shape[1::-1]) != camera["resolution"]:
                            raise ValueError("Image resolution differs from raw intrinsics")
                        pixels = detect_board(rgb, board)
                        if pixels is None:
                            raise ValueError("All 20 circles must be visible")
                        paths[row["group"]] = path
                    except (ValueError, TypeError, KeyError, OSError, cv2.error) as error:
                        observation["error"] = str(error)
                        flange, pixels = np.eye(4), np.full((20, 2), np.nan)
                    views.append(WristView(flange, pixels, row["group"]))
                    observations.append(observation)
                report["observations"][side] = observations
                handeye, fixed_board, quality = solve_wrist(views, camera, board, limits)
                angle = camera.get("output_rotation_deg", 0)
                if angle not in (0, 180):
                    raise ValueError("Only raw or 180 degree display orientation is supported")
                output_rotation = transform(Rotation.from_euler("z", angle, degrees=True).as_matrix(), [0, 0, 0])
                report["calibrations"][side] = {
                    "passed": quality["passed"], "camera": camera, "reference_frame": f"tianji_{side}_flange",
                    "T_flange_sensor_opencv": handeye.tolist(), "T_base_board": fixed_board.tolist(),
                    "T_flange_output_opencv": (handeye @ output_rotation).tolist(), "quality": quality}
                if not quality["passed"]:
                    report["errors"][side] = "; ".join(quality["failure_reasons"])
                lens = (camera_matrix(camera), opencv_distortion(camera)[0])
                for view in views:
                    if not validation_group(view.group):
                        continue
                    prediction = projected_points(board.points(), np.linalg.inv(view.base_flange @ handeye) @ fixed_board, *lens)
                    rgb = np.array(Image.open(paths[view.group]).convert("RGB"))
                    for seen, predicted in zip(view.pixels, prediction):
                        cv2.circle(rgb, tuple(np.round(seen).astype(int)), 4, (0, 255, 0), 1)
                        cv2.drawMarker(rgb, tuple(np.round(predicted).astype(int)), (255, 0, 255), cv2.MARKER_CROSS, 7, 1)
                    Image.fromarray(rgb).save(output / f"{side}_validation_{view.group:04d}.png")
            except (ValueError, TypeError, KeyError, RuntimeError, OSError, cv2.error) as error:
                report["errors"][side] = str(error)
        report["passed"] = bool(report["calibrations"]) and not report["errors"]
    except (ValueError, TypeError, KeyError, RuntimeError, OSError, cv2.error) as error:
        report["errors"]["session"] = str(error)
    write_json(output / "result.json", report)
    summary = ["# 腕部相机手眼标定报告", "", f"验收：{'通过' if report['passed'] else '未通过'}", ""]
    if not report["passed"] and report["calibrations"]:
        summary.extend(["已保存候选外参和逐组误差；当前结果未通过验收，不能导出。", ""])
    for side, item in report["calibrations"].items():
        quality = item["quality"]
        summary.extend([f"{side}：{json.dumps(quality['validation'], ensure_ascii=False)}", "",
                        "| 组号 | 用途 | 单张 PnP RMS (px) | 联合 RMS (px) | 平移差 (mm) | 旋转差 (°) | 备注 |",
                        "| --- | --- | --- | --- | --- | --- | --- |"])
        for row in report["observations"][side]:
            group = str(row["group"])
            pnp = quality["per_group_pnp"].get(group, {}).get("pnp_rms_px")
            values = quality["per_group_rms_px_m_deg"].get(group)
            pnp_text = f"{pnp:.3f}" if pnp is not None else "—"
            numbers = f"{values[0]:.3f} | {values[1] * 1000:.2f} | {values[2]:.3f}" if values else "— | — | —"
            reason = (row.get("error") or quality["rejected_train_groups"].get(group)
                      or quality["initial_inconsistent_train_groups"].get(group, ""))
            summary.append(f"| {group} | {'验证' if row['validation'] else '训练'} | {pnp_text} | {numbers} | {reason} |")
        summary.append("")
    summary.extend(f"{side}：{message}" for side, message in report["errors"].items())
    summary.append("结果为所选臂法兰到相机的变换，平移单位为米。")
    (output / "report.md").write_text("\n".join(summary) + "\n")
    return report


def export_result(result_file, output):
    report = json.loads(Path(result_file).read_text())
    if (report.get("schema") != "wrist_handeye_result_v2" or report.get("passed") is not True
            or report.get("synthetic") or not report.get("calibrations")):
        raise ValueError("Only measured wrist results that passed validation can be exported")
    for item in report["calibrations"].values():
        if item.get("passed") is not True or item.get("quality", {}).get("passed") is not True:
            raise ValueError("Every exported wrist calibration must pass validation")
        checked_pose(item["T_flange_sensor_opencv"])
        checked_pose(item["T_flange_output_opencv"])
    payload = {"schema": "wrist_camera_extrinsics_v2", "robot_model_sha256": report["robot_model_sha256"],
               "source_result_sha256": digest(result_file), "calibrations": report["calibrations"]}
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return output
