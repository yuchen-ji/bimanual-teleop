"""File-based external-camera solve, diagnostics and explicit result export."""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .board import (
    CircleBoard,
    camera_matrix,
    detect_board,
    estimate_board_pose,
    opencv_distortion,
    projected_points,
)
from .external import Limits, View, solve_external, validation_group
from .geometry import checked_pose
from .robot import load_model, world_flange
from .storage import digest, load_wrist_reference, write_json


def check_wrist_camera(actual, reference):
    if (
        actual["serial"] != reference["serial"]
        or actual["resolution"] != reference["resolution"]
    ):
        raise ValueError("Wrist serial/resolution differs from the hand-eye reference")
    a, b = actual["rgb_intrinsics"], reference["rgb_intrinsics"]
    if (
        a["model"] != b["model"]
        or not np.allclose(camera_matrix(actual), camera_matrix(reference), atol=1e-7)
        or not np.allclose(a["coeffs"], b["coeffs"], atol=1e-7)
    ):
        raise ValueError("Wrist lens differs from the calibrated reference")


def solve_session(source, wrist_result, output, limits=None):
    limits = limits or Limits()
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "external_camera_result_v1",
        "created_at": datetime.now().isoformat(),
        "passed": False,
        "errors": [],
        "limits": asdict(limits),
    }
    try:
        session = json.loads((source / "session.json").read_text())
        if (
            session.get("schema") != "external_camera_capture_v1"
            or session.get("status") != "captured"
            or not session.get("stationary_only")
            or session.get("images") != "raw RGB sensor orientation"
        ):
            raise ValueError(
                "Expected a finished stationary external-camera capture with raw RGB images"
            )
        side = session["side"]
        reference, wrist = load_wrist_reference(wrist_result, side)
        if session["wrist_calibration_sha256"] != digest(wrist_result):
            raise ValueError(
                "Capture and solve must use exactly the same wrist calibration"
            )
        board = CircleBoard(**session["board"])
        if session["board"] != reference["board"]:
            raise ValueError("Board dimensions differ from the wrist calibration")
        model, urdf_hash = load_model(reference["urdf_sha256"])
        cameras = session["cameras"]
        check_wrist_camera(cameras["wrist"], wrist["camera"])
        if cameras["external"]["serial"] == cameras["wrist"]["serial"]:
            raise ValueError("Two different cameras are required")
        records = [
            json.loads(line)
            for line in (source / "samples.jsonl").read_text().splitlines()
            if line
        ]
        groups = [row["group"] for row in records]
        if any(type(g) is not int or g < 0 for g in groups) or len(set(groups)) != len(
            groups
        ):
            raise ValueError("Capture group IDs must be unique nonnegative integers")
        handeye = checked_pose(wrist["T_flange_sensor_opencv"])
        views, observations = [], []
        for row in records:
            observation = {
                "group": row["group"],
                "validation": validation_group(row["group"]),
            }
            try:
                if (
                    not np.isfinite(
                        [
                            row["time_error_ns"],
                            row["pair_time_error_ns"],
                            *row["joint_window_span_rad"],
                        ]
                    ).all()
                    or abs(row["time_error_ns"]) > 25_000_000
                    or abs(row["pair_time_error_ns"]) > 25_000_000
                    or len(row["joint_window_span_rad"]) != 7
                    or min(row["joint_window_span_rad"]) < 0
                    or max(row["joint_window_span_rad"]) > 0.002
                    or row["external_time_ns"] - row["wrist_time_ns"]
                    != row["pair_time_error_ns"]
                    or row["joint_time_ns"] - row["wrist_time_ns"]
                    != row["time_error_ns"]
                ):
                    raise ValueError("Sample is moving or timestamps do not correspond")
                world_wrist = world_flange(model, side, row["joints_rad"]) @ handeye
                pixels = {}
                for name in ("wrist", "external"):
                    path = (source / row[f"{name}_image"]).resolve()
                    path.relative_to(source)
                    if digest(path) != row[f"{name}_sha256"]:
                        raise ValueError(f"{name} image hash changed")
                    rgb = np.array(Image.open(path).convert("RGB"))
                    if list(rgb.shape[1::-1]) != cameras[name]["resolution"]:
                        raise ValueError(
                            f"{name} image resolution differs from its intrinsics"
                        )
                    pixels[name] = detect_board(rgb, board)
                    if pixels[name] is None:
                        raise ValueError(f"{name}: all 20 circles must be visible")
                views.append(
                    View(world_wrist, pixels["wrist"], pixels["external"], row["group"])
                )
            except (ValueError, TypeError, OSError, KeyError, cv2.error) as error:
                observation["error"] = str(error)
                # Keep the original group and split. The solver rejects invalid
                # training data within its quota; invalid validation data fails.
                missing = np.full((20, 2), np.nan)
                views.append(View(np.eye(4), missing, missing, row["group"]))
            observations.append(observation)
        report.update(
            synthetic=bool(session.get("synthetic", False)),
            side=side,
            camera=cameras["external"],
            reference_frame="project_world",
            robot_base_frame=f"RobotModel.base_pose({side})",
            urdf_sha256=urdf_hash,
            wrist_calibration_sha256=digest(wrist_result),
            session_sha256=digest(source / "session.json"),
            samples_sha256=digest(source / "samples.jsonl"),
            board=session["board"],
            observations=observations,
        )
        camera, quality = solve_external(
            views, cameras["wrist"], cameras["external"], board, limits
        )
        world_base = model.base_pose(side)
        report.update(
            passed=quality["passed"],
            T_world_camera_opencv=camera.tolist(),
            T_base_camera_opencv=(np.linalg.inv(world_base) @ camera).tolist(),
            T_world_robot_base=world_base.tolist(),
            quality=quality,
        )
        if not quality["passed"]:
            report["errors"].append(
                "Independent validation exceeds the configured limits"
            )
        for view, row in zip(views, records):
            if not validation_group(view.group):
                continue
            wrist_board, _ = estimate_board_pose(
                view.wrist_pixels, cameras["wrist"], board
            )
            predicted = projected_points(
                board.points(),
                np.linalg.inv(camera) @ view.world_wrist @ wrist_board,
                camera_matrix(cameras["external"]),
                opencv_distortion(cameras["external"])[0],
            )
            image = np.array(Image.open(source / row["external_image"]).convert("RGB"))
            for seen, projected in zip(view.external_pixels, predicted):
                cv2.circle(image, tuple(np.round(seen).astype(int)), 4, (0, 255, 0), 1)
                cv2.drawMarker(
                    image,
                    tuple(np.round(projected).astype(int)),
                    (255, 0, 255),
                    cv2.MARKER_CROSS,
                    7,
                    1,
                )
            Image.fromarray(image).save(output / f"validation_{view.group:04d}.png")
    except (ValueError, TypeError, KeyError, OSError, cv2.error) as error:
        report["passed"] = False
        report["errors"].append(str(error))
    write_json(output / "result.json", report)
    summary = [
        "# 主视角（第三视角）标定报告",
        "",
        f"验收：{'通过' if report['passed'] else '未通过'}",
        "",
    ]
    if "quality" in report:
        for key, value in report["quality"]["validation"].items():
            summary.append(f"- {key}: {value:.6f}")
    summary.extend(
        ["", "像素误差及位姿一致性不代表机器人绝对定位精度。", *report["errors"]]
    )
    (output / "report.md").write_text("\n".join(summary) + "\n")
    return report


def export_result(result_file, output, simulation=False):
    report = json.loads(Path(result_file).read_text())
    if (
        report.get("schema") != "external_camera_result_v1"
        or not report.get("passed")
        or report.get("synthetic")
        or not report.get("quality", {}).get("passed")
    ):
        raise ValueError(
            "Only measured results that passed independent validation can be exported"
        )
    camera = checked_pose(report["T_world_camera_opencv"])
    checked_pose(report["T_base_camera_opencv"])
    payload = {
        key: report[key]
        for key in (
            "schema",
            "camera",
            "side",
            "reference_frame",
            "robot_base_frame",
            "T_world_camera_opencv",
            "T_base_camera_opencv",
            "T_world_robot_base",
            "urdf_sha256",
            "wrist_calibration_sha256",
        )
    }
    payload["validation"] = report["quality"]["validation"]
    payload["uncertainty"] = report["quality"]["uncertainty"]
    payload["schema"] = "camera_extrinsics_v1"
    payload["source_result_sha256"] = digest(result_file)
    if simulation:
        from bimanual_teleop.sim.config import load_config

        load_model(report["urdf_sha256"])
        config = load_config()
        meta = report["camera"]
        if meta["resolution"] != [
            config["cameras"]["width"],
            config["cameras"]["height"],
        ]:
            raise ValueError(
                "Camera resolution differs from simulation; export a generic result instead"
            )
        matrix = camera_matrix(meta)
        payload = {
            "cameras": {
                "head": {
                    "position_world_m": camera[:3, 3].tolist(),
                    "quaternion_xyzw": Rotation.from_matrix(camera[:3, :3])
                    .as_quat()
                    .tolist(),
                    "intrinsics": {
                        "fx": float(matrix[0, 0]),
                        "fy": float(matrix[1, 1]),
                        "cx": float(matrix[0, 2]),
                        "cy": float(matrix[1, 2]),
                    },
                    "distortion_coefficients": opencv_distortion(meta)[0].tolist(),
                }
            }
        }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".json" and not simulation:
        raw = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    else:
        import yaml

        raw = (
            "# Explicit camera calibration export; OpenCV optical coordinates, meters.\n"
            + yaml.safe_dump(payload, sort_keys=False)
        )
    with output.open("x") as stream:
        stream.write(raw)
    return output
