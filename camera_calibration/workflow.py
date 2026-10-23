"""Offline wrist solving and reprojection validation; opens no devices."""

import json
import shutil
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
from .geometry import transform
from .storage import digest
from .wrist import solve_handeye


def output_lens(metadata):
    matrix = camera_matrix(metadata)
    coefficients, error = opencv_distortion(metadata)
    intrinsics = {
        "fx": float(matrix[0, 0]),
        "fy": float(matrix[1, 1]),
        "cx": float(matrix[0, 2]),
        "cy": float(matrix[1, 2]),
    }
    angle = metadata["output_rotation_deg"]
    if angle == 180:
        width, height = metadata["resolution"]
        intrinsics.update(
            cx=width - 1 - intrinsics["cx"], cy=height - 1 - intrinsics["cy"]
        )
        coefficients = coefficients.copy()
        coefficients[2:4] *= -1
    elif angle != 0:
        raise ValueError(
            "Only unrotated and 180 degree video orientations are supported"
        )
    return intrinsics, coefficients.tolist(), error


def solve_session(source, output):
    from bimanual_teleop.sim.config import load_config, sim_path
    from bimanual_teleop.sim.kinematics import RobotModel

    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    session = json.loads((source / "session.json").read_text())
    if session.get("schema") != "wrist_handeye_capture_v1" or not session.get(
        "stationary_only"
    ):
        raise ValueError("Expected a stationary, read-only wrist calibration capture")
    if session.get("status") != "captured":
        raise ValueError("Capture did not finish successfully")
    board = CircleBoard(**session["board"])
    objects = board.points()
    records = [
        json.loads(line)
        for line in (source / "samples.jsonl").read_text().splitlines()
        if line
    ]
    manifest_path = source / "source_manifest.json"
    source_manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    )
    origins = {row["image"]: row for row in source_manifest.get("samples", [])}
    config = load_config()
    urdf = sim_path(config["robot"]["urdf"])
    model = RobotModel(urdf, config["robot"]["world_yaw_deg"])
    report = {
        "schema": "wrist_handeye_result_v1",
        "created_at": datetime.now().isoformat(),
        "passed": False,
        "synthetic": bool(session.get("synthetic", False)),
        "source_session": str(source),
        "session_sha256": digest(source / "session.json"),
        "samples_sha256": digest(source / "samples.jsonl"),
        "urdf_sha256": digest(urdf),
        "board": session["board"],
        "pose_source": "URDF FK of measured stationary joints, world -> flange",
        "capture_provenance": {
            "source": session.get("source", "local read-only stationary capture"),
            "assembled_from": session.get("assembled_from", {}),
            "source_manifest_sha256": digest(manifest_path)
            if manifest_path.exists()
            else None,
        },
        "observations": {},
        "calibrations": {},
        "errors": {},
    }
    for side, metadata in session["cameras"].items():
        try:
            if metadata["resolution"] != [
                config["cameras"]["width"],
                config["cameras"]["height"],
            ]:
                raise ValueError("Capture and simulation image dimensions differ")
            grippers, boards, image_points, accepted, rejected = [], [], [], [], []
            K = camera_matrix(metadata)
            distortion, lens_error = opencv_distortion(metadata)
            for record in [row for row in records if row["side"] == side]:
                try:
                    if (
                        abs(record["time_error_ns"]) > 25_000_000
                        or max(record["joint_window_span_rad"]) > 0.002
                    ):
                        raise ValueError("Sample is moving or not synchronized")
                    image_path = (source / record["image"]).resolve()
                    image_path.relative_to(source)
                    image_digest = digest(image_path)
                    origin = origins.get(record["image"], {})
                    if origin and image_digest != origin["sha256"]:
                        raise ValueError(
                            "Image hash differs from the original capture manifest"
                        )
                    rgb = np.array(Image.open(image_path).convert("RGB"))
                    centers = detect_board(rgb, board)
                    if centers is None:
                        raise ValueError("All 20 circles could not be detected")
                    camera_board, quality = estimate_board_pose(
                        centers, metadata, board
                    )
                    q = np.asarray(record["joints_rad"], dtype=float)
                    if q.shape != (7,) or not np.isfinite(q).all():
                        raise ValueError("Invalid measured joints")
                    full = np.zeros(54)
                    offset = 0 if side == "left" else 7
                    full[offset : offset + 7] = q
                    if np.any(
                        full[offset : offset + 7]
                        < model.lower[offset : offset + 7] - 0.01
                    ) or np.any(
                        full[offset : offset + 7]
                        > model.upper[offset : offset + 7] + 0.01
                    ):
                        raise ValueError(
                            "Measured joints exceed the prepared URDF limits"
                        )
                    grippers.append(model.flange(side, full, world=True))
                    boards.append(camera_board)
                    image_points.append(centers)
                    accepted.append(
                        {
                            **record,
                            **quality,
                            "image_sha256": image_digest,
                            "source_session": origin.get("source_session"),
                            "source_index": origin.get("source_index"),
                        }
                    )
                except (ValueError, OSError) as error:
                    rejected.append(
                        {"image": record.get("image"), "reason": str(error)}
                    )
            report["observations"][side] = {
                "accepted_samples": len(accepted),
                "pnp_rejected": rejected,
            }
            handeye, target, quality = solve_handeye(grippers, boards)
            pixel_errors = []
            for gripper, pixels in zip(grippers, image_points):
                predicted = projected_points(
                    objects, np.linalg.inv(gripper @ handeye) @ target, K, distortion
                )
                pixel_errors.append(
                    float(np.sqrt(np.mean(np.sum((predicted - pixels) ** 2, axis=1))))
                )
            held_out = quality["held_out_indices"]
            quality.update(
                reprojection_rms_px=pixel_errors,
                held_out_max_reprojection_px=max(pixel_errors[i] for i in held_out),
                held_out_median_reprojection_px=float(
                    np.median([pixel_errors[i] for i in held_out])
                ),
                pnp_rejected=rejected,
                accepted_samples=len(accepted),
            )
            quality["passed"] &= (
                quality["held_out_max_reprojection_px"] < 4
                and quality["held_out_median_reprojection_px"] < 2
            )
            zero = np.zeros(54)
            prefix = "l" if side == "left" else "r"
            bracket_flange = np.linalg.inv(
                model.link_pose(prefix + "_camera_bracket", zero)
            ) @ model.flange(side, zero, world=True)
            bracket_sensor = bracket_flange @ handeye
            sensor_output = transform(
                Rotation.from_euler(
                    "z", metadata["output_rotation_deg"], degrees=True
                ).as_matrix(),
                [0, 0, 0],
            )
            bracket_output = bracket_sensor @ sensor_output
            intrinsics, output_distortion, _ = output_lens(metadata)
            wrist_settings = {
                "intrinsics": intrinsics,
                "distortion_coefficients": output_distortion,
                "pose_in_bracket": {
                    "position_m": bracket_output[:3, 3].tolist(),
                    "quaternion_xyzw": Rotation.from_matrix(bracket_output[:3, :3])
                    .as_quat()
                    .tolist(),
                },
            }
            reference_index = min(held_out, key=lambda index: pixel_errors[index])
            reference = accepted[reference_index]
            shutil.copy2(source / reference["image"], output / f"{side}_reference.png")
            for index in held_out:
                rgb = np.array(
                    Image.open(source / accepted[index]["image"]).convert("RGB")
                )
                predicted = projected_points(
                    objects,
                    np.linalg.inv(grippers[index] @ handeye) @ target,
                    K,
                    distortion,
                )
                for measured, reprojection in zip(image_points[index], predicted):
                    cv2.circle(
                        rgb, tuple(np.round(measured).astype(int)), 4, (0, 255, 0), 1
                    )
                    cv2.drawMarker(
                        rgb,
                        tuple(np.round(reprojection).astype(int)),
                        (255, 0, 255),
                        cv2.MARKER_CROSS,
                        7,
                        1,
                    )
                Image.fromarray(rgb).save(output / f"{side}_heldout_{index:03d}.png")
            report["calibrations"][side] = {
                "passed": bool(quality["passed"]),
                "camera": metadata,
                "T_flange_sensor_opencv": handeye.tolist(),
                "T_world_board": target.tolist(),
                "T_bracket_output_opencv": bracket_output.tolist(),
                "settings": wrist_settings,
                "lens_conversion_max_error_px": lens_error,
                "quality": quality,
                "reference": {
                    **reference,
                    "index": reference_index,
                    "image": f"{side}_reference.png",
                },
                "samples": accepted,
            }
        except (ValueError, cv2.error, OSError) as error:
            report["errors"][side] = str(error)
    report["passed"] = (
        not report["errors"]
        and bool(report["calibrations"])
        and all(value["passed"] for value in report["calibrations"].values())
    )
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
