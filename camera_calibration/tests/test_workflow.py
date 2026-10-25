"""Raw images, file integrity, native FK, and measured-only result export."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from camera_calibration.board import CircleBoard, camera_matrix, projected_points
from camera_calibration.external_workflow import checked_record, export_result, solve_session
from camera_calibration.geometry import pose_errors
from camera_calibration.robot import base_flange, load_model
from camera_calibration.storage import digest, write_json
from camera_calibration.tests.test_external import scene


def render_board(pose, camera):
    image = np.full((1440, 1920, 3), 30, dtype=np.uint8)
    theta = np.linspace(0, 2 * np.pi, 80)
    for point in CircleBoard().points():
        circle = point + np.c_[.012 * np.cos(theta), .012 * np.sin(theta), np.zeros(len(theta))]
        pixels = projected_points(circle, pose, camera_matrix(camera), camera["rgb_intrinsics"]["coeffs"])
        cv2.fillPoly(image, [np.round(pixels * 3).astype(np.int32)], (220, 220, 220))
    return cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)


def create_capture(root):
    views, camera, truth, mount = scene()
    (root / "external").mkdir(parents=True)
    session = {"schema": "external_eye_to_hand_capture_v3", "method": "eye_to_hand",
               "status": "captured", "board_mount": "rigid_on_selected_flange", "stationary_confirmation": "operator",
               "capture_mode": "manual", "images": "raw RGB sensor orientation", "side": "left",
               "reference_frame": "tianji_left_base", "robot_model_sha256": "test-model",
               "board": asdict(CircleBoard()), "cameras": {"external": camera},
               "counts": {"groups": len(views)}, "synthetic": True}
    records = []
    for view in views:
        pose = np.linalg.inv(truth) @ view.base_flange @ mount
        path = root / "external" / f"{view.group:04d}.png"
        Image.fromarray(render_board(pose, camera)).save(path)
        timestamp = (view.group + 1) * 1_000_000_000
        records.append({"group": view.group, "joints_rad": [float(view.group)] * 7,
                        "T_base_flange": view.base_flange.tolist(),
                        "capture_request_ns": timestamp - 20_000_000, "external_time_ns": timestamp,
                        "joint_time_ns": timestamp, "time_error_ns": 0,
                        "external_frame_number": view.group + 100,
                        "external_image": str(path.relative_to(root)), "external_sha256": digest(path)})
    write_json(root / "session.json", session)
    (root / "samples.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    return views, truth, session, records


class WorkflowTests(unittest.TestCase):
    def test_raw_images_offline_solve_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            views, truth, session, records = create_capture(root / "capture")
            with patch("camera_calibration.external_workflow.load_model",
                       return_value=SimpleNamespace(model=SimpleNamespace(digest="test-model"))), \
                    patch("camera_calibration.external_workflow.base_flange",
                          side_effect=lambda model, side, q: views[int(q[0])].base_flange):
                report = solve_session(root / "capture", root / "solve")
            self.assertTrue(report["passed"], report["errors"])
            self.assertLess(pose_errors(np.array(report["T_base_camera_opencv"]), truth)[0], .006)
            self.assertEqual(len(list((root / "solve").glob("validation_*.png"))), 5)
            with self.assertRaisesRegex(ValueError, "measured"):
                export_result(root / "solve/result.json", root / "out.json")
            report["synthetic"] = False
            write_json(root / "solve/result.json", report)
            export_result(root / "solve/result.json", root / "out.json")
            exported = json.loads((root / "out.json").read_text())
            self.assertEqual(exported["reference_frame"], "tianji_left_base")
            self.assertNotIn("wrist_calibration_sha256", exported)
            self.assertNotIn("T_world_camera_opencv", exported)
            with self.assertRaises(FileExistsError):
                export_result(root / "solve/result.json", root / "out.json")
            write_json(root / "world.json", np.eye(4).tolist())
            export_result(root / "solve/result.json", root / "world-result.json", root / "world.json")
            world = json.loads((root / "world-result.json").read_text())
            np.testing.assert_allclose(world["T_world_camera_opencv"], report["T_base_camera_opencv"])

    def test_old_capture_method_is_rejected_with_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source").mkdir()
            write_json(root / "source/session.json", {"schema": "external_camera_capture_v1"})
            report = solve_session(root / "source", root / "solve")
            self.assertFalse(report["passed"])
            self.assertIn("old wrist-reference", report["errors"][0])
            self.assertTrue((root / "solve/report.md").is_file())

    def test_tampered_validation_image_is_not_silently_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            views, _, _, _ = create_capture(root / "source")
            (root / "source/external/0000.png").write_bytes(b"changed")
            with patch("camera_calibration.external_workflow.load_model",
                       return_value=SimpleNamespace(model=SimpleNamespace(digest="test-model"))), \
                    patch("camera_calibration.external_workflow.base_flange",
                          side_effect=lambda model, side, q: views[int(q[0])].base_flange):
                report = solve_session(root / "source", root / "solve")
            self.assertFalse(report["passed"])
            self.assertIn("Held-out group 0", report["errors"][0])
            self.assertIn("hash changed", report["observations"][0]["error"])

    def test_manual_records_have_no_automatic_stop_or_time_window_requirements(self):
        row = {"external_time_ns": 1_000_000_000, "capture_request_ns": 980_000_000,
               "joint_time_ns": 1_000_000_000, "time_error_ns": 0,
               "external_frame_number": 100}
        checked_record(row)
        checked_record(dict(row, capture_request_ns=1_000_000_001))
        checked_record(dict(row, joint_time_ns=1_300_000_000, time_error_ns=300_000_000))
        checked_record(row, row)  # Two explicit key presses may save the same displayed frame.
        for update in ({"time_error_ns": 1}, {"external_time_ns": -1}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                checked_record(dict(row, **update))
        with self.assertRaisesRegex(ValueError, "backwards"):
            checked_record(dict(row, external_frame_number=99), row)

    def test_native_fk_is_offline_and_model_is_identified(self):
        model = load_model()
        for side in ("left", "right"):
            result = base_flange(model, side, [0.] * 7)
            self.assertEqual(result.shape, (4, 4))
            np.testing.assert_allclose(result[3], [0, 0, 0, 1])
        with self.assertRaisesRegex(ValueError, "model changed"):
            load_model(expected_hash="wrong")


if __name__ == "__main__":
    unittest.main()
