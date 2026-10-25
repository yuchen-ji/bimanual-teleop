"""Known eye-in-hand geometry and raw-image manual capture workflow."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from camera_calibration.board import CircleBoard
from camera_calibration.geometry import pose_errors
from camera_calibration.storage import digest, write_json
from camera_calibration.tests.test_external import scene
from camera_calibration.tests.test_workflow import render_board
from camera_calibration.wrist import WristView, solve_wrist
from camera_calibration.wrist_workflow import export_result, solve_session


def observations(noise=0.):
    views, camera, handeye, target = scene(noise)
    return ([WristView(np.linalg.inv(v.base_flange), v.external_pixels.copy(), v.group) for v in views],
            camera, handeye, target)


class WristTests(unittest.TestCase):
    def test_known_handeye_and_fixed_board_are_jointly_recovered(self):
        for noise in (0., .08):
            with self.subTest(noise=noise):
                views, camera, handeye, target = observations(noise)
                actual, board, quality = solve_wrist(views, camera)
                self.assertTrue(quality["passed"], quality["validation"])
                self.assertLess(pose_errors(actual, handeye)[0], .003)
                self.assertLess(pose_errors(board, target)[0], .003)
                self.assertEqual(quality["validation_groups"], [0, 5, 10, 15, 20])

    def test_heldout_pose_does_not_influence_the_fit(self):
        views, camera, handeye, target = observations()
        views[0].base_flange[:3, 3] += [.05, 0, 0]
        actual, _, quality = solve_wrist(views, camera)
        self.assertFalse(quality["passed"])
        np.testing.assert_allclose(actual, handeye, atol=1e-8)

    def test_raw_manual_images_solve_without_stop_windows_and_export(self):
        views, camera, handeye, target = observations()
        camera["output_rotation_deg"] = 180
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "capture"
            (capture / "left").mkdir(parents=True)
            session = {"schema": "wrist_handeye_capture_v2", "status": "captured", "capture_mode": "manual",
                       "method": "eye_in_hand", "board_mount": "fixed_in_environment", "synthetic": True,
                       "images": "raw RGB sensor orientation", "robot_model_sha256": "test-model",
                       "board": asdict(CircleBoard()), "cameras": {"left": camera}, "counts": {"left": 25}}
            rows = []
            for view in views:
                image_path = capture / "left" / f"{view.group:04d}.png"
                board = np.linalg.inv(view.base_flange @ handeye) @ target
                Image.fromarray(render_board(board, camera)).save(image_path)
                timestamp = (view.group + 1) * 1_000_000_000
                rows.append({"side": "left", "group": view.group, "joints_rad": [float(view.group)] * 7,
                             "T_base_flange": view.base_flange.tolist(), "capture_request_ns": timestamp + 20_000_000,
                             "joint_time_ns": timestamp + 300_000_000, "image_time_ns": timestamp,
                             "image_frame_number": view.group + 100, "time_error_ns": 300_000_000,
                             "image": str(image_path.relative_to(capture)), "image_sha256": digest(image_path)})
            write_json(capture / "session.json", session)
            (capture / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            with patch("camera_calibration.wrist_workflow.load_model",
                       return_value=SimpleNamespace(model=SimpleNamespace(digest="test-model"))), \
                    patch("camera_calibration.wrist_workflow.base_flange",
                          side_effect=lambda model, side, q: views[int(q[0])].base_flange):
                report = solve_session(capture, root / "solve")
            self.assertTrue(report["passed"], report["errors"])
            result = report["calibrations"]["left"]
            self.assertLess(pose_errors(np.array(result["T_flange_sensor_opencv"]), handeye)[0], .006)
            np.testing.assert_allclose(np.array(result["T_flange_sensor_opencv"]) @ np.diag([-1., -1., 1., 1.]),
                                       result["T_flange_output_opencv"], atol=1e-10)
            self.assertEqual(len(list((root / "solve").glob("left_validation_*.png"))), 5)
            with self.assertRaisesRegex(ValueError, "measured wrist"):
                export_result(root / "solve/result.json", root / "export.json")
            report["synthetic"] = False
            write_json(root / "solve/result.json", report)
            export_result(root / "solve/result.json", root / "export.json")
            self.assertEqual(json.loads((root / "export.json").read_text())["schema"], "wrist_camera_extrinsics_v2")


if __name__ == "__main__":
    unittest.main()
