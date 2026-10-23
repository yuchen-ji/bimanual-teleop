"""Synthetic calibration checks; no physical SDK or camera is opened."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from bimanual_teleop.sim import ROOT
from bimanual_teleop.sim.camera_geometry import wrist_camera_local_pose
from bimanual_teleop.sim.config import load_config, validate
from camera_calibration.board import (
    CircleBoard,
    camera_matrix,
    detect_board,
    estimate_board_pose,
    projected_points,
)
from camera_calibration.collect import FeedbackReader, collect
from camera_calibration.geometry import pose_errors, transform
from camera_calibration.simulation import apply_result
from camera_calibration.workflow import output_lens
from camera_calibration.wrist import solve_handeye


def synthetic_lens():
    return {
        "resolution": [640, 480],
        "output_rotation_deg": 180,
        "rgb_intrinsics": {
            "width": 640,
            "height": 480,
            "fx": 400.0,
            "fy": 398.0,
            "ppx": 315.0,
            "ppy": 244.0,
            "model": "opencv_brown",
            "coeffs": [-0.04, 0.02, 0.0007, -0.0003, 0.0],
        },
    }


def handeye_observations():
    rng = np.random.default_rng(42)
    handeye = transform(
        Rotation.from_euler("xyz", [0.2, -0.1, 0.5]).as_matrix(), [0.03, 0.06, 0.07]
    )
    target = transform(
        Rotation.from_euler("xyz", [0.3, 0.1, -0.4]).as_matrix(), [0.5, 0.1, 0.8]
    )
    grippers = [
        transform(
            Rotation.from_rotvec(rng.normal(size=3) * 0.4).as_matrix(),
            [0.3, 0.1, 0.6] + rng.normal(size=3) * 0.08,
        )
        for _ in range(25)
    ]
    boards = [np.linalg.inv(g @ handeye) @ target for g in grippers]
    return grippers, boards, handeye, target


class WristHandeyeTests(unittest.TestCase):
    def test_board_scale_and_detection_under_image_roll(self):
        points = CircleBoard().points()
        np.testing.assert_allclose(points[1] - points[0], [0.1, 0, 0])
        np.testing.assert_allclose(points[4] - points[0], [0.05, 0.05, 0])
        image = np.full((480, 640, 3), 32, dtype=np.uint8)
        expected = points[:, :2] * 900 + [150, 140]
        for point in expected:
            cv2.circle(
                image, tuple(np.round(point).astype(int)), 21, (225, 225, 225), -1
            )
        cv2.circle(image, (600, 430), 3, (225, 225, 225), -1)
        for _ in range(4):
            centers = detect_board(image)
            self.assertIsNotNone(centers)
            np.testing.assert_allclose(centers, expected, atol=0.05)
            expected = np.c_[expected[:, 1], image.shape[1] - 1 - expected[:, 0]]
            image = np.rot90(image).copy()

    def test_pnp_recovers_tilted_board_with_distortion(self):
        metadata = synthetic_lens()
        points = CircleBoard().points()
        truth = transform(
            Rotation.from_euler("xyz", [0.3, -0.2, 0.1]).as_matrix(),
            [-0.17, -0.08, 0.65],
        )
        pixels = projected_points(
            points, truth, camera_matrix(metadata), metadata["rgb_intrinsics"]["coeffs"]
        )
        recovered, quality = estimate_board_pose(pixels, metadata)
        distance, angle = pose_errors(recovered, truth)
        self.assertLess(distance, 1e-6)
        self.assertLess(angle, 0.001)
        self.assertLess(quality["pnp_rms_px"], 0.001)

    def test_oblique_ellipses_with_background_clutter(self):
        metadata = synthetic_lens()
        points = CircleBoard().points()
        pose = transform(
            Rotation.from_euler("xyz", [1.15, 0.1, -0.1]).as_matrix(),
            [-0.15, -0.04, 0.55],
        )
        image = np.full((960, 1280, 3), 32, dtype=np.uint8)
        theta = np.linspace(0, 2 * np.pi, 80)
        for point in points:
            circle = (
                point
                + np.c_[
                    0.025 * np.cos(theta), 0.025 * np.sin(theta), np.zeros(len(theta))
                ]
            )
            pixel = projected_points(
                circle,
                pose,
                camera_matrix(metadata),
                metadata["rgb_intrinsics"]["coeffs"],
            )
            cv2.fillPoly(image, [np.round(pixel * 2).astype(np.int32)], (175, 175, 175))
        cv2.rectangle(image, (5, 5), (250, 180), (250, 250, 250), -1)
        image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
        centers = detect_board(image)
        self.assertIsNotNone(centers)
        truth = projected_points(
            points, pose, camera_matrix(metadata), metadata["rgb_intrinsics"]["coeffs"]
        )
        self.assertLess(float(np.linalg.norm(centers - truth, axis=1).max()), 1.0)
        _, quality = estimate_board_pose(centers, metadata)
        self.assertLess(quality["pnp_rms_px"], 0.3)

    def test_handeye_recovers_known_mount(self):
        grippers, boards, truth, target = handeye_observations()
        recovered, fitted_target, quality = solve_handeye(grippers, boards)
        self.assertTrue(quality["passed"])
        np.testing.assert_allclose(recovered, truth, atol=1e-7)
        np.testing.assert_allclose(fitted_target, target, atol=1e-7)
        self.assertFalse(
            set(quality["train_indices"]) & set(quality["held_out_indices"])
        )

    def test_noisy_observations_and_training_outlier(self):
        grippers, boards, truth, _ = handeye_observations()
        rng = np.random.default_rng(7)
        for board in boards:
            board[:3, 3] += rng.normal(size=3) * 0.0002
            board[:3, :3] = (
                Rotation.from_rotvec(rng.normal(size=3) * 0.0004).as_matrix()
                @ board[:3, :3]
            )
        boards[3][:3, 3] += [0.08, 0, 0]
        recovered, _, quality = solve_handeye(grippers, boards)
        self.assertTrue(quality["passed"])
        self.assertEqual(quality["rejected_train_indices"], [3])
        distance, angle = pose_errors(recovered, truth)
        self.assertLess(distance, 0.001)
        self.assertLess(angle, 0.1)

    def test_held_out_error_cannot_be_fitted_or_discarded(self):
        grippers, boards, truth, _ = handeye_observations()
        boards[0][:3, 3] += [0.05, 0, 0]
        recovered, _, quality = solve_handeye(grippers, boards)
        self.assertFalse(quality["passed"])
        self.assertIn(0, quality["held_out_indices"])
        self.assertGreater(quality["held_out_max_translation_m"], 0.049)
        np.testing.assert_allclose(recovered, truth, atol=1e-7)

    def test_single_rotation_axis_is_rejected(self):
        grippers, boards, _, _ = handeye_observations()
        for index, gripper in enumerate(grippers):
            gripper[:3, :3] = Rotation.from_euler("z", index * 0.05).as_matrix()
        with self.assertRaisesRegex(ValueError, "nonparallel"):
            solve_handeye(grippers, boards)

    def test_rotated_lens_and_usd_camera_axes(self):
        metadata = synthetic_lens()
        intrinsics, coefficients, _ = output_lens(metadata)
        self.assertEqual((intrinsics["cx"], intrinsics["cy"]), (324.0, 235.0))
        np.testing.assert_allclose(coefficients[2:4], [-0.0007, 0.0003])
        rotation = Rotation.from_euler("xyz", [0.2, 0.5, -0.8])
        measured = {
            "position_m": [0.01, 0.06, -0.03],
            "quaternion_xyzw": rotation.as_quat().tolist(),
        }
        position, quaternion = wrist_camera_local_pose({}, measured)
        np.testing.assert_allclose(position, measured["position_m"])
        usd = Rotation.from_quat(quaternion)
        np.testing.assert_allclose(usd.apply([0, 0, -1]), rotation.apply([0, 0, 1]))
        np.testing.assert_allclose(usd.apply([0, 1, 0]), rotation.apply([0, -1, 0]))
        config = load_config()
        config["cameras"]["wrist"]["left"]["pose_in_bracket"] = measured
        validate(config)
        config["cameras"]["wrist"]["left"]["pose_in_bracket"]["quaternion_xyzw"] = [
            0,
            0,
            0,
            0,
        ]
        with self.assertRaises(ValueError):
            validate(config)

    def test_failed_calibration_never_changes_config(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.json"
            result.write_text(
                json.dumps({"schema": "wrist_handeye_result_v1", "passed": False})
            )
            with self.assertRaisesRegex(ValueError, "held-out"):
                apply_result(result)

    def test_apply_uses_separate_simulation_config_and_rejects_synthetic(self):
        from bimanual_teleop.sim._dependencies import yaml
        from camera_calibration.workflow import digest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            (root / "assets/generated").mkdir(parents=True)
            base = load_config()
            base["cameras"]["wrist_calibration_file"] = None
            (root / "configs/default.yaml").write_text(yaml.safe_dump(base))
            shutil.copy2(
                ROOT / "assets/generated/robot.urdf",
                root / "assets/generated/robot.urdf",
            )
            settings = {
                "pose_in_bracket": {
                    "position_m": [0.01, 0.06, -0.03],
                    "quaternion_xyzw": [0, 0, 0, 1],
                },
                "intrinsics": {"fx": 400.0, "fy": 398.0, "cx": 324.0, "cy": 235.0},
                "distortion_coefficients": [-0.04, 0.02, -0.0007, 0.0003, 0.0],
            }
            result = root / "result.json"
            report = {
                "schema": "wrist_handeye_result_v1",
                "passed": True,
                "synthetic": True,
                "urdf_sha256": digest(root / "assets/generated/robot.urdf"),
                "board": {},
                "calibrations": {"left": {"passed": True, "settings": settings}},
            }
            result.write_text(json.dumps(report))
            with (
                patch("bimanual_teleop.sim.config.ROOT", root),
                patch("camera_calibration.simulation.SIM_ROOT", root),
            ):
                before = load_config()
                with self.assertRaises(ValueError):
                    apply_result(result)
                # Exercise successful writes only inside this isolated temporary simulation tree.
                report["synthetic"] = False
                result.write_text(json.dumps(report))
                destination = apply_result(result)
                after = load_config()
                self.assertEqual(
                    destination, root / "configs/wrist_camera_calibrated.yaml"
                )
                self.assertEqual(after["cameras"]["wrist"]["left"], settings)
                self.assertEqual(before["cameras"]["head"], after["cameras"]["head"])
                self.assertEqual(
                    before["cameras"]["wrist"]["right"],
                    after["cameras"]["wrist"]["right"],
                )

    def test_feedback_only_uses_read_connection_methods(self):
        class ReadOnlySDK:
            def __init__(self):
                self.calls = []
                self.sequence = 0

            def open(self, ip):
                self.calls.append(("open", ip))

            def poll_feedback(self):
                self.calls.append(("poll_feedback",))
                self.sequence += 1
                if self.sequence == 3:
                    reader.stopped.set()
                return SimpleNamespace(
                    sequence=[self.sequence] * 2,
                    received_ns=self.sequence * 5_000_000,
                    q=[180.0] * 7 + [90.0] * 7,
                )

            def close(self):
                self.calls.append(("close",))

        sdk = ReadOnlySDK()
        reader = FeedbackReader(sdk, "example")
        reader.start()
        reader.thread.join(1)
        reader.close()
        self.assertFalse(reader.thread.is_alive())
        self.assertIsNone(reader.error)
        np.testing.assert_allclose(reader.buffers["left"][0][2], np.pi)
        np.testing.assert_allclose(reader.buffers["right"][0][2], np.pi / 2)
        self.assertEqual(
            [row[0] for row in sdk.calls], ["open"] + ["poll_feedback"] * 3 + ["close"]
        )

    def test_stability_rejects_moving_and_unsynchronized_samples(self):
        reader = FeedbackReader(None, "example")
        center = 1_000_000_000
        for index in range(-40, 41):
            reader.buffers["left"].append(
                (center + index * 5_000_000, index + 41, np.zeros(7))
            )
        value, _ = reader.stationary("left", center)
        self.assertIsNotNone(value)
        reader.buffers["left"][-1][2][0] = 0.003
        self.assertIsNone(reader.stationary("left", center)[0])
        self.assertIsNone(reader.stationary("left", center + 1_000_000_000)[0])

    def test_collector_without_usb_fails_before_creating_drivers(self):
        with (
            patch("platform.system", return_value="Linux"),
            patch("platform.machine", return_value="x86_64"),
            patch.object(Path, "glob", return_value=[]),
            self.assertRaisesRegex(RuntimeError, "No local video"),
        ):
            collect(SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
