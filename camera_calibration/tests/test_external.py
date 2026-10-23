"""Numerical behavior and failure detection, without physical devices."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from camera_calibration.board import CircleBoard, camera_matrix, projected_points
from camera_calibration.external import View, solve_external
from camera_calibration.geometry import pose_errors, transform


def lens(serial, fx=600.0, fy=595.0):
    return {
        "serial": serial,
        "resolution": [640, 480],
        "fps": 30,
        "output_rotation_deg": 0,
        "rgb_intrinsics": {
            "width": 640,
            "height": 480,
            "fx": fx,
            "fy": fy,
            "ppx": 319.0,
            "ppy": 241.0,
            "model": "opencv_brown",
            "coeffs": [-0.04, 0.012, 0.0005, -0.0003, 0.002],
        },
    }


def scene(noise=0.0, count=25):
    rng = np.random.default_rng(146)
    wrist, external = lens("wrist"), lens("external", 570.0, 580.0)
    truth = transform(
        Rotation.from_euler("xyz", [0.03, -0.08, 0.15]).as_matrix(), [0.24, -0.05, 0.02]
    )
    views = []
    for group in range(count):
        world_wrist = transform(
            Rotation.from_rotvec(rng.uniform(-0.04, 0.04, 3)).as_matrix(),
            rng.uniform(-0.015, 0.015, 3),
        )
        world_board = transform(
            Rotation.from_euler("xyz", rng.uniform(-0.4, 0.4, 3)).as_matrix(),
            [-0.14, -0.08, 1.0]
            + rng.uniform([-0.04, -0.04, -0.12], [0.04, 0.04, 0.12]),
        )
        pixels = []
        for camera, meta in ((world_wrist, wrist), (truth, external)):
            image = projected_points(
                CircleBoard().points(),
                np.linalg.inv(camera) @ world_board,
                camera_matrix(meta),
                meta["rgb_intrinsics"]["coeffs"],
            )
            pixels.append(image + rng.normal(0, noise, image.shape))
        views.append(View(world_wrist, *pixels, group))
    return views, wrist, external, truth


class ExternalTests(unittest.TestCase):
    def test_known_pose_and_different_lenses(self):
        views, wrist, external, truth = scene()
        recovered, report = solve_external(views, wrist, external)
        self.assertTrue(report["passed"], report["validation"])
        distance, angle = pose_errors(recovered, truth)
        self.assertLess(distance, 1e-6)
        self.assertLess(angle, 1e-4)
        self.assertFalse(set(report["train_groups"]) & set(report["validation_groups"]))
        self.assertEqual(report["validation_groups"], [0, 5, 10, 15, 20])
        self.assertIn("excludes systematic", report["uncertainty"]["scope"])

    def test_noise_and_joint_training_outlier(self):
        views, wrist, external, truth = scene(0.08)
        views[3].world_wrist[:3, 3] += [0.08, 0, 0]
        recovered, report = solve_external(views, wrist, external)
        self.assertTrue(report["passed"], report["validation"])
        self.assertIn("3", report["rejected_train_groups"])
        distance, angle = pose_errors(recovered, truth)
        self.assertLess(distance, 0.003)
        self.assertLess(angle, 0.2)

    def test_held_out_joint_error_cannot_change_solution(self):
        views, wrist, external, _ = scene()
        baseline, _ = solve_external(views, wrist, external)
        views[0].world_wrist[:3, 3] += [0.05, 0, 0]
        recovered, report = solve_external(views, wrist, external)
        self.assertFalse(report["passed"])
        self.assertNotIn("0", report["rejected_train_groups"])
        np.testing.assert_allclose(recovered, baseline, atol=1e-9)

    def test_rejected_training_pose_behind_camera_does_not_break_report(self):
        views, wrist, external, truth = scene()
        views[3].world_wrist[:3, :3] = Rotation.from_euler("y", np.pi).as_matrix()
        recovered, report = solve_external(views, wrist, external)
        self.assertTrue(report["passed"])
        self.assertIn("3", report["rejected_train_groups"])
        np.testing.assert_allclose(recovered, truth, atol=1e-6)

    def test_invalid_held_out_pixels_are_not_dropped(self):
        views, wrist, external, _ = scene()
        views[0].external_pixels[:] = np.nan
        with self.assertRaisesRegex(ValueError, "Held-out group 0"):
            solve_external(views, wrist, external)

    def test_training_invalid_view_keeps_original_validation_split(self):
        views, wrist, external, _ = scene()
        views[3].external_pixels[:] = np.nan
        _, report = solve_external(views, wrist, external)
        self.assertEqual(report["validation_groups"], [0, 5, 10, 15, 20])
        self.assertIn("3", report["rejected_train_groups"])

    def test_repeated_views_and_single_axis_coverage_fail(self):
        views, wrist, external, _ = scene()
        repeated = [
            View(
                views[0].world_wrist.copy(),
                views[0].wrist_pixels.copy(),
                views[0].external_pixels.copy(),
                i,
            )
            for i in range(25)
        ]
        with self.assertRaisesRegex(ValueError, "20%|repeats"):
            solve_external(repeated, wrist, external)
        fixed, _, _, camera = scene()
        for i, view in enumerate(fixed):
            board = transform(
                (
                    Rotation.from_euler("y", 0.3) * Rotation.from_euler("z", i * 0.035)
                ).as_matrix(),
                [-0.14 + i * 0.001, -0.08, 1.0],
            )
            view.world_wrist = np.eye(4)
            for attr, pose, meta in (
                ("wrist_pixels", board, wrist),
                ("external_pixels", np.linalg.inv(camera) @ board, external),
            ):
                setattr(
                    view,
                    attr,
                    projected_points(
                        CircleBoard().points(),
                        pose,
                        camera_matrix(meta),
                        meta["rgb_intrinsics"]["coeffs"],
                    ),
                )
        with self.assertRaisesRegex(ValueError, "coverage|20%"):
            solve_external(fixed, wrist, external)

    def test_camera_moved_in_validation_fails(self):
        views, wrist, external, truth = scene()
        moved = truth.copy()
        moved[:3, 3] += [0.04, 0, 0]
        for view in views:
            if view.group % 5 == 0:
                # Moving only the external camera preserves a valid per-frame PnP.
                from camera_calibration.board import estimate_board_pose

                wrist_board, _ = estimate_board_pose(view.wrist_pixels, wrist)
                board = np.linalg.inv(moved) @ view.world_wrist @ wrist_board
                view.external_pixels = projected_points(
                    CircleBoard().points(),
                    board,
                    camera_matrix(external),
                    external["rgb_intrinsics"]["coeffs"],
                )
        recovered, report = solve_external(views, wrist, external)
        self.assertFalse(report["passed"])
        np.testing.assert_allclose(recovered, truth, atol=1e-6)

    def test_too_many_training_errors_fail(self):
        views, wrist, external, _ = scene()
        for i in (1, 2, 3, 4, 6):
            views[i].external_pixels[:] = np.nan
        with self.assertRaisesRegex(ValueError, "20%"):
            solve_external(views, wrist, external)

    def test_minimum_counts_valid_groups_after_rejection(self):
        views, wrist, external, _ = scene(count=15)
        views[3].external_pixels[:] = np.nan
        with self.assertRaisesRegex(ValueError, "valid independent"):
            solve_external(views, wrist, external)


if __name__ == "__main__":
    unittest.main()
