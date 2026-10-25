"""Known eye-to-hand geometry, independent validation, and degenerate motions."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from camera_calibration.board import CircleBoard, camera_matrix, projected_points
from camera_calibration.external import View, solve_external
from camera_calibration.geometry import pose_errors, transform


def lens(serial="external"):
    return {"serial": serial, "resolution": [640, 480], "output_rotation_deg": 0,
            "rgb_intrinsics": {"width": 640, "height": 480, "fx": 600., "fy": 595.,
                               "ppx": 319., "ppy": 241., "model": "opencv_brown",
                               "coeffs": [-.04, .012, .0005, -.0003, .002]}}


def scene(noise=0.):
    camera = lens()
    truth = transform(Rotation.from_euler("xyz", [.1, -.2, .15]).as_matrix(), [.2, -.3, .1])
    mount = transform(Rotation.from_euler("xyz", [.15, .1, -.1]).as_matrix(), [.03, -.05, .14])
    rng = np.random.default_rng(163)
    views = []
    for group in range(25):
        board = transform(Rotation.from_euler("xyz", rng.uniform(-.4, .4, 3)).as_matrix(),
                          [-.16, -.1, 1.1] + rng.uniform([-.05, -.04, -.15], [.05, .04, .15]))
        flange = truth @ board @ np.linalg.inv(mount)
        pixels = projected_points(CircleBoard().points(), board, camera_matrix(camera),
                                  camera["rgb_intrinsics"]["coeffs"])
        views.append(View(flange, pixels + rng.normal(0, noise, pixels.shape), group))
    return views, camera, truth, mount


class ExternalTests(unittest.TestCase):
    def test_jointly_recovers_camera_and_unknown_mount(self):
        for noise in (0., .08):
            with self.subTest(noise=noise):
                views, camera, truth, mount = scene(noise)
                actual, mounting, report = solve_external(views, camera)
                self.assertTrue(report["passed"], report["validation"])
                self.assertLess(pose_errors(actual, truth)[0], .003)
                self.assertLess(pose_errors(actual, truth)[1], .2)
                self.assertLess(pose_errors(mounting, mount)[0], .003)
                self.assertEqual(report["validation_groups"], [0, 5, 10, 15, 20])
                self.assertFalse(set(report["train_groups"]) & set(report["validation_groups"]))

    def test_held_out_flange_error_fails_without_changing_fit(self):
        views, camera, _, _ = scene()
        baseline, mount, _ = solve_external(views, camera)
        views[0].base_flange[:3, 3] += [.05, 0, 0]
        result, result_mount, report = solve_external(views, camera)
        self.assertFalse(report["passed"])
        np.testing.assert_allclose(result, baseline, atol=1e-9)
        np.testing.assert_allclose(result_mount, mount, atol=1e-9)
        self.assertNotIn("0", report["rejected_train_groups"])

    def test_training_outlier_is_rejected_and_split_keeps_group_numbers(self):
        views, camera, truth, _ = scene(.05)
        views[3].base_flange[:3, 3] += [.08, 0, 0]
        result, _, report = solve_external(views, camera)
        self.assertTrue(report["passed"], report["validation"])
        self.assertIn("3", report["rejected_train_groups"])
        self.assertEqual(report["validation_groups"], [0, 5, 10, 15, 20])
        self.assertLess(pose_errors(result, truth)[0], .003)

    def test_held_out_detection_failure_cannot_be_dropped(self):
        views, camera, _, _ = scene()
        views[0].external_pixels[:] = np.nan
        with self.assertRaisesRegex(ValueError, "Held-out group 0"):
            solve_external(views, camera)

    def test_too_many_training_errors_fail(self):
        views, camera, _, _ = scene()
        for index in (1, 2, 3, 4, 6):
            views[index].external_pixels[:] = np.nan
        with self.assertRaisesRegex(ValueError, "20%"):
            solve_external(views, camera)

    def test_inconsistent_fit_preserves_candidate_and_diagnostics_without_passing(self):
        views, camera, _, _ = scene()
        bad = (1, 2, 3, 4, 6, 7)
        for index in bad:
            views[index].base_flange[:3, 3] += [.04, 0, 0]
        result, mount, report = solve_external(views, camera)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(np.isfinite(mount).all())
        self.assertFalse(report["passed"])
        self.assertFalse(report["training_consistency_passed"])
        self.assertIn("6/20", report["failure_reasons"][0])
        self.assertEqual(set(report["initial_inconsistent_train_groups"]), {str(i) for i in bad})
        self.assertEqual(report["rejected_train_groups"], {})
        self.assertEqual(len(report["train_groups"]), 20)
        self.assertEqual(report["validation_groups"], [0, 5, 10, 15, 20])
        self.assertEqual(len(report["per_group_rms_px_m_deg"]), 25)
        self.assertEqual(len(report["per_group_pnp"]), 25)
        # Good held-out metrics cannot override excessive training inconsistency.
        self.assertLess(report["validation"]["max_px"], 3.)

    def test_single_axis_and_translation_only_motion_fail(self):
        for only_translate in (True, False):
            views, camera, truth, mount = scene()
            for index, view in enumerate(views):
                board = transform(Rotation.from_euler("z", 0 if only_translate else index * .035).as_matrix(),
                                  [-.16 + index * .002, -.1, 1.1])
                view.base_flange = truth @ board @ np.linalg.inv(mount)
                view.external_pixels = projected_points(CircleBoard().points(), board, camera_matrix(camera),
                                                        camera["rgb_intrinsics"]["coeffs"])
            with self.subTest(only_translate=only_translate), self.assertRaisesRegex(ValueError, "nonparallel|20%"):
                solve_external(views, camera)

    def test_board_slippage_in_validation_fails(self):
        views, camera, truth, mount = scene()
        changed = mount.copy()
        changed[0, 3] += .03
        for view in views:
            if view.group % 5 == 0:
                board = np.linalg.inv(truth) @ view.base_flange @ changed
                view.external_pixels = projected_points(CircleBoard().points(), board, camera_matrix(camera),
                                                        camera["rgb_intrinsics"]["coeffs"])
        result, _, report = solve_external(views, camera)
        self.assertFalse(report["passed"])
        np.testing.assert_allclose(result, truth, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
