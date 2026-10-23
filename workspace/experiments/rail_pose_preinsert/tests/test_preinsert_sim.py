import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[2])]

from common import (
    cad_to_sim,
    opposed_grasp,
    perception_errors,
    pose,
    sample_path,
    seated_rail,
)


class GeometryTests(unittest.TestCase):
    def test_direction_ambiguity_is_not_hidden_by_small_centre_error(self):
        centre = np.array([0, .24, 0, 1.])
        truth = pose([0, 0, 1])
        flipped = truth @ pose([0, .48, 0], [0, 0, 180])
        frames = [{"T_W_C": np.eye(4).tolist(), "T_W_S": truth.tolist(), "state": {"time_ns": 5}}]
        records = [{"T_C_CAD": flipped.tolist(), "timestamp_ns": 5}]
        error = perception_errors(frames, records, np.eye(4), centre)[0]
        self.assertAlmostEqual(error["cad_origin_error_m"], .48)
        self.assertAlmostEqual(error["rotation_error_rad"], np.pi)
        self.assertLess(error["centre_error_m"], 1e-12)
        records[0]["timestamp_ns"] = 6
        with self.assertRaises(ValueError):
            perception_errors(frames, records, np.eye(4), centre)

    def test_contact_force_without_opposed_directions_is_not_a_grip(self):
        state = {"rail_contacts": {"l_thumb_tip": 2, "l_index_finger_tip": 2},
                 "rail_contact_vectors": {"l_thumb_tip": [1, 0, 0], "l_index_finger_tip": [1, 0, 0]}}
        self.assertFalse(opposed_grasp(state))
        state["rail_contact_vectors"]["l_index_finger_tip"] = [-1, 0, 0]
        self.assertFalse(opposed_grasp(state))  # Two fingers cannot certify the power wrap.
        for finger in ("middle_finger", "ring_finger", "pinky"):
            state["rail_contacts"][f"l_{finger}_tip"] = 2
            state["rail_contact_vectors"][f"l_{finger}_tip"] = [-1, 0, 0]
        self.assertTrue(opposed_grasp(state))
        state["rail_contacts"]["l_pinky_tip"] = .1
        self.assertFalse(opposed_grasp(state))

    def test_wrap_uses_recorded_hand_shape_and_palm_axes_without_extra_roll(self):
        import yaml
        from grasp_template import fit_demonstrated_wrap, load_reference

        from bimanual_teleop.sim.config import load_config, sim_path
        from bimanual_teleop.sim.kinematics import RobotModel
        config = load_config()
        task = yaml.safe_load((HERE / "config.yaml").read_text())["task"]
        model = RobotModel(sim_path(config["robot"]["urdf"]))
        reference = load_reference()
        template = fit_demonstrated_wrap(model, reference, task)
        T = np.asarray(template["T_S_F"])
        np.testing.assert_allclose(T @ template["T_F_S"], np.eye(4), atol=1e-12)
        self.assertEqual(template["extra_wrist_roll_deg"], 0)
        self.assertLessEqual(np.abs(template["joint_adjustment_rad"]).max(), task["wrap_max_joint_adjustment_rad"])
        self.assertEqual(len(template["tip_fit_error_m"]), 5)
        self.assertLess(max(template["tip_fit_error_m"].values()), .002)
        # Geometry fitting is explicitly not a physical grip certificate.
        self.assertIn("not a physical grip certification", template["validation"])

    def test_cad_conversion_preserves_asymmetric_hole_locations(self):
        import json

        from bimanual_teleop.sim import ROOT
        manifest = json.loads((ROOT / "assets/generated/manifest.json").read_text())
        T = cad_to_sim(manifest)
        np.testing.assert_allclose(T[:3, :3].T @ T[:3, :3], np.eye(3))
        # Original CAD Y is the long axis; the converted model's X is long.
        np.testing.assert_allclose(T @ [0, .24, 0, 1],
            np.r_[np.array([.24, 0, 0])-manifest["rail"]["origin_in_rotated_cad_m"], 1])

    def test_preinsert_axis_and_hole_clearance_use_scene_not_sdk_z(self):
        import json

        from bimanual_teleop.sim import ROOT
        from bimanual_teleop.sim.config import load_config
        manifest = json.loads((ROOT / "assets/generated/manifest.json").read_text())
        config = load_config(); seated = seated_rail(config, manifest)
        pre = seated.copy(); pre[:3, 3] += .02*pre[:3, 2]
        holes = np.asarray(manifest["fixture_geometry"]["rail_hole_bottoms_m"])
        bottoms = holes @ pre[:3, :3].T + pre[:3, 3]
        top = config["scene"]["fixture_origin"][2]+manifest["fixture_geometry"]["pin_top_height_m"]
        self.assertGreater(bottoms[:, 2].min()-top, .005)
        np.testing.assert_allclose((np.linalg.inv(seated) @ pre)[:3, 3], [0, 0, .02], atol=1e-12)

    def test_runtime_sampler_matches_real_planner_during_multiple_segments(self):
        sys.path[:0] = [str(HERE.parents[3] / "diffusion_policy/workspace/rail_pose_insert")]
        from rail_preinsert.planner import JointPath
        path = JointPath([np.zeros(7), np.arange(7)*.03, np.arange(7)*-.02], .12, .25)
        plan = {"joint_waypoints_rad": path.points.tolist(), "segment_durations_s": path.durations.tolist()}
        for t in np.linspace(-.1, path.duration+.1, 51):
            np.testing.assert_allclose(sample_path(plan, t), path.at(t))


class ControllerTests(unittest.TestCase):
    def controller(self, directory):
        import yaml
        from controller import PreinsertController

        from bimanual_teleop.sim.config import load_config, sim_path
        from bimanual_teleop.sim.kinematics import RobotModel
        config = load_config(); task = yaml.safe_load((HERE / "config.yaml").read_text())["task"]
        model = RobotModel(sim_path(config["robot"]["urdf"]))
        runtime = {"open": [0.]*20, "grasp": [.5]*20, "goal_rail": np.eye(4).tolist()}
        return PreinsertController(config, model, task, directory, runtime, mode="auto", headless=True,
                                   perception="oracle", grasp="fixed")

    def test_hold_does_not_insert_or_release_and_requires_actual_rail_accuracy(self):
        import json

        from bimanual_teleop.sim.kinematics import pose_vector
        with tempfile.TemporaryDirectory() as directory:
            source = self.controller(directory)
            source.transition("hold")
            command = source.command.copy()
            state = {"rail": pose_vector(pose([0, 0, .02])).tolist()}
            result = source.step(state, 2.)
            np.testing.assert_equal(result, command)
            self.assertEqual(source.status, "simulation_preinsert_passed")
            report = json.loads((Path(directory) / "result.json").read_text())
            self.assertFalse(report["insertion_executed"])
            self.assertFalse(report["physical_grasp_validated"])
            self.assertTrue(report["test_data"])
            source.close()
        with tempfile.TemporaryDirectory() as directory:
            source = self.controller(directory); source.transition("hold")
            source.step({"rail": pose_vector(pose([0, 0, 0])).tolist()}, 2.)
            self.assertEqual(source.status, "failed")
            source.close()

    def test_dynamic_grasp_slip_fails_without_commanding_a_release(self):
        from bimanual_teleop.sim.kinematics import pose_vector
        with tempfile.TemporaryDirectory() as directory:
            source = self.controller(directory)
            source.grasp_mode = "physics"; source.grip = np.eye(4)
            state = {"time_ns": 1, "tick": 1, "q": source.command.tolist(),
                     "eef": [pose_vector(np.eye(4)).tolist()], "rail": pose_vector(pose([2, 0, 0])).tolist()}
            before = source.command.copy(); source.observe(state)
            self.assertEqual(source.status, "failed")
            np.testing.assert_equal(source.step(state, .01), before)
            source.close()


if __name__ == "__main__":
    unittest.main()
