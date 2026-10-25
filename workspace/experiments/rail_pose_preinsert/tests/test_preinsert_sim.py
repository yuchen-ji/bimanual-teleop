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
    def test_flange_z_alignment_keeps_position_for_parallel_antiparallel_and_general_axes(self):
        from common import align_flange_z
        initial = pose([.3, -.1, .9], [15, -30, 40])
        for direction in (initial[:3, 2], -initial[:3, 2], np.array([2., 0., 0.])):
            target = align_flange_z(initial, direction)
            np.testing.assert_allclose(target[:3, 3], initial[:3, 3], atol=1e-12)
            np.testing.assert_allclose(target[:3, 2], direction / np.linalg.norm(direction), atol=1e-12)
            np.testing.assert_allclose(target[:3, :3].T @ target[:3, :3], np.eye(3), atol=1e-12)

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
        task = yaml.safe_load((HERE / "config.demonstrated.yaml").read_text())["task"]
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
    def controller(self, directory, headless=True):
        import yaml
        from controller import PreinsertController

        from bimanual_teleop.sim.config import load_config, sim_path
        from bimanual_teleop.sim.kinematics import RobotModel
        config = load_config(); task = yaml.safe_load((HERE / "config.yaml").read_text())["task"]
        task["orientation_locked_grasp"] = False
        model = RobotModel(sim_path(config["robot"]["urdf"]))
        runtime = {"open": [0.]*20, "grasp": [.5]*20, "goal_rail": np.eye(4).tolist()}
        return PreinsertController(config, model, task, directory, runtime, mode="auto", headless=headless,
                                   perception="oracle", grasp="fixed")

    def test_gui_failure_keeps_application_and_next_trial_preserves_previous_result(self):
        import json

        with tempfile.TemporaryDirectory() as directory:
            source = self.controller(directory, headless=False)
            source.mode = "step"
            source.reset()
            self.assertTrue(source.waiting)
            source.finish("failed", "grip test")
            self.assertFalse(source.should_stop)
            self.assertTrue(source.pause_physics_requested)
            previous = source.trial_directory
            source.reset()
            self.assertFalse(source.pause_physics_requested)
            self.assertTrue(source.waiting)
            result = json.loads((previous / "result.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["reason"], "grip test")
            source.close()

    def test_gui_buttons_queue_physics_operations_outside_render_callbacks(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.controller(directory, headless=False)
            source.bound = True
            source.world = object()  # Any synchronous play/pause call would fail.
            source.next_stage()
            self.assertEqual(source.physics_action_requested, "play")
            source.toggle_physics()
            self.assertEqual(source.physics_action_requested, "toggle")
            source.request_reset()
            self.assertTrue(source.reset_requested)
            self.assertEqual(source.physics_action_requested, "play")
            self.assertTrue(source.start_on_reset)
            source.bound = False
            source.mode = "step"
            source.reset()
            self.assertFalse(source.waiting)
            self.assertFalse(source.start_on_reset)
            source.close()

    def test_simulation_controls_have_restart_without_a_parameter_panel(self):
        class UI:
            def __init__(self):
                self.labels = []
                self.buttons = {}
            def Label(self, text, **kwargs):
                self.labels.append(text)
                return type("Label", (), {"text": text})()
            def Button(self, text, clicked_fn):
                self.buttons[text] = clicked_fn
        with tempfile.TemporaryDirectory() as directory:
            source = self.controller(directory, headless=False)
            ui = UI()
            source.build_ui(ui)
            self.assertIn("Restart task", ui.buttons)
            self.assertIn("Apply config / Restart task", ui.buttons)
            self.assertIn("Restore default layout", ui.buttons)
            self.assertNotIn("Open task settings", ui.buttons)
            self.assertFalse(hasattr(source, "panel"))
            source.close()

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
        import json

        from bimanual_teleop.sim.kinematics import pose_vector
        with tempfile.TemporaryDirectory() as directory:
            source = self.controller(directory)
            source.grasp_mode = "physics"; source.grip = np.eye(4)
            state = {"time_ns": 1, "tick": 1, "q": source.command.tolist(),
                     "eef": [pose_vector(np.eye(4)).tolist()], "rail": pose_vector(pose([2, 0, 0])).tolist()}
            before = source.command.copy(); source.observe(state)
            self.assertEqual(source.status, "failed")
            failure = json.loads((Path(directory) / "failure_state.json").read_text())
            self.assertEqual(failure["state"]["time_ns"], 1)
            self.assertIn("limit", failure["reason"])
            self.assertEqual(failure["metrics"]["grasp_translation_limit_m"], source.task["max_grasp_translation_m"])
            np.testing.assert_equal(source.step(state, .01), before)
            # Later physics feedback can differ from the rendered screenshot;
            # it must never replace the state which triggered the stop.
            later = {**state, "time_ns": 2}
            source.observe(later)
            saved = json.loads((Path(directory) / "failure_state.json").read_text())
            self.assertEqual(saved["state"]["time_ns"], 1)
            source.close()


if __name__ == "__main__":
    unittest.main()
