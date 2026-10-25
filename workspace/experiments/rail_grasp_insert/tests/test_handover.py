"""Handover interlocks and scene/provenance contracts, independent of Isaac."""

import json
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from run import load_experiment
from handover import HandoverController, HandoverReplayController, PHASES, extraction_lift_pose, preflight, scene_signature
from grasping import FINGERS, gripping, make_grasp
from bimanual_teleop.sim.config import load_config, sim_path
from bimanual_teleop.sim.kinematics import RobotModel, pose_matrix, pose_vector, transform


class HandoverTests(unittest.TestCase):
    def setUp(self):
        self.config, self.task = load_experiment(HERE/"handover.yaml")
        self.model = RobotModel(sim_path(self.config["robot"]["urdf"]))
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        with patch("handover.ArmPlanner"):
            self.controller = HandoverController(self.config, self.model, self.task, self.temp.name, mode="auto")
        self.addCleanup(self.controller.close)
        self.state = {"q": self.controller.ready.tolist(), "q_target": self.controller.ready.tolist(),
                      "rail": pose_vector(transform(self.task["handover_center_m"], np.deg2rad(self.task["handover_rpy_deg"]))),
                      "rail_contacts": {}, "rail_contact_vectors": {}}

    def contacts(self, *sides):
        for side in sides:
            prefix = "l_" if side == "left" else "r_"
            self.state["rail_contacts"].update({prefix+"thumb_distal": 2., prefix+"ring_finger_distal": 2.})
            self.state["rail_contacts"][prefix+"index_finger_distal"] = 1.
            self.state["rail_contacts"][prefix+"middle_finger_distal"] = 1.
            self.state["rail_contact_vectors"].update({prefix+"thumb_distal": [0., 2., 0.], prefix+"ring_finger_distal": [0., -2., 0.]})

    def test_right_middle_and_left_end_are_distinct(self):
        self.assertAlmostEqual(self.task["right_grasp_axis_m"], 0.)
        self.assertGreater(self.task["left_grasp_axis_m"], .12)
        for side, offset in (("left", 14), ("right", 34)):
            np.testing.assert_allclose(self.controller.templates[side]["open"],
                                       np.clip(self.task["hand_reference_rad"][side]["open"], self.model.lower[offset:offset+20], self.model.upper[offset:offset+20]))

    def test_left_cannot_release_before_right_is_verified(self):
        self.controller.phase_index = PHASES.index("left_release")
        self.controller.closed["left"] = True
        self.controller.step(self.state, .01)
        self.assertEqual(self.controller.status, "failed")
        self.assertEqual(self.controller.reason, "left_release_before_verified_right_grip")
        self.assertTrue(self.controller.closed["left"])

    def test_contact_spread_keeps_reach_preshape_and_contacts_inside_rail(self):
        task = copy.deepcopy(self.task)
        task.pop("closed_hand_profiles_rad", None)
        baseline = make_grasp(self.model, task, "left")
        task["tip_axis_offsets_m"] = {"left": [0., -.018, -.010, -.002, .008]}
        grasp = make_grasp(self.model, task, "left")
        np.testing.assert_allclose(grasp["flange_in_rail"], baseline["flange_in_rail"])
        np.testing.assert_allclose(grasp["preshape"], baseline["preshape"])
        q = np.zeros(54); q[14:34] = grasp["closed"]
        inverse = np.linalg.inv(self.model.flange("left", q, True))
        tips = np.array([(grasp["flange_in_rail"]@inverse@self.model.link_pose("l_"+finger+"_tip", q))[:3, 3] for finger in FINGERS])
        self.assertTrue(np.all(abs(tips[:, 0]) < self.config["rail"]["size_m"][0]/2-.015))
        self.assertTrue(np.all(np.diff(tips[1:, 0]) > .025))
        self.assertLess(max(grasp["tip_fit_errors_m"].values()), .001)

    def test_explicit_reach_preshape_respects_mechanical_joint_limits(self):
        task = copy.deepcopy(self.task)
        profile = self.controller.templates["left"]["open"].copy()
        task["preshape_hand_rad"] = {"left": profile.tolist()}
        np.testing.assert_allclose(make_grasp(self.model, task, "left")["preshape"], profile)
        task["preshape_hand_rad"]["left"][0] = self.model.upper[14]+.01
        with self.assertRaisesRegex(ValueError, "within hand joint limits"):
            make_grasp(self.model, task, "left")

    def test_explicit_demonstrated_closed_hand_is_not_refitted_to_tip_targets(self):
        task = copy.deepcopy(self.task)
        profile = np.asarray(task["hand_reference_rad"]["left"]["closed"])
        task["closed_hand_profiles_rad"] = {"left": profile.tolist()}
        grasp = make_grasp(self.model, task, "left")
        np.testing.assert_allclose(grasp["closed"], profile)
        self.assertFalse(any(key.startswith("closed_") for key in grasp["tip_fit_errors_m"]))
        task["closed_hand_profiles_rad"]["left"][0] = self.model.upper[14]+.01
        with self.assertRaisesRegex(ValueError, "within hand joint limits"):
            make_grasp(self.model, task, "left")

    def test_per_hand_closed_profile_loads_alongside_legacy_single_hand_profile(self):
        import yaml
        document = yaml.safe_load((HERE/"handover.yaml").read_text())
        profile = document["task"]["hand_reference_rad"]["left"]["closed"]
        document["task"]["closed_hand_profiles_rad"] = {"left": profile}
        path = Path(self.temp.name)/"profile.yaml"
        path.write_text(yaml.safe_dump(document))
        config, task = load_experiment(path)
        self.assertEqual(len(task["closed_hand_rad"]), 20)
        np.testing.assert_allclose(make_grasp(self.model, task, "left")["closed"], profile)

    def test_preflight_saves_unreachable_pose_as_failed_with_its_stage(self):
        task = {**self.task, "convex_decomposition_links": ["l_mount", "r_mount"]}
        with patch("handover.ArmPlanner") as factory:
            factory.return_value.solve.side_effect = ValueError("unreachable test pose")
            result = preflight(self.config, task, self.temp.name)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failed_stage"], "left_grasp")
        self.assertEqual(result["scene_signature"], scene_signature(self.config, task))
        self.assertNotEqual(result["scene_signature"], scene_signature(self.config))
        self.assertEqual(json.loads((Path(self.temp.name)/"result.json").read_text())["reason"], "unreachable test pose")

    def test_left_hand_stays_closed_if_right_grip_is_lost(self):
        self.controller.phase_index = PHASES.index("left_release")
        self.controller.closed["left"] = True; self.controller.right_verified = True
        self.controller.step(self.state, .2)
        self.assertEqual(self.controller.reason, "right_grip_lost")
        self.assertTrue(self.controller.closed["left"])

    def test_release_opens_hand_instead_of_returning_to_approach_hook(self):
        self.controller.phase_index = PHASES.index("left_release")
        self.controller.closed["left"] = self.controller.preshaped["left"] = True
        self.controller.right_verified = True
        self.controller.command[14:34] = self.controller.templates["left"]["closed"]
        self.contacts("right")
        self.controller._step(self.state, .01)
        for _ in range(100):
            self.controller.update_hands(.05, self.state)
        np.testing.assert_allclose(self.controller.command[14:34], self.controller.templates["left"]["open"])

    def test_right_release_clear_thumb_is_preserved_until_retreat_finishes(self):
        self.controller.phase_index = PHASES.index("right_release")
        self.controller.closed["right"] = False
        target = self.controller.templates["right"]["release"]
        self.assertGreater(np.max(abs(target-self.controller.templates["right"]["open"])), .5)
        for _ in range(100):
            self.controller.update_hands(.05, self.state)
        np.testing.assert_allclose(self.controller.command[34:54], target)
        self.controller.phase_index = PHASES.index("right_retreat")
        self.controller.update_hands(.05, self.state)
        np.testing.assert_allclose(self.controller.command[34:54], target)
        self.controller.phase_index = PHASES.index("verify")
        for _ in range(100):
            self.controller.update_hands(.05, self.state)
        np.testing.assert_allclose(self.controller.command[34:54], self.controller.templates["right"]["open"])

    def test_right_release_cannot_advance_with_residual_finger_pressure(self):
        self.controller.phase_index = PHASES.index("right_release")
        self.controller.command[34:54] = self.controller.templates["right"]["release"]
        self.state["rail_contacts"] = {"r_thumb_proximal_abd": .1}
        self.controller._step(self.state, .01)
        self.assertEqual(self.controller.phase, "right_release")
        self.state["rail_contacts"] = {}
        self.controller._step(self.state, .01)
        self.assertEqual(self.controller.phase, "right_retreat")

    def test_release_profile_rejects_angles_outside_mechanical_limits(self):
        task = copy.deepcopy(self.task)
        task["release_hand_rad"]["right"][0] = self.model.upper[34]+.01
        with self.assertRaisesRegex(ValueError, "Release profile.*within hand joint limits"):
            make_grasp(self.model, task, "right")

    def test_handover_requires_both_hands_and_stability(self):
        self.controller.phase_index = PHASES.index("handover_hold")
        self.contacts("left", "right")
        for _ in range(60):
            if self.controller.phase != "handover_hold":
                break
            self.controller.step(self.state, .01)
        self.assertTrue(self.controller.right_verified)
        np.testing.assert_array_equal(self.controller.grasp_commands["right"], self.controller.command[34:54])
        self.assertEqual(self.controller.phase, "left_release")

    def test_verified_right_grasp_keeps_position_preload_as_weight_increases(self):
        self.controller.task["right_adaptive_closing"] = True
        self.controller.closed["right"] = True
        self.controller.command[34:54] = self.controller.templates["right"]["open"]
        self.controller.update_hands(.1, self.state)
        preload = self.controller.command[34:54].copy()
        self.controller.grasp_commands["right"] = preload.copy()
        self.controller.right_verified = True
        self.state["rail_contacts"] = {"r_"+finger+"_distal": 20. for finger in FINGERS}
        for _ in range(10):
            self.controller.update_hands(.1, self.state)
        np.testing.assert_allclose(self.controller.command[34:54], preload)

    def test_ring_and_pinky_can_support_a_grasp_but_mounts_cannot(self):
        self.contacts("right")
        self.assertTrue(gripping(self.state, "right", .5))
        self.assertFalse(gripping(self.state, "left", .5))
        self.state["rail_contacts"] = {"r_camera_bracket": 20.}
        self.assertFalse(gripping(self.state, "right", .5))

    def test_loaded_fingers_can_oppose_palm_but_palm_alone_is_not_a_grasp(self):
        self.state["rail_contacts"] = {"r_thumb_distal": 2., "r_index_finger_distal": 2., "r_wrist": 4.}
        self.state["rail_contact_vectors"] = {"r_thumb_distal": [0., 0., -2.], "r_index_finger_distal": [0., 0., -2.], "r_wrist": [0., 0., 4.]}
        self.assertTrue(gripping(self.state, "right", .5))
        del self.state["rail_contacts"]["r_thumb_distal"]
        self.assertFalse(gripping(self.state, "right", .5))
        self.state["rail_contacts"]["r_thumb_distal"] = 2.
        del self.state["rail_contacts"]["r_wrist"]
        self.state["rail_contacts"]["r_camera_bracket"] = 4.
        self.assertFalse(gripping(self.state, "right", .5))

    def test_verified_palm_contact_obeys_existing_grip_loss_grace_period(self):
        self.controller.phase_index = PHASES.index("right_close")
        self.controller.held["left"] = np.linalg.inv(self.model.flange("left", self.state["q"], True))@pose_matrix(self.state["rail"])
        self.state["rail_contacts"] = {"l_wrist": 7., "l_thumb_distal": 2.}
        self.assertIsNone(self.controller.check_carry(self.state, .01))
        self.assertEqual(self.controller.status, "running")
        self.assertAlmostEqual(self.controller.loss_s["left"], .01)
        self.controller.check_carry(self.state, self.task["carry_contact_loss_s"])
        self.assertEqual(self.controller.reason, "left_grip_lost")

    def test_release_allows_palm_unloading_but_retreat_rejects_new_palm_contact(self):
        self.controller.phase_index = PHASES.index("left_release")
        self.controller.right_verified = True
        self.contacts("right")
        self.state["rail_contacts"]["l_wrist"] = 7.
        self.assertIsNone(self.controller.check_carry(self.state, .01))
        self.controller.phase_index = PHASES.index("left_retreat")
        self.controller.check_carry(self.state, .01)
        self.assertEqual(self.controller.reason, "rail_hit_arm: l_wrist")

    def test_released_left_finger_cannot_push_rail_during_right_load_verification(self):
        self.controller.phase_index = PHASES.index("left_retreat")
        self.controller.right_verified = True
        self.contacts("right")
        self.state["rail_contacts"]["l_thumb_distal"] = 6.
        self.controller.check_carry(self.state, .01)
        self.assertEqual(self.controller.reason, "released_left_hand_recontact: l_thumb_distal")

    def test_wrong_grasp_region_does_not_pass_close(self):
        self.controller.phase_index = PHASES.index("right_close")
        self.contacts("left", "right")
        self.controller.command[34:54] = self.controller.templates["right"]["closed"]
        self.controller.closed["right"] = True
        with patch("handover.grasp_axis_position", return_value=.2):
            for _ in range(60):
                self.controller.step(self.state, .01)
        self.assertEqual(self.controller.phase, "right_close")
        self.assertEqual(self.controller.status, "running")

    def test_contacts_cannot_start_lifting_before_fingers_finish_closing(self):
        self.controller.phase_index = PHASES.index("left_close")
        self.contacts("left")
        self.controller.command[14:34] = self.controller.templates["left"]["open"]
        with patch("handover.grasp_axis_position", return_value=self.task["left_grasp_axis_m"]):
            for _ in range(12):
                self.controller.step(self.state, .05)
            self.assertEqual(self.controller.phase, "left_close")
            for _ in range(140):
                if self.controller.phase != "left_close":
                    break
                self.controller.step(self.state, .05)
        self.assertEqual(self.controller.phase, "left_lift")

    def test_wrist_pose_feedback_continues_during_finger_closing(self):
        self.controller.phase_index = PHASES.index("left_close")
        self.state["q"][1] += .03
        before = self.controller.command[:7].copy()
        self.controller.step(self.state, 1/120)
        self.assertGreater(float(np.max(abs(self.controller.command[:7]-before))), 1e-6)
        self.assertLessEqual(float(np.max(abs(self.controller.command[:7]-before))), 1/120+1e-12)

    def test_closing_does_not_chase_a_rail_pushed_sideways(self):
        self.controller.phase_index = PHASES.index("left_close")
        held_wrist = self.controller.arm_holds["left"].copy()
        self.controller.step(self.state, .01)
        self.state["rail"][1] -= .01
        self.controller.step(self.state, .01)
        np.testing.assert_array_equal(self.controller.arm_holds["left"], held_wrist)

    def test_pregrasp_waits_for_fingers_to_preshape_before_approach(self):
        self.controller.phase_index = PHASES.index("left_pregrasp")
        pose = self.model.flange("left", self.state["q"], True)
        self.controller.command[14:34] = self.controller.templates["left"]["open"]
        self.controller.command[34:54] = self.controller.templates["right"]["open"]
        self.state["q"] = self.controller.command.tolist()
        self.controller.motion = {"side": "left", "start": pose, "goal": pose, "object": None,
                                  "duration": .001, "trajectory": None}
        self.controller.step(self.state, 1/120)
        self.assertTrue(self.controller.preshaped["left"])
        self.assertEqual(self.controller.phase, "left_pregrasp")
        for _ in range(400):
            self.state["q"] = self.state["q_target"] = self.controller.command.tolist()
            self.controller.step(self.state, 1/120)
            if self.controller.phase != "left_pregrasp":
                break
        self.assertEqual(self.controller.phase, "left_approach")

    def test_one_supporting_finger_cannot_start_end_lifting(self):
        self.controller.phase_index = PHASES.index("left_close")
        self.controller.closed["left"] = True
        self.controller.command[14:34] = self.controller.templates["left"]["closed"]
        self.contacts("left")
        del self.state["rail_contacts"]["l_index_finger_distal"]
        del self.state["rail_contacts"]["l_middle_finger_distal"]
        with patch("handover.grasp_axis_position", return_value=self.task["left_grasp_axis_m"]):
            for _ in range(60):
                self.controller.step(self.state, .01)
        self.assertEqual(self.controller.phase, "left_close")
        self.assertEqual(self.controller.status, "running")

    def test_right_approach_waits_for_right_hook_and_does_not_select_left_hand(self):
        controller = self.controller
        controller.phase_index = PHASES.index("right_pregrasp")
        controller.templates["right"]["preshape"] = controller.templates["right"]["open"].copy()
        self.contacts("left")
        pose = self.model.flange("right", self.state["q"], True)
        controller.motion = {"side": "right", "start": pose, "goal": pose, "object": None,
                             "duration": .001, "trajectory": None}
        controller.step(self.state, 1/120)
        self.assertTrue(controller.preshaped["right"])
        self.assertFalse(controller.preshaped["left"])
        self.assertEqual(controller.phase, "right_pregrasp")
        for _ in range(400):
            self.state["q"] = self.state["q_target"] = controller.command.tolist()
            controller.step(self.state, 1/120)
            if controller.phase != "right_pregrasp":break
        self.assertEqual(controller.phase, "right_approach")

    def test_contact_feedback_waits_for_fingers_still_clear_of_the_rail(self):
        self.controller.task["left_adaptive_closing"] = True
        self.controller.phase_index = PHASES.index("left_close")
        self.controller.closed["left"] = True
        self.controller.command[14:34] = self.controller.templates["left"]["open"]
        self.state["rail_contacts"] = {"l_index_finger_distal": 10.}
        for _ in range(100):
            self.controller.update_hands(.01, self.state)
        progress = self.controller.adaptive_motion["left"]["progress"]
        self.assertEqual(progress[1], 0.)
        self.assertGreater(progress[2], .3)
        self.assertFalse(self.controller.closing_complete("left", self.state))
        self.state["rail_contacts"] = {}
        self.controller.update_hands(.01, self.state)
        self.assertGreater(self.controller.adaptive_motion["left"]["progress"][1], 0.)

    def test_close_failure_reports_missing_support_separately_from_opposition(self):
        self.controller.task["left_adaptive_closing"] = False
        self.controller.phase_index = PHASES.index("left_close")
        self.controller.closed["left"] = True
        self.controller.command[14:34] = self.controller.templates["left"]["closed"]
        self.contacts("left")
        del self.state["rail_contacts"]["l_ring_finger_distal"]
        self.controller.elapsed = 8.
        with patch("handover.grasp_axis_position", return_value=self.task["left_grasp_axis_m"]):
            self.controller.step(self.state, .01)
        self.assertEqual(self.controller.reason, "left_insufficient_supporting_fingers")

    def test_contact_feedback_continues_under_load(self):
        self.controller.task["left_adaptive_closing"] = True
        self.controller.phase_index = PHASES.index("left_close")
        self.controller.closed["left"] = True
        self.controller.command[14:34] = self.controller.templates["left"]["open"]
        self.controller.update_hands(.1, self.state)
        before = self.controller.adaptive_motion["left"]["progress"].copy()
        self.controller.phase_index = PHASES.index("left_lift")
        self.controller.update_hands(.1, self.state)
        self.assertTrue(np.all(self.controller.adaptive_motion["left"]["progress"] > before))

    def test_end_grasp_rotation_is_measured_at_grip_and_real_sliding_stops(self):
        self.controller.phase_index = PHASES.index("left_lift")
        self.contacts("left")
        flange = self.model.flange("left", self.state["q"], True)
        held = np.linalg.inv(flange)@pose_matrix(self.state["rail"])
        self.controller.held["left"] = held.copy()
        pivot = transform([self.task["left_grasp_axis_m"], 0., 0.])
        rotated = held@pivot@transform(rpy=[0., .1, 0.])@np.linalg.inv(pivot)
        self.assertGreater(np.linalg.norm(rotated[:3, 3]-held[:3, 3]), self.task["grip_slip_m"])
        self.state["rail"] = pose_vector(flange@rotated)
        self.assertIsNone(self.controller.check_carry(self.state, .01))
        self.assertEqual(self.controller.status, "running")
        rotated[0, 3] += self.task["loading_grip_slip_m"]+.001
        self.state["rail"] = pose_vector(flange@rotated)
        self.controller.check_carry(self.state, .01)
        self.assertEqual(self.controller.status, "failed")
        self.assertTrue(self.controller.reason.startswith("left_grasp_slip:"))

    def test_vertical_extraction_clears_entire_bin_without_rotating_rail(self):
        from itertools import product
        from bimanual_teleop.sim.bin_geometry import bin_geometry
        task = copy.deepcopy(self.task); task["left_lift_vertical_m"] = .20
        start = transform(self.config["rail"]["initial_center"], np.deg2rad(self.config["rail"]["initial_rpy_deg"]))
        half = np.asarray(self.config["rail"]["size_m"])/2
        corners = np.array([np.r_[half*np.asarray(signs), 1.] for signs in product([-1., 1.], repeat=3)])
        lowest = []
        for fraction in np.linspace(0., 1., 41):
            pose = extraction_lift_pose(self.config, task, start, fraction)
            np.testing.assert_allclose(pose[:3, :3], start[:3, :3])
            np.testing.assert_allclose(pose[:2, 3], start[:2, 3])
            lowest.append(float((pose@corners.T)[2].min()))
        self.assertTrue(np.all(np.diff(lowest) >= 0.))
        rim = self.config["scene"]["bin_centers"][0][2]+bin_geometry(self.config["scene"])["rim_height_m"]
        self.assertGreater(lowest[-1], rim+.02)

    def test_vertical_extraction_rejects_invalid_height(self):
        for value in (0., -.1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                extraction_lift_pose(self.config, {**self.task, "left_lift_vertical_m": value}, np.eye(4))

    def prepare_vertical_lift(self):
        controller = self.controller
        controller.task["left_lift_vertical_m"] = .28
        controller.phase_index = PHASES.index("left_lift")
        self.contacts("left")
        flange = self.model.flange("left", self.state["q"], True)
        rail = pose_matrix(self.state["rail"])
        controller.initial_rail = rail.copy()
        goal_flange = flange.copy(); goal_flange[2, 3] += .28
        goal_rail = rail.copy(); goal_rail[2, 3] += .28
        controller.phase_pose = rail.copy()
        controller.motion = {"side": "left", "start": flange, "goal": goal_flange,
                             "object": goal_rail, "duration": 1., "trajectory": None}
        return flange, rail

    def test_vertical_lift_does_not_twist_wrist_to_chase_load_seating(self):
        flange, rail = self.prepare_vertical_lift()
        self.state["rail"] = pose_vector(rail@transform(rpy=[.1, 0., 0.]))
        with patch.object(self.controller, "servo") as servo:
            self.controller._step(self.state, .1)
        target = servo.call_args.args[2]
        np.testing.assert_allclose(target[:3, :3], flange[:3, :3], atol=1e-12)

    def test_vertical_lift_cannot_pass_with_low_end_still_below_bin_rim(self):
        flange, rail = self.prepare_vertical_lift()
        self.controller.motion["goal"] = flange.copy()
        rail[2, 3] = .90
        self.state["rail"] = pose_vector(rail)
        self.controller.initial_rail[2, 3] = .70
        self.controller.elapsed = 1.
        self.controller.linear_speed = self.controller.angular_speed = 0.
        self.controller.stable = 1.
        with patch.object(self.controller, "servo"):
            self.controller._step(self.state, .01)
        self.assertNotIn("left_lift", self.controller.completed_phases)
        self.assertEqual(self.controller.stable, 0.)

    def test_extraction_arc_keeps_far_end_above_its_initial_floor_height(self):
        task = {k: v for k, v in self.task.items() if k != "left_lift_vertical_m"}
        start = transform(self.config["rail"]["initial_center"], np.deg2rad(self.config["rail"]["initial_rpy_deg"]))
        far_end = np.array([-self.config["rail"]["size_m"][0]/2, 0., -self.config["rail"]["size_m"][2]/2, 1.])
        initial_height = (start@far_end)[2]
        for fraction in np.linspace(0., 1., 31):
            pose = extraction_lift_pose(self.config, task, start, fraction)
            self.assertGreaterEqual((pose@far_end)[2], initial_height-1e-10)
        np.testing.assert_allclose(extraction_lift_pose(self.config, task, start, 0.), start)
        goal = extraction_lift_pose(self.config, task, start)
        self.assertGreater(goal[2, 3]-start[2, 3], self.task["lift_m"])
        self.assertAlmostEqual((goal@far_end)[2]-initial_height, self.task["left_lift_floor_clearance_m"])

    def test_extraction_tilts_before_pulling_and_preserves_floor_support(self):
        task = {k: v for k, v in self.task.items() if k != "left_lift_vertical_m"}
        start = transform(self.config["rail"]["initial_center"], np.deg2rad(self.config["rail"]["initial_rpy_deg"]))
        corner = np.array([-self.config["rail"]["size_m"][0]/2, 0., -self.config["rail"]["size_m"][2]/2, 1.])
        pivot = start@corner
        early = extraction_lift_pose(self.config, task, start, .35)
        np.testing.assert_allclose(early@corner, pivot)
        self.assertGreater(early[2, 3], start[2, 3]+.01)
        late = extraction_lift_pose(self.config, task, start, .75)
        self.assertAlmostEqual((late@corner)[2], pivot[2])
        self.assertLess((late@corner)[0], pivot[0])

    def test_loading_reference_is_captured_before_stricter_carry_check(self):
        self.controller.phase_index = PHASES.index("left_lift")
        self.contacts("left")
        self.controller.last_state = self.state
        self.controller.complete_phase()
        self.assertEqual(self.controller.phase, "left_clear")
        np.testing.assert_allclose(self.controller.held["left"],
                                   np.linalg.inv(self.model.flange("left", self.state["q"], True))@pose_matrix(self.state["rail"]))
        shifted = pose_matrix(self.state["rail"])
        shifted[0, 3] += .010
        self.assertLess(.010, self.task["loading_grip_slip_m"])
        self.state["rail"] = pose_vector(shifted)
        self.controller.check_carry(self.state, .01)
        self.assertEqual(self.controller.status, "failed")
        self.assertTrue(self.controller.reason.startswith("left_grasp_slip:"))

    def test_scene_signature_detects_changed_fixture_and_leaves_defaults_alone(self):
        before = load_config()
        digest = scene_signature(self.config)
        self.config["scene"]["fixture_origin"][0] += .01
        self.assertNotEqual(scene_signature(self.config), digest)
        self.assertEqual(load_config(), before)
        path = Path(self.temp.name)/"incompatible.json"
        path.write_text(json.dumps({"scene_signature": digest}))
        with self.assertRaisesRegex(ValueError, "Replay scene differs"):
            HandoverReplayController(self.config, self.model, self.task, self.temp.name, path, mode="auto")

    def test_demo_manifest_identity_is_preserved_in_generated_trajectory(self):
        self.controller.save_trajectory()
        document = json.loads((self.controller.trial/"trajectory.json").read_text())
        self.assertEqual(document["demo_provenance"], self.task["demo_provenance"])
        self.assertEqual(document["scene_signature"], scene_signature(self.config, self.task))
        self.assertEqual(document["strategy"], "bimanual_handover")

    def test_replay_rejects_changed_runtime_collision_geometry(self):
        task = {**self.task, "convex_decomposition_links": ["l_mount", "r_mount"]}
        self.assertNotEqual(scene_signature(self.config, task), scene_signature(self.config))
        self.assertEqual(scene_signature(self.config, task), scene_signature(self.config, {**task, "convex_decomposition_links": ["r_mount", "l_mount"]}))
        path = Path(self.temp.name)/"old-collision-geometry.json"
        path.write_text(json.dumps({"scene_signature": scene_signature(self.config)}))
        with self.assertRaisesRegex(ValueError, "Replay scene differs"):
            HandoverReplayController(self.config, self.model, task, self.temp.name, path, mode="auto")

    def test_replay_rejects_changed_common_physics_code_with_identical_config(self):
        path = Path(self.temp.name)/"old-physics.json"
        path.write_text(json.dumps({"scene_signature": scene_signature(self.config, self.task)}))
        changed = Path(self.temp.name)/"changed-collisions.py"
        changed.write_text("# Different common collision implementation\n")
        with patch("handover.collision_module.__file__", str(changed)):
            with self.assertRaisesRegex(ValueError, "Replay scene differs"):
                HandoverReplayController(self.config, self.model, self.task, self.temp.name, path, mode="auto")

    def test_trajectory_start_drift_cannot_create_a_joint_target_jump(self):
        self.controller.phase_index = PHASES.index("left_pregrasp")
        pose = self.model.flange("left", self.state["q"], True)
        class Trajectory:
            def eval(_self, time):
                return np.array(self.state["q"][:7])+1.
        self.controller.motion = {"side": "left", "start": pose, "goal": pose,
                                  "object": None, "duration": 2., "trajectory": Trajectory(),
                                  "tracking_bias": np.zeros(7)}
        before = self.controller.command.copy()
        result = self.controller.step(self.state, 1/120)
        self.assertLessEqual(float(np.max(abs(result[:14]-before[:14]))), 1/120+1e-12)
        self.assertTrue(np.all(result >= self.model.lower) and np.all(result <= self.model.upper))


if __name__ == "__main__":
    unittest.main()
