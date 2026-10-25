"""Physical-scene defaults, precision presets and legacy task compatibility."""

import copy
import unittest

from bimanual_teleop.sim import ROOT
from bimanual_teleop.sim._dependencies import np
from bimanual_teleop.sim.collisions import collision_policy
from bimanual_teleop.sim.config import load_config, sim_path, validate
from bimanual_teleop.sim.kinematics import RobotModel, transform
from bimanual_teleop.sim.bin_geometry import bin_geometry


class CollisionPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.model = RobotModel(sim_path(cls.config["robot"]["urdf"]))

    def test_measured_mass_replaces_assumption_without_changing_default_pose(self):
        self.assertEqual(self.config["rail"]["mass_kg"], .7642)
        self.assertFalse(self.config["rail"]["mass_is_assumption"])
        np.testing.assert_allclose(self.config["rail"]["initial_center"], [.527767116, .006474946, .821278683])
        self.assertEqual(self.config["rail"]["finger_friction"], .8)

    def test_mount_recesses_are_preserved_independently_of_finger_precision(self):
        policy = collision_policy(self.config, self.model)
        for name in ("l_mount", "r_mount"):
            self.assertEqual(policy[name]["approximation"], "convexDecomposition")
        for name in ("base_link", "l_camera_bracket", "r_force_sensor"):
            self.assertEqual(policy[name]["approximation"], "convexHull")
        self.assertEqual(policy["l_index_finger_distal"]["approximation"], "convexHull")

    def test_hand_contact_preset_only_changes_finger_precision(self):
        detailed = load_config(ROOT/"configs/hand_contact.yaml")
        baseline = copy.deepcopy(self.config)
        baseline["robot"]["collision"]["finger_approximation"] = "convexDecomposition"
        self.assertEqual(detailed, baseline)
        policy = collision_policy(detailed, self.model)
        self.assertEqual(policy["r_pinky_distal"]["approximation"], "convexDecomposition")

    def test_bin_preset_rail_is_mostly_inside_robot_side_narrow_rim(self):
        config = load_config(ROOT/"configs/rail_bin_pick.yaml")
        rail = config["rail"];scene = config["scene"]
        self.assertEqual(rail["mass_kg"], .7642)
        self.assertEqual(rail["finger_friction"], .8)
        pose = transform(rail["initial_center"], np.deg2rad(rail["initial_rpy_deg"]))
        ends = np.array([[-rail["size_m"][0]/2, 0, 0], [rail["size_m"][0]/2, 0, 0]])@pose[:3,:3].T+pose[:3,3]
        low,high = ends[np.argsort(ends[:,2])]
        box = np.asarray(scene["bin_centers"][0]);near = box[0]-scene["bin_inner_size"][0]/2
        self.assertLess(high[0], near);self.assertGreater(low[0], near)
        self.assertLess(near-high[0], .01)
        self.assertLess(abs(low[1]-box[1]), .01)
        rim = box[2]+bin_geometry(scene)["rim_height_m"]
        crossing = low+(near-low[0])/(high[0]-low[0])*(high-low)
        self.assertLess(abs(crossing[2]-rim), .02)

    def test_legacy_task_can_request_precise_fingers_without_changing_default(self):
        policy = collision_policy(self.config,self.model,["l_index_finger_distal"])
        self.assertEqual(policy["l_index_finger_distal"]["approximation"],"convexDecomposition")
        self.assertEqual(policy["r_index_finger_distal"]["approximation"],"convexHull")
        self.assertEqual(self.config["robot"]["collision"]["finger_approximation"],"convexHull")
        with self.assertRaisesRegex(ValueError,"Unknown collision decomposition links"):
            collision_policy(self.config,self.model,["missing_finger"])

    def test_invalid_physics_settings_rejected_before_isaac(self):
        for section,name,value in (("collision","finger_approximation","none"),
                                   ("rail","finger_friction",float("nan")),
                                   ("rail","rest_offset_m",.001),
                                   ("rail","mass_is_assumption","false")):
            config=copy.deepcopy(self.config)
            (config["robot"][section] if section=="collision" else config[section])[name]=value
            with self.assertRaises(ValueError):validate(config)
