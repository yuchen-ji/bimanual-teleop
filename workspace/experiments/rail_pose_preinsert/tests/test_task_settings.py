import copy
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[2])]

from task_settings import apply_overrides, positive_settings

from bimanual_teleop.sim.config import load_config, sim_path
from bimanual_teleop.sim.kinematics import RobotModel


class SettingsTests(unittest.TestCase):
    def setUp(self):
        config = load_config()
        self.model = RobotModel(sim_path(config["robot"]["urdf"]))
        self.grasp = np.clip(np.full(20, .4), self.model.lower[14:34], self.model.upper[14:34])

    def test_existing_yaml_grasp_overrides_remain_loadable_without_a_panel(self):
        task = {"until": "grasp", "joint_speed_rad_s": .06,
                "hand_grasp_override_rad": self.grasp.tolist(),
                "rail_in_flange_override": {"xyz_m": [.012, -.008, .155], "rpy_deg": [8., -6., 20.]}}
        references = apply_overrides(task, {"grasp": self.grasp}, self.model)
        np.testing.assert_allclose(references["grasp"], self.grasp)
        np.testing.assert_allclose(task["rail_in_flange_xyz_m"], [.012, -.008, .155])

    def test_invalid_yaml_does_not_change_the_active_task(self):
        task = {"hand_grasp_override_rad": self.grasp.tolist()}
        task["hand_grasp_override_rad"][0] = self.model.upper[14]+1
        original = copy.deepcopy(task)
        with self.assertRaisesRegex(ValueError, "joint limits"):
            apply_overrides(task, {"grasp": self.grasp}, self.model)
        self.assertEqual(task, original)

    def test_host_rejects_invalid_planning_speeds(self):
        for speed in (0., -.1, float("inf"), float("nan")):
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                positive_settings({"joint_speed_rad_s": speed})


if __name__ == "__main__":
    unittest.main()
