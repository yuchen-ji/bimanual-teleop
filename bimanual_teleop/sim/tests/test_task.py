import copy
import json
import unittest

from bimanual_teleop.sim._dependencies import np

from bimanual_teleop.sim.config import load_config
from bimanual_teleop.sim.task import TaskMonitor


class TaskTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        # This synthetic fixture deliberately exercises the original 211 mm pair.
        self.config["rail"].update(hole_spacing_m=.211, size_m=[.4, .045, .025])
        self.geometry = {"pin_centers_m": [[0, -.1055, .04], [0, .1055, .04]], "support_height_m": .04, "pin_top_height_m": .049}
        self.monitor = TaskMonitor(self.config, self.geometry)

    def fixture_state(self, stamp, contacts=(0, 0), offset=0):
        position = np.array(self.config["scene"]["fixture_origin"])+[offset, 0, .04+.0125]
        return {"time_ns": int(stamp*1e9), "rail": [*position, 0, 0, np.sin(np.pi/4), np.cos(np.pi/4)],
                "hand_contact_force": list(contacts)}

    def test_release_and_stability_required(self):
        for i in range(80):
            metrics = self.monitor.update(self.fixture_state(i/30, contacts=(1, 0)))
        self.assertFalse(metrics["fixture_completed"])
        for i in range(80, 112):
            metrics = self.monitor.update(self.fixture_state(i/30))
        self.assertTrue(metrics["fixture_completed"])

    def test_offset_and_unknown_contact_do_not_succeed(self):
        for i in range(80):
            state = self.fixture_state(i/30, offset=.003)
            metrics = self.monitor.update(state)
        self.assertFalse(metrics["fixture_completed"])
        for i in range(80, 160):
            state = self.fixture_state(i/30); state.pop("hand_contact_force")
            metrics = self.monitor.update(state)
        self.assertFalse(metrics["fixture_completed"])

    def test_metrics_are_json_serializable_above_and_on_fixture(self):
        for height in (.18, 0):
            state = self.fixture_state(1)
            state["rail"][2] += height
            metrics = self.monitor.update(state)
            self.assertIsInstance(metrics["fixture_seated"], bool)
            json.dumps(metrics)

    def test_full_task_requires_prior_fixture_stage(self):
        position = np.array(self.config["scene"]["bin_centers"][1])+[0, 0, .0225]
        for i in range(80):
            metrics = self.monitor.update({"time_ns": int(i/30*1e9), "rail": [*position, 0, 0, np.sin(np.pi/4), np.cos(np.pi/4)], "hand_contact_force": [0, 0]})
        self.assertFalse(metrics["task_completed"])
