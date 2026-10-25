import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[2])]
from unsupported_hold import UnsupportedHold


class UnsupportedHoldTests(unittest.TestCase):
    def run_case(self, elapsed, drop=0., attached=False):
        with tempfile.TemporaryDirectory() as directory:
            view = NS(shared_metatype=NS(link_names=[]), get_link_transforms=lambda: np.empty((1, 0, 7)))
            finished = []
            source = NS(output=Path(directory), command=np.zeros(54), metrics={}, support_active=False,
                        flange=lambda s: np.eye(4), joint=NS(GetJointEnabledAttr=lambda: NS(Get=lambda: attached)),
                        body=NS(GetKinematicEnabledAttr=lambda: NS(Get=lambda: False)),
                        robot=NS(_articulation_view=NS(_physics_view=view)), finish=lambda *a: finished.append(a))
            initial = {"time_ns": 0, "rail": [0., 0., 1., 0., 0., 0., 1.]}
            control = UnsupportedHold(source, initial)
            state = {"time_ns": int(elapsed*1e9), "rail": [0., 0., 1.-drop, 0., 0., 0., 1.]}
            control.step(state, .01)
            return finished

    def test_slip_stops_test(self):
        self.assertIn("slipped", self.run_case(.1, drop=.003)[0][1])

    def test_enabled_attachment_invalidates_test(self):
        self.assertIn("invalid", self.run_case(.1, attached=True)[0][1])

    def test_static_hold_requires_full_duration(self):
        self.assertEqual(self.run_case(4.9), [])
        self.assertEqual(self.run_case(5.1)[0][0], "unsupported_static_hold_passed")
