import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[2])]
from tighten_grasp import TightenGrasp


class TighteningTests(unittest.TestCase):
    def create(self, output):
        source = SimpleNamespace(command=np.zeros(54), metrics={}, support_active=True,
                                 event=lambda *a, **k: None, finish=lambda *a: None)
        state = {'rail': [0, 0, 1, 0, 0, 0, 1], 'rail_contacts': {}}
        return source, state, TightenGrasp(source, state, Path(output))

    def test_intersection_rolls_back_without_removing_support(self):
        with tempfile.TemporaryDirectory() as output:
            source, state, grasp = self.create(output)
            grasp.target[18] = .02
            grasp.phase = 'review'
            grasp.response = lambda: {'geometry': {'intersecting_links': ['l_index_finger_tip_sensor_frame']},
                                      'stop': 'Mesh intersection during tightening'}
            target = grasp.step(state, .01)
            np.testing.assert_array_equal(target, grasp.initial)
            self.assertEqual(grasp.phase, 'rollback')
            self.assertTrue(source.support_active)

    def test_overload_stops_further_closure(self):
        with tempfile.TemporaryDirectory() as output:
            source, state, grasp = self.create(output)
            state['rail_contacts']['l_thumb_tip_sensor_frame'] = 4.1
            grasp.target[14] = .02
            target = grasp.step(state, .01)
            np.testing.assert_array_equal(target, grasp.initial)
            self.assertTrue(source.support_active)
            self.assertIn('4 N', grasp.failure_reason)

    def test_unavailable_checker_does_not_loop_forever_during_rollback(self):
        with tempfile.TemporaryDirectory() as output:
            source, _, grasp = self.create(output)
            stopped = []
            source.finish = lambda *args: stopped.append(args)
            grasp.phase = 'rollback_review'
            grasp.request_id = 'missing'
            grasp.request_started = 0.
            with patch('tighten_grasp.time.monotonic', return_value=31.):
                self.assertIsNone(grasp.response())
            self.assertEqual(stopped[0][0], 'failed')
            self.assertTrue(source.support_active)
            self.assertTrue(source.pause_physics_requested)
