"""Contact trials must stop on slip/overload and reject rail rotation."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import adapt_contact_control as control


def state(x=0., force=0.):
    return {'rail': [x, 0., 0., 0., 0., 0., 1.], 'eef': [[0., 0., 0., 0., 0., 0., 1.]],
            'rail_contacts': {'l_index_finger_tip_sensor_frame': force}}


def source(path, supported):
    result = SimpleNamespace(command=np.zeros(54), support_active=supported, last_state=state(),
                             adaptation_output=path, finished=False, flange=lambda _: np.eye(4))
    def finish(status, reason):
        result.finished = True
        result.reason = reason
    result.finish = finish
    return result


class ContactSafetyTests(unittest.TestCase):
    def test_initial_grasp_restores_measured_pose_and_separate_drive_preload(self):
        s = control.AdaptContactController.__new__(control.AdaptContactController)
        s.initial_checkpoint = {'q': [0.]*54, 'rail': state()['rail']}
        s.command = np.full(54, .1)
        s.model = SimpleNamespace(names=[f'j{i}' for i in range(54)])
        s.robot, s.rail, s.joint, s.body, s.ccd = [MagicMock() for _ in range(5)]
        s.robot.get_dof_index.side_effect = s.model.names.index
        with patch.object(control.PreinsertController, 'scene_ready'), patch.dict(
                sys.modules, {'isaacsim.core.utils.types': SimpleNamespace(ArticulationAction=SimpleNamespace)}):
            s.scene_ready({})
        np.testing.assert_array_equal(s.robot.set_joint_positions.call_args.args[0], np.zeros(54))
        action = s.robot.get_articulation_controller().apply_action.call_args.args[0]
        np.testing.assert_array_equal(action.joint_positions, np.full(54, .1))
        s.body.CreateKinematicEnabledAttr.assert_called_once_with(False)
        s.joint.CreateJointEnabledAttr.assert_called_once_with(False)
        self.assertFalse(s.support_active)

    def test_supported_checkpoint_is_rejected(self):
        saved = {'q': [0.]*54, 'q_target': [0.]*54, 'rail': state()['rail'],
                 'fixed_constraint_enabled': False, 'rail_kinematic': True}
        model = SimpleNamespace(lower=np.full(54, -2.), upper=np.full(54, 2.))
        with self.assertRaisesRegex(ValueError, 'neither attachment nor support'):
            control.validate_initial_checkpoint(saved, model)

    def test_continuous_hold_does_not_finish_at_duration_and_bounds_history(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(control, 'snapshot', side_effect=lambda source, sample: sample):
            s = source(Path(directory), False)
            segment = control.ContactSegment(s, {'id': 'continuous', 'q': [0.]*54,
                                                'rail': state()['rail'], 'continuous': True})
            for _ in range(1300):
                segment.step(state(), .01)
            self.assertFalse(s.finished)
            self.assertEqual(len(segment.history), 1200)

    def test_unsupported_slip_is_a_failure(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(control, 'snapshot', side_effect=lambda source, sample: sample):
            s = source(Path(directory), False)
            segment = control.ContactSegment(s, {'id': 'slip', 'q': [0.]*54, 'rail': state()['rail'], 'duration_s': 5})
            segment.step(state(x=.0021), .01)
            self.assertTrue(s.finished and s.pause_physics_requested)
            self.assertIn('slipped', s.reason)

    def test_contact_overload_stops_before_duration(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(control, 'snapshot', side_effect=lambda source, sample: sample):
            s = source(Path(directory), True)
            segment = control.ContactSegment(s, {'id': 'overload', 'q': [0.]*54, 'rail': state()['rail'], 'duration_s': 5, 'force_limit_n': 4.})
            segment.step(state(force=4.1), .01)
            self.assertTrue(s.finished and s.pause_physics_requested)
            self.assertIn('force', s.reason)

    def test_rail_rotation_rejected_before_physics_changes(self):
        s = control.AdaptContactController.__new__(control.AdaptContactController)
        s.model = SimpleNamespace(lower=np.full(54, -2.), upper=np.full(54, 2.))
        s.command = np.zeros(54)
        s.adaptation_orientation = [0., 0., 0., 1.]
        with self.assertRaisesRegex(ValueError, 'rotation is locked'):
            s.start_contact_segment({'id': 'rotation', 'q': [0.]*54, 'rail': [0., 0., 0., 0., 0., .01, .99995]})

    def test_right_hand_target_rejected_before_physics_changes(self):
        s = control.AdaptContactController.__new__(control.AdaptContactController)
        s.model = SimpleNamespace(lower=np.full(54, -2.), upper=np.full(54, 2.))
        s.command = np.zeros(54)
        q = np.zeros(54); q[34] = .1
        with self.assertRaises(AssertionError):
            s.start_contact_segment({'id': 'right', 'q': q.tolist(), 'rail': state()['rail']})


if __name__ == '__main__':
    unittest.main()
