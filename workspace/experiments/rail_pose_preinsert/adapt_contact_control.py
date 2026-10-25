"""Measured short simulation segments, controlled only by reviewed local targets."""
import json
from collections import deque
from pathlib import Path

import numpy as np
from common import finger_forces, pose_error, write_json
from scipy.spatial.transform import Rotation

from bimanual_teleop.sim.kinematics import pose_matrix

INITIAL_CHECKPOINT = Path(__file__).parent/'hand_profiles/left_grasp_initial_state.json'


def validate_initial_checkpoint(value, model):
    q, target, rail = (np.asarray(value[key], dtype=float) for key in ('q', 'q_target', 'rail'))
    if q.shape != (54,) or target.shape != (54,) or rail.shape != (7,) or not np.isfinite(np.r_[q, target, rail]).all():
        raise ValueError('Invalid saved grasp state')
    if np.any(target < model.lower) or np.any(target > model.upper):
        raise ValueError('Saved grasp drive targets exceed limits')
    if np.any(q < model.lower-.001) or np.any(q > model.upper+.001):
        raise ValueError('Saved measured joints exceed limits')
    if not np.isclose(np.linalg.norm(rail[3:]), 1., atol=1e-5):
        raise ValueError('Invalid saved rail quaternion')
    if value.get('fixed_constraint_enabled', True) or value.get('rail_kinematic', True):
        raise ValueError('Initial grasp must have neither attachment nor support')
    return value


def snapshot(source, state):
    result = dict(state)
    view = source.robot._articulation_view._physics_view
    names = list(view.shared_metatype.link_names)
    if names and isinstance(names[0], (tuple, list)):
        names = list(names[0])
    poses = view.get_link_transforms()[0]
    result['hand_link_poses'] = {n: np.asarray(poses[i]).tolist() for i, n in enumerate(names)
                               if n == 'l_wrist' or n.startswith(('l_thumb_', 'l_index_finger_', 'l_middle_finger_', 'l_ring_finger_', 'l_pinky_'))}
    result['fixed_constraint_enabled'] = bool(source.joint.GetJointEnabledAttr().Get())
    result['rail_kinematic'] = bool(source.body.GetKinematicEnabledAttr().Get())
    result['rail_linear_velocity_m_s'] = np.asarray(source.rail.get_linear_velocity()).tolist()
    result['rail_angular_velocity_rad_s'] = np.asarray(source.rail.get_angular_velocity()).tolist()
    return result


class ContactSegment:
    def __init__(self, source, request):
        self.source = source
        self.request = request
        self.start = source.command.copy()
        self.target = np.asarray(request['q'])
        self.start_rail_xyz = np.asarray(source.last_state['rail'][:3])
        self.target_rail_xyz = np.asarray(request['rail'][:3])
        self.rail_xyz = self.start_rail_xyz.copy()
        self.quaternion = np.asarray(request['rail'][3:])
        self.elapsed = 0.
        self.request_id = None
        self.duration = request.get('duration_s', 1.)
        self.settle = request.get('settle_s', 1.)
        self.history = deque(maxlen=1200) if request.get('continuous') else []
        self.last_status_sample = 0.
        self.initial_relative = np.linalg.inv(source.flange(source.last_state)) @ pose_matrix(source.last_state['rail'])
        self.maximum = np.zeros(2)

    def step(self, state, dt):
        self.elapsed += dt
        u = min(1., self.elapsed/max(.01, self.duration))
        u = u*u*u*(10-15*u+6*u*u)
        self.rail_xyz = (1-u)*self.start_rail_xyz+u*self.target_rail_xyz
        desired = (1-u)*self.start+u*self.target
        force = finger_forces(state)
        self.history.append({'time_s': self.elapsed, 'forces': force, 'rail_contacts': state.get('rail_contacts'),
                             'rail': state['rail'], 'eef': state['eef'][0]})
        reason = None
        if max(force.values()) > self.request.get('force_limit_n', 8.):
            reason = 'Finger force exceeded segment limit'
        if max(state.get('environment_contacts', {}).values(), default=0.) > .5:
            reason = 'Robot touched environment'
        if not self.source.support_active:
            relative = np.linalg.inv(self.source.flange(state)) @ pose_matrix(state['rail'])
            err = pose_error(relative, self.initial_relative)
            self.maximum = np.maximum(self.maximum, err)
            if err[0] > .002 or err[1] > np.deg2rad(3.):
                reason = 'Rail slipped in unsupported test'
            if max(state.get('rail_support_contacts', {}).values(), default=0.) > .05:
                reason = 'Rail touched external support'
        if self.request.get('continuous') and self.elapsed-self.last_status_sample >= 1.:
            self.last_status_sample = self.elapsed
            write_json(self.source.adaptation_output/'initial_grasp_live_state.json', snapshot(self.source, state))
        if reason or (not self.request.get('continuous') and self.elapsed >= self.duration+self.settle):
            measured = snapshot(self.source, state)
            identifier = self.request['id']
            write_json(self.source.adaptation_output/(identifier+'_state.json'), measured)
            write_json(self.source.adaptation_output/(identifier+'_trace.json'), list(self.history))
            write_json(self.source.adaptation_output/(identifier+'_result.json'), {'complete': True, 'reason': reason,
                       'elapsed_s': self.elapsed, 'unsupported': not self.source.support_active,
                       'max_relative_error_m_rad': self.maximum.tolist(), 'state_file': identifier+'_state.json'})
            self.source.finish('adaptation_failed' if reason else 'adaptation_checkpoint', reason or identifier)
            self.source.snapshot_requested = self.source.failure_snapshot_pending = False
            self.source.pause_physics_requested = True
        return desired


from controller import PreinsertController


class AdaptContactController(PreinsertController):
    def __init__(self, *args, adaptation_output, **kwargs):
        self.adaptation_output = Path(adaptation_output)
        self.adaptation_ready = False
        self.adaptation_seen = None
        self.initial_checkpoint = json.loads(INITIAL_CHECKPOINT.read_text()) if INITIAL_CHECKPOINT.exists() else None
        super().__init__(*args, **kwargs)

    def load_initial_checkpoint(self):
        self.initial_checkpoint = validate_initial_checkpoint(json.loads(INITIAL_CHECKPOINT.read_text()), self.model)
        self.ready = np.asarray(self.initial_checkpoint['q_target']).copy()
        self.config['rail']['initial_center'] = self.initial_checkpoint['rail'][:3]
        self.config['rail']['initial_rpy_deg'] = Rotation.from_quat(self.initial_checkpoint['rail'][3:]).as_euler('xyz', degrees=True).tolist()

    def before_reset(self):
        if INITIAL_CHECKPOINT.exists():
            self.load_initial_checkpoint()
        super().before_reset()

    def reset(self):
        super().reset()
        self.adaptation_ready = False
        self.waiting = True
        self.initial_checkpoint_pending = False
        if self.initial_checkpoint is not None:
            self.load_initial_checkpoint()
            self.command = self.ready.copy()
            self.support_active = False
            self.waiting = False
            self.phase = 'initial_grasp_hold'
            self.adaptation_ready = True
            self.adaptation_orientation = np.asarray(self.initial_checkpoint['rail'][3:])
            path = self.adaptation_output/'segment.json'
            self.adaptation_seen = json.loads(path.read_text())['id'] if path.exists() else None

    def scene_ready(self, drives):
        super().scene_ready(drives)
        if self.initial_checkpoint is None:
            return
        from isaacsim.core.utils.types import ArticulationAction
        saved = self.initial_checkpoint
        indices = np.array([self.robot.get_dof_index(name) for name in self.model.names], dtype=np.int32)
        # Engine ready/drive targets retain the preload. Initialize the physical
        # joints separately so the commanded compression is not teleported in.
        self.robot.set_joint_positions(np.asarray(saved['q']), joint_indices=indices)
        self.robot.set_joint_velocities(np.zeros(54), joint_indices=indices)
        self.robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=self.command, joint_indices=indices))
        rail = np.asarray(saved['rail'])
        self.joint.CreateJointEnabledAttr(False)
        self.body.CreateKinematicEnabledAttr(False)
        self.ccd.CreateEnableCCDAttr(True)
        self.support_active = False
        self.rail.set_world_pose(rail[:3], rail[[6, 3, 4, 5]])
        self.rail.set_linear_velocity(np.zeros(3))
        self.rail.set_angular_velocity(np.zeros(3))
        self.initial_checkpoint_pending = True

    def service_reload(self):
        super().service_reload()
        if not self.bound or self.last_state is None:
            return
        if self.initial_checkpoint_pending:
            self.initial_checkpoint_pending = False
            request = {'id': f'initial_grasp_{self.reset_index}', 'q': self.command.tolist(),
                       'rail': self.initial_checkpoint['rail'], 'supported': False,
                       'continuous': True, 'force_limit_n': 12.}
            self.locked_grasp = ContactSegment(self, request)
            self.phase = 'locked_grasp_close'
            self.status = 'initial_grasp_ready'
            self.config_message = 'Saved grasp initial state. Support OFF, attachment OFF. Play to inspect; Restart restores this grasp.'
            self.physics_action_requested = 'pause'
            write_json(self.adaptation_output/'initial_grasp_restored_state.json', snapshot(self, self.last_state))
            self.report()
            return
        if not self.adaptation_ready:
            self.adaptation_ready = True
            self.finished = True
            self.physics_action_requested = 'pause'
            self.adaptation_orientation = np.asarray(self.last_state['rail'][3:])
            self.status = 'adaptation_ready'
            self.config_message = 'Demonstration adaptation: waiting for verified contact target'
            write_json(self.adaptation_output/'ready_state.json', snapshot(self, self.last_state))
            import os
            write_json(self.adaptation_output/'ready.json', {'pid': os.getpid(), 'ready': True})
            self.report()
            return
        if not self.finished or self.world.is_playing():
            return
        path = self.adaptation_output/'segment.json'
        if not path.exists():
            return
        request = json.loads(path.read_text())
        if request['id'] == self.adaptation_seen:
            return
        self.adaptation_seen = request['id']
        try:
            self.start_contact_segment(request)
            write_json(self.adaptation_output/'command_result.json', {'id': request['id'], 'ok': True})
        except (ValueError, AssertionError) as error:
            write_json(self.adaptation_output/'command_result.json', {'id': request['id'], 'ok': False, 'error': str(error)})

    def start_contact_segment(self, request):
        q = np.asarray(request['q'])
        rail = np.asarray(request['rail'])
        if q.shape != (54,) or rail.shape != (7,) or not np.isfinite(np.r_[q, rail]).all():
            raise ValueError('Invalid segment shape or nonfinite target')
        if np.any(q < self.model.lower) or np.any(q > self.model.upper):
            raise ValueError('Joint target exceeds model limits')
        np.testing.assert_allclose(q[7:14], self.command[7:14], atol=1e-7)
        np.testing.assert_allclose(q[34:], self.command[34:], atol=1e-7)
        rotation = Rotation.from_quat(rail[3:]).inv()*Rotation.from_quat(self.adaptation_orientation)
        if rotation.magnitude() > 1e-5:
            raise ValueError('Pregrasp rail rotation is locked')
        if request.get('initialize'):
            raise ValueError('Live teleportation is not allowed; initialize through scene configuration')
        self.joint.CreateJointEnabledAttr(False)
        was_supported = self.support_active
        self.support_active = bool(request.get('supported', True))
        self.body.CreateKinematicEnabledAttr(self.support_active)
        self.ccd.CreateEnableCCDAttr(not self.support_active)
        if was_supported and not self.support_active:
            self.rail.set_linear_velocity(np.zeros(3))
            self.rail.set_angular_velocity(np.zeros(3))
        self.reset_index += 1
        self.phase = 'locked_grasp_close'
        self.status, self.reason = 'running', None
        self.finished = self.waiting = self.paused = False
        self.snapshot_requested = self.failure_snapshot_pending = self.pause_physics_requested = False
        self.pending_attachment = self.pending_placement = False
        self.grasp_mode = 'physics'
        self.grip = None
        self.locked_grasp = ContactSegment(self, request)
        self.config_message = 'Adapting demonstrated grasp; fixed constraint disabled'
        self.event('recorded_grasp_segment_started', segment=request['id'], supported=self.support_active)
        self.physics_action_requested = 'play'
        self.report()
