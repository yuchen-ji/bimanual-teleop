"""Apply small measured closure steps; reject penetration before an unsupported test."""
import json
import time

import numpy as np
from common import finger_forces, write_json
from locked_grasp import LockedGrasp
from unsupported_hold import UnsupportedHold


class TightenGrasp(LockedGrasp):
    def __init__(self, source, state, output):
        self.rpc_output = output
        self.source = source
        self.rail_xyz = np.asarray(state['rail'][:3]).copy()
        self.quaternion = np.asarray(state['rail'][3:]).copy()
        self.initial = source.command.copy()
        self.last_safe = self.initial.copy()
        self.target = self.initial.copy()
        self.phase = 'settle'
        self.elapsed = 0.
        self.request_id = None
        self.iteration = 0
        self.max_forces = dict.fromkeys(finger_forces(state), 0.)
        self.stable_checks = 0
        self.failure_reason = None

    def request(self, stage, state):
        # Reuse the actual PhysX link-pose snapshot without switching the main
        # task's output directory or interrupting its existing host service.
        output = self.source.output
        try:
            self.source.output = self.rpc_output
            super().request(stage, state)
        finally:
            self.source.output = output

    def response(self):
        path = self.rpc_output/'rpc/response.json'
        value = json.loads(path.read_text()) if path.exists() else {}
        if value.get('id') != self.request_id:
            if time.monotonic()-self.request_started > 30:
                if self.phase == 'rollback_review':
                    write_json(self.rpc_output/'tighten_result.json', {'complete': True, 'grip_tightened': False,
                               'restored_clear_geometry': False, 'reason': 'Rollback geometry checker unavailable',
                               'support_retained': True})
                    self.source.finish('failed', 'Initial target restored; geometry checker unavailable; rail supported')
                    self.source.pause_physics_requested = True
                else:
                    self.rollback('Tightening checker did not respond')
            return None
        self.request_id = None
        if not value.get('ok'):
            self.rollback(value.get('error', 'Tightening checker failed'))
            return None
        return value

    def rollback(self, reason):
        self.failure_reason = reason
        self.phase, self.elapsed = 'rollback', 0.
        # Return all fingers to the verified nonintersecting initial grasp.
        self.target = self.initial.copy()
        self.request_id = None
        self.source.event('tightening_rolled_back', reason=reason)

    def step(self, state, dt):
        self.elapsed += dt
        forces = finger_forces(state)
        for f, value in forces.items():
            self.max_forces[f] = max(self.max_forces[f], value)
        self.source.metrics['tightening_peak_force_n'] = self.max_forces.copy()
        if self.phase not in ('rollback', 'rollback_review', 'hold') and max(forces.values()) > 4.:
            self.rollback('Finger contact force exceeded 4 N')
        if self.phase == 'settle' and self.elapsed >= 1.2:
            self.request('tighten_check', state)
            self.phase, self.elapsed = 'review', 0.
        elif self.phase == 'review':
            response = self.response()
            if response:
                self.source.metrics['tightening_geometry'] = response['geometry']
                if response.get('stop'):
                    self.rollback(response['stop'])
                elif response.get('ready'):
                    self.stable_checks += 1
                    if self.stable_checks >= 3:
                        self.source.support_active = False
                        self.source.body.CreateKinematicEnabledAttr(False)
                        self.source.ccd.CreateEnableCCDAttr(True)
                        self.phase, self.elapsed = 'hold', 0.
                        self.hold = UnsupportedHold(self.source, state)
                        self.source.event('tightened_grasp_support_removed')
                    else:
                        self.phase, self.elapsed = 'settle', 0.
                elif self.iteration >= 40:
                    self.rollback('No stable force grip within conservative closure budget')
                else:
                    self.stable_checks = 0
                    self.last_safe = self.target.copy()
                    self.target = np.asarray(response['command'])
                    self.iteration += 1
                    self.phase, self.elapsed = 'settle', 0.
        elif self.phase == 'rollback' and self.elapsed >= 2.:
            self.request('tighten_verify', state)
            self.phase, self.elapsed = 'rollback_review', 0.
        elif self.phase == 'rollback_review':
            response = self.response()
            if response:
                geometry = response['geometry']
                clear = not geometry['intersecting_links'] and not geometry['interfinger_intersections']
                write_json(self.rpc_output/'tighten_result.json', {'complete': True, 'grip_tightened': False,
                           'reason': self.failure_reason, 'restored_clear_geometry': clear,
                           'geometry': geometry, 'peak_force_n': self.max_forces, 'steps': self.iteration,
                           'collision_settings_unchanged': True, 'support_retained': True})
                self.source.metrics['tightening_rollback_geometry'] = geometry
                self.source.finish('failed', self.failure_reason+'; initial hand target restored, rail supported')
                self.source.pause_physics_requested = True
        elif self.phase == 'hold':
            desired = self.hold.step(state, dt)
            if self.source.finished:
                write_json(self.rpc_output/'tighten_result.json', {'complete': True,
                           'grip_tightened': self.source.status == 'unsupported_static_hold_passed',
                           'reason': self.source.reason, 'peak_force_n': self.max_forces, 'steps': self.iteration,
                           'collision_settings_unchanged': True, 'support_retained': False,
                           'hold': self.source.metrics.get('unsupported_hold')})
            return desired
        return self.target.copy()
