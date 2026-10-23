import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from run import load_experiment, read_runtime_result
from controller import PHASES, RailController, RailReplayController, interpolate_pose
from evaluate import initial_pose_valid, insertion_metrics, is_seated, opposing_contacts, pose_error, seated_pose, unintended_robot_contacts
from bimanual_teleop.sim import ROOT
from bimanual_teleop.sim.config import load_config, sim_path
from bimanual_teleop.sim.kinematics import RobotModel, pose_matrix, pose_vector, transform
from bimanual_teleop.sim.actions import JointTrajectory
from bimanual_teleop.sim.local_session import LocalSession


class RailExperimentTests(unittest.TestCase):
    def setUp(self):
        self.config, self.task = load_experiment(HERE/'config.yaml')
        self.geometry = json.loads((ROOT/'assets/generated/manifest.json').read_text())['fixture_geometry']
        self.model = RobotModel(sim_path(self.config['robot']['urdf']))

    def test_experiment_override_does_not_mutate_default(self):
        before = load_config()
        self.config['rail']['initial_center'][0] += .1
        self.assertEqual(load_config(), before)
        self.assertNotEqual(load_config()['rail']['initial_rpy_deg'], self.config['rail']['initial_rpy_deg'])

    def test_runtime_failure_before_startup_is_saved_as_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = read_runtime_result(output, 255)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['runtime_returncode'], 255)
            self.assertEqual(json.loads((output/'result.json').read_text()), result)

    def test_both_pin_correspondences_register_real_hole_centers(self):
        for reverse in (False, True):
            pose = seated_pose(self.config, self.geometry, reverse)
            metrics = insertion_metrics({'rail': pose_vector(pose)}, self.config, self.geometry)
            self.assertLess(metrics['pin_error_m'], 1e-5)
            self.assertLess(metrics['support_gap_m'], 1e-8)

    def test_seated_requires_release_depth_and_stability(self):
        pose = seated_pose(self.config, self.geometry)
        state = {'rail': pose_vector(pose), 'hand_contact_force': [0, 0]}
        self.assertTrue(is_seated(state, self.config, self.geometry, self.task, 0, 0)[0])
        state['hand_contact_force'] = [.1, 0]
        self.assertFalse(is_seated(state, self.config, self.geometry, self.task, 0, 0)[0])
        state['hand_contact_force'] = [0, 0]
        state['rail_contacts'] = {'l_index_finger_proximal': 4.2}
        self.assertFalse(is_seated(state, self.config, self.geometry, self.task, 0, 0)[0])
        state['rail_contacts'] = {}
        self.assertFalse(is_seated(state, self.config, self.geometry, self.task, .1, 0)[0])
        state['rail'][2] += .02
        self.assertFalse(is_seated(state, self.config, self.geometry, self.task, 0, 0)[0])

    def test_mount_contact_cannot_substitute_for_opposing_fingers(self):
        self.assertFalse(opposing_contacts({'rail_contacts': {'l_camera_bracket': 100, 'l_thumb_distal': 1}}, .1))
        state = {'rail_contacts': {'l_index_finger_distal': 1, 'l_thumb_distal': 1},
                 'rail_contact_vectors': {'l_index_finger_distal': [-1, 0, 0], 'l_thumb_distal': [1, 0, 0]}}
        self.assertTrue(opposing_contacts(state, .1))
        state['rail_contacts']['l_wrist'] = .62
        self.assertFalse(unintended_robot_contacts(state, .1))
        state['rail_contacts']['l_camera_bracket'] = 1.
        self.assertEqual(unintended_robot_contacts(state, .1), {'l_camera_bracket': 1.})
        del state['rail_contacts']['l_camera_bracket']
        state['rail_contact_vectors']['l_index_finger_distal'] = [1, 0, 0]
        self.assertFalse(opposing_contacts(state, .1))
        self.assertEqual(unintended_robot_contacts(state, .1), {'l_wrist': .62})

    def test_scene_gate_reset_and_replay_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = RailController(self.config, self.model, self.task, directory, mode='auto', until='scene')
            state = {'q': controller.ready.tolist(), 'q_target': controller.ready.tolist(),
                     'rail': pose_vector(transform(self.config['rail']['initial_center'], np.deg2rad(self.config['rail']['initial_rpy_deg'])))}
            controller.linear_speed = controller.angular_speed = 0.
            for _ in range(121):
                controller.step(state, 1/120)
            self.assertEqual(controller.status, 'milestone_passed')
            controller.reset()
            self.assertIsNone(controller.last_state)
            self.assertIsNone(controller.grip)
            self.assertEqual(controller.phase, 'settle')
            controller.samples.append({'time_s': 1, 'positions_rad': controller.ready.tolist()})
            controller.close()
            trajectory = JointTrajectory(controller.trial/'trajectory.json', self.model, self.config['runtime'])
            np.testing.assert_array_equal(trajectory.sample(0), controller.ready)

    def test_timeout_and_stop_are_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = RailController(self.config, self.model, self.task, directory, mode='auto', exit_on_complete=True)
            state = {'q': controller.ready.tolist(), 'rail': [0, 0, 0, 0, 0, 0, 1]}
            controller.step(state, self.task['settle_timeout_s']+1)
            self.assertEqual(controller.status, 'failed')
            self.assertTrue(controller.should_stop)
            controller.close()
        with tempfile.TemporaryDirectory() as directory:
            controller = RailController(self.config, self.model, self.task, directory)
            controller.stop_reason = 'received SIGINT'
            controller.close()
            self.assertEqual(controller.status, 'interrupted')
            self.assertEqual(controller.reason, 'received SIGINT')
            result = {'status': 'running', 'trial': controller.trial.name}
            (Path(directory)/'result.json').write_text(json.dumps(result))
            self.assertEqual(read_runtime_result(Path(directory), 130)['status'], 'interrupted')
            self.assertEqual(json.loads((Path(directory)/'unfinished-result.json').read_text()), result)
        with tempfile.TemporaryDirectory() as directory:
            controller = RailController(self.config, self.model, self.task, directory, mode='auto', exit_on_complete=True)
            controller.phase_index = PHASES.index('wrap')
            target = controller.ready.copy(); target[0] += .5
            state = {'q': controller.ready.tolist(), 'q_target': target.tolist(),
                     'rail': pose_vector(transform(self.config['rail']['initial_center']))}
            for _ in range(60):
                controller.step(state, 1/120)
                if controller.finished:
                    break
            self.assertEqual(controller.status, 'failed')
            self.assertTrue(controller.reason.startswith('arm_tracking_error: Joint1_L'))
            controller.close()

    def test_action_source_uses_actual_state_and_rejects_invalid_targets(self):
        class Source:
            name = 'test_source'
            def step(source, state, dt):
                source.received = state
                return source.target
            def close(source, error):
                pass
        with tempfile.TemporaryDirectory() as directory:
            config = copy.deepcopy(self.config)
            config['paths']['recordings'] = directory
            source = Source()
            session = LocalSession(config, self.model, status_path=Path(directory)/'status.json', action_source=source)
            session.last_state = {'q': session.ready.tolist(), 'q_target': session.ready.tolist()}
            source.target = session.ready.copy()
            np.testing.assert_array_equal(session.targets(2), session.ready)
            self.assertIs(source.received, session.last_state)
            source.target = source.target.copy(); source.target[0] += .5
            with self.assertRaisesRegex(ValueError, 'speed'):
                session.targets(4)
            source.target = np.full(54, np.nan)
            with self.assertRaisesRegex(ValueError, 'finite'):
                session.targets(4)
            session.last_state = None
            session.close()

    def test_dropped_or_flat_rail_cannot_pass_scene_gate(self):
        state = {'rail': pose_vector(transform(self.config['rail']['initial_center'], np.deg2rad(self.config['rail']['initial_rpy_deg'])))}
        self.assertTrue(initial_pose_valid(state, self.config))
        state['rail'][2] = .02
        self.assertFalse(initial_pose_valid(state, self.config))
        state['rail'] = pose_vector(transform(self.config['rail']['initial_center']))
        self.assertFalse(initial_pose_valid(state, self.config))

    def test_only_robot_facing_narrow_edge_can_pass_scene_gate(self):
        # Previously used Y-facing long edge, then the opposite narrow edge.
        for center, rpy in (([.68, .49, .86], [0, -30, -90]),
                            ([.78, .56, .833], [0, -24, 0])):
            state = {'rail': pose_vector(transform(center, np.deg2rad(rpy)))}
            self.assertFalse(initial_pose_valid(state, self.config))

    def test_turning_carried_rail_keeps_its_center_on_the_planned_path(self):
        start = transform([.46, 0, 1.10])
        target = transform([.46, 0, 1.10], [np.pi, 0, 0])
        grip = transform([.07, .05, .185], [-np.pi/4, 0, np.pi])
        for t in np.linspace(0, 1, 21):
            flange = interpolate_pose(start, target, t)@np.linalg.inv(grip)
            rail = flange@grip
            np.testing.assert_allclose(rail[:3, 3], [.46, 0, 1.10], atol=1e-10)

    def test_object_rotation_respects_flange_speed_along_the_full_arc(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = RailController(self.config, self.model, self.task, directory, mode='auto')
            flange = self.model.flange('left', controller.ready, world=True)
            rail = flange@transform([0, 0, .2])
            controller.goal_rail = rail@transform(rpy=(np.pi, 0, 0))
            controller.transfer_center[2] = rail[2, 3]
            controller.phase_index = PHASES.index('level')
            state = {'q': controller.ready.tolist(), 'rail': pose_vector(rail).tolist()}
            with patch.object(controller, 'solve_arm', return_value=(controller.ready[:7].copy(), 0.)):
                controller.setup_motion(state)
            poses = []
            for fraction in np.linspace(0, 1, 1001):
                smooth = fraction**3*(10-15*fraction+6*fraction*fraction)
                poses.append(interpolate_pose(controller.start_object, controller.target_object, smooth)@np.linalg.inv(controller.motion_grip))
            dt = controller.duration/1000
            speeds = [pose_error(a, b)[0]/dt for a, b in zip(poses, poses[1:])]
            self.assertLessEqual(max(speeds), self.task['linear_speed_m_s']+1e-5)
            controller.close()

    def test_narrow_edge_approach_waypoints_are_reachable_with_joint_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = RailController(self.config, self.model, self.task, directory, mode='auto')
            state = {'q': controller.ready.tolist(),
                     'rail': pose_vector(transform(self.config['rail']['initial_center'], np.deg2rad(self.config['rail']['initial_rpy_deg'])))}
            for phase in ('pregrasp', 'slot', 'approach'):
                controller.phase_index = PHASES.index(phase)
                count = len(self.task['approach_waypoints_in_rail_m']) if phase == 'approach' else 1
                for index in range(count):
                    controller.approach_index = index
                    controller.setup_motion(state)
                    self.assertFalse(controller.finished, controller.reason)
                    q = controller.ready.copy(); q[:7] = controller.ik_goal
                    position, angle = pose_error(self.model.flange('left', q, world=True), controller.target_pose)
                    self.assertLess(position, .002)
                    self.assertLess(angle, .01)
                    self.assertTrue(np.all(q[:7] >= controller.ik_lower-1e-6))
                    self.assertTrue(np.all(q[:7] <= controller.ik_upper+1e-6))
            state['q'] = q.tolist()
            controller.command = q.copy()
            controller.grip = np.linalg.inv(self.model.flange('left', q, world=True))@pose_matrix(state['rail'])
            for phase in ('clear', 'transfer', 'level', 'align', 'insert'):
                controller.phase_index = PHASES.index(phase)
                controller.setup_motion(state)
                self.assertFalse(controller.finished, f'{phase}: {controller.reason}')
                q[:7] = controller.ik_goal
                position, angle = pose_error(self.model.flange('left', q, world=True), controller.target_pose)
                self.assertLess(position, .002)
                self.assertLess(angle, .01)
                state.update(q=q.tolist(), rail=pose_vector(controller.target_object).tolist())
                controller.command = q.copy()
            controller.close()

    def test_lift_gate_rejects_support_and_rotation_but_tolerates_contact_microvibration(self):
        for scenario in ('microvibration', 'supported', 'rotating'):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory:
                controller = RailController(self.config, self.model, self.task, directory, mode='auto', until='grasp')
                controller.phase_index = PHASES.index('lift')
                controller.start_pose = controller.target_pose = self.model.flange('left', controller.ready, world=True)
                controller.ik_goal = controller.ready[:7].copy()
                controller.duration = .1
                controller.lift_reference = transform([.66, .56, .83])
                state = {'q': controller.ready.tolist(), 'q_target': controller.ready.tolist(), 'eef': [],
                         'environment_contacts': {}, 'rail_contacts': {'l_thumb_distal': 8., 'l_index_finger_distal': 9.},
                         'rail_contact_vectors': {'l_thumb_distal': [0, 0, 8.], 'l_index_finger_distal': [0, 0, -9.]}}
                for tick in range(1, 241):
                    angle = .1*tick/120 if scenario == 'rotating' else (.0005 if tick % 37 == 0 else 0.)
                    state['rail'] = pose_vector(transform([.66, .56, .85], [angle, 0, 0])).tolist()
                    state['rail_support_contacts'] = {'input_bin': 1. if scenario == 'supported' else (.15 if tick % 41 == 0 else .04)}
                    state.update(tick=tick*2, time_ns=round(tick*1e9/120))
                    controller.observe(state)
                    controller.step(state, 1/120)
                    if controller.finished:
                        break
                self.assertEqual(controller.status == 'milestone_passed', scenario == 'microvibration')
                controller.close()

    def test_carried_object_feedback_compensates_a_changed_hand_object_transform(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = RailController(self.config, self.model, self.task, directory, mode='auto')
            actual_flange = self.model.flange('left', controller.ready, world=True)
            actual_grip = controller.grip_template_base.copy()
            actual_object = actual_flange@actual_grip
            desired_object = actual_object@transform(rpy=(.05, 0, 0))
            controller.phase_index = PHASES.index('level')
            controller.grip = actual_grip
            controller.motion_grip = actual_grip@transform(rpy=(.04, 0, 0))
            controller.start_object, controller.target_object = actual_object, desired_object
            controller.start_pose = actual_flange
            controller.target_pose = desired_object@np.linalg.inv(controller.motion_grip)
            controller.ik_goal = controller.ready[:7].copy()
            controller.duration = controller.elapsed = 1.
            state = {'q': controller.ready.tolist(), 'q_target': controller.ready.tolist(), 'rail': pose_vector(actual_object).tolist(),
                     'rail_contacts': {'l_thumb_distal': 8., 'l_index_finger_distal': 9.},
                     'rail_contact_vectors': {'l_thumb_distal': [0, 0, 8.], 'l_index_finger_distal': [0, 0, -9.]}}
            with patch.object(self.model, 'velocity_ik', return_value=controller.ready[:7].copy()) as velocity:
                controller.step(state, 1/120)
            commanded_flange = velocity.call_args.args[2]
            np.testing.assert_allclose(commanded_flange@actual_grip, desired_object, atol=1e-10)
            self.assertFalse(np.allclose(commanded_flange, controller.target_pose))
            self.assertNotIn('level', controller.completed_phases)
            controller.phase_index = PHASES.index('insert')
            with patch.object(self.model, 'velocity_ik', return_value=controller.ready[:7].copy()) as velocity:
                controller.step(state, 1/120)
            self.assertEqual(velocity.call_args.args[4]['max_linear_speed_m_s'], self.task['insertion_speed_m_s'])
            controller.grip = actual_grip@transform(rpy=(.30, 0, 0))
            controller.phase_index = PHASES.index('level')
            with patch.object(self.model, 'velocity_ik', return_value=controller.ready[:7].copy()):
                controller.step(state, 1/120)
            self.assertFalse(controller.finished)
            controller.phase_index = PHASES.index('align')
            controller.step(state, 1/120)
            self.assertTrue(controller.reason.startswith('grasp_slip'))
            controller.close()

    def test_reorientation_rejects_object_motion_outside_the_planned_rotation(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = RailController(self.config, self.model, self.task, directory, mode='auto')
            flange = self.model.flange('left', controller.ready, world=True)
            actual = flange@controller.grip_template_base
            goal = actual@transform(rpy=(.30, 0, 0))
            controller.phase_index = PHASES.index('level')
            controller.grip = controller.motion_grip = controller.grip_template_base.copy()
            controller.start_object = controller.target_object = goal
            controller.start_pose = flange
            controller.target_pose = goal@np.linalg.inv(controller.motion_grip)
            controller.ik_goal = controller.ready[:7].copy()
            controller.duration = 1.
            state = {'q': controller.ready.tolist(), 'q_target': controller.ready.tolist(), 'rail': pose_vector(actual).tolist(),
                     'rail_contacts': {'l_thumb_distal': 8., 'l_index_finger_distal': 9.},
                     'rail_contact_vectors': {'l_thumb_distal': [0, 0, 8.], 'l_index_finger_distal': [0, 0, -9.]}}
            with patch.object(self.model, 'velocity_ik', return_value=controller.ready[:7].copy()):
                for _ in range(60):
                    controller.step(state, 1/120)
                    if controller.finished:
                        break
            self.assertEqual(controller.status, 'failed')
            self.assertEqual(controller.reason, 'reorientation_tracking_error')
            controller.close()

    def test_release_hand_opens_away_from_the_actual_plate_after_rolling(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = RailController(self.config, self.model, self.task, directory)
            for roll in (0., .30):
                q = controller.ready.copy(); q[14:34] = self.task['closed_hand_rad']
                rail = self.model.flange('left', q, world=True)@controller.grip_template_base@transform(rpy=(roll, 0, 0))
                target = controller.generate_release_hand({'q': q.tolist(), 'rail': pose_vector(rail).tolist()})
                self.assertIsNotNone(target, controller.reason)
                q[14:34] = target
                self.assertTrue(np.all(q >= self.model.lower))
                self.assertTrue(np.all(q <= self.model.upper))
                for link, direction in (('l_thumb_tip', 1), ('l_index_finger_tip', -1), ('l_middle_finger_tip', -1)):
                    point = (np.linalg.inv(rail)@self.model.link_pose(link, q))[:3, 3]
                    self.assertGreater(direction*point[2], .024)
                state = {'q': q.tolist(), 'rail': pose_vector(rail).tolist(),
                         'rail_contacts': {'l_index_finger_proximal': 4.2}}
                self.assertTrue(controller.release_exit_ready(state))
                state['rail_contacts']['l_thumb_distal'] = .2
                self.assertFalse(controller.release_exit_ready(state))
                state['rail_contacts'] = {'l_index_finger_proximal': 6.}
                self.assertFalse(controller.release_exit_ready(state))
                state['rail_contacts'] = {}
                q[14:34] = self.task['closed_hand_rad']; state['q'] = q.tolist()
                self.assertFalse(controller.release_exit_ready(state))
            controller.close()

    def test_level_pose_window_accepts_small_jitter_and_rejects_continuous_drift(self):
        for scenario in ('microjitter', 'drift'):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory:
                controller = RailController(self.config, self.model, self.task, directory, mode='auto')
                flange = self.model.flange('left', controller.ready, world=True)
                reference = flange@controller.grip_template_base
                controller.phase_index = PHASES.index('level')
                controller.grip = controller.motion_grip = controller.grip_template_base.copy()
                controller.start_object = controller.target_object = reference
                controller.start_pose = controller.target_pose = flange
                controller.ik_goal = controller.ready[:7].copy()
                controller.duration = .1
                state = {'q': controller.ready.tolist(), 'q_target': controller.ready.tolist(), 'eef': [], 'environment_contacts': {},
                         'rail_contacts': {'l_thumb_distal': 8., 'l_index_finger_distal': 9.},
                         'rail_contact_vectors': {'l_thumb_distal': [0, 0, 8.], 'l_index_finger_distal': [0, 0, -9.]}}
                with patch.object(self.model, 'velocity_ik', return_value=controller.ready[:7].copy()):
                    for tick in range(1, 241):
                        actual = reference@transform(rpy=((-.0003 if tick % 2 else .0003), 0, 0))
                        if scenario == 'drift':
                            actual[0, 3] += .0015*np.sin(np.pi*tick/120)
                        state.update(rail=pose_vector(actual).tolist(), tick=tick*2, time_ns=round(tick*1e9/120))
                        controller.observe(state)
                        controller.step(state, 1/120)
                        if 'level' in controller.completed_phases:
                            break
                self.assertEqual('level' in controller.completed_phases, scenario == 'microjitter')
                if scenario == 'microjitter':
                    self.assertGreater(controller.filtered_motion[1], self.task['stable_angular_speed_rad_s'])
                controller.close()

    def test_repeat_waits_for_reset_and_keeps_diagnostics_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = RailController(self.config, self.model, self.task, directory, mode='auto', until='scene',
                                        exit_on_complete=True, start_phase='approach')
            controller.repeat = 2
            state = {'q': controller.ready.tolist(), 'q_target': controller.ready.tolist(),
                     'rail': pose_vector(transform(self.config['rail']['initial_center'], np.deg2rad(self.config['rail']['initial_rpy_deg'])))}
            for trial in range(2):
                controller.linear_speed = controller.angular_speed = 0.
                for _ in range(121):
                    controller.step(state, 1/120)
                self.assertEqual(controller.status, 'diagnostic_passed')
                self.assertEqual(controller.reset_requested, trial == 0)
                self.assertEqual(controller.should_stop, trial == 1)
                if trial == 0:
                    controller.reset()
            summary = json.loads((Path(directory)/'summary.json').read_text())
            self.assertEqual(summary['passed_trials'], 2)
            controller.close()

    def test_export_preserves_application_time_and_loads_targets_at_full_joint_speed(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = RailController(self.config, self.model, self.task, directory)
            state = {'rail': pose_vector(transform(self.config['rail']['initial_center'],
                        np.deg2rad(self.config['rail']['initial_rpy_deg']))).tolist(),
                     'eef': [], 'rail_contacts': {}, 'environment_contacts': {}}
            for command_tick in range(0, 22, 2):
                q = controller.ready.copy(); q[0] += command_tick/self.config['runtime']['physics_hz']
                state.update(q=q.tolist(), q_target=q.tolist(), command_tick=command_tick,
                             tick=command_tick+2, time_ns=round((command_tick+2)*1e9/self.config['runtime']['physics_hz']))
                controller.observe(state)
            controller.close()
            trajectory = JointTrajectory(controller.trial/'trajectory.json', self.model, self.config['runtime'])
            self.assertAlmostEqual(trajectory.sample(10/240)[0], controller.ready[0]+10/240, places=12)

    def test_replay_completion_is_diagnostic_and_requires_matching_initial_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            ready = np.r_[np.deg2rad(self.config['robot']['ready_deg']['left']),
                          np.deg2rad(self.config['robot']['ready_deg']['right']), np.zeros(40)]
            target = ready.copy(); target[0] += .01
            document = {'joint_names': list(self.model.names), 'test_data': True,
                        'source_run': 'original', 'source_trial': 'trial_007', 'source_demo_ids': ['original-demo-id'],
                        'waypoints': [{'time_s': 0., 'positions_rad': ready.tolist()},
                                      {'time_s': 1., 'positions_rad': target.tolist()}]}
            path = Path(directory)/'input.json'; path.write_text(json.dumps(document))
            controller = RailReplayController(self.config, self.model, self.task, Path(directory)/'replay', path, mode='auto')
            actual = controller.step({}, .5)
            self.assertAlmostEqual(actual[0], ready[0]+.005)
            controller.step({}, .5)
            self.assertEqual(controller.status, 'diagnostic_passed')
            self.assertEqual(controller.reason, 'trajectory_replayed_only')
            self.assertTrue(controller.diagnostic)
            self.assertEqual(controller.recording_metadata()['parent_sources'][0]['source_demo_ids'], ['original-demo-id'])
            controller.close()
            document['waypoints'][0]['positions_rad'][0] += .02
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, 'first waypoint'):
                RailReplayController(self.config, self.model, self.task, Path(directory)/'bad', path, mode='auto')

    def test_finger_templates_reach_the_plate_instead_of_the_channel_void(self):
        template = transform(self.task['rail_in_flange_xyz_m'], np.deg2rad(self.task['rail_in_flange_rpy_deg']))
        def point(posture, link):
            q = np.zeros(54); q[14:34] = self.task[posture]
            object_world = self.model.flange('left', q, world=True)@template
            return (np.linalg.inv(object_world)@self.model.link_pose(link, q))[:3, 3]
        opened = point('open_hand_rad', 'l_thumb_tip')
        closed = point('closed_hand_rad', 'l_thumb_tip')
        self.assertGreater(opened[2], .020)
        self.assertLess(abs(opened[1]), .002)
        self.assertLess(closed[2], -.012)
        self.assertLess(abs(closed[1]), .002)
        for link in ('l_index_finger_tip', 'l_middle_finger_tip'):
            wrapped = point('wrap_hand_rad', link)
            self.assertLess(wrapped[2], -.020)
            self.assertLess(abs(wrapped[1]), .006)


if __name__ == '__main__':
    unittest.main()
