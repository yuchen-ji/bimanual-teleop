import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[2])]
from common import write_json
from host_rpc import RequestServer, WorkerLease, process_identity


class HostServiceTests(unittest.TestCase):
    def test_results_do_not_end_service_and_stale_solves_cannot_replace_new_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requests = []
            def handle(request):
                requests.append(request['id'])
                if request['id'] == 'old':
                    write_json(root/'rpc/request.json', {'id': 'new', 'stage': 'locked_grasp_plan'})
                return {'ok': True}
            lease = SimpleNamespace(state={})
            server = RequestServer(root, handle, lease)
            for status in ('failed', 'constrained_grasp_complete'):
                write_json(root/'result.json', {'status': status})
                write_json(root/'rpc/request.json', {'id': status, 'stage': 'locked_grasp_plan'})
                self.assertTrue(server.poll())
            write_json(root/'rpc/request.json', {'id': 'old', 'stage': 'locked_grasp_plan'})
            server.poll()
            self.assertNotEqual(json.loads((root/'rpc/response.json').read_text())['id'], 'old')
            self.assertTrue(server.poll())
            self.assertEqual(requests[-1], 'new')
            self.assertFalse(RequestServer(root, handle, lease).poll())

    def test_planner_exception_does_not_prevent_next_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def handle(request):
                if request['id'] == 'bad':
                    raise ValueError('invalid geometry')
                return {'ok': True}
            server = RequestServer(root, handle, SimpleNamespace(state={}))
            for request_id, expected in [('bad', False), ('good', True)]:
                write_json(root/'rpc/request.json', {'id': request_id, 'stage': 'locked_grasp_plan'})
                server.poll()
                self.assertEqual(json.loads((root/'rpc/response.json').read_text())['ok'], expected)

    def test_lease_has_single_owner_and_heartbeat_survives_busy_solver(self):
        with tempfile.TemporaryDirectory() as directory, WorkerLease(directory) as lease:
            with self.assertRaisesRegex(RuntimeError, 'already owns'), WorkerLease(directory):
                pass
            lease.state = {'status': 'busy', 'request_id': 'long-solve'}
            time.sleep(1.1)
            health = json.loads((Path(directory)/'rpc/worker.json').read_text())
            self.assertEqual(health['status'], 'busy')
            self.assertLess(time.time()-health['updated_at'], 1.)
        self.assertIsNotNone(process_identity(os.getpid()))
        self.assertIsNone(process_identity(999999999))


    def test_reconnect_restores_saved_configuration_revisions(self):
        from bottom_grasp_worker import bootstrap
        from test_reload import ReloadTests
        with tempfile.TemporaryDirectory() as directory:
            _, _, reloads, source = ReloadTests().setup_run(directory)
            prepared = reloads.prepare('installed')
            source.pending_configuration = prepared['configuration']
            source.install_pending_settings()
            source.reset()
            restored = bootstrap(Path(directory))
            self.assertEqual(restored.bridge_for({'revision': 'installed'}).task, source.task)
            self.assertTrue(restored.prepare('next', current_revision='installed')['ok'])
            source.close()


class LifecycleTests(unittest.TestCase):
    def source(self, directory):
        from test_preinsert_sim import ControllerTests
        return ControllerTests().controller(directory, headless=False)

    def test_reset_restores_configured_grasp_mode_and_clears_offline_message(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(directory)
            source.grasp_mode = 'physics'
            source.config_message = 'Planner offline'
            source.reset()
            self.assertEqual(source.grasp_mode, 'fixed')
            self.assertEqual(source.config_message, '')
            source.close()

    def test_completed_toolbar_play_is_blocked_but_start_queues_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(directory)
            source.bound = True
            source.world = SimpleNamespace(is_playing=lambda: True)
            source.finish('failed', 'grasp slipped')
            source.service_lifecycle()
            self.assertEqual(source.physics_action_requested, 'pause')
            source.next_stage()
            self.assertTrue(source.reset_requested)
            self.assertEqual(source.physics_action_requested, 'play')
            source.service_lifecycle()
            self.assertEqual(source.physics_action_requested, 'play')
            source.close()

    def test_failure_snapshot_can_finish_before_play_guard_pauses(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(directory)
            source.bound = True
            source.world = SimpleNamespace(is_playing=lambda: True)
            source.last_state = {'time_ns': 1}
            source.finish('failed', 'slipped')
            self.assertTrue(source.snapshot_requested)
            source.service_lifecycle()
            self.assertIsNone(source.physics_action_requested)
            source.close()

    def test_offline_host_is_detected_while_paused_including_reload(self):
        for reloading in (False, True):
            with self.subTest(reloading=reloading), tempfile.TemporaryDirectory() as directory:
                source = self.source(directory)
                source.bound = True
                source.world = SimpleNamespace(is_playing=lambda: False)
                if reloading:
                    source.finish('interrupted', 'Applying configuration')
                    source.reload_id = 'pending'
                else:
                    source.locked_grasp = SimpleNamespace(request_id='pending')
                path = Path(directory)/'rpc/request.json'
                write_json(path, {'id': 'pending'})
                with patch('controller.time.time', return_value=path.stat().st_mtime+6):
                    source.service_lifecycle()
                self.assertEqual(source.status, 'failed')
                self.assertIn('Planner offline', source.reason)
                self.assertEqual(source.physics_action_requested, 'pause')
                self.assertFalse(source.snapshot_requested)
                source.close()

    def test_busy_heartbeat_and_matching_response_prevent_false_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(directory)
            source.locked_grasp = SimpleNamespace(request_id='pending')
            root = Path(directory)
            write_json(root/'rpc/request.json', {'id': 'pending'})
            now = time.time()+10
            write_json(root/'rpc/worker.json', {'status': 'busy', 'updated_at': now})
            with patch('controller.time.time', return_value=now):
                source.service_lifecycle()
            self.assertFalse(source.finished)
            write_json(root/'rpc/response.json', {'id': 'pending', 'ok': True})
            with patch('controller.time.time', return_value=now+60):
                source.service_lifecycle()
            self.assertFalse(source.finished)
            source.close()
