"""Configurable operator controls, with no hardware connections."""

import threading
import unittest
from unittest.mock import Mock

from bimanual_teleop.cli.runtime import TeleopUI, run_loop
from bimanual_teleop.recording.ui import RecordingUI
from bimanual_teleop.system import SystemState
from bimanual_teleop.types import Health


class ControlsTests(unittest.TestCase):
    def setUp(self):
        self.runtime = Mock(state=SystemState.READY, last_error=None)
        self.runtime.health.return_value = Health(True, 0)
        self.runtime.engage.side_effect = lambda _: setattr(self.runtime, "state", SystemState.ENGAGED)
        self.runtime.pause.side_effect = lambda _: setattr(self.runtime, "state", SystemState.PAUSED)
        self.messages = []
        self.gesture = Mock()
        self.ui = TeleopUI(self.runtime, None, toggle_engagement_key="t", ready_pose_key="r",
                           home_enabled=True, gesture=self.gesture, gesture_engagement_enabled=False,
                           emit=self.messages.append)
        self.addCleanup(self.ui.close)

    def test_toggle_engages_pauses_and_reengages(self):
        self.ui.handle("e")
        self.runtime.engage.assert_not_called()
        self.ui.handle("T")
        self.assertEqual(self.runtime.state, SystemState.ENGAGED)
        self.ui.handle("t")
        self.assertEqual(self.runtime.state, SystemState.PAUSED)
        self.ui.handle("t")
        self.assertEqual(self.runtime.engage.call_count, 2)
        self.assertEqual(self.runtime.state, SystemState.ENGAGED)

    def test_toggle_cancels_pending_engagement(self):
        self.runtime.health.return_value = Health(False, 0, "waiting")
        self.ui.handle("t")
        self.assertTrue(self.ui.engage_pending)
        self.ui.handle("t")
        self.assertFalse(self.ui.engage_pending)
        self.runtime.health.return_value = Health(True, 0)
        self.runtime.engage.assert_not_called()
        self.ui.handle("t")
        self.runtime.engage.assert_called_once()

    def test_toggle_cancels_background_engagement(self):
        self.ui.toggle_engagement_key = "enter"
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        def engage(_):
            entered.set()
            release.wait(1.)
        self.runtime.engage.side_effect = engage
        self.ui.background_engage = True
        self.ui.handle("\n")
        self.assertTrue(entered.wait(.5))
        self.ui.handle("\r")
        self.assertTrue(self.ui._operation_cancel.is_set())
        self.runtime.pause.assert_called_once()
        release.set()
        self.ui._operation_thread.join(1.)
        self.ui.poll_operation()
        self.assertEqual(self.runtime.state, SystemState.PAUSED)

    def test_custom_home_key_stops_following_and_does_not_resume(self):
        self.runtime.state = SystemState.ENGAGED
        self.ui.handle("h")
        self.runtime.home.assert_not_called()
        self.ui.handle("R")
        self.ui._operation_thread.join(1.)
        self.ui.poll_operation()
        self.runtime.home.assert_called_once()
        self.runtime.pause.assert_called_once()
        self.runtime.engage.assert_not_called()
        self.assertEqual(self.runtime.state, SystemState.PAUSED)

    def test_failed_stop_does_not_start_homing(self):
        self.runtime.state = SystemState.ENGAGED
        self.runtime.pause.side_effect = [RuntimeError("stop failed"), None]
        self.ui.handle("r")
        self.runtime.home.assert_not_called()
        self.assertIsNone(self.ui._operation_thread)
        self.assertEqual(self.ui.last_motion_error, "stop failed")

    def test_home_gesture_still_requires_detached_state(self):
        self.runtime.state = SystemState.ENGAGED
        self.ui.gesture_engagement_enabled = True
        self.assertEqual(self.ui.handle_gesture("home"), "ignored")
        self.runtime.pause.assert_not_called()
        self.runtime.home.assert_not_called()

    def test_home_key_finishes_recording_before_stopping_and_moving(self):
        self.runtime.state = SystemState.ENGAGED
        recorder = Mock(poll=Mock(return_value=None), notices=[])
        recorder.state = "recording"
        calls = Mock()
        calls.attach_mock(recorder.pause, "recorder_pause")
        calls.attach_mock(self.runtime.pause, "pause")
        calls.attach_mock(self.runtime.home, "home")
        ui = RecordingUI(self.runtime, None, recorder=recorder, home_enabled=True,
                         toggle_engagement_key="enter", emit=lambda _: None)
        self.addCleanup(ui.close)
        ui.handle("H")
        ui._operation_thread.join(1.)
        ui.poll_operation()
        names = [call[0] for call in calls.mock_calls]
        self.assertLess(names.index("pause"), names.index("recorder_pause"))
        self.assertLess(names.index("recorder_pause"), names.index("home"))
        recorder.end.assert_not_called()
        self.runtime.home.assert_called_once()
        self.runtime.engage.assert_not_called()
        self.assertEqual(self.runtime.state, SystemState.PAUSED)

    def test_toggle_cancels_running_home(self):
        self.ui.toggle_engagement_key = "enter"
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        def home(cancel):
            self.runtime.state = SystemState.HOMING
            entered.set()
            release.wait(1.)
        self.runtime.home.side_effect = home
        self.runtime.state = SystemState.PAUSED
        self.ui.handle("r")
        self.assertTrue(entered.wait(.5))
        self.ui.handle("\r")
        self.assertTrue(self.runtime.home.call_args.args[0].is_set())
        release.set()
        self.ui._operation_thread.join(1.)
        self.ui.poll_operation()
        self.assertEqual(self.runtime.state, SystemState.PAUSED)
        self.runtime.engage.assert_not_called()

    def test_disabled_gestures_preserve_keyboard_controls(self):
        for state in (SystemState.READY, SystemState.ENGAGED, SystemState.PAUSED, SystemState.HOMING):
            self.runtime.state = state
            for command in ("engage", "pause", "home"):
                with self.subTest(state=state, command=command):
                    self.assertEqual(self.ui.handle_gesture(command), "ignored")
        self.runtime.engage.assert_not_called()
        self.runtime.pause.assert_not_called()
        self.runtime.home.assert_not_called()
        self.runtime.state = SystemState.READY
        self.assertNotIn("比 V", self.ui.help_text)
        self.assertNotIn("比 V", self.ui.start_hint)
        self.assertNotIn("摇滚", self.ui.help_text)
        self.assertNotIn("张开", self.ui.help_text)
        self.ui.handle("\n")
        self.runtime.engage.assert_called_once()
        self.ui.handle(" ")
        self.runtime.pause.assert_called_once()
        self.ui.handle("r")
        self.ui._operation_thread.join(1.)
        self.ui.poll_operation()
        self.runtime.home.assert_called_once()

    def test_enter_variants_toggle_and_do_not_duplicate_help(self):
        self.ui.toggle_engagement_key = "enter"
        self.ui.handle("\n")
        self.assertEqual(self.runtime.state, SystemState.ENGAGED)
        self.ui.handle("\r")
        self.assertEqual(self.runtime.state, SystemState.PAUSED)
        self.ui.handle("\r")
        self.assertEqual(self.runtime.state, SystemState.ENGAGED)
        self.ui.handle("\n")
        self.assertEqual(self.runtime.state, SystemState.PAUSED)
        self.assertEqual(self.runtime.engage.call_count, 2)
        self.assertEqual(self.runtime.pause.call_count, 2)
        self.assertEqual(self.ui.start_hint, "按 Enter")
        self.assertEqual(self.ui.help_text.count("Enter"), 1)
        self.assertIn("Enter 接合/脱离", self.ui.help_text)

    def test_second_enter_cancels_wait_and_readiness_does_not_resume(self):
        self.ui.toggle_engagement_key = "enter"
        self.runtime.health.return_value = Health(False, 0, "waiting")
        self.ui.handle("\n")
        self.assertTrue(self.ui.engage_pending)
        self.ui.handle("\r")
        self.assertFalse(self.ui.engage_pending)
        self.runtime.health.return_value = Health(True, 0)
        self.runtime.status.return_value = {"state": "paused"}
        run_loop(self.runtime, self.ui, Mock(read=Mock(return_value="q")))
        self.runtime.engage.assert_not_called()

    def test_readiness_completes_enter_request_without_toggling_it_off(self):
        self.ui.toggle_engagement_key = "enter"
        self.runtime.health.return_value = Health(False, 0, "waiting")
        self.ui.handle("\r")
        self.assertTrue(self.ui.engage_pending)
        self.runtime.health.return_value = Health(True, 0)
        self.runtime.status.return_value = {"state": "engaged"}
        run_loop(self.runtime, self.ui, Mock(read=Mock(return_value="q")))
        self.runtime.engage.assert_called_once()
        self.runtime.pause.assert_not_called()
        self.assertFalse(self.ui.engage_pending)

    def test_failed_pending_engagement_retains_error_and_pauses(self):
        self.ui.toggle_engagement_key = "enter"
        self.runtime.health.return_value = Health(False, 0, "waiting")
        self.ui.handle("\n")
        self.runtime.health.return_value = Health(True, 0)
        self.runtime.engage.side_effect = RuntimeError("engagement failed")
        self.runtime.status.return_value = {"state": "paused"}
        result = run_loop(self.runtime, self.ui, Mock(read=Mock(return_value="q")))
        self.runtime.pause.assert_called_once_with("engagement failed")
        self.assertEqual(result["last_motion_error"], "engagement failed")
        self.assertFalse(self.ui.engage_pending)

    def test_disabled_gestures_are_not_polled(self):
        self.runtime.status.return_value = {"state": "ready"}
        run_loop(self.runtime, self.ui, Mock(read=Mock(return_value="q")))
        self.gesture.poll.assert_not_called()

    def test_disabled_gestures_do_not_end_recording(self):
        recorder = Mock()
        ui = RecordingUI(self.runtime, None, recorder=recorder, gesture=self.gesture,
                         gesture_engagement_enabled=False, emit=lambda _: None)
        for command in ("engage", "pause", "home"):
            self.assertEqual(ui.handle_gesture(command), "ignored")
        recorder.end.assert_not_called()
        self.runtime.engage.assert_not_called()
        self.runtime.pause.assert_not_called()
        self.runtime.home.assert_not_called()

    def test_toggle_stop_finishes_recording_before_pause(self):
        recorder = Mock()
        recorder.state = "recording"
        self.runtime.state = SystemState.ENGAGED
        calls = Mock()
        calls.attach_mock(recorder.pause, "recorder_pause")
        calls.attach_mock(self.runtime.pause, "pause")
        ui = RecordingUI(self.runtime, None, recorder=recorder, toggle_engagement_key="enter",
                         emit=lambda _: None)
        ui.handle("\r")
        self.assertEqual(calls.mock_calls[0][0], "pause")
        self.assertEqual(calls.mock_calls[-1][0], "recorder_pause")
        recorder.end.assert_not_called()
        self.assertEqual(self.runtime.state, SystemState.PAUSED)


if __name__ == "__main__":
    unittest.main()
