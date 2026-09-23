"""Arm/hand coordination and keyboard entrypoints with no device access."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import builtins
import io
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bimanual_teleop.system import SystemState
from bimanual_teleop.control.arm import quest as arm_runtime
from bimanual_teleop.control.combined import QuestTianjiWujiTeleop
from bimanual_teleop.cli import teleop_quest_tianji as cli, teleop_wuji_hand2 as hand_cli
from bimanual_teleop.cli import runtime as runtime_ui
from bimanual_teleop.types import Health


from tests.support.combined import Runtime


def fail(message):
    raise RuntimeError(message)


class CoordinationTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.arms, self.hands = Runtime("arms", self.calls), Runtime("hands", self.calls)
        self.runtime = QuestTianjiWujiTeleop(self.arms, self.hands)
        self.runtime.start()
        self.calls.clear()

    def test_hands_hold_before_arm_engagement_and_only_follow_afterwards(self):
        self.arms.engage_hook = lambda: self.assertTrue(self.hands.holding)
        self.runtime.engage("arm profile")
        self.assertEqual(self.calls, ["hands.prepare", "arms.engage", "hands.follow"])
        self.assertEqual(self.runtime.state, SystemState.ENGAGED)
        self.assertEqual(self.runtime.tick(123), "arm target")
        self.assertNotIn("hands.tick", self.calls)  # Hand scheduling belongs to its worker.

    def test_partial_hand_preparation_failure_never_engages_arms(self):
        self.hands.prepare_hook = lambda: fail("right hand enable failed")
        with self.assertRaisesRegex(RuntimeError, "right hand enable failed"):
            self.runtime.engage()
        self.assertEqual(self.calls, ["hands.prepare", "arms.pause", "hands.pause"])
        self.assertTrue(self.hands.holding)
        self.assertEqual(self.runtime.state, SystemState.PAUSED)

    def test_arm_engagement_failure_rolls_back_hand_following(self):
        self.arms.engage_hook = lambda: fail("arm enable failed")
        with self.assertRaisesRegex(RuntimeError, "arm enable failed"):
            self.runtime.engage()
        self.assertNotIn("hands.follow", self.calls)
        self.assertEqual(self.calls[-2:], ["arms.pause", "hands.pause"])

    def test_arm_self_pause_propagates_to_healthy_hand_hold(self):
        self.runtime.engage()
        self.arms.tick_hook = lambda: self.arms.pause("IK failed")
        self.runtime.tick(123)
        self.assertEqual(self.runtime.state, SystemState.PAUSED)
        self.assertEqual(self.hands.last_error, "IK failed")
        self.assertTrue(self.hands.holding)

    def test_hand_fault_blocks_next_arm_tick(self):
        self.runtime.engage()
        self.hands.pause("left hand feedback lost")
        self.calls.clear()
        self.assertIsNone(self.runtime.tick(123))
        self.assertNotIn("arms.tick", self.calls)
        self.assertEqual(self.arms.last_error, "left hand feedback lost")

    def test_unhealthy_hand_blocks_arm_tick_before_hand_worker_changes_state(self):
        self.runtime.engage()
        self.hands.health = lambda: Health(False, 0, "left glove invalid sample latched")
        self.assertEqual(self.hands.state, SystemState.ENGAGED)
        self.calls.clear()
        self.assertIsNone(self.runtime.tick(123))
        self.assertNotIn("arms.tick", self.calls)
        self.assertEqual(self.arms.last_error, "left glove invalid sample latched")

    def test_hand_fault_during_arm_tick_is_detected_before_next_cycle(self):
        self.runtime.engage()
        self.arms.tick_hook = lambda: self.hands.pause("glove invalid burst")
        self.runtime.tick(123)
        self.assertEqual(self.runtime.state, SystemState.PAUSED)
        self.calls.clear()
        self.runtime.tick(124)
        self.assertEqual(self.calls, [])

    def test_pause_failure_still_requests_the_other_group_to_stop(self):
        self.runtime.engage()
        self.hands.pause_hook = lambda: fail("hand stop unavailable")
        with self.assertRaisesRegex(RuntimeError, "hand stop unavailable"):
            self.runtime.pause("operator stop")
        self.assertEqual(self.calls[-2:], ["arms.pause", "hands.pause"])
        self.assertEqual(self.runtime.state, SystemState.PAUSED)

    def test_arm_hold_is_requested_before_waiting_for_blocked_hand_pause(self):
        self.runtime.engage()
        entered, release = threading.Event(), threading.Event()
        def blocked_pause():
            entered.set()
            if not release.wait(2):
                raise RuntimeError("test did not release hand pause")
        self.hands.pause_hook = blocked_pause
        worker = threading.Thread(target=lambda: self.runtime.pause("operator stop"))
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            self.assertEqual(self.arms.state, SystemState.PAUSED)
            self.assertEqual(self.arms.last_error, "operator stop")
            self.assertTrue(worker.is_alive())
        finally:
            release.set()
            worker.join(1)

    def test_arm_pause_failure_still_requests_hand_hold(self):
        self.runtime.engage()
        self.arms.pause_hook = lambda: fail("arm stop unavailable")
        with self.assertRaisesRegex(RuntimeError, "arm stop unavailable"):
            self.runtime.pause("operator stop")
        self.assertEqual(self.calls[-2:], ["arms.pause", "hands.pause"])
        self.assertEqual(self.hands.state, SystemState.PAUSED)

    def test_close_stops_arms_before_disabling_hands_and_releases_both_on_error(self):
        self.runtime.engage()
        self.calls.clear()
        self.hands.close_hook = lambda: fail("hand parameter restore failed")
        with self.assertRaisesRegex(RuntimeError, "parameter restore failed"):
            self.runtime.close()
        self.assertEqual(self.calls, ["arms.close", "hands.close"])
        self.assertEqual(self.runtime.state, SystemState.CLOSED)
        self.runtime.close()
        self.assertEqual(len(self.calls), 2)

    def test_partial_start_failure_closes_both_groups(self):
        calls = []
        arms, hands = Runtime("arms", calls), Runtime("hands", calls)
        arms.start_hook = lambda: fail("arm connection failed")
        runtime = QuestTianjiWujiTeleop(arms, hands)
        with self.assertRaisesRegex(RuntimeError, "arm connection failed"):
            runtime.start()
        self.assertEqual(calls, ["hands.start", "arms.start", "arms.close", "hands.close"])
        self.assertEqual(runtime.state, SystemState.CLOSED)

    def test_arm_shutdown_failure_survives_hand_cleanup_failure(self):
        self.arms.close_hook = lambda: fail("arm disable unconfirmed")
        self.hands.close_hook = lambda: fail("hand close failed")
        with self.assertRaisesRegex(RuntimeError, "arm disable unconfirmed.*hand close failed"):
            self.runtime.close()
        self.assertEqual(self.calls, ["arms.close", "hands.close"])
        self.assertEqual(self.runtime.state, SystemState.CLOSED)


class BackgroundKeyboardTests(unittest.TestCase):
    # These tests use real thread cancellation around fake, blocked device work.
    def setUp(self):
        self.calls = []
        self.arms, self.hands = Runtime("arms", self.calls), Runtime("hands", self.calls)
        self.runtime = QuestTianjiWujiTeleop(self.arms, self.hands)
        self.runtime.start()
        self.calls.clear()
        self.messages = []
        self.ui = runtime_ui.TeleopUI(self.runtime, None,
                                  background_engage=True, gesture=Mock(), emit=self.messages.append,
                                  following_message="双臂与双手正在跟随")

    def exercise_cancel(self, *, key, stage):
        entered, release = threading.Event(), threading.Event()

        def block():
            entered.set()
            if not release.wait(2.):
                raise RuntimeError("test did not release engagement")

        if stage == "hands":
            self.hands.prepare_hook = block
        else:
            self.arms.engage_hook = block
        try:
            if key == "rock":
                self.ui.gesture = Mock()
                self.ui.handle_gesture("engage")
            else:
                self.ui.handle("\n")
            self.assertTrue(entered.wait(1.))
            self.assertTrue(self.ui._operation_thread.is_alive())
            self.ui.handle("\n")  # Repeated Enter cannot start another enable operation.
            self.ui.handle_gesture("pause") if key == "rock" else self.ui.handle(key)
            if key == "q":
                self.runtime.close()
            self.assertEqual(self.runtime.state, SystemState.CLOSED if key == "q" else SystemState.PAUSED)
            self.assertEqual(self.ui.quit, key == "q")
        finally:
            release.set()
            if self.ui._operation_thread is not None:
                self.ui._operation_thread.join(2.)
                self.assertFalse(self.ui._operation_thread.is_alive())
                self.ui.poll_operation()
        self.assertEqual(self.calls.count("hands.prepare"), 1)
        self.assertNotIn("hands.follow", self.calls)
        if stage == "hands":
            self.assertNotIn("arms.engage", self.calls)
        self.assertNotIn("双臂与双手正在跟随", self.messages)
        self.assertEqual(self.runtime.state, SystemState.CLOSED if key == "q" else SystemState.PAUSED)
        self.ui.close()

    def test_space_cancels_blocked_hand_enable_without_waiting(self):
        self.exercise_cancel(key=" ", stage="hands")

    def test_q_cancels_blocked_arm_enable_without_starting_hand_follow(self):
        self.exercise_cancel(key="q", stage="arms")

    def test_ui_closes_arms_before_joining_blocked_background_worker(self):
        entered, release = threading.Event(), threading.Event()
        self.hands.prepare_hook = lambda: (entered.set(), release.wait(1.))
        self.arms.close_hook = release.set
        self.ui.handle("\n")
        self.assertTrue(entered.wait(.5))
        self.ui.handle("q")
        self.ui.close()
        self.assertEqual(self.calls, ["hands.prepare", "arms.close", "hands.close"])
        self.assertIsNone(self.ui._operation_thread)
        self.assertEqual(self.runtime.state, SystemState.CLOSED)

    def test_rock_gesture_cancels_hand_preparation_without_engaging_arms(self):
        self.exercise_cancel(key="rock", stage="hands")

    def test_rock_gesture_cancels_arm_engagement_without_starting_hand_follow(self):
        self.exercise_cancel(key="rock", stage="arms")

    def test_background_success_reports_combined_follow_once(self):
        self.ui.handle("\n")
        self.ui._operation_thread.join(2.)
        self.ui.poll_operation()
        self.ui.poll_operation()
        self.ui.handle("\r")
        self.assertIsNone(self.ui._operation_thread)
        self.assertEqual(self.ui.handle_gesture("engage"), "ignored")
        self.assertEqual(self.calls, ["hands.prepare", "arms.engage", "hands.follow"])
        self.assertEqual(self.runtime.state, SystemState.ENGAGED)
        self.assertEqual(self.messages.count("双臂与双手正在跟随"), 1)
        self.ui.close()


class GestureControlTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.arms, self.hands = Runtime("arms", self.calls), Runtime("hands", self.calls)
        self.runtime = QuestTianjiWujiTeleop(self.arms, self.hands)
        self.runtime.start()
        self.calls.clear()
        self.addCleanup(self.runtime.close)

    def run_phases(self, phases, *, healthy=lambda index: True, stop_at=None, fault_at=None):
        from bimanual_teleop.control.hand.gesture import GestureCommands
        from tests.support.gesture import frame

        frames = [poses for poses in phases for _ in range(10)]
        index, now = 0, 1_000_000_000
        self.runtime.health = lambda: Health(healthy(index), now, "test tracking")
        gesture = GestureCommands({side: lambda side=side, i=i:
            frame(side, frames[index][i], now, index) for i, side in enumerate(("left", "right"))})
        ui = runtime_ui.TeleopUI(self.runtime, None, gesture=gesture, emit=lambda _: None)
        events = []
        poll = gesture.poll
        sides_seen = [()]

        def observed_poll(*args, **kwargs):
            command = poll(*args, **kwargs)
            if command is not None:
                sides_seen[0] = command[1]
            return command

        gesture.poll = observed_poll
        handle = ui.handle_gesture

        def observed_handle(command):
            action = handle(command)
            events.append(SimpleNamespace(details={"sides": sides_seen[0],
                                                   "command": command, "action": action},
                                          observed_monotonic_ns=now))
            return action

        ui.handle_gesture = observed_handle

        def read(timeout):
            nonlocal index, now
            index += 1
            now += 50_000_000
            if index == len(frames):
                return "q"
            if index == fault_at:
                self.runtime.pause("test tracking fault")
            if index == stop_at:
                return " "
            return None

        with patch.object(runtime_ui.time, "monotonic_ns", side_effect=lambda: now):
            runtime_ui.run_loop(self.runtime, ui, Mock(read=read))
        self.gesture_events = events
        return [event.details for event in self.gesture_events]

    def test_both_v_start_either_rock_stops_and_both_v_resume_once(self):
        from tests.support.gesture import OPEN, ROCK, V
        events = self.run_phases([(V, OPEN), (V, V), (V, V), (OPEN, ROCK),
                                 (V, V), (V, V), (ROCK, OPEN)])
        self.assertEqual(events, [
            {"sides": ("left", "right"), "command": "engage", "action": "engage"},
            {"sides": ("right",), "command": "pause", "action": "pause"},
            {"sides": ("left", "right"), "command": "engage", "action": "engage"},
            {"sides": ("left",), "command": "pause", "action": "pause"},
        ])
        self.assertEqual(self.calls.count("arms.engage"), 2)
        self.assertEqual(self.calls.count("hands.follow"), 2)
        self.assertEqual(self.calls.count("arms.pause"), 2)
        self.assertEqual(self.calls.count("hands.pause"), 2)

    def test_startup_waits_for_ready_then_confirms_fresh_v_without_consuming_it_early(self):
        from tests.support.gesture import V
        events = self.run_phases([(V, V), (V, V), (V, V)],
                                 healthy=lambda index: index >= 10)
        self.assertEqual([e["action"] for e in events], ["engage"])
        self.assertGreaterEqual(self.gesture_events[0].observed_monotonic_ns, 1_800_000_000)
        self.assertEqual(self.calls.count("arms.engage"), 1)
        self.assertEqual(self.calls.count("hands.follow"), 1)

    def test_health_lost_between_detection_and_engage_does_not_queue_motion(self):
        ui = runtime_ui.TeleopUI(self.runtime, None, gesture=Mock(), emit=lambda _: None)
        self.runtime.health = lambda: Health(False, 0, "tracking unavailable")
        self.assertEqual(ui.handle_gesture("engage"), "not_ready")
        self.assertFalse(ui.engage_pending)
        self.runtime.health = lambda: Health(True, 1)
        ui.poll_operation()
        self.assertNotIn("arms.engage", self.calls)

    def test_recorded_v_starts_after_readiness_and_recorded_rock_stops(self):
        from tests.support.gesture import recorded_pose
        left, right = recorded_pose("v_left_early"), recorded_pose("v_right_early")
        events = self.run_phases([(left, right)] * 3 +
                                 [(left, recorded_pose("rock_right_recorded"))],
                                 healthy=lambda index: index >= 10)
        self.assertEqual([event["action"] for event in events], ["engage", "pause"])
        self.assertEqual(self.calls.count("arms.engage"), 1)
        self.assertEqual(self.calls.count("hands.follow"), 1)

    def test_fault_recovery_cannot_restart_held_v_until_a_new_pose(self):
        from tests.support.gesture import OPEN, V
        events = self.run_phases([(V, V)] * 3 + [(OPEN, OPEN), (V, V)], fault_at=15,
                                 healthy=lambda index: index < 15 or index >= 25)
        self.assertEqual([event["action"] for event in events], ["engage", "engage"])
        self.assertGreaterEqual(self.gesture_events[1].observed_monotonic_ns, 3_300_000_000)
        self.assertEqual(self.calls.count("arms.engage"), 2)

    def test_space_before_dwell_completes_cannot_be_undone_by_the_held_gesture(self):
        from tests.support.gesture import OPEN, V
        events = self.run_phases([(V, V), (V, V), (OPEN, OPEN), (V, V)], stop_at=5)
        self.assertEqual([e["action"] for e in events], ["engage"])
        self.assertEqual(self.calls[0:2], ["arms.pause", "hands.pause"])
        self.assertEqual(self.calls.count("hands.follow"), 1)

    def test_rock_while_paused_does_not_start_and_v_while_engaged_does_not_reengage(self):
        from tests.support.gesture import OPEN, ROCK, V
        events = self.run_phases([(ROCK, ROCK), (V, V), (OPEN, OPEN), (V, V)])
        self.assertEqual([event["action"] for event in events], ["pause", "engage", "ignored"])
        self.assertEqual(self.calls.count("arms.engage"), 1)
        self.assertEqual(self.calls.count("hands.follow"), 1)

    def test_combined_enter_starts_and_resumes_both_groups(self):
        self.arms, self.hands = Runtime("arms", self.calls), Runtime("hands", self.calls)
        self.runtime = QuestTianjiWujiTeleop(self.arms, self.hands)
        self.runtime.start()
        self.calls.clear()
        ui = runtime_ui.TeleopUI(self.runtime, None, gesture=Mock(),
                          emit=lambda _: None)
        ui.handle("\n")
        self.assertEqual(self.calls, ["hands.prepare", "arms.engage", "hands.follow"])
        self.assertEqual(self.runtime.state, SystemState.ENGAGED)
        ui.handle("\r")
        self.assertEqual(ui.handle_gesture("engage"), "ignored")
        self.assertEqual(len(self.calls), 3)
        ui.handle(" ")
        self.assertEqual(self.runtime.state, SystemState.PAUSED)
        ui.handle("\r")
        self.assertEqual(self.runtime.state, SystemState.ENGAGED)
        ui.handle("q")
        self.assertEqual(self.calls, ["hands.prepare", "arms.engage", "hands.follow",
                                     "arms.pause", "hands.pause",
                                     "hands.prepare", "arms.engage", "hands.follow"])
        self.assertTrue(ui.quit)

    def test_combined_enter_waits_for_readiness_and_run_loop_engages_once(self):
        ready, now, reads = False, 1_000_000_000, 0
        self.runtime.health = lambda: Health(ready, now, "tracking unavailable")
        ui = runtime_ui.TeleopUI(self.runtime, None, gesture=Mock(poll=Mock(return_value=None)),
                          toggle_engagement_key="enter", emit=lambda _: None)

        def read(timeout):
            nonlocal ready, now, reads
            now += runtime_ui.PERIOD_NS
            reads += 1
            if reads == 1:
                return "\r"
            if reads == 2:
                self.assertTrue(ui.engage_pending)
                self.assertEqual(self.calls, [])
                ready = True
                return None
            self.assertFalse(ui.engage_pending)
            self.assertEqual(self.runtime.state, SystemState.ENGAGED)
            return "q"

        with patch.object(runtime_ui.time, "monotonic_ns", side_effect=lambda: now):
            runtime_ui.run_loop(self.runtime, ui, Mock(read=read))
        self.assertEqual(self.calls.count("hands.prepare"), 1)
        self.assertEqual(self.calls.count("arms.engage"), 1)
        self.assertEqual(self.calls.count("hands.follow"), 1)

    def test_combined_pending_enter_can_be_cancelled_by_space_rock_or_q(self):
        for key in (" ", "rock", "q"):
            with self.subTest(key=key):
                self.calls.clear()
                self.runtime.health = lambda: Health(False, 0, "tracking unavailable")
                ui = runtime_ui.TeleopUI(self.runtime, None, gesture=Mock(), emit=lambda _: None)
                ui.handle("\n")
                self.assertTrue(ui.engage_pending)
                self.assertEqual(self.calls, [])
                ui.handle_gesture("pause") if key == "rock" else ui.handle(key)
                self.assertFalse(ui.engage_pending)
                self.assertEqual(self.calls, [] if key == "q" else ["arms.pause", "hands.pause"])
                self.assertEqual(ui.quit, key == "q")


class EntryPointTests(unittest.TestCase):
    def test_viewer_is_opt_in_and_both_entries_close_the_preview(self):
        for entry, args in ((cli, ["--arms-only"]), (hand_cli, [])):
            for enabled in (False, True):
                with self.subTest(entry=entry.__name__, enabled=enabled), \
                        patch.object(entry, "NonblockingTerminal"), \
                        patch.object(cli, "create_runtime", return_value=self.runtime), \
                        patch.object(entry, "run_loop", return_value={"elapsed_s": .1, "motion_pauses": 0}), \
                        patch("bimanual_teleop.visualization.realsense.RealSensePreview") as preview, \
                        redirect_stderr(io.StringIO()):
                    self.assertEqual(entry.main(args + (["--viewer"] if enabled else [])), 0)
                self.assertEqual(preview.called, enabled)
                if enabled:
                    preview.return_value.start.assert_called_once()
                    preview.return_value.close.assert_called_once()

    def setUp(self):
        self.module = ModuleType("bimanual_teleop.control.hand.follow")
        self.config = {"control_hz": 120}
        self.runtime = Runtime("runtime", [])
        self.load_config = Mock(return_value=self.config)
        config_patch = patch("bimanual_teleop.devices.wuji.config.load_config", self.load_config)
        config_patch.start()
        self.addCleanup(config_patch.stop)
        self.module.preflight = Mock()
        self.module.create_wuji_teleop = Mock(return_value=self.runtime)
        self.process_module = ModuleType("bimanual_teleop.control.hand.process")
        self.process_module.WujiProcess = Mock(return_value=self.runtime)
        self.patch_module = patch.dict(sys.modules, {self.module.__name__: self.module,
                                                     self.process_module.__name__: self.process_module})
        self.patch_module.start()
        self.addCleanup(self.patch_module.stop)
        prepare = patch.object(cli, "prepare_initial_pose")
        prepare.start()
        self.addCleanup(prepare.stop)
        for entry in (cli, hand_cli):
            confirm = patch.object(entry, "confirm_motion", return_value=True)
            confirm.start()
            self.addCleanup(confirm.stop)
            logging = patch.object(entry, "configure_runtime_logging")
            logging.start()
            self.addCleanup(logging.stop)

    def test_arm_only_factory_does_not_import_wuji_or_construct_hand_runtime(self):
        args = SimpleNamespace(serial=None, robot_ip="unused", sdk_root=None,
                               arms_only=True,
                               wuji_config=Path("absent-wuji.yaml"), coordinate_frame="headset")
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "wuji_sdk" or name.startswith("bimanual_teleop.control.hand.follow"):
                raise AssertionError("arm-only startup imported Wuji")
            return original_import(name, *args, **kwargs)

        with ExitStack() as stack:
            for name in ("devices.quest.adapter.QuestSource", "devices.tianji.driver.TianjiDriver",
                         "devices.tianji.model.TianjiKinematics",
                         "control.arm.cartesian.TianjiCartesianExecutor"):
                stack.enter_context(patch(f"bimanual_teleop.{name}"))
            factory = stack.enter_context(patch.object(arm_runtime, "QuestTianjiTeleop"))
            stack.enter_context(patch("builtins.__import__", side_effect=guarded_import))
            self.assertIs(cli.create_runtime(args, None, None), factory.return_value)
            self.assertEqual(factory.call_args.kwargs["coordinate_frame"], "headset")
        self.module.create_wuji_teleop.assert_not_called()

    def test_combined_factory_constructs_both_arm_and_hand_runtimes(self):
        args = SimpleNamespace(serial=None, robot_ip="unused", sdk_root=None, side="both",
                               arms_only=False, wuji_config=cli.DEFAULT_WUJI_CONFIG,
                               wuji_settings=self.config, coordinate_frame="headset")
        with ExitStack() as stack:
            for name in ("devices.quest.adapter.QuestSource", "devices.tianji.driver.TianjiDriver",
                         "devices.tianji.model.TianjiKinematics",
                         "control.arm.cartesian.TianjiCartesianExecutor"):
                stack.enter_context(patch(f"bimanual_teleop.{name}"))
            arms = stack.enter_context(patch.object(arm_runtime, "QuestTianjiTeleop"))
            combined = stack.enter_context(patch("bimanual_teleop.control.combined.QuestTianjiWujiTeleop"))
            self.assertIs(cli.create_runtime(args, None, None), combined.return_value)
        self.process_module.WujiProcess.assert_called_once_with(
            self.config, verbose=False)
        self.module.create_wuji_teleop.assert_not_called()
        self.assertEqual(arms.call_args.kwargs["side"], "both")
        combined.assert_called_once_with(arms.return_value, self.runtime)

    def test_arms_only_ignores_missing_wuji_config_and_skips_sdk_preflight(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(cli, "NonblockingTerminal"), \
                patch.object(cli, "create_runtime", return_value=self.runtime) as create, \
                patch.object(cli, "run_loop", return_value={"elapsed_s": .1, "motion_pauses": 0}) as loop, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            absent = Path(directory) / "absent-wuji.yaml"
            self.assertEqual(cli.main(["--arms-only", "--wuji-config", str(absent)]), 0)
        args = create.call_args.args[0]
        self.assertTrue(args.arms_only)
        self.assertEqual(args.wuji_config, absent)
        self.assertIsNone(loop.call_args.args[1].gesture)
        self.load_config.assert_not_called()
        self.module.preflight.assert_not_called()
        self.module.create_wuji_teleop.assert_not_called()
        cli.configure_runtime_logging.assert_called_once()
        self.assertEqual(cli.configure_runtime_logging.call_args.kwargs["wuji"], False)
        self.assertEqual(cli.configure_runtime_logging.call_args.kwargs["verbose"], False)
        self.assertEqual(Path(cli.configure_runtime_logging.call_args.kwargs["log_file"]).parent,
                         Path("logs"))

    def test_combined_single_arm_selection_fails_before_configuration_or_devices(self):
        for side in ("left", "right"):
            with self.subTest(side=side), patch.object(cli, "load_config") as config, \
                    patch.object(cli, "create_runtime") as create, \
                    redirect_stderr(io.StringIO()) as output:
                with self.assertRaises(SystemExit) as result:
                    cli.main(["--side", side])
                self.assertEqual(result.exception.code, 2)
                self.assertIn("--arms-only", output.getvalue())
                config.assert_not_called()
                create.assert_not_called()
        self.load_config.assert_not_called()

    def test_wuji_preflight_failure_precedes_arm_ready_pose_motion(self):
        self.module.preflight.side_effect = RuntimeError("Wuji SDK unavailable")
        with patch.object(cli, "prepare_initial_pose") as prepare, \
             patch.object(cli, "NonblockingTerminal") as terminal, \
             patch.object(cli, "create_runtime") as create, \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = cli.main(["--wuji-config", "unused.yaml"])
        self.assertEqual(code, 1)
        prepare.assert_not_called()
        terminal.assert_not_called()
        create.assert_not_called()

    def test_invalid_wuji_configuration_does_not_enter_motion_preparation(self):
        self.load_config.side_effect = ValueError("missing right glove")
        with patch.object(cli, "prepare_initial_pose") as prepare, \
             patch.object(cli, "NonblockingTerminal") as terminal, \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["--wuji-config", "unused.yaml"]), 1)
        prepare.assert_not_called()
        terminal.assert_not_called()
        self.module.preflight.assert_not_called()

    def test_missing_optional_sdk_is_reported_without_motion_or_traceback(self):
        self.module.preflight.side_effect = ImportError("No module named wuji_sdk")
        with patch.object(cli, "prepare_initial_pose") as prepare, \
             patch.object(cli, "NonblockingTerminal") as terminal, \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as output:
            self.assertEqual(cli.main(["--wuji-config", "unused.yaml"]), 1)
        prepare.assert_not_called()
        terminal.assert_not_called()
        self.assertIn("wuji_sdk", output.getvalue())

    def test_default_combined_entry_prepares_and_waits_for_gesture_or_enter(self):
        settings = cli.load_config(cli.DEFAULT_CONFIG)
        settings["controls"]["gesture_engagement_enabled"] = True
        samples = {side: Mock() for side in ("left", "right")}
        self.runtime.hands = SimpleNamespace(glove_samples=lambda: samples, glove_timeout_ns=250_000_000)
        with patch.object(cli, "NonblockingTerminal"), \
             patch.object(cli, "prepare_initial_pose") as prepare, \
             patch.object(cli, "create_runtime", return_value=self.runtime) as create, \
             patch.object(cli, "load_config", return_value=settings) as tianji_load, \
             patch.object(cli, "run_loop", return_value={"elapsed_s": .1, "motion_pauses": 0}) as loop, \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = cli.main([])
        self.assertEqual(code, 0)
        tianji_load.assert_called_once_with(cli.DEFAULT_CONFIG, None)
        self.load_config.assert_called_once_with(cli.DEFAULT_WUJI_CONFIG)
        self.assertEqual(create.call_args.args[0].config, cli.DEFAULT_CONFIG)
        self.assertEqual(create.call_args.args[0].wuji_config, cli.DEFAULT_WUJI_CONFIG)
        self.assertFalse(create.call_args.args[0].arms_only)
        self.assertIs(create.call_args.args[0].wuji_settings, self.config)
        prepare.assert_called_once()
        ui = loop.call_args.args[1]
        self.assertTrue(ui.background_engage)
        self.assertEqual({side: read() for side, read in ui.gesture.sources.items()}, samples)
        self.assertNotIn("runtime.engage", self.runtime.calls)

    def test_combined_entry_disables_all_gestures_from_config(self):
        settings = cli.load_config(cli.DEFAULT_CONFIG)
        settings["controls"]["gesture_engagement_enabled"] = False
        with patch.object(cli, "NonblockingTerminal"), \
             patch.object(cli, "prepare_initial_pose"), \
             patch.object(cli, "create_runtime", return_value=self.runtime), \
             patch.object(cli, "load_config", return_value=settings), \
             patch("bimanual_teleop.control.hand.gesture.GestureCommands") as gestures, \
             patch.object(cli, "run_loop", return_value={"elapsed_s": .1, "motion_pauses": 0}) as loop, \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as output:
            self.assertEqual(cli.main([]), 0)
        gestures.assert_not_called()
        self.assertIsNone(loop.call_args.args[1].gesture)
        self.assertNotIn("比 V", output.getvalue())
        self.assertNotIn("摇滚", output.getvalue())
        self.assertNotIn("张开", output.getvalue())

    def test_combined_entry_forwards_explicit_config_paths_and_user_name(self):
        settings = cli.load_config(cli.DEFAULT_CONFIG)
        self.runtime.hands = SimpleNamespace(glove_samples=lambda: {side: Mock() for side in ("left", "right")},
                                             glove_timeout_ns=250_000_000)
        with patch.object(cli, "NonblockingTerminal"), \
                patch.object(cli, "create_runtime", return_value=self.runtime) as create, \
                patch.object(cli, "load_config", return_value=settings) as tianji_load, \
                patch.object(cli, "run_loop", return_value={"elapsed_s": .1, "motion_pauses": 0}), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["--tianji-config", "custom-tianji.yaml",
                                       "--wuji-config", "custom-wuji.yaml", "--user-name", "Alice"]), 0)
        tianji_load.assert_called_once_with(Path("custom-tianji.yaml"), None)
        self.load_config.assert_called_once_with(Path("custom-wuji.yaml"))
        self.assertEqual(create.call_args.args[0].config, Path("custom-tianji.yaml"))
        self.assertEqual(create.call_args.args[0].wuji_config, Path("custom-wuji.yaml"))
        self.assertEqual(create.call_args.args[0].wuji_settings["sdk_user_name"], "Alice")

    def test_blank_user_name_is_rejected_before_motion_confirmation(self):
        for entry in (cli, hand_cli):
            with self.subTest(entry=entry.__name__), \
                    patch.object(entry, "confirm_motion") as confirm, \
                    redirect_stderr(io.StringIO()):
                self.assertEqual(entry.main(["--user-name", " "]), 1)
                confirm.assert_not_called()
                self.module.create_wuji_teleop.assert_not_called()

    def run_hand_cli(self, arguments, *, start_error=None):
        if start_error:
            self.runtime.start_hook = lambda: fail(start_error)
        with patch.object(hand_cli, "NonblockingTerminal") as terminal, \
             patch.object(hand_cli, "run_loop", return_value={"elapsed_s": .1}) as loop, \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            terminal.return_value.__enter__.return_value = Mock()
            code = hand_cli.main(arguments)
        return code, loop

    def test_standalone_waits_for_engagement_with_both_hands_at_120_hz(self):
        code, loop = self.run_hand_cli([])
        self.assertEqual(code, 0)
        self.module.create_wuji_teleop.assert_called_once_with(
            self.config, sides=("left", "right"), sink=None)
        self.assertEqual(loop.call_args.kwargs["period_ns"], round(1e9 / 120))
        self.assertEqual(self.runtime.calls, ["runtime.start", "runtime.close"])

    def test_standalone_accepts_named_wuji_config_and_legacy_alias(self):
        for flag in ("--wuji-config", "--config"):
            with self.subTest(flag=flag):
                self.load_config.reset_mock()
                self.assertEqual(self.run_hand_cli([flag, "custom-wuji.yaml"])[0], 0)
                self.load_config.assert_called_once_with(Path("custom-wuji.yaml"))

    def test_standalone_confirmation_does_not_implicitly_engage(self):
        code, _ = self.run_hand_cli(["--side", "right"])
        self.assertEqual(code, 0)
        self.module.create_wuji_teleop.assert_called_once_with(
            self.config, sides=("right",), sink=None)
        self.assertNotIn("runtime.engage", self.runtime.calls)

    def test_standalone_partial_start_failure_still_closes_runtime(self):
        code, loop = self.run_hand_cli([], start_error="hand not connected")
        self.assertEqual(code, 1)
        loop.assert_not_called()
        self.assertEqual(self.runtime.calls, ["runtime.start", "runtime.close"])

    def test_motion_stub_and_failure_create_no_runtime_files(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            os.chdir(temporary)
            try:
                self.assertEqual(self.run_hand_cli([])[0], 0)
                self.runtime.start_hook = lambda: fail("模拟断流")
                self.assertEqual(self.run_hand_cli([])[0], 1)
                self.assertEqual(list(Path(temporary).rglob("*")), [])
            finally:
                os.chdir(original)

    def test_native_connection_timeout_is_reported_by_cli(self):
        from bimanual_teleop.devices.wuji.adapter import WujiHandDriver

        class WujiException(Exception):
            pass

        manager = Mock()
        manager.connect.side_effect = WujiException("Connection timeout")
        sdk = SimpleNamespace(ConnectOptions=lambda **kwargs: SimpleNamespace(**kwargs))
        hand = WujiHandDriver("left", "192.168.1.110:7447", manager=manager, sdk=sdk)
        self.runtime.start_hook = lambda: hand.start(None)
        self.runtime.close_hook = hand.close
        with patch.object(hand_cli, "print_message") as output:
            code, loop = self.run_hand_cli([])
        self.assertEqual(code, 1)
        loop.assert_not_called()
        self.assertTrue(hand._closed)
        message, severity = output.call_args.args
        self.assertEqual(severity, "error")
        self.assertIn("wuji_left_hand (192.168.1.110:7447) connect failed", message)
        self.assertIn("WujiException: Connection timeout", message)
        self.assertEqual(self.runtime.calls[-1], "runtime.close")


if __name__ == "__main__":
    unittest.main()
