"""Keyboard and scheduler checks with fake devices; never load a native SDK."""

from contextlib import redirect_stderr, redirect_stdout
import logging
import io
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml

from bimanual_teleop.types import ControlProfile, Health
from bimanual_teleop.common.console import (LiveProgress, StatusConsole,
    configure_runtime_logging, format_message, print_message)
from bimanual_teleop.cli import teleop_quest_tianji as cli
from bimanual_teleop.cli import runtime as runtime_ui
from bimanual_teleop.common import terminal as terminal_module
from bimanual_teleop.control.arm import preparation


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ControlProfile("test-profile", "cartesian_impedance", {
    "active_arms": ["left", "right"], "arms": {"left": {}, "right": {}},
})


class SideSelectionTests(unittest.TestCase):
    def test_profile_selects_only_requested_arm_without_mutating_loaded_profile(self):
        for side in ("left", "right"):
            selected = cli.select_profile_side(PROFILE, side)
            self.assertEqual(selected.parameters["active_arms"], [side])
            self.assertEqual(selected.profile_id, f"test-profile-{side}")
        self.assertEqual(PROFILE.parameters["active_arms"], ["left", "right"])
        self.assertIs(cli.select_profile_side(PROFILE, "both"), PROFILE)


class Runtime:
    def __init__(self):
        self.calls = []
        self.ticks = []

    def status(self, *, include_target=True):
        return {"state": "PAUSED", "last_error": None}

    def tick(self, now):
        self.ticks.append(now)

    def health(self): return Health(True, 0)
    def start(self): self.calls.append(("start",))
    def close(self): self.calls.append(("close",))
    def engage(self, profile): self.calls.append(("engage", profile))
    def pause(self, reason):
        self.calls.append(("pause", reason))


class KeyboardTests(unittest.TestCase):
    def setUp(self):
        self.runtime = Runtime()
        self.messages = []
        self.ui = runtime_ui.TeleopUI(self.runtime, PROFILE, emit=self.messages.append)

    def test_enter_engages_and_space_pauses(self):
        self.ui.handle("\n")
        self.assertEqual(self.runtime.calls[-1], ("engage", PROFILE))
        self.ui.handle(" ")
        self.assertEqual(self.runtime.calls[-1][0], "pause")

    def test_removed_calibration_keys_have_no_device_effect(self):
        for key in "hb j1234567rs".replace(" ", ""):
            self.ui.handle(key)
        self.assertEqual(self.runtime.calls, [])

    def test_handled_runtime_pause_is_reported_once_and_survives_keyboard_exit(self):
        self.runtime.state = "paused"
        self.runtime.last_error = "right: IK failed"
        self.ui.report_runtime_pause()
        self.ui.report_runtime_pause()
        self.assertEqual(self.ui.motion_pauses, 1)
        self.assertEqual(self.ui.last_motion_error, "right: IK failed")
        self.assertIn("遥操作已暂停", self.messages[-1])
        self.ui.handle("q")
        self.assertEqual(self.ui.last_motion_error, "right: IK failed")

    def test_control_diagnostic_is_optional_without_losing_pause_recovery_hint(self):
        reason = 'right: IK failed\n[控制诊断] {"target":[1,2,3]}'
        for verbose in (False, True):
            with self.subTest(verbose=verbose):
                messages = []
                ui = runtime_ui.TeleopUI(self.runtime, PROFILE, emit=messages.append, verbose=verbose)
                self.runtime.state = "paused"
                self.runtime.last_error = reason
                ui.report_runtime_pause()
                self.assertEqual("[控制诊断]" in messages[0], verbose)
                self.assertIn("right: IK failed", messages[0])
                self.assertIn("恢复时按 Enter", messages[0])
                self.assertEqual(ui.last_motion_error, reason)

    def test_runtime_rejection_pauses_but_quit_closes_without_another_pause(self):
        self.runtime.engage = Mock(side_effect=ValueError("tracking lost"))
        self.ui.handle("\n")
        self.assertEqual(self.runtime.calls[-1], ("pause", "tracking lost"))
        self.ui.handle("q")
        self.assertTrue(self.ui.quit)
        self.ui.close()
        self.assertEqual(self.runtime.calls, [("pause", "tracking lost"), ("close",)])

    def test_status_ignores_clock_counters_and_reports_readiness_changes_once(self):
        status = {"state": "ready", "health": {"ready": False, "detail": "left tracking unavailable"}}
        self.ui.report_status(status)
        for index in range(5):
            self.ui.report_status({**status, "cycles": index, "last_compute_ns": index*100,
                                   "scheduler_cycles": index, "skipped_deadlines": index})
        self.assertEqual(len(self.messages), 1)
        status["health"] = {"ready": True, "detail": "ready", "observed_monotonic_ns": 123}
        self.ui.report_status(status)
        self.ui.report_status({**status, "health": {**status["health"], "observed_monotonic_ns": 456}})
        self.assertEqual(len(self.messages), 2)
        self.assertIn("设备已就绪", self.messages[-1])
        status["health"] = {"ready": False, "detail": "right tracking unavailable"}
        self.ui.report_status(status)
        self.assertEqual(len(self.messages), 3)
        self.assertIn("right tracking unavailable", self.messages[-1])

    def test_pending_enter_prompt_is_not_repeated_by_status_or_extra_enter(self):
        self.runtime.health = lambda: Health(False, 0, "left tracking unavailable")
        self.ui.handle("\n")
        self.ui.handle("\n")
        self.ui.report_status({"state": "ready", "health": {"ready": False, "detail": "left tracking unavailable"}})
        self.assertEqual(len(self.messages), 1)
        self.assertIn("就绪后自动接合", self.messages[0])
        self.ui.handle(" ")
        self.assertFalse(self.ui.engage_pending)
        self.assertIn("暂停", self.messages[-1])

    def test_common_wait_reasons_are_short_chinese_without_changing_diagnostics(self):
        examples = (
            ("Waiting for Quest frames", "等待 Quest 数据"),
            ("Quest left tracking unavailable (flags=0, active=False)", "左手柄未被追踪"),
            ("Quest right tracking unavailable (flags=0)", "右手柄未被追踪"),
            ("Quest XR session not focused; close the headset system menu", "请关闭 Quest 系统菜单"),
            ("Quest reference space is changing; wait for a new origin frame, then re-engage", "坐标系正在切换"),
            ("Quest reference changed; press Enter to re-engage", "坐标系已改变"),
            ("Quest input stream gap reached 100 ms; re-engage after fresh frames return", "中断达到超时阈值"),
            ("Quest input is silent or additionally queued beyond 100 ms", "100 毫秒未更新或排队超时"),
            ("Quest additional input backlog exceeded 100 ms", "排队已超过 100 毫秒"),
            ("no Quest frame arrived for 1 second; host stream is silent", "连续 1 秒没有新帧"),
            ("No Tianji feedback", "等待机器人反馈"),
        )
        for raw, expected in examples:
            self.assertIn(expected, self.ui.waiting_message(raw))
            self.assertNotIn("flags=", self.ui.waiting_message(raw))
        raw = "tj_move_joints: OnSetTargetState_A returned 1"
        self.assertEqual(runtime_ui.brief_reason(raw), raw)


class TrackingNoticeTests(unittest.TestCase):
    def setUp(self):
        self.now = 0
        self.messages = []
        self.runtime = SimpleNamespace(state="engaged", last_error=None,
            tracking_status={"left": self.metric()},
            status=Mock(side_effect=AssertionError("full status must not be computed")))
        self.ui = runtime_ui.TeleopUI(self.runtime, PROFILE, verbose=True,
                              emit=self.messages.append)
        clock = patch.object(runtime_ui.time, "monotonic_ns", side_effect=lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)

    @staticmethod
    def metric(*, limited=True, position=.002, angle=0.):
        return {"limited": limited, "progress": .25, "position_error_m": position,
                "orientation_error_rad": angle}

    def poll(self, milliseconds):
        self.now = round(milliseconds*1e6)
        self.ui.report_tracking()

    def test_default_output_hides_tracking_fluctuations_but_keeps_fault_pause(self):
        self.ui = runtime_ui.TeleopUI(self.runtime, PROFILE, emit=self.messages.append)
        for start in (0, 1200):
            self.runtime.tracking_status["left"] = self.metric()
            self.poll(start)
            self.poll(start + 500)
            self.runtime.tracking_status["left"] = self.metric(limited=False)
            self.poll(start + 600)
            self.poll(start + 1100)
        self.assertEqual(self.messages, [])
        self.runtime.state, self.runtime.last_error = "paused", "Quest tracking lost"
        self.ui.report_runtime_pause()
        self.ui.report_runtime_pause()
        self.assertEqual(len(self.messages), 1)
        self.assertIn("遥操作已暂停：Quest tracking lost", self.messages[0])
        self.assertIn("按 Enter", self.messages[0])
        self.assertEqual(self.ui.motion_pauses, 1)

    def test_sustained_limitation_reports_once_and_keeps_engagement_distinct_from_pause(self):
        self.poll(0)
        self.poll(499.999)
        self.assertEqual(self.messages, [])
        for milliseconds in range(500, 2500, 5):
            self.poll(milliseconds)
        self.assertEqual(len(self.messages), 1)
        self.assertIn("目标推进受限（左臂）", self.messages[0])
        self.assertIn("仍处于接合状态", self.messages[0])
        self.assertIn("手柄位置/姿态移回可达范围", self.messages[0])
        self.assertNotIn("关节越界", self.messages[0])
        self.assertNotIn("已暂停", self.messages[0])
        self.assertEqual(self.runtime.state, "engaged")
        self.assertEqual(self.ui.motion_pauses, 0)
        self.assertIsNone(self.ui.last_motion_error)
        self.runtime.status.assert_not_called()

    def test_short_limitation_and_subthreshold_errors_do_not_report(self):
        self.poll(0)
        self.poll(400)
        self.runtime.tracking_status["left"] = self.metric(limited=False)
        self.poll(450)
        self.poll(1200)
        self.runtime.tracking_status["left"] = self.metric(position=.00099, angle=math.radians(.49))
        self.poll(1300)
        self.poll(2500)
        self.runtime.tracking_status["left"] = self.metric(limited=False, position=.1, angle=.2)
        self.poll(2600)
        self.poll(4000)
        self.assertEqual(self.messages, [])

    def test_position_or_orientation_threshold_can_trigger_for_combined_runtime(self):
        arms = self.runtime
        self.runtime = SimpleNamespace(state=SimpleNamespace(value="engaged"), arms=arms,
                                       status=Mock(side_effect=AssertionError("full status called")))
        self.ui.runtime = self.runtime
        arms.tracking_status = {"left": self.metric(position=.001),
                                "right": self.metric(position=0., angle=math.radians(.5))}
        self.poll(0)
        self.poll(500)
        self.assertEqual(len(self.messages), 1)
        self.assertIn("左臂、右臂", self.messages[0])
        self.runtime.status.assert_not_called()
        arms.status.assert_not_called()

    def test_alternating_brief_limitation_on_each_side_does_not_combine_durations(self):
        for index, side in enumerate(("left", "right", "left", "right")):
            self.runtime.tracking_status = {side: self.metric()}
            self.poll(index*400)
        self.assertEqual(self.messages, [])
        self.poll(1700)
        self.assertEqual(len(self.messages), 1)
        self.assertIn("（右臂）", self.messages[0])
        self.runtime.tracking_status["left"] = self.metric()
        self.poll(1800)
        self.poll(2400)
        self.assertEqual(len(self.messages), 1)

    def test_recovery_requires_stable_half_second_and_allows_a_later_notice(self):
        self.poll(0)
        self.poll(500)
        self.runtime.tracking_status["left"] = self.metric(limited=False)
        self.poll(600)
        self.poll(1099.999)
        self.assertEqual(len(self.messages), 1)
        self.runtime.tracking_status["left"] = self.metric()
        self.poll(1100)
        self.runtime.tracking_status["left"] = self.metric(limited=False)
        self.poll(1200)
        self.poll(1699.999)
        self.assertEqual(len(self.messages), 1)
        for milliseconds in range(1700, 3000, 5):
            self.poll(milliseconds)
        self.assertEqual(len(self.messages), 2)
        self.assertEqual(self.messages[-1], "目标推进已恢复正常，遥操作继续跟随。")
        self.runtime.tracking_status["left"] = self.metric()
        self.poll(3100)
        self.poll(3600)
        self.assertEqual(len(self.messages), 3)
        self.assertIn("目标推进受限", self.messages[-1])

    def test_pause_resets_notice_without_reporting_recovery_or_automatic_resume(self):
        self.poll(0)
        self.poll(500)
        self.runtime.state, self.runtime.last_error = "paused", "Quest tracking lost"
        self.ui.report_runtime_pause()
        self.runtime.tracking_status["left"] = self.metric(limited=False)
        self.poll(2000)
        self.assertEqual(len(self.messages), 2)
        self.assertIn("遥操作已暂停", self.messages[-1])
        self.assertEqual(self.ui.motion_pauses, 1)
        self.runtime.state, self.runtime.last_error = "engaged", None
        self.ui._engaged()
        self.poll(2100)
        self.poll(2700)
        self.assertEqual(len(self.messages), 3)
        self.runtime.tracking_status["left"] = self.metric()
        self.poll(3000)
        self.poll(3499.999)
        self.assertEqual(len(self.messages), 3)
        self.poll(3500)
        self.assertEqual(len(self.messages), 4)
        self.assertIn("目标推进受限", self.messages[-1])

    def test_missing_tracking_snapshot_or_paused_state_does_not_emit_motion_notice(self):
        self.poll(0)
        self.runtime.tracking_status = {}
        self.poll(500)
        self.runtime.tracking_status = {"left": self.metric()}
        self.poll(700)
        self.poll(1199.999)
        self.assertEqual(self.messages, [])
        self.runtime.state = "paused"
        self.poll(1200)
        self.poll(2000)
        self.assertEqual(self.messages, [])

    def test_scheduler_reports_limitation_without_increasing_full_status_poll_rate(self):
        runtime = Runtime()
        runtime.state = "engaged"
        runtime.tracking_status = {"left": self.metric()}
        runtime.status = Mock(return_value={"state": "engaged", "health": {"ready": True}})
        ui = runtime_ui.TeleopUI(runtime, PROFILE, verbose=True,
                         emit=self.messages.append)
        def read(timeout):
            self.now += 5_000_000
            return "q" if self.now >= 800_000_000 else None
        runtime_ui.run_loop(runtime, ui, Mock(read=read))
        self.assertEqual(len(runtime.ticks), 160)
        self.assertEqual(runtime.status.call_count, 1)
        self.assertEqual(len(self.messages), 1)
        self.assertIn("目标推进受限", self.messages[0])


class ConsoleTests(unittest.TestCase):
    def test_tty_colours_levels_but_redirect_and_no_color_are_plain(self):
        stream = io.StringIO()
        stream.isatty = lambda: True
        with patch.dict(os.environ, {}, clear=True):
            for level, label in (("info", "提示"), ("ready", "就绪"), ("warning", "警告"),
                                 ("error", "错误"), ("done", "完成")):
                self.assertIn("\x1b[", format_message("测试", level, stream=stream))
                self.assertIn(label, format_message("测试", level, stream=stream))
            print_message("测试", "ready", stream=stream)
            self.assertIn("测试", stream.getvalue())
            self.assertEqual(format_message("测试", "ready", stream=io.StringIO()), "[就绪] 测试")
        with patch.dict(os.environ, {"NO_COLOR": ""}):
            self.assertEqual(format_message("测试", "error", stream=stream), "[错误] 测试")

    def test_state_dedup_warning_limit_and_tty_progress(self):
        stream = io.StringIO()
        stream.isatty = lambda: True
        console = StatusConsole(stream=stream, warning_interval_s=5)
        progress = LiveProgress(stream=stream, interval_s=.2)
        with patch("bimanual_teleop.common.console.time.monotonic", side_effect=[0, 1, 6, 7, 7.1, 7.3]):
            console.state("就绪", "ready")
            console.state("就绪", "ready")
            console.warning("重复故障")
            console.warning("重复故障")
            console.warning("重复故障")
            progress.update("目标 1")
            progress.update("目标 2")
            progress.update("目标 3")
            progress.clear()
        output = stream.getvalue()
        self.assertEqual(output.count("重复故障"), 2)
        self.assertEqual(output.count("就绪"), 2)  # coloured label and message
        self.assertIn("\r", output)
        self.assertNotIn("目标 2", output)

    def test_default_logging_keeps_sdk_errors_and_python_warnings(self):
        sdk = SimpleNamespace(set_log_level=Mock())
        logger = logging.getLogger("bimanual_teleop")
        handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
        try:
            with patch.dict(sys.modules, {"wuji_sdk": sdk}):
                configure_runtime_logging(wuji=True)
                configure_runtime_logging(wuji=True)
            self.assertEqual(sdk.set_log_level.call_args_list, [
                unittest.mock.call("error"), unittest.mock.call("error")])
            self.assertEqual(logger.level, logging.WARNING)
        finally:
            logger.handlers[:] = handlers
            logger.setLevel(level)
            logger.propagate = propagate


class SchedulerTests(unittest.TestCase):
    def test_terminal_escape_sequences_do_not_become_motion_or_direction_keys(self):
        terminal = cli.NonblockingTerminal(io.StringIO())
        self.assertIsNone(terminal._keys("\x1b["))
        self.assertIsNone(terminal._keys("B\x1b[6~\x1bOP"))
        self.assertEqual(terminal._keys("h j"), "h j")

    def test_keyboard_does_not_block_ticks_and_delayed_cycles_are_skipped(self):
        clock = [0]
        runtime = Runtime()
        ui = runtime_ui.TeleopUI(runtime, PROFILE, emit=lambda _: None)
        keys = iter(("\n", None, None, None, None, "q"))

        def read(timeout):
            self.assertLessEqual(timeout, .005)
            key = next(keys)
            clock[0] += 30_000_000 if len(runtime.ticks) == 3 else 5_000_000
            return key

        with patch.object(runtime_ui.time, "monotonic_ns", side_effect=lambda: clock[0]):
            timing = runtime_ui.run_loop(runtime, ui, Mock(read=read))
        self.assertEqual(len(runtime.ticks), 6)
        self.assertEqual(timing["skipped_deadlines"], 5)
        self.assertEqual(sum(call[0] == "engage" for call in runtime.calls), 1)
        self.assertEqual(len(set(runtime.ticks)), len(runtime.ticks))

    def test_late_wakeup_and_slow_tick_do_not_cause_catch_up_bursts(self):
        for work_ns in (0, 12_000_000):
            with self.subTest(work_ns=work_ns):
                clock, completed = [0], []
                runtime = Runtime()
                ui = runtime_ui.TeleopUI(runtime, PROFILE, emit=lambda _: None)

                def tick(now):
                    runtime.ticks.append(now)
                    clock[0] += work_ns if len(runtime.ticks) == 2 else 0
                    completed.append(clock[0])

                def read(timeout):
                    clock[0] += round(timeout * 1e9)
                    if len(runtime.ticks) == 1:
                        clock[0] += 4_700_000
                    return "q" if len(runtime.ticks) == 4 else None

                runtime.tick = tick
                with patch.object(runtime_ui.time, "monotonic_ns", side_effect=lambda: clock[0]):
                    timing = runtime_ui.run_loop(runtime, ui, Mock(read=read))
                self.assertEqual(len(runtime.ticks), 4)
                self.assertTrue(all(b - a >= runtime_ui.PERIOD_NS
                                    for a, b in zip(runtime.ticks, runtime.ticks[1:])))
                # A slow tick must finish before the next one starts; elapsed
                # deadlines are counted instead of executing catch-up work.
                self.assertGreater(runtime.ticks[2], completed[1])
                self.assertEqual(timing["skipped_deadlines"], 2 if work_ns else 0)

    def test_repeated_status_is_quiet_without_records(self):
        clock = [0]
        runtime, messages = Runtime(), []
        ui = runtime_ui.TeleopUI(runtime, PROFILE, emit=messages.append)

        def read(timeout):
            clock[0] += 1_000_000_000
            return "q" if clock[0] >= 4_000_000_000 else None

        with patch.object(runtime_ui.time, "monotonic_ns", side_effect=lambda: clock[0]):
            runtime_ui.run_loop(runtime, ui, Mock(read=read))
        self.assertEqual(len(messages), 1)
        self.assertIn("设备已就绪", messages[0])
        self.assertNotIn("{", "".join(messages))

    def test_gesture_status_poll_does_not_repeat_console_output(self):
        clock = [0]
        runtime, messages = Runtime(), []
        diagnostics = {"hands": {side: {"v": False, "rock": False, "detail": "等待 V 手势"}
                                 for side in ("left", "right")},
                       "start_armed": True, "hold_ms": 0.}
        gesture = Mock(poll=Mock(return_value=None), status=Mock(return_value=diagnostics))
        ui = runtime_ui.TeleopUI(runtime, PROFILE, gesture=gesture, emit=messages.append)

        def read(timeout):
            clock[0] += 1_000_000_000
            return "q" if clock[0] >= 4_000_000_000 else None

        with patch.object(runtime_ui.time, "monotonic_ns", side_effect=lambda: clock[0]):
            runtime_ui.run_loop(runtime, ui, Mock(read=read))
        gesture.status.assert_not_called()
        self.assertEqual(sum("手势：" in message for message in messages), 0)

    def test_terminal_restores_settings_after_failure_and_treats_eof_as_exit(self):
        stream = Mock()
        stream.fileno.return_value = 12
        stream.isatty.return_value = True
        with patch.object(terminal_module.termios, "tcgetattr", return_value=["original"]), \
             patch.object(terminal_module.termios, "tcsetattr") as restore, \
             patch.object(terminal_module.tty, "setcbreak"):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                with cli.NonblockingTerminal(stream):
                    raise RuntimeError("test failure")
            restore.assert_called_once_with(12, terminal_module.termios.TCSADRAIN, ["original"])
        runtime = Runtime()
        ui = runtime_ui.TeleopUI(runtime, PROFILE, emit=lambda _: None)
        runtime_ui.run_loop(runtime, ui, Mock(read=lambda timeout: ""))
        ui.close()
        self.assertEqual(runtime.calls, [("close",)])


class MainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / "config.yaml"
        self.settings = {"controller_ip": "192.0.2.7", "profile": {
            "profile_id": PROFILE.profile_id, "mode": PROFILE.mode, "parameters": PROFILE.parameters}}
        self.config.write_text(yaml.safe_dump(self.settings))
        self.args = ["--arms-only", "--config", str(self.config)]
        self.runtime = Runtime()
        self.stdout, self.stderr = io.StringIO(), io.StringIO()

    def invoke(self, *, start_error=None, prepare_error=None, keys="q"):
        if start_error:
            self.runtime.start = Mock(side_effect=start_error)
        terminal = Mock()
        terminal.__enter__ = Mock(return_value=Mock(read=lambda timeout: keys))
        terminal.__exit__ = Mock(return_value=False)
        with patch.object(cli, "NonblockingTerminal", return_value=terminal), \
             patch.object(cli, "confirm_motion", return_value=True), \
             patch.object(cli, "prepare_initial_pose", side_effect=prepare_error,
                          return_value={"side": "both"}) as prepare, \
             patch.object(cli, "create_runtime", return_value=self.runtime) as create, \
             patch.object(cli, "configure_runtime_logging"), \
             redirect_stdout(self.stdout), redirect_stderr(self.stderr):
            self.order = Mock()
            self.order.attach_mock(prepare, "prepare")
            self.order.attach_mock(create, "create")
            result = cli.main(self.args)
        self.prepare, self.create = prepare, create
        if prepare_error:
            self.assertEqual(self.runtime.calls, [])
            create.assert_not_called()
        else:
            self.assertEqual(self.runtime.calls[-1], ("close",))
        terminal.__exit__.assert_called_once()
        return result, create.call_args.args[0] if create.called else None

    def test_arms_only_prepares_then_waits_for_engagement_and_closes_cleanly(self):
        result, args = self.invoke()
        self.assertEqual(result, 0)
        self.assertTrue(args.arms_only)
        self.assertEqual(args.coordinate_frame, "headset")
        self.prepare.assert_called_once()
        self.assertFalse(any(call[0] in ("engage", "load", "jog") for call in self.runtime.calls))
        self.assertIn("实机遥操作", self.stderr.getvalue())
        self.assertIn("Enter 接合/脱离", self.stderr.getvalue())
        self.assertNotIn("手柄映射", self.stderr.getvalue())
        self.assertIn("已退出", self.stderr.getvalue())
        self.assertEqual(self.stdout.getvalue(), "")
        self.assertNotIn("\x1b[", self.stderr.getvalue())

    def test_shared_config_supplies_controller_ip_and_coordinate_frame(self):
        self.settings.update(controller_ip="192.0.2.8", quest={"coordinate_frame": "world"})
        self.config.write_text(yaml.safe_dump(self.settings))
        result, args = self.invoke()
        self.assertEqual(result, 0)
        self.assertEqual(args.robot_ip, "192.0.2.8")
        self.assertEqual(args.coordinate_frame, "world")

    def test_configured_controls_reach_ui_and_help(self):
        self.settings["controls"] = {"toggle_engagement_key": "t", "ready_pose_key": "r",
                                     "gesture_engagement_enabled": False}
        self.config.write_text(yaml.safe_dump(self.settings))
        with patch.object(cli, "run_loop", return_value={"elapsed_s": .1, "motion_pauses": 0}) as loop:
            result, _ = self.invoke()
        self.assertEqual(result, 0)
        ui = loop.call_args.args[1]
        self.assertEqual(ui.toggle_engagement_key, "t")
        self.assertEqual(ui.ready_pose_key, "r")
        self.assertFalse(ui.gesture_engagement_enabled)
        self.assertIn("T 接合/脱离", self.stderr.getvalue())
        self.assertIn("R 停止跟随并回位", self.stderr.getvalue())

    def test_tianji_config_option_reaches_profile_and_initial_pose_preparation(self):
        self.settings.update(controller_ip="192.0.2.8", quest={"coordinate_frame": "world"})
        self.settings["profile"]["profile_id"] = "custom-tianji"
        self.config.write_text(yaml.safe_dump(self.settings))
        self.args = ["--arms-only", "--tianji-config", str(self.config)]
        result, args = self.invoke()
        self.assertEqual(result, 0)
        self.assertEqual(args.config, self.config)
        self.assertEqual(args.robot_ip, "192.0.2.8")
        self.assertEqual(args.coordinate_frame, "world")
        self.assertEqual(self.create.call_args.args[1].profile_id, "custom-tianji")
        self.assertIs(self.prepare.call_args.args[0], args)

    def test_cli_ip_overrides_shared_configuration(self):
        self.args += ["--robot-ip", "192.0.2.9"]
        result, args = self.invoke()
        self.assertEqual(result, 0)
        self.assertEqual(args.robot_ip, "192.0.2.9")

    def test_invalid_coordinate_frame_fails_before_preparation_or_connection(self):
        self.settings["quest"] = {"coordinate_frame": "grip"}
        self.config.write_text(yaml.safe_dump(self.settings))
        with patch.object(cli, "prepare_initial_pose") as prepare, \
                patch.object(cli, "create_runtime") as create, \
                redirect_stderr(self.stderr):
            self.assertEqual(cli.main([
                *self.args, "--log-file", str(self.root / "invalid-config.jsonl")]), 1)
        prepare.assert_not_called()
        create.assert_not_called()
        self.assertIn("coordinate_frame", self.stderr.getvalue())

    def test_motion_prepares_before_creating_runtime_and_still_waits_for_enter(self):
        result, args = self.invoke()
        self.assertEqual(result, 0)
        self.assertEqual([call[0] for call in self.order.mock_calls], ["prepare", "create"])
        self.assertEqual(args.config, self.config)
        self.assertEqual(self.runtime.calls, [("start",), ("close",)])
        self.assertEqual(self.create.call_args.args[2], None)
        self.assertIn("已退出", self.stderr.getvalue())

    def test_shutdown_failure_is_visible_and_returns_nonzero_for_q_and_interrupt(self):
        for interrupt in (False, True):
            with self.subTest(interrupt=interrupt):
                self.stderr.seek(0)
                self.stderr.truncate()
                def close():
                    self.runtime.calls.append(("close",))
                    raise RuntimeError("天机退出停机未确认，请按实体急停")
                self.runtime.close = close
                if interrupt:
                    with patch.object(cli, "run_loop", side_effect=KeyboardInterrupt):
                        result, _ = self.invoke()
                else:
                    result, _ = self.invoke()
                self.assertEqual(result, 1)
                self.assertIn("天机退出停机未确认", self.stderr.getvalue())
                self.assertNotIn("[完成]", self.stderr.getvalue())

    def test_single_side_cli_passes_selected_profile(self):
        self.args += ["--side", "left"]
        result, args = self.invoke()
        self.assertEqual(result, 0)
        self.assertEqual(args.side, "left")
        selected = self.create.call_args.args[1]
        self.assertEqual(selected.parameters["active_arms"], ["left"])
        self.assertEqual(selected.profile_id, "test-profile-left")

    def test_preparation_failure_prevents_all_teleop_device_creation(self):
        result, _ = self.invoke(prepare_error=RuntimeError("initial pose rejected"))
        self.assertEqual(result, 1)
        self.assertIn("initial pose rejected", self.stderr.getvalue())

    def test_start_failure_still_closes_both_components(self):
        result, _ = self.invoke(start_error=RuntimeError("USB unavailable"))
        self.assertEqual(result, 1)
        self.assertIn("USB unavailable", self.stderr.getvalue())
        self.assertNotIn("Traceback", self.stderr.getvalue())

    def test_noninteractive_stdin_fails_before_creating_devices(self):
        with patch.object(sys, "stdin", io.StringIO()), \
             patch.object(cli, "create_runtime") as runtime, \
             patch.object(cli, "configure_runtime_logging"), \
             redirect_stdout(self.stdout), redirect_stderr(self.stderr):
            self.assertEqual(cli.main(self.args), 1)
        runtime.assert_not_called()
        self.assertIn("交互终端", self.stderr.getvalue())

    def test_cli_has_no_output_option_or_runtime_records(self):
        with patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                cli.main(self.args + ["--output", str(self.root / "logs")])
        self.assertFalse((self.root / "logs").exists())


class PreparationStartupTests(unittest.TestCase):
    """The position subprocess must finish before teleop opens a device."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.args = SimpleNamespace(robot_ip="192.0.2.7", config=self.root / "custom-config.yaml",
                                    sdk_root=self.root / "custom.so")
        self.process = Mock()
        self.process.poll.return_value = 0
        self.process.wait.return_value = 0

    def spawn(self, command, **kwargs):
        self.command, self.options = command, kwargs
        return self.process

    def run_preparation(self, keys=(None, "\n", None)):
        terminal = Mock(read=Mock(side_effect=keys))
        with patch.object(preparation, "ROOT", self.root), \
                patch.object(preparation.subprocess, "Popen", side_effect=self.spawn) as spawn, \
                redirect_stderr(io.StringIO()):
            result = cli.prepare_initial_pose(self.args, terminal)
        return result, spawn

    def test_complete_child_releases_connection_before_return(self):
        result, _ = self.run_preparation()
        self.process.wait.assert_called_once_with()
        self.process.send_signal.assert_not_called()
        self.assertEqual(self.command[0], sys.executable)
        for flag, expected in (("--ip", self.args.robot_ip), ("--tianji-config", self.args.config),
                               ("--sdk-root", self.args.sdk_root)):
            self.assertEqual(self.command[self.command.index(flag)+1], str(expected))
        self.assertEqual(self.command[1], "-c")
        self.assertIn("confirmed=True", self.command[2])
        self.assertNotIn("--output", self.command)
        self.assertTrue(self.options["start_new_session"])
        self.assertEqual(self.options["stdin"], preparation.subprocess.DEVNULL)
        self.assertNotIn("env", self.options)
        self.assertEqual(result["side"], "both")
        self.assertFalse((self.root / "data").exists())

    def test_single_side_preparation_forwards_selection(self):
        self.args.side = "right"
        result, _ = self.run_preparation()
        self.assertEqual(self.command[self.command.index("--side")+1], "right")
        self.assertEqual(result["side"], "right")

    def test_verbose_preparation_forwards_output_flag(self):
        self.args.verbose = True
        self.run_preparation()
        self.assertIn("--verbose", self.command)

    def test_failed_exit_blocks_teleop_without_summary_file(self):
        self.process.poll.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "准备失败.*退出码 1"):
            self.run_preparation()
        self.assertFalse((self.root / "data").exists())

    def test_space_quit_eof_and_ctrl_c_stop_child_and_wait_for_cleanup(self):
        for key in (" ", "q", "Q", "", "\x03", KeyboardInterrupt()):
            with self.subTest(key=key):
                self.process.reset_mock()
                self.process.poll.return_value = None
                self.process.wait.side_effect = [preparation.subprocess.TimeoutExpired("prepare", .1),
                                                 KeyboardInterrupt(), 1]
                exception = KeyboardInterrupt if isinstance(key, BaseException) else RuntimeError
                with self.assertRaises(exception):
                    self.run_preparation((None, key))
                self.process.send_signal.assert_called_once_with(preparation.signal.SIGINT)
                self.assertEqual(self.process.wait.call_count, 3)

    def test_cancel_before_launch_never_starts_preparation(self):
        with patch.object(preparation.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(RuntimeError, "已取消"):
                cli.prepare_initial_pose(self.args, Mock(read=lambda _: "q"))
        spawn.assert_not_called()

    def test_real_subprocess_cancellation_waits_for_child_finally(self):
        package = self.root / "bimanual_teleop"
        scripts = package / "cli"
        scripts.mkdir(parents=True)
        (package / "__init__.py").touch()
        (scripts / "__init__.py").touch()
        (scripts / "home_tianji.py").write_text(
            "import argparse\n"
            "from pathlib import Path\n"
            "import time\n"
            "def parser():\n"
            "    result = argparse.ArgumentParser()\n"
            "    for name in ('--ip', '--tianji-config', '--side', '--sdk-root'):\n"
            "        result.add_argument(name)\n"
            "    return result\n"
            "def run(args, *, confirmed=False):\n"
            "    assert confirmed\n"
            "    try:\n"
            "        Path('started').touch()\n"
            "        time.sleep(10)\n"
            "    except KeyboardInterrupt:\n"
            "        pass\n"
            "    finally:\n"
            "        time.sleep(.05)\n"
            "        Path('held-and-closed').touch()\n"
            "    return 0\n")
        started = self.root / "started"
        deadline = time.monotonic()+3

        def read(timeout):
            if started.exists():
                return " "
            if time.monotonic() > deadline:
                raise RuntimeError("test child did not start")
            time.sleep(timeout)
            return None

        with patch.object(preparation, "ROOT", self.root), redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "已取消"):
                cli.prepare_initial_pose(self.args, Mock(read=read))
        self.assertTrue((self.root / "held-and-closed").exists())


class RuntimeIntegrationTests(unittest.TestCase):
    def test_enter_before_tracking_is_ready_engages_once_after_controller_wakes(self):
        from tests.support.quest import RuntimeFixture

        fx = RuntimeFixture()
        self.addCleanup(fx.runtime.close)
        fx.quest.emit(invalid="left")
        ui = runtime_ui.TeleopUI(fx.runtime, fx.profile, toggle_engagement_key="enter", emit=lambda _: None)
        ui.handle("\n")
        self.assertTrue(ui.engage_pending)
        self.assertFalse(fx.driver.engaged)
        reads = 0

        def read(timeout):
            nonlocal reads
            reads += 1
            fx.advance()
            if reads == 4:
                self.assertTrue(fx.driver.engaged)
                self.assertFalse(ui.engage_pending)
                self.assertGreater(fx.runtime.cycles, 0)
                return "q"

        with patch.object(runtime_ui.time, "monotonic_ns", side_effect=fx.clock):
            runtime_ui.run_loop(fx.runtime, ui, Mock(read=read))
        self.assertEqual(fx.driver.calls.count("engage"), 1)

    def test_space_and_quit_cancel_pending_engagement(self):
        for key in (" ", "q"):
            runtime = Runtime()
            runtime.health = lambda: Health(False, 0, "left tracking unavailable")
            ui = runtime_ui.TeleopUI(runtime, PROFILE, emit=lambda _: None)
            ui.handle("\n")
            self.assertTrue(ui.engage_pending)
            ui.handle(key)
            self.assertFalse(ui.engage_pending)
            self.assertFalse(any(call[0] == "engage" for call in runtime.calls))

    def test_real_runtime_enter_engages_without_calibration_files_or_gestures(self):
        from tests.support.quest import RuntimeFixture

        fx = RuntimeFixture()
        self.addCleanup(fx.runtime.close)
        with patch("bimanual_teleop.control.arm.cartesian.time.monotonic_ns", side_effect=fx.clock):
            ui = runtime_ui.TeleopUI(fx.runtime, fx.profile, toggle_engagement_key="enter", emit=lambda _: None)
            self.assertEqual(fx.driver.calls, ["start"])
            ui.handle("\n")
            self.assertEqual(fx.runtime.status()["mode"], "follow")
            self.assertIsNotNone(fx.runtime.tick())
            ui.handle("\r")
            self.assertFalse(fx.driver.engaged)
            fx.advance()
            ui.handle("\n")
            self.assertEqual(fx.runtime.status()["mode"], "follow")


if __name__ == "__main__":
    unittest.main()
