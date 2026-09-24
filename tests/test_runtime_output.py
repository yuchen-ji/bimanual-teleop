"""Terminal and no-recording checks; all device interaction is replaced."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import json
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml

from bimanual_teleop.common.console import (
    LiveProgress, StatusConsole, configure_runtime_logging, format_message, runtime_message,
)
from bimanual_teleop.common.runlog import RuntimeLog


class TtyBuffer(io.StringIO):
    def isatty(self):
        return True


@contextmanager
def temporary_working_directory():
    original = Path.cwd()
    with tempfile.TemporaryDirectory() as temporary:
        os.chdir(temporary)
        try:
            yield Path(temporary)
        finally:
            os.chdir(original)


class ConsoleBehaviorTests(unittest.TestCase):
    def test_compact_diagnostics_preserve_other_errors_and_verbose_keeps_original(self):
        for label in ("IK诊断", "控制诊断", "停机诊断"):
            message = f'目标失败\n[{label}] {{"reason":"不可达"}}; 关闭失败\n连接中断'
            self.assertEqual(runtime_message(message), "目标失败; 关闭失败\n连接中断")
            self.assertEqual(runtime_message(message, verbose=True), message)
        unknown = "目标失败\n[控制诊断] 无法编码诊断"
        self.assertEqual(runtime_message(unknown), unknown)

    def test_verbose_logging_can_be_enabled_and_reset_without_duplicate_handlers(self):
        sdk = SimpleNamespace(set_log_level=Mock())
        logger = logging.getLogger("bimanual_teleop.test_output")
        for verbose in (False, True, False):
            with self.subTest(verbose=verbose), redirect_stderr(io.StringIO()) as output, \
                    patch.dict("sys.modules", {"wuji_sdk": sdk}):
                configure_runtime_logging(wuji=True, verbose=verbose)
                logger.debug("调试细节")
                logger.warning('故障原因\n[控制诊断] {"target":[1,2,3]}')
                logger.error("连接失败")
            self.assertEqual("调试细节" in output.getvalue(), verbose)
            self.assertEqual("[控制诊断]" in output.getvalue(), verbose)
            self.assertEqual(output.getvalue().count("故障原因"), 1)
            self.assertIn("连接失败", output.getvalue())
            sdk.set_log_level.assert_called_with("debug" if verbose else "error")

    def test_structured_runtime_log_captures_debug_records_and_closes_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.jsonl"
            run_log = configure_runtime_logging(log_file=path)
            run_log.start(command="test", arguments={"path": Path("config.yaml")})
            logging.getLogger("bimanual_teleop.test_output").debug("周期细节")
            logging.getLogger("bimanual_teleop.test_output").warning("故障细节")
            run_log.event("runtime_status", value=float("nan"))
            run_log.close(result="ok")
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(rows[0]["event"], "session_start")
        self.assertIn("thread_schedstat", rows[0]["process"])
        self.assertIn("cpu_pressure", rows[0]["process"])
        self.assertIn("io_pressure", rows[0]["process"])
        self.assertIn("process_io", rows[0]["process"])
        self.assertIn("cgroup_cpu", rows[0]["process"])
        self.assertFalse(any(row["event"] == "python_log" and
                             row["message"] == "周期细节" for row in rows))
        self.assertTrue(any(row["event"] == "python_log" and
                            row["message"] == "故障细节" for row in rows))
        self.assertTrue(any(row["event"] == "runtime_status" and
                            row["value"] == "nan" for row in rows))
        self.assertEqual(rows[-1]["event"], "session_end")

    def test_status_changes_are_deduplicated_and_warnings_are_throttled(self):
        stream = io.StringIO()
        console = StatusConsole(stream=stream, warning_interval_s=5)
        with patch("bimanual_teleop.common.console.time.monotonic",
                   side_effect=(0, 1, 6, 7, 8, 9, 13)):
            console.state("连接中")
            console.state("连接中")
            console.state("设备就绪", "ready")
            console.warning("队列溢出", key="overflow")
            console.warning("队列溢出", key="overflow")
            console.warning("队列溢出", key="overflow")
            console.error("故障 A")
            console.error("故障 A")
            console.error("故障 B")
            console.error("故障 A")
        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 7)
        self.assertEqual(sum("连接中" in line for line in lines), 1)
        self.assertEqual(sum("队列溢出" in line for line in lines), 2)
        self.assertEqual(sum("故障 A" in line for line in lines), 2)
        self.assertEqual(sum("故障 B" in line for line in lines), 1)

    def test_color_only_on_interactive_terminal_without_no_color(self):
        with patch.dict(os.environ, {}, clear=True):
            colored = format_message("暂停", "warning", stream=TtyBuffer())
            self.assertIn("\x1b[33m", colored)
            self.assertEqual(format_message("暂停", "warning", stream=io.StringIO()),
                             "[警告] 暂停")
        with patch.dict(os.environ, {"NO_COLOR": ""}):
            self.assertEqual(format_message("暂停", "warning", stream=TtyBuffer()),
                             "[警告] 暂停")

    def test_live_progress_is_interactive_and_limited_to_one_line(self):
        terminal = TtyBuffer()
        progress = LiveProgress(stream=terminal, interval_s=.2)
        with patch("bimanual_teleop.common.console.time.monotonic", side_effect=(0, .1, .3)):
            progress.update("目标 1")
            progress.update("目标 2")
            progress.update("目标 3")
        self.assertNotIn("目标 2", terminal.getvalue())
        self.assertIn("\r", terminal.getvalue())
        progress.clear()
        self.assertEqual(terminal.getvalue().count("\n"), 1)
        redirected = io.StringIO()
        LiveProgress(stream=redirected).update("不应输出")
        self.assertEqual(redirected.getvalue(), "")

    def test_wuji_sdk_defaults_to_errors_without_duplicate_handlers(self):
        sdk = SimpleNamespace(set_log_level=Mock())
        with patch.dict("sys.modules", {"wuji_sdk": sdk}):
            configure_runtime_logging(wuji=True)
            configure_runtime_logging(wuji=True)
        self.assertEqual([call.args[0] for call in sdk.set_log_level.call_args_list],
                         ["error", "error"])
        logger = logging.getLogger("bimanual_teleop")
        self.assertEqual(len(logger.handlers), 1)

    def test_pause_log_contains_retained_control_cycles_and_device_snapshots(self):
        from bimanual_teleop.cli.runtime import TeleopUI

        events = []
        run_log = SimpleNamespace(event=lambda event, **details: events.append((event, details)))
        executor = SimpleNamespace(timing_status={"elapsed_ns": 7})
        arms = SimpleNamespace(
            state="paused", cycle_timing={"elapsed_ns": 9}, executor=executor,
            last_pause_diagnostic={"schema": "teleop_pause_v1",
                                   "driver_stop": {"target_age_ms": 51.}})
        hands = SimpleNamespace(status=lambda **_: {"worker_age_ns": 3,
                                                     "snapshot_sequence": 4})
        runtime = SimpleNamespace(
            state="paused", last_error="watchdog expired", arms=arms, hands=hands,
            status=lambda **_: {"state": "paused",
                                "arms": {"last_error": "watchdog expired"}})
        ui = TeleopUI(runtime, None, emit=lambda _: None, runtime_log=run_log)
        ui.loop_timing = {"wake_lateness_ns": 5}
        ui.record_cycle(runtime, 12)
        ui.log_runtime_status(runtime.status())
        ui.report_runtime_pause()
        pauses = [details for event, details in events if event == "motion_pause"]
        self.assertEqual(len(pauses), 1)
        self.assertEqual(pauses[0]["preceding_cycles"][0]["cycle_index"], 12)
        self.assertEqual(pauses[0]["diagnostic"]["driver_stop"]["target_age_ms"], 51.)
        self.assertEqual(pauses[0]["diagnostic"]["hands"]["snapshot_sequence"], 4)
        self.assertEqual([event for event, _ in events], ["motion_pause"])


class EntryBehaviorTests(unittest.TestCase):
    def setUp(self):
        from bimanual_teleop.cli import home_tianji, teleop_quest_tianji, teleop_wuji_hand2
        from bimanual_teleop.control.arm import jog
        from bimanual_teleop.control.hand import home
        for module in (home_tianji, teleop_quest_tianji, teleop_wuji_hand2, jog, home):
            for name in ("NonblockingTerminal", "confirm_motion"):
                options = {"return_value": True} if name == "confirm_motion" else {}
                patcher = patch.object(module, name, **options)
                patcher.start()
                self.addCleanup(patcher.stop)

    def test_teleop_output_flags_reach_logging_and_ui(self):
        from bimanual_teleop.cli import teleop_quest_tianji, teleop_wuji_hand2

        for entry, required, factory in (
            (teleop_quest_tianji, ["--arms-only"],
             "bimanual_teleop.cli.teleop_quest_tianji.create_runtime"),
            (teleop_wuji_hand2, [],
             "bimanual_teleop.control.hand.follow.create_wuji_teleop"),
        ):
            for flags in ([], ["-v"], ["--verbose"]):
                with self.subTest(entry=entry.__name__, flags=flags), \
                        patch.object(entry, "configure_runtime_logging") as logging_setup, \
                        patch(factory, return_value=SimpleNamespace(start=Mock(), close=Mock())), \
                        patch.object(teleop_quest_tianji, "prepare_initial_pose"), \
                        patch("bimanual_teleop.control.hand.follow.preflight"), \
                        patch.object(entry, "run_loop", return_value={"elapsed_s": .1, "motion_pauses": 0}) as loop, \
                        redirect_stderr(io.StringIO()):
                    self.assertEqual(entry.main([*required, *flags]), 0)
                    kwargs = logging_setup.call_args.kwargs
                    self.assertEqual(kwargs["wuji"], entry is teleop_wuji_hand2)
                    self.assertEqual(kwargs["verbose"], bool(flags))
                    if entry is teleop_quest_tianji:
                        self.assertEqual(Path(kwargs["log_file"]).parent, Path("logs"))
                    else:
                        self.assertNotIn("log_file", kwargs)
                    self.assertEqual(loop.call_args.args[1].verbose, bool(flags))

    def test_retained_entry_help_has_no_runtime_output_option(self):
        from bimanual_teleop.cli import (
            calibrate_wuji_glove, teleop_wuji_hand2, teleop_quest_tianji,
            view_quest, view_wuji_glove,
        )
        from bimanual_teleop.cli.home_tianji import parser as ready_parser
        from bimanual_teleop.control.arm import jog
        from bimanual_teleop.control.hand import home

        for name, entry in (
            ("Quest 可视化", view_quest.main),
            ("手套可视化", view_wuji_glove.main),
            ("手套标定", calibrate_wuji_glove.main),
            ("单手控制", teleop_wuji_hand2.main),
            ("机械臂遥操作", teleop_quest_tianji.main),
            ("机械臂点动", jog.main),
            ("Hand2 回零", home.main),
        ):
            with self.subTest(entry=name), redirect_stdout(io.StringIO()) as output:
                with self.assertRaises(SystemExit) as result:
                    entry(["--help"])
                self.assertEqual(result.exception.code, 0)
                self.assertNotIn("--output", output.getvalue())
        with redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as result:
                ready_parser().parse_args(["--help"])
        self.assertEqual(result.exception.code, 0)
        self.assertNotIn("--output", output.getvalue())

        for name, entry, required in (
            ("Quest 可视化", view_quest.main, []),
            ("手套可视化", view_wuji_glove.main, ["--side", "left"]),
            ("手套标定", calibrate_wuji_glove.main,
             ["--side", "left", "--kind", "joints", "--user-name", "named"]),
            ("单手控制", teleop_wuji_hand2.main, []),
            ("机械臂遥操作", teleop_quest_tianji.main, []),
            ("机械臂点动", jog.main, ["--side", "left", "--ip", "192.0.2.1"]),
            ("Hand2 回零", home.main, ["--side", "left"]),
        ):
            with self.subTest(entry=name), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    entry([*required, "--output", "run.jsonl"])
                self.assertEqual(result.exception.code, 2)
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                ready_parser().parse_args(["--output", "run.jsonl"])
        self.assertEqual(result.exception.code, 2)

    def test_ready_pose_cancellation_and_failure_do_not_create_runtime_records(self):
        from bimanual_teleop.cli import home_tianji as ready

        with temporary_working_directory() as directory, redirect_stderr(io.StringIO()):
            with patch.object(ready, "confirm_motion", return_value=False):
                self.assertEqual(ready.run(ready.parser().parse_args([])), 0)
            bad = ready.parser().parse_args(["--config", str(directory / "absent.yaml")])
            self.assertEqual(ready.run(bad), 1)
            self.assertEqual(list(directory.rglob("*")), [])

    def test_quest_teleop_and_start_failure_create_only_diagnostic_logs(self):
        from bimanual_teleop.cli import teleop_quest_tianji as teleop

        with temporary_working_directory() as directory, redirect_stderr(io.StringIO()):
            runtime = SimpleNamespace(start=Mock(), close=Mock())
            timing = {"elapsed_s": .1, "motion_pauses": 0}
            with patch.object(teleop, "NonblockingTerminal") as terminal, \
                    patch.object(teleop, "create_runtime", return_value=runtime), \
                    patch.object(teleop, "run_loop", return_value=timing), \
                    patch.object(teleop, "prepare_initial_pose") as prepare:
                self.assertEqual(teleop.main(["--arms-only"]), 0)
                self.assertTrue(terminal.called)
                self.assertTrue(runtime.close.called)
                prepare.assert_called_once()
                runtime.start.side_effect = RuntimeError("模拟断流")
                self.assertEqual(teleop.main(["--arms-only"]), 1)
            files = sorted(path.relative_to(directory) for path in directory.rglob("*") if path.is_file())
            self.assertEqual(len(files), 2)
            self.assertTrue(all(path.parent == Path("logs") and path.suffix == ".jsonl"
                                for path in files))

    def test_jog_and_home_motion_stub_and_failure_leave_no_records(self):
        from bimanual_teleop.control.arm import jog
        from bimanual_teleop.control.hand import home

        with temporary_working_directory() as directory, redirect_stderr(io.StringIO()):
            arm = SimpleNamespace(start=Mock(), close=Mock())
            with patch.object(jog, "TianjiDriver", return_value=arm), \
                    patch.object(jog, "TianjiKinematics"), \
                    patch.object(jog, "NonblockingTerminal"), \
                    patch.object(jog, "prepare_initial_pose") as prepare, \
                    patch.object(jog, "run_jog") as run_jog:
                args = ["--side", "left", "--ip", "192.0.2.1"]
                self.assertEqual(jog.main(args), 0)
                prepare.assert_called_once()
                run_jog.side_effect = RuntimeError("模拟越限")
                self.assertEqual(jog.main(args), 1)
            self.assertEqual(list(directory.rglob("*")), [])

            hand = SimpleNamespace(
                start=Mock(), close=Mock(), metadata={"hardware_version": "test"},
                health=Mock(return_value=SimpleNamespace(ready=True, detail="ready")),
                get_latest=Mock(return_value=SimpleNamespace(
                    payload=SimpleNamespace(position_rad=(0.,) * 20))),
            )
            with patch.object(home, "configure_runtime_logging"), \
                    patch.object(home, "WujiHandDriver", return_value=hand), \
                    patch.object(home, "run_home") as run_home:
                args = ["--side", "left"]
                self.assertEqual(home.main(args), 0)
                run_home.assert_called_once()
                run_home.side_effect = RuntimeError("模拟反馈错误")
                self.assertEqual(home.main(args), 1)
            self.assertEqual(list(directory.rglob("*")), [])

    def test_single_hand_motion_and_failure_leave_no_records(self):
        from bimanual_teleop.cli import teleop_wuji_hand2
        from bimanual_teleop.control.hand import follow

        with temporary_working_directory() as directory, redirect_stderr(io.StringIO()):
            runtime = SimpleNamespace(start=Mock(), close=Mock())
            with patch.object(teleop_wuji_hand2, "configure_runtime_logging"), \
                    patch.object(teleop_wuji_hand2, "NonblockingTerminal"), \
                    patch.object(teleop_wuji_hand2, "run_loop", return_value={"elapsed_s": .1}), \
                    patch.object(follow, "preflight"), \
                    patch.object(follow, "create_wuji_teleop", return_value=runtime) as create:
                self.assertEqual(teleop_wuji_hand2.main(["--side", "left"]), 0)
                with patch("bimanual_teleop.devices.wuji.config.load_config", return_value={"sdk_user_name": "old"}):
                    self.assertEqual(teleop_wuji_hand2.main(
                        ["--side", "left", "--user-name", "yuchen"]), 0)
                    self.assertEqual(create.call_args.args[0], {"sdk_user_name": "yuchen"})
                runtime.start.side_effect = RuntimeError("模拟手套断流")
                self.assertEqual(teleop_wuji_hand2.main(["--side", "left"]), 1)
            self.assertEqual(list(directory.rglob("*")), [])

    def test_visualization_and_calibration_stubs_leave_no_runtime_records(self):
        from bimanual_teleop.cli import calibrate_wuji_glove, view_quest, view_wuji_glove
        from bimanual_teleop.visualization import quest as quest_view
        from bimanual_teleop.visualization import wuji as wuji_view
        import matplotlib.pyplot as plt

        timer = SimpleNamespace(add_callback=Mock(), start=Mock(), stop=Mock())
        view = SimpleNamespace(figure=SimpleNamespace(
            canvas=SimpleNamespace(new_timer=Mock(return_value=timer))))
        with temporary_working_directory() as directory, redirect_stderr(io.StringIO()), \
                patch.object(plt, "show"), patch.object(plt, "close"), \
                patch.object(quest_view, "QuestPoseView", return_value=view), \
                patch.object(wuji_view, "WujiGloveView", return_value=view), \
                patch.object(view_quest, "configure_runtime_logging"), \
                patch.object(view_wuji_glove, "configure_runtime_logging"), \
                patch.object(calibrate_wuji_glove, "configure_runtime_logging"):
            source = SimpleNamespace(start=Mock(), close=Mock(),
                                     get_latest=Mock(), health=Mock(), metadata={})
            session = SimpleNamespace(open=Mock(), close=Mock(), manager=Mock(), sdk=Mock(),
                                      metadata={"sdk_user_name": "named", "sdk_user_id": "id"})
            session.open.return_value = session
            with patch.object(view_quest, "QuestSource", return_value=source), \
                    patch.object(view_wuji_glove, "WujiSdkSession", return_value=session) as view_session, \
                    patch.object(view_wuji_glove, "WujiGloveSource", return_value=source) as glove_source:
                self.assertEqual(view_quest.main([]), 0)
                self.assertEqual(view_wuji_glove.main(["--side", "left"]), 0)
                self.assertEqual(view_wuji_glove.main(
                    ["--side", "left", "--user-name", "named"]), 0)
                self.assertEqual(view_session.call_args.kwargs,
                                 {"user_name": "named"})
                self.assertEqual(glove_source.call_args.kwargs["streams"],
                                 ("skeleton", "tactile", "contact"))
                source.start.side_effect = RuntimeError("模拟设备异常")
                self.assertEqual(view_quest.main([]), 1)
                self.assertEqual(view_wuji_glove.main(["--side", "left"]), 1)
            result = {"sdk_user": {"user_id": "id", "display_name": "named"},
                      "model": "WujiGlove"}
            with patch.object(calibrate_wuji_glove, "calibrate_glove", return_value=result) as guided:
                args = ["--side", "left", "--kind", "joints", "--user-name", "named"]
                self.assertEqual(calibrate_wuji_glove.main(args), 0)
                guided.side_effect = RuntimeError("模拟标定失败")
                self.assertEqual(calibrate_wuji_glove.main(args), 1)
            self.assertEqual(list(directory.rglob("*")), [])

    def test_wuji_user_name_must_match_one_existing_user(self):
        from bimanual_teleop.devices.wuji.adapter import resolve_user_id

        manager = Mock()
        manager.list_users.return_value = []
        with self.assertRaisesRegex(ValueError, "未找到 SDK 用户名"):
            resolve_user_id(manager, user_name="missing")
        manager.list_users.return_value = [
            {"user_id": "u_a", "display_name": "same"},
            {"user_id": "u_b", "display_name": "same"},
        ]
        with self.assertRaisesRegex(ValueError, "对应多个用户"):
            resolve_user_id(manager, user_name="same")
        manager.create_user.assert_not_called()

    def test_wuji_viewer_configuration_and_explicit_user_override(self):
        from bimanual_teleop.devices.wuji.config import glove_settings

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wuji.yaml"
            for selection in ({"sdk_user_name": "yuchen"}, {}):
                path.write_text(yaml.safe_dump({"devices": {"left": {"glove": "test"}}, **selection}))
                self.assertEqual(glove_settings(path, "left"), ("test", {
                    "user_name": selection.get("sdk_user_name", "")}))
                self.assertEqual(glove_settings(path, "left", user_name="Alice")[1],
                                 {"user_name": "Alice"})
            for selection in ({"sdk_user_name": "yuchen", "sdk_user_id": "legacy"},
                              {"sdk_user_id": "legacy"},
                              {"sdk_user_name": 123}, {"sdk_user_name": " "}):
                path.write_text(yaml.safe_dump({"devices": {"left": {"glove": "test"}}, **selection}))
                with self.assertRaises(ValueError):
                    glove_settings(path, "left")


if __name__ == "__main__":
    unittest.main()
