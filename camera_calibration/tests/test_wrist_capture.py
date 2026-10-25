"""Keyboard wrist capture preserves manual sampling and raw sensor orientation."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from bimanual_teleop.devices.tianji.driver import decode_feedback
from bimanual_teleop.devices.tianji.sdk import FeedbackSnapshot, load_sdk
from camera_calibration import cli, collect as wrist_capture, external_collect, motion
from camera_calibration.robot import base_flange, load_model
from camera_calibration.tests.test_external import lens


class WristCaptureTests(unittest.TestCase):
    def test_sdk_actual_joints_drive_snapshot_and_fk_instead_of_command_targets(self):
        raw = load_sdk("robot").DCSS()
        for index, side in enumerate(("left", "right")):
            out = raw.m_Out[index]
            out.m_OutFrameSerial = 11 + index
            out.m_FB_Joint_Pos[:] = [1.2345678 + joint + index for joint in range(7)]
            out.m_FB_Joint_Cmd[:] = [30. + joint + index for joint in range(7)]
            out.m_FB_Joint_PosE[:] = [60. + joint + index for joint in range(7)]
        packet = FeedbackSnapshot.from_dcss(raw, 123456789, 1)
        sample = decode_feedback(packet, "test-feedback")
        model = load_model()
        for index, side in enumerate(("left", "right")):
            with self.subTest(side=side):
                args = cli.parser().parse_args(["wrist", "collect", "--side", side])
                robot = motion.KeyboardRobot(args, model)
                robot.driver = Mock()
                robot.driver.get_latest.return_value = sample
                robot.driver.health.return_value = SimpleNamespace(ready=True)
                actual = np.deg2rad(list(raw.m_Out[index].m_FB_Joint_Pos))
                target = np.deg2rad(list(raw.m_Out[index].m_FB_Joint_Cmd))
                snapshot = robot.snapshot()
                np.testing.assert_allclose(snapshot["joints_rad"], actual, rtol=0, atol=1e-15)
                np.testing.assert_allclose(sample.payload.arms[side].controller_target_rad, target,
                                           rtol=0, atol=1e-15)
                self.assertEqual(snapshot["joint_time_ns"], 123456789)
                self.assertEqual(snapshot["joint_sequence"], 11 + index)
                measured_pose = base_flange(model, side, snapshot["joints_rad"])
                np.testing.assert_allclose(measured_pose, base_flange(model, side, actual))
                self.assertFalse(np.allclose(measured_pose, base_flange(model, side, target)))

    def test_main_and_wrist_share_keyboard_position_control(self):
        self.assertIs(wrist_capture.KeyboardRobot, external_collect.KeyboardRobot)
        self.assertIs(wrist_capture.KeyboardInput, external_collect.KeyboardInput)
        args = cli.parser().parse_args(["wrist", "collect", "--side", "right",
                                        "--step-mm", "3", "--step-deg", "1", "--transition-s", ".4"])
        driver = Mock()

        def jog(*positional, **options):
            self.assertTrue(options["configure_on_engage"])
            self.assertEqual(options["control_mode"], "position")
            self.assertEqual(options["side"], "right")
            self.assertAlmostEqual(options["translation_m"], .003)
            self.assertAlmostEqual(options["rotation_rad"], np.deg2rad(1))
            self.assertAlmostEqual(options["transition_s"], .4)
            with options["terminal"] as terminal:
                terminal.stopped.wait(3)

        with patch.object(motion, "TianjiDriver", return_value=driver), \
                patch.object(motion, "run_jog", side_effect=jog):
            with wrist_capture.KeyboardRobot(args, Mock()) as robot:
                driver.configure.assert_not_called()
                driver.move_joints.assert_not_called()
            driver.close.assert_called_once()
            self.assertFalse(robot.thread.is_alive())

    def test_terminal_and_preview_keys_move_capture_pause_and_quit(self):
        terminal = Mock(read=Mock(side_effect=["\rW", "c", "i ", "C", "q"]))
        robot, capture = Mock(), Mock()
        with wrist_capture.KeyboardInput(terminal, robot, capture) as keys:
            self.assertTrue(keys.stopped.wait(1))
            keys.check()
        self.assertEqual(capture.call_count, 2)
        self.assertEqual([call.args[0] for call in robot.send_key.call_args_list[:4]], ["\r", "w", "i", " "])
        self.assertFalse(keys.thread.is_alive())
        robot.reset_mock()
        capture.reset_mock()
        keys = wrist_capture.KeyboardInput(Mock(), robot, capture)
        for key in ("enter", "W", "I", "space", "C", "q", "w", "c"):
            keys.key(key)
        self.assertEqual([call.args[0] for call in robot.send_key.call_args_list], ["\r", "w", "i", " ", " "])
        capture.assert_called_once()
        self.assertTrue(keys.stopped.is_set())

    def test_selected_camera_saves_only_c_requests_repeated_pose_and_raw_pixels(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                self._capture(side, preview=False)

    def test_left_preview_rotates_before_detection_but_saved_pixels_remain_raw(self):
        self._capture("left", preview=True)

    def _capture(self, side, preview):
        args = cli.parser().parse_args(["wrist", "collect", "--side", side])
        args.no_preview = not preview
        args.left_serial, args.right_serial = "left-camera", "right-camera"
        clock = SimpleNamespace(now=1_000_000_000, cycle=0)

        class Keys:
            def __init__(self, terminal, robot, on_capture):
                self.robot, self.on_capture, self.stopped = robot, on_capture, self

            def __enter__(self):
                return self

            def check(self):
                pass

            def is_set(self):
                clock.now += 50_000_000
                clock.cycle += 1
                if clock.cycle in (1, 3):
                    self.on_capture(clock.now)
                return clock.cycle >= 5

            def key(self, key):
                pass

            def __exit__(self, *exc):
                self.robot.send_key(" ")

        pixels = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)
        camera = Mock(metadata={side: lens(side)})
        camera.poll.side_effect = lambda: {side: (clock.now - 20_000_000, clock.cycle, pixels)}
        robot = Mock(ip="192.168.1.190")
        robot.snapshot.side_effect = lambda: {"joints_rad": [.1] * 7, "joint_sequence": 1,
                                               "joint_time_ns": clock.now}
        model = SimpleNamespace(model=SimpleNamespace(digest="test-model", path=Path("test.cfg")))
        figure, axis = Mock(), Mock()
        with tempfile.TemporaryDirectory() as directory:
            args.output = Path(directory) / "capture"
            with patch.object(wrist_capture, "require_local_usb"), \
                    patch.object(wrist_capture, "NonblockingTerminal"), \
                    patch.object(wrist_capture, "confirm_motion", return_value=True), \
                    patch.object(wrist_capture, "load_model", return_value=model), \
                    patch.object(wrist_capture, "CameraStreams") as cameras, \
                    patch.object(wrist_capture, "KeyboardRobot") as robots, \
                    patch.object(wrist_capture, "KeyboardInput", Keys), \
                    patch.object(wrist_capture, "base_flange", return_value=np.eye(4)), \
                    patch.object(wrist_capture, "detect_board", return_value=None) as detect, \
                    patch.object(wrist_capture.time, "monotonic_ns", side_effect=lambda: clock.now), \
                    patch.object(wrist_capture.time, "sleep"), \
                    patch("matplotlib.pyplot.get_backend", return_value="mock"), \
                    patch("matplotlib.pyplot.subplots", return_value=(figure, np.array([[axis]], dtype=object))), \
                    patch("matplotlib.pyplot.show"), patch("matplotlib.pyplot.pause"), \
                    patch("matplotlib.pyplot.close"), redirect_stdout(io.StringIO()):
                cameras.return_value.__enter__.return_value = camera
                robots.return_value.__enter__.return_value = robot
                wrist_capture.collect(args)
            cameras.assert_called_once_with({side: f"{side}-camera"}, {side: [640, 480]})
            rows = [json.loads(line) for line in (args.output / "samples.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual([row["side"] for row in rows], [side, side])
            self.assertEqual([row["group"] for row in rows], [0, 1])
            self.assertEqual(robot.snapshot.call_count, 2)
            for row in rows:
                np.testing.assert_array_equal(np.array(Image.open(args.output / row["image"])), pixels)
                self.assertEqual(row["joints_rad"], [.1] * 7)
                self.assertNotIn("joint_window_span_rad", row)
            session = json.loads((args.output / "session.json").read_text())
            self.assertEqual(session["status"], "captured")
            self.assertEqual(session["counts"], {side: 2})
            self.assertEqual(session["sides"], [side])
            self.assertEqual(session["robot_control"], "keyboard")
            self.assertEqual(session["robot_control_mode"], "position")
            self.assertEqual(session["stationary_confirmation"], "operator")
            self.assertNotIn("feedback_source", session)
            self.assertEqual(session["cameras"][side]["output_rotation_deg"], 180 if side == "left" else 0)
            if preview:
                self.assertTrue(detect.called)
                for call in detect.call_args_list:
                    np.testing.assert_array_equal(call.args[0], pixels[::-1, ::-1])
                self.assertTrue(axis.imshow.return_value.set_data.called)
                events = [call.args[0] for call in figure.canvas.mpl_connect.call_args_list]
                self.assertIn("key_press_event", events)
                self.assertIn("close_event", events)
            else:
                detect.assert_not_called()
            robots.return_value.__exit__.assert_called_once()
            cameras.return_value.__exit__.assert_called_once()

    def test_cancel_does_not_start_robot_camera_or_create_output(self):
        args = cli.parser().parse_args(["wrist", "collect"])
        with tempfile.TemporaryDirectory() as directory:
            args.output = Path(directory) / "capture"
            with patch.object(wrist_capture, "require_local_usb"), patch.object(wrist_capture, "load_model"), \
                    patch.object(wrist_capture, "NonblockingTerminal"), \
                    patch.object(wrist_capture, "confirm_motion", return_value=False), \
                    patch.object(wrist_capture, "CameraStreams") as cameras, \
                    patch.object(wrist_capture, "KeyboardRobot") as robots:
                self.assertIsNone(wrist_capture.collect(args))
            self.assertFalse(args.output.exists())
            cameras.assert_not_called()
            robots.assert_not_called()

    def test_control_failure_closes_robot_and_preserves_failed_status(self):
        args = cli.parser().parse_args(["wrist", "collect", "--no-preview"])
        robot = Mock(ip="192.168.1.190")
        robot.check.side_effect = RuntimeError("control fault")
        model = SimpleNamespace(model=SimpleNamespace(digest="test-model", path=Path("test.cfg")))
        with tempfile.TemporaryDirectory() as directory:
            args.output = Path(directory) / "capture"
            with patch.object(wrist_capture, "require_local_usb"), \
                    patch.object(wrist_capture, "load_model", return_value=model), \
                    patch.object(wrist_capture, "NonblockingTerminal"), \
                    patch.object(wrist_capture, "confirm_motion", return_value=True), \
                    patch.object(wrist_capture, "CameraStreams") as cameras, \
                    patch.object(wrist_capture, "KeyboardRobot") as robots, \
                    patch.object(wrist_capture, "KeyboardInput"), redirect_stdout(io.StringIO()):
                cameras.return_value.__enter__.return_value.metadata = {"left": lens()}
                robots.return_value.__enter__.return_value = robot
                with self.assertRaisesRegex(RuntimeError, "control fault"):
                    wrist_capture.collect(args)
            self.assertEqual(json.loads((args.output / "session.json").read_text())["status"], "failed")
            robots.return_value.__exit__.assert_called_once()
            cameras.return_value.__exit__.assert_called_once()

    def test_cli_removes_passive_source_and_limits_control_to_one_arm(self):
        args = cli.parser().parse_args(["wrist", "collect"])
        self.assertEqual(args.side, "left")
        self.assertEqual((args.step_mm, args.step_deg, args.transition_s), (5., 2., .25))
        self.assertFalse(hasattr(args, "feedback_source"))
        for options in (["--feedback-source", "passive"], ["--side", "both"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.parser().parse_args(["wrist", "collect", *options])
        for option in ("--step-mm", "--step-deg", "--transition-s"):
            with patch.object(wrist_capture, "collect") as collect, \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["wrist", "collect", option, "0"]), 1)
            collect.assert_not_called()

    def test_cli_routes_wrist_and_main_camera_to_their_own_collectors(self):
        with patch.object(wrist_capture, "collect") as wrist, \
                patch.object(external_collect, "collect") as external:
            self.assertEqual(cli.main(["wrist", "collect", "--side", "left", "--samples", "1"]), 0)
            self.assertEqual(wrist.call_args.args[0].left_serial, "408322072104")
            external.assert_not_called()
            self.assertEqual(cli.main(["external", "collect", "--samples", "1"]), 0)
            self.assertEqual(external.call_args.args[0].external_serial, "405622076394")


if __name__ == "__main__":
    unittest.main()
