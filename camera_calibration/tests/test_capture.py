"""Manual capture and annotated preview, without physical devices."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image

from camera_calibration import cli, external_collect as capture, motion
from camera_calibration.tests.test_external import lens


class ManualTests(unittest.TestCase):
    def test_keyboard_c_always_requests_capture_without_motion_gates(self):
        terminal = Mock(read=Mock(side_effect=["c", "w", "c", " ", "q"]))
        robot, on_capture = Mock(), Mock()
        with capture.KeyboardInput(terminal, robot, on_capture) as keyboard:
            self.assertTrue(keyboard.stopped.wait(1))
            keyboard.check()
        self.assertEqual(on_capture.call_count, 2)
        self.assertEqual(robot.send_key.call_args_list[0].args, ("w",))
        self.assertTrue(any(call.args == (" ",) for call in robot.send_key.call_args_list))
        self.assertFalse(keyboard.thread.is_alive())

    def test_pause_discards_only_motion_backlog(self):
        keys = motion.JogKeys()
        with keys, patch.object(motion.time, "monotonic_ns", return_value=1_000_000_000):
            keys.put("w")
            keys.put("i")
            keys.put(" ")
            self.assertEqual(keys.read(0), " ")
            self.assertIsNone(keys.read(0))
            keys.queue.put((1, "w"))
            self.assertIsNone(keys.read(0))
            keys.stopped.set()
            self.assertEqual(keys.read(0), "")

    def test_motion_starts_unengaged_in_position_mode_and_closes_driver(self):
        args = cli.parser().parse_args(["external", "collect", "--side", "right"])
        driver = Mock()
        events = []

        def jog(*positional, **options):
            self.assertTrue(options["configure_on_engage"])
            self.assertEqual(options["control_mode"], "position")
            self.assertEqual(options["side"], "right")
            with options["terminal"] as terminal:
                events.append("jog")
                terminal.stopped.wait(3)

        with patch.object(motion, "TianjiDriver", return_value=driver), \
                patch.object(motion, "run_jog", side_effect=jog):
            with motion.KeyboardRobot(args, Mock()) as robot:
                self.assertEqual(events, ["jog"])
                driver.configure.assert_not_called()
                driver.move_joints.assert_not_called()
            driver.start.assert_called_once_with(record_reported_config=False)
            driver.close.assert_called_once()
            self.assertFalse(robot.thread.is_alive())

    def test_snapshot_needs_one_valid_feedback_sample_and_ignores_velocity(self):
        args = cli.parser().parse_args(["external", "collect"])
        robot = motion.KeyboardRobot(args, Mock())
        joints = [0.1] * 7
        arm = SimpleNamespace(error=0, source_sequence=9,
                              joints=SimpleNamespace(position_rad=joints, velocity_rad_s=[1.] * 7))
        sample = SimpleNamespace(header=SimpleNamespace(received_monotonic_ns=123),
                                 payload=SimpleNamespace(arms={"left": arm}))
        robot.driver = Mock()
        robot.driver.get_latest.return_value = sample
        robot.driver.health.return_value = SimpleNamespace(ready=True)
        result = robot.snapshot()
        joints[0] = 2.
        self.assertEqual(result, {"joint_time_ns": 123, "joint_sequence": 9, "joints_rad": [.1] * 7})
        self.assertFalse(hasattr(robot, "feedback"))
        robot.driver.get_latest.return_value = None
        with self.assertRaisesRegex(ValueError, "尚未收到"):
            robot.snapshot()
        robot.driver.get_latest.return_value = sample
        robot.driver.health.return_value = SimpleNamespace(ready=False, detail="source counter stalled")
        with self.assertRaisesRegex(ValueError, "source counter stalled"):
            robot.snapshot()

    def test_cancel_prevents_driver_cameras_and_output_creation(self):
        args = cli.parser().parse_args(["external", "collect"])
        args.external_serial = "main"
        with tempfile.TemporaryDirectory() as directory:
            args.output = Path(directory) / "capture"
            with patch.object(capture, "require_local_usb"), patch.object(capture, "load_model"), \
                    patch.object(capture, "NonblockingTerminal"), patch.object(capture, "confirm_motion", return_value=False), \
                    patch.object(capture, "CameraStreams") as cameras, patch.object(capture, "KeyboardRobot") as robot:
                self.assertIsNone(capture.collect(args))
                self.assertFalse(args.output.exists())
                cameras.assert_not_called()
                robot.assert_not_called()

    def test_old_auto_capture_options_are_removed(self):
        for option in ("--wrist-result", "--capture-mode", "--feedback-source"):
            with self.subTest(option=option), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    cli.parser().parse_args(["external", "collect", option, "unused"])

    def test_collect_saves_only_c_requests_and_allows_repeated_pose_without_detection(self):
        args = cli.parser().parse_args(["external", "collect", "--no-preview"])
        args.external_serial = "main"
        clock = SimpleNamespace(now=1_000_000_000, cycle=0)

        class Input:
            def __init__(self, terminal, robot, on_capture):
                self.robot, self.on_capture, self.stopped = robot, on_capture, self

            def __enter__(self):
                return self

            def is_set(self):
                clock.cycle += 1
                clock.now += 50_000_000
                if clock.cycle in (1, 3):
                    self.on_capture(clock.now)
                return clock.cycle >= 5

            def check(self):
                pass

            def __exit__(self, *exc):
                self.robot.send_key(" ")

        rgb = np.full((480, 640, 3), 87, dtype=np.uint8)
        cameras = Mock(metadata={"external": lens("main")})
        cameras.poll.side_effect = lambda: {"external": (clock.now - 20_000_000, clock.cycle, rgb)}
        robot = Mock()
        robot.snapshot.side_effect = lambda: {"joints_rad": [0.] * 7, "joint_time_ns": clock.now,
                                              "joint_sequence": 1}
        with tempfile.TemporaryDirectory() as directory:
            args.output = Path(directory) / "capture"
            model = SimpleNamespace(model=SimpleNamespace(digest="model", path=Path("model.cfg")))
            with patch.object(capture, "require_local_usb"), patch.object(capture, "load_model", return_value=model), \
                    patch.object(capture, "NonblockingTerminal"), \
                    patch.object(capture, "confirm_motion", return_value=True), \
                    patch.object(capture, "CameraStreams") as factory, \
                    patch.object(capture, "KeyboardRobot") as robots, \
                    patch.object(capture, "KeyboardInput", Input), \
                    patch.object(capture, "base_flange", return_value=np.eye(4)), \
                    patch.object(capture, "detect_board") as detector, \
                    patch.object(capture.time, "monotonic_ns", side_effect=lambda: clock.now), \
                    patch.object(capture.time, "sleep"), redirect_stdout(io.StringIO()):
                factory.return_value.__enter__.return_value = cameras
                robots.return_value.__enter__.return_value = robot
                capture.collect(args)
            rows = [json.loads(line) for line in (args.output / "samples.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(robot.snapshot.call_count, 2)
            for row in rows:
                self.assertEqual(row["joints_rad"], [0.] * 7)
                self.assertNotIn("joint_window_span_rad", row)
                np.testing.assert_array_equal(np.asarray(Image.open(args.output / row["external_image"])), rgb)
            session = json.loads((args.output / "session.json").read_text())
            self.assertEqual(session["status"], "captured")
            self.assertEqual(session["stationary_confirmation"], "operator")
            self.assertEqual(session["robot_control_mode"], "position")
            self.assertEqual(len(list((args.output / "external").glob("*.png"))), 2)
            detector.assert_not_called()
            factory.return_value.__exit__.assert_called_once()
            robots.return_value.__exit__.assert_called_once()

    def test_collect_failure_closes_robot_and_preserves_failed_status(self):
        args = cli.parser().parse_args(["external", "collect", "--no-preview"])
        args.external_serial = "main"
        robot = Mock()
        robot.check.side_effect = RuntimeError("control fault")
        with tempfile.TemporaryDirectory() as directory:
            args.output = Path(directory) / "capture"
            model = SimpleNamespace(model=SimpleNamespace(digest="model", path=Path("model.cfg")))
            with patch.object(capture, "require_local_usb"), patch.object(capture, "load_model", return_value=model), \
                    patch.object(capture, "NonblockingTerminal"), \
                    patch.object(capture, "confirm_motion", return_value=True), \
                    patch.object(capture, "CameraStreams") as factory, \
                    patch.object(capture, "KeyboardRobot") as robots, \
                    patch.object(capture, "KeyboardInput"), redirect_stdout(io.StringIO()):
                factory.return_value.__enter__.return_value.metadata = {"external": lens("main")}
                robots.return_value.__enter__.return_value = robot
                with self.assertRaisesRegex(RuntimeError, "control fault"):
                    capture.collect(args)
            session = json.loads((args.output / "session.json").read_text())
            self.assertEqual(session["status"], "failed")
            robots.return_value.__exit__.assert_called_once()

    def test_preview_draws_centers_and_every_index_on_a_copy(self):
        raw = np.zeros((480, 640, 3), dtype=np.uint8)
        centers = np.array([[100 + column * 60, 100 + row * 60] for row in range(5) for column in range(4)], dtype=float)
        with patch.object(cv2, "circle", wraps=cv2.circle) as circles, \
                patch.object(cv2, "putText", wraps=cv2.putText) as labels:
            display = capture.detection_overlay(raw, centers)
        self.assertEqual(circles.call_count, 20)
        self.assertEqual(circles.call_args_list[0].args[1], (100, 100))
        numbered = [call.args[1] for call in labels.call_args_list if call.args[1].isdigit()]
        self.assertEqual(numbered, [str(index) for index in range(20) for _ in range(2)])
        self.assertEqual(circles.call_args_list[0].args[3], (255, 70, 40))
        self.assertTrue(np.any(display))
        self.assertFalse(np.any(raw))
        with patch.object(cv2, "circle", wraps=cv2.circle) as circles:
            missing = capture.detection_overlay(raw, None)
        circles.assert_not_called()
        self.assertFalse(np.any(missing[95:105, 95:105]))


if __name__ == "__main__":
    unittest.main()
