"""Timing, stop detection and cleanup with simulated cameras/SDK only."""

import json
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from camera_calibration.external_collect import collect
from camera_calibration.streams import BoardStability, CameraStreams, paired_frame
from camera_calibration.tests.test_external import lens


class CaptureTests(unittest.TestCase):
    def test_pairing_obeys_skew_age_and_consumes_frames(self):
        now = 1_000_000_000
        queues = {
            "wrist": deque([(now - 250_000_000, 1, None)]),
            "external": deque([(now - 240_000_000, 2, None)]),
        }
        pair = paired_frame(queues, now)
        self.assertEqual((pair[0][1], pair[1][1]), (1, 2))
        self.assertIsNone(paired_frame(queues, now))
        for lag, skew in ((600_000_000, 0), (250_000_000, 26_000_000)):
            queues = {
                "wrist": deque([(now - lag, 1, None)]),
                "external": deque([(now - lag + skew, 2, None)]),
            }
            self.assertIsNone(paired_frame(queues, now))

    def test_board_motion_is_independent_of_robot_stop_detection(self):
        stability = BoardStability()
        points = np.arange(40).reshape(20, 2)
        for i in range(5):
            stopped = stability.update(i * 100_000_000, [points, points])
        self.assertTrue(stopped)
        self.assertFalse(stability.update(500_000_000, [points + 3, points]))
        stability.clear()
        self.assertFalse(stability.update(600_000_000, [points, points]))

    def test_failed_second_camera_start_stops_first(self):
        first, second = MagicMock(), MagicMock()
        second.start.side_effect = RuntimeError("USB failure")
        intrinsics = SimpleNamespace(
            width=640,
            height=480,
            fx=400.0,
            fy=400.0,
            ppx=320.0,
            ppy=240.0,
            model="none",
            coeffs=[0.0] * 5,
        )
        first.start.return_value.get_stream.return_value.as_video_stream_profile.return_value.get_intrinsics.return_value = intrinsics
        devices = []
        for serial in ("w", "e"):
            device = MagicMock()
            device.get_info.side_effect = lambda field, serial=serial: serial
            device.query_sensors.return_value = []
            devices.append(device)
        rs = MagicMock()
        rs.context.return_value.query_devices.return_value = devices
        rs.pipeline.side_effect = [first, second]
        with (
            patch("camera_calibration.streams.require_local_usb"),
            patch.dict("sys.modules", {"pyrealsense2": rs}),
            self.assertRaisesRegex(RuntimeError, "USB failure"),
            CameraStreams(
                {"wrist": "w", "external": "e"},
                {"wrist": [640, 480], "external": [640, 480]},
            ),
        ):
            self.fail("Camera setup must fail")
        first.stop.assert_called_once()

    def test_external_capture_failure_releases_cameras_and_feedback(self):
        camera = MagicMock()
        camera.metadata = {"wrist": lens("w"), "external": lens("e")}
        camera.poll.side_effect = RuntimeError("simulated disconnect")
        context = MagicMock()
        context.__enter__.return_value = camera
        feedback = MagicMock()
        feedback.error = None
        reference = {"board": {"columns": 4, "rows": 5, "same_row_spacing_m": 0.1}}
        wrist = {"camera": lens("w")}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            path.write_text("{}")
            args = SimpleNamespace(
                wrist_result=path,
                side="left",
                output=Path(directory) / "capture",
                external_serial="e",
                external_resolution=[640, 480],
                robot_ip="test",
                no_preview=True,
                samples=25,
            )
            with (
                patch("camera_calibration.external_collect.require_local_usb"),
                patch(
                    "camera_calibration.external_collect.load_wrist_reference",
                    return_value=(reference, wrist),
                ),
                patch("camera_calibration.external_collect.socket.socket"),
                patch(
                    "camera_calibration.external_collect.CameraStreams",
                    return_value=context,
                ),
                patch(
                    "camera_calibration.external_collect.FeedbackReader",
                    return_value=feedback,
                ),
                patch("bimanual_teleop.devices.tianji.sdk.ControlSDK") as sdk,
                self.assertRaisesRegex(RuntimeError, "simulated disconnect"),
            ):
                collect(args)
            sdk.return_value.open.assert_not_called()  # The mocked reader owns the connection.
            feedback.start.assert_called_once()
            feedback.close.assert_called_once()
            context.__exit__.assert_called_once()
            session = json.loads((args.output / "session.json").read_text())
            self.assertEqual(session["status"], "failed")

    def test_server_guard_runs_before_device_creation(self):
        with (
            patch(
                "camera_calibration.external_collect.require_local_usb",
                side_effect=RuntimeError("No local video"),
            ),
            patch("camera_calibration.external_collect.CameraStreams") as streams,
            self.assertRaisesRegex(RuntimeError, "No local video"),
        ):
            collect(SimpleNamespace())
        streams.assert_not_called()


if __name__ == "__main__":
    unittest.main()
