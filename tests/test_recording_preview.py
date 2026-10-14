"""Preview backpressure must never block the raw recording consumer."""

import multiprocessing as mp
import unittest
from unittest.mock import Mock, patch

import numpy as np

from bimanual_teleop.recording.preview import (
    _GAP_PX, _Images, _PREVIEW_PERIOD_S, RecordingPreview, _fit_screen, _lower_priority,
    _window_size,
)


class RecordingPreviewTests(unittest.TestCase):
    def test_window_fills_one_monitor_width_and_leaves_room_below(self):
        width, height, panel = _window_size(1920, 1080)
        self.assertEqual(width, 1920)
        self.assertAlmostEqual(panel * 3 + 2 * _GAP_PX, width)
        self.assertAlmostEqual(height, panel * 3 / 4)
        self.assertLess(height, 1080 * .5)
        self.assertEqual(_window_size(3840, 1080)[0], 1920)
        self.assertLessEqual(_window_size(1080, 1920)[1], 1920 * .55)
        self.assertLessEqual(_window_size(2400, 600)[1], 600 * .55 + 1e-9)

    def test_layout_has_no_outer_margins(self):
        from matplotlib.figure import Figure
        figure = Figure()
        figure.subplots(1, 3)
        _fit_screen(figure)
        np.testing.assert_allclose(figure.get_size_inches() * figure.dpi, _window_size(1920, 1080)[:2])
        params = figure.subplotpars
        self.assertEqual((params.left, params.right, params.bottom, params.top), (0, 1, 0, 1))

    def test_mailbox_retains_only_latest_owned_images(self):
        images = _Images(mp.get_context("spawn"))
        frame = np.full((480, 640, 3), 7, dtype="u1")
        images.publish({"camera_0": frame})
        frame[:] = 8
        images.publish({"camera_0": frame, "camera_2": frame})
        frame[:] = 9
        pixels, stamps = images.read()
        self.assertTrue(np.all(pixels[0] == 8))
        self.assertTrue(np.all(pixels[1] == 0))
        self.assertTrue(np.all(pixels[2] == 8))
        self.assertGreater(stamps[0], 0)
        self.assertEqual(stamps[1], 0)
        pixels[:] = 0
        self.assertTrue(np.all(images.read()[0][0] == 8))

    def test_busy_or_closed_gui_does_not_block_or_accumulate_frames(self):
        with patch("multiprocessing.context.SpawnContext.Process") as process:
            process.return_value.is_alive.return_value = True
            preview = RecordingPreview()
        preview.images.lock.acquire()
        try:
            preview.update({"camera_0": np.ones((480, 640, 3), dtype="u1")})
            self.assertEqual(tuple(preview.images.stamps), (0, 0, 0))
            self.assertIsNone(preview.images.read())
        finally:
            preview.images.lock.release()
        preview.next_update = 0
        with patch("bimanual_teleop.recording.preview.time.monotonic", return_value=123.):
            preview.update({"camera_0": np.ones((480, 640, 3), dtype="u1")})
        self.assertGreater(preview.images.stamps[0], 0)
        self.assertEqual(preview.next_update, 123. + _PREVIEW_PERIOD_S)
        preview.next_update = 0
        preview.process.is_alive.return_value = False
        with patch.object(preview.images, "publish") as publish:
            preview.update({})
            publish.assert_not_called()
        preview.close()

    def test_preview_priority_is_best_effort(self):
        with patch("bimanual_teleop.recording.preview.os.nice") as nice:
            _lower_priority()
            nice.assert_called_once_with(10)
        with patch("bimanual_teleop.recording.preview.os.nice",
                   side_effect=OSError("unsupported")):
            _lower_priority()

    def test_stuck_gui_shutdown_is_bounded(self):
        preview = RecordingPreview.__new__(RecordingPreview)
        preview.stopped, preview.process = Mock(), Mock()
        preview.process.is_alive.return_value = True
        preview.close()
        preview.stopped.set.assert_called_once()
        preview.process.terminate.assert_called_once()
        preview.process.kill.assert_called_once()
        self.assertEqual([call.args for call in preview.process.join.call_args_list], [(1.,)] * 3)
