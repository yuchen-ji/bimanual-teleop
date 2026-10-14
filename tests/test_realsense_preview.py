"""Camera discovery, stale images and process cleanup without USB or a GUI."""

from contextlib import redirect_stderr
import io
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from bimanual_teleop.visualization import realsense as preview


def sdk(serials=("2", "1")):
    rs = Mock()
    rs.camera_info = SimpleNamespace(serial_number="serial", name="name")
    rs.stream, rs.format = SimpleNamespace(color="color"), SimpleNamespace(rgb8="rgb8")
    devices = [SimpleNamespace(get_info=lambda key, s=s: s if key == "serial" else "D435")
               for s in serials]
    rs.context.return_value.query_devices.return_value = devices
    pipelines = [Mock() for _ in serials]
    rs.pipeline.side_effect = pipelines
    return rs, pipelines


class PreviewTests(unittest.TestCase):
    def test_discovery_binds_each_serial_and_enables_only_color(self):
        rs, pipelines = sdk()
        cameras = preview.open_cameras(rs)
        self.assertEqual([c.label for c in cameras], ["D435 · 1", "D435 · 2"])
        self.assertEqual([c.args[0] for c in rs.config.return_value.enable_device.call_args_list], ["1", "2"])
        self.assertEqual(rs.config.return_value.enable_stream.call_args.args,
                         ("color", 640, 480, "rgb8", 30))

    def test_default_serial_mapping_keeps_left_rgb_rotated_in_standalone_preview(self):
        rs, _pipelines = sdk(("408322070938", "348122072137", "408322072104"))
        cameras = preview.open_cameras(rs)
        image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        expected = {
            "348122072137": ("camera_0", image),
            "408322072104": ("camera_1", image[::-1, ::-1]),
            "408322070938": ("camera_2", image),
        }
        self.assertEqual(len(cameras), 3)
        for camera in cameras:
            serial = camera.label.split(" · ")[-1]
            name, pixels = expected[serial]
            self.assertEqual(camera.name, name)
            camera.pipeline.poll_for_frames.return_value = Mock(
                get_color_frame=lambda: Mock(get_data=lambda: image))
            artist = Mock()
            preview.update_camera(camera, artist, Mock(), 1.)
            np.testing.assert_array_equal(artist.set_data.call_args.args[0], pixels)

    def test_no_camera_warns_without_creating_a_window(self):
        rs, _ = sdk(())
        plt = Mock()
        with redirect_stderr(io.StringIO()) as output:
            preview.run_preview(lambda: True, rs=rs, plt=plt)
        self.assertIn("未找到", output.getvalue())
        plt.subplots.assert_not_called()

    def test_busy_camera_does_not_hide_available_camera(self):
        rs, pipelines = sdk()
        pipelines[0].start.side_effect = RuntimeError("busy")
        with redirect_stderr(io.StringIO()) as output:
            cameras = preview.open_cameras(rs)
        self.assertEqual(len(cameras), 1)
        self.assertIn("busy", output.getvalue())

    def test_stale_and_disconnected_images_are_hidden(self):
        pipeline, artist, axes = Mock(), Mock(), Mock()
        camera = preview._Camera(pipeline, "D435", 0.)
        pipeline.poll_for_frames.return_value = None
        with redirect_stderr(io.StringIO()):
            preview.update_camera(camera, artist, axes, 2.)
        artist.set_visible.assert_called_with(False)
        data = np.ones((480, 640, 3), dtype=np.uint8)
        pipeline.poll_for_frames.return_value = Mock(get_color_frame=lambda: Mock(get_data=lambda: data))
        preview.update_camera(camera, artist, axes, 3.)
        artist.set_visible.assert_called_with(True)
        self.assertFalse(camera.stale)
        pipeline.poll_for_frames.side_effect = RuntimeError("disconnected")
        with redirect_stderr(io.StringIO()):
            preview.update_camera(camera, artist, axes, 4.)
        self.assertTrue(camera.failed)
        artist.set_visible.assert_called_with(False)

    def test_window_close_and_parent_stop_release_all_cameras(self):
        for close_window in (False, True):
            rs, pipelines = sdk()
            plt, figure = Mock(), Mock(number=1)
            plt.get_backend.return_value = "TkAgg"
            plt.subplots.return_value = figure, SimpleNamespace(flat=[Mock(), Mock()])
            plt.fignum_exists.return_value = not close_window
            preview.run_preview(lambda: not close_window, rs=rs, plt=plt)
            for pipeline in pipelines:
                pipeline.stop.assert_called_once()
            plt.close.assert_called_once_with(figure)

    def test_unavailable_gui_releases_started_streams(self):
        rs, pipelines = sdk()
        plt = Mock()
        plt.get_backend.return_value = "Agg"
        with redirect_stderr(io.StringIO()) as output:
            preview.run_preview(lambda: True, rs=rs, plt=plt)
        self.assertIn("图形窗口", output.getvalue())
        for pipeline in pipelines:
            pipeline.stop.assert_called_once()

    def test_process_parent_pipe_closes_and_hung_child_is_reaped(self):
        with patch.object(preview.subprocess, "Popen") as popen:
            view = preview.RealSensePreview()
            view.start()
            process = popen.return_value
            process.wait.side_effect = [subprocess.TimeoutExpired("preview", 3), None]
            view.close()
            process.stdin.close.assert_called_once()
            process.terminate.assert_called_once()
            self.assertIsNone(view.process)


if __name__ == "__main__":
    unittest.main()
