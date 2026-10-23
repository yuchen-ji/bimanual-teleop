"""Optional color-only preview in a process that never imports robot control."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bimanual_teleop.visualization.style import styled
from bimanual_teleop.recording.camera import orient_rgb
from bimanual_teleop.recording.config import load_config as load_recording_config
from bimanual_teleop.recording.schema import CAMERAS

from bimanual_teleop.common.console import print_message


class RealSensePreview:
    """The parent's pipe lifetime bounds acquisition, including parent crashes."""

    def __init__(self):
        self.process = None

    def start(self):
        try:
            self.process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve())], stdin=subprocess.PIPE,
                start_new_session=True)
        except OSError as error:
            print_message(f"RealSense 预览未启动：{error}", "warning")

    def close(self):
        process, self.process = self.process, None
        if process is None:
            return
        process.stdin.close()
        try:
            process.wait(timeout=3.)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2.)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@dataclass
class _Camera:
    pipeline: object
    label: str
    last_frame: float
    failed: bool = False
    stale: bool = False
    name: str | None = None


def open_cameras(rs):
    """A failed camera does not suppress other connected color cameras."""
    cameras = []
    names = dict(zip(load_recording_config().cameras, CAMERAS))
    context = rs.context()
    devices = sorted(context.query_devices(), key=lambda d: d.get_info(rs.camera_info.serial_number))
    for device in devices:
        serial = device.get_info(rs.camera_info.serial_number)
        label = f"{device.get_info(rs.camera_info.name)} · {serial}"
        pipeline = rs.pipeline(context)
        try:
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            pipeline.start(config)
        except RuntimeError as error:
            try:
                pipeline.stop()
            except RuntimeError:
                pass
            print_message(f"RealSense {serial} 无法预览：{error}", "warning")
            continue
        cameras.append(_Camera(pipeline, label, time.monotonic(), name=names.get(serial)))
    return cameras


def update_camera(camera, artist, axes, now):
    """Nonblocking reads; a stale/disconnected image is never left visible."""
    import numpy as np

    if camera.failed:
        return
    try:
        frames = camera.pipeline.poll_for_frames()
        color = frames.get_color_frame() if frames else None
        if color:
            artist.set_data(orient_rgb(camera.name, np.asanyarray(color.get_data())))
            artist.set_visible(True)
            axes.set_title(camera.label)
            camera.last_frame, camera.stale = now, False
        elif now - camera.last_frame >= 1. and not camera.stale:
            artist.set_visible(False)
            axes.set_title(camera.label + "\nNo new frames")
            camera.stale = True
            print_message(f"{camera.label} 无新图像，已清除旧画面。", "warning")
    except RuntimeError as error:
        camera.failed = True
        artist.set_visible(False)
        axes.set_title(camera.label + "\nDisconnected")
        print_message(f"{camera.label} 预览中断：{error}", "warning")


def run_preview(stop_requested, *, rs=None, plt=None):
    cameras, figure = [], None
    try:
        if rs is None:
            import pyrealsense2 as rs
        if plt is None:
            import matplotlib.pyplot as plt
        import numpy as np

        cameras = open_cameras(rs)
        if not cameras:
            print_message("未找到可预览的 RealSense 相机，遥操作继续。", "warning")
            return
        if "agg" == plt.get_backend().lower():
            raise RuntimeError("没有可用的图形窗口；请在桌面会话中运行")
        figure, axes = styled(plt.subplots)(1, len(cameras), squeeze=False,
                                   figsize=(6.4 * len(cameras), 5.))
        figure.canvas.manager.set_window_title("RealSense color preview")
        artists = []
        for camera, axis in zip(cameras, axes.flat):
            axis.set_axis_off()
            axis.set_title(camera.label)
            artists.append(axis.imshow(np.zeros((480, 640, 3), dtype=np.uint8)))
        figure.tight_layout()
        plt.show(block=False)
        while not stop_requested() and plt.fignum_exists(figure.number):
            for camera, artist, axis in zip(cameras, artists, axes.flat):
                update_camera(camera, artist, axis, time.monotonic())
            figure.canvas.draw_idle()
            plt.pause(1 / 30)
    except Exception as error:
        # This optional GUI process reports backend-specific failures too (for
        # example Tk's TclError), without changing the robot's lifecycle.
        print_message(f"RealSense 预览不可用：{error}；遥操作继续。", "warning")
    finally:
        for camera in cameras:
            try:
                camera.pipeline.stop()
            except RuntimeError:
                pass
        if figure is not None:
            plt.close(figure)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    def parent_closed():
        return stopped or bool(select.select([sys.stdin], [], [], 0)[0])

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    run_preview(parent_closed)


if __name__ == "__main__":
    main()
