"""Optional GUI with a bounded latest-image mailbox, separate from recording."""

import math
import multiprocessing as mp
import os
import signal
import time

from .schema import CAMERAS, RGB_SHAPE


_PREVIEW_PERIOD_S = .2
_GAP_PX = 4
# The rest of the screen height stays free for the operator's terminal.
_MAX_HEIGHT_FRACTION = .55


def _window_size(screen_width, screen_height):
    """Side-by-side panels at the camera aspect ratio, as wide as one monitor."""
    count = len(CAMERAS)
    image_height, image_width = RGB_SHAPE[:2]
    gaps = _GAP_PX * (count - 1)
    # X reports one screen spanning all monitors; stay within a 16:9 monitor.
    width = min(screen_width, screen_height * 16 / 9,
                screen_height * _MAX_HEIGHT_FRACTION * image_width / image_height * count + gaps)
    panel_width = (width - gaps) / count
    return width, panel_width * image_height / image_width, panel_width


def _fit_screen(figure):
    window = getattr(figure.canvas.manager, "window", None)
    tk = hasattr(window, "winfo_screenwidth")
    screen = (window.winfo_screenwidth(), window.winfo_screenheight()) if tk else (1920, 1080)
    width, height, panel_width = _window_size(*screen)
    figure.set_size_inches(width / figure.dpi, height / figure.dpi)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=_GAP_PX / panel_width)
    if tk:
        # Pinned to the top so the free space below stays contiguous.
        window.geometry("+0+0")


def _lower_priority(increment=10):
    """Keep optional GUI drawing below capture and motion-control priority."""
    try:
        os.nice(increment)
    except (AttributeError, OSError):
        pass


class _Images:
    def __init__(self, context):
        self.pixels = context.RawArray("B", len(CAMERAS) * math.prod(RGB_SHAPE))
        self.stamps = context.RawArray("q", len(CAMERAS))
        self.lock = context.Lock()

    def _array(self):
        import numpy as np
        return np.frombuffer(self.pixels, dtype=np.uint8).reshape(len(CAMERAS), *RGB_SHAPE)

    def publish(self, frames):
        # A stalled GUI may miss preview updates, never recording frames.
        if not self.lock.acquire(block=False):
            return
        try:
            pixels = self._array()
            for camera, image in frames.items():
                index = CAMERAS.index(camera)
                pixels[index] = image
                self.stamps[index] = time.monotonic_ns()
        finally:
            self.lock.release()

    def read(self):
        if not self.lock.acquire(block=False):
            return None
        try:
            return self._array().copy(), tuple(self.stamps)
        finally:
            self.lock.release()


def _run(images, stopped):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # The preview is spawned by the already de-prioritized recorder, so this
    # additional increment makes GUI redraws the first work to yield under load.
    _lower_priority()
    figure = None
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        if plt.get_backend().lower() == "agg":
            raise RuntimeError("录制预览需要桌面图形环境")
        plt.rcParams["toolbar"] = "None"
        figure, axes = plt.subplots(1, len(CAMERAS))
        figure.canvas.manager.set_window_title("录制预览")
        _fit_screen(figure)
        artists = []
        for camera, axis in zip(CAMERAS, axes):
            axis.set_axis_off()
            artists.append(axis.imshow(np.zeros(RGB_SHAPE, dtype="u1")))
            axis.text(.01, .98, camera, transform=axis.transAxes, ha="left", va="top",
                      color="white", fontsize=9,
                      bbox={"facecolor": "black", "alpha": .5, "edgecolor": "none", "pad": 2})
        plt.show(block=False)
        parent = mp.parent_process()
        while not stopped.is_set() and plt.fignum_exists(figure.number):
            if parent is not None and not parent.is_alive():
                break
            latest = images.read()
            if latest is not None:
                pixels, stamps = latest
                now = time.monotonic_ns()
                for index, artist in enumerate(artists):
                    artist.set_visible(stamps[index] > 0 and now - stamps[index] < 1_000_000_000)
                    artist.set_data(pixels[index])
            figure.canvas.draw_idle()
            plt.pause(.001)
            stopped.wait(_PREVIEW_PERIOD_S)
    except Exception as error:
        from bimanual_teleop.common.console import print_message
        print_message(f"录制预览不可用：{error}；采集继续。", "warning")
    finally:
        if figure is not None:
            plt.close(figure)


class RecordingPreview:
    def __init__(self):
        context = mp.get_context("spawn")
        self.images = _Images(context)
        self.stopped = context.Event()
        self.process = context.Process(target=_run, args=(self.images, self.stopped),
                                       name="recording-preview")
        self.process.start()
        self.next_update = 0.

    def update(self, frames):
        now = time.monotonic()
        if now < self.next_update or not self.process.is_alive():
            return
        self.images.publish(frames)
        self.next_update = now + _PREVIEW_PERIOD_S

    def close(self):
        self.stopped.set()
        self.process.join(1.)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(1.)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(1.)
