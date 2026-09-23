"""Optional GUI with a bounded latest-image mailbox, separate from recording."""

import multiprocessing as mp
import os
import signal
import time


_PREVIEW_PERIOD_S = .2


def _lower_priority(increment=10):
    """Keep optional GUI drawing below capture and motion-control priority."""
    try:
        os.nice(increment)
    except (AttributeError, OSError):
        pass


class _Images:
    def __init__(self, context):
        self.pixels = context.RawArray("B", 3 * 480 * 640 * 3)
        self.stamps = context.RawArray("q", 3)
        self.lock = context.Lock()

    def _array(self):
        import numpy as np
        return np.frombuffer(self.pixels, dtype=np.uint8).reshape(3, 480, 640, 3)

    def publish(self, frames):
        # A stalled GUI may miss preview updates, never recording frames.
        if not self.lock.acquire(block=False):
            return
        try:
            pixels = self._array()
            for camera, image in frames.items():
                index = int(camera.removeprefix("camera_"))
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
        figure, axes = plt.subplots(1, 3, figsize=(15, 4))
        artists = []
        for index, axis in enumerate(axes):
            axis.set_title(f"camera_{index}")
            axis.set_axis_off()
            artists.append(axis.imshow(np.zeros((480, 640, 3), dtype="u1")))
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
