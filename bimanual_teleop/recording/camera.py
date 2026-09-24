"""Three independently timestamped RealSense streams; no disk or GUI work."""

import math
from queue import Empty, Full, Queue
import threading
import time

from .sink import Record


class _ClockBridge:
    def __init__(self, clock_ns, wall_ns):
        self.clock_ns, self.wall_ns = clock_ns, wall_ns
        self.offset_ns = self._offset()

    def _offset(self):
        before = self.clock_ns()
        wall = self.wall_ns()
        after = self.clock_ns()
        return wall - (before + after) // 2

    def map(self, timestamp_ms):
        if not math.isfinite(timestamp_ms):
            raise RuntimeError("Camera timestamp is not finite")
        if abs(self._offset() - self.offset_ns) > 10_000_000:
            raise RuntimeError("Host realtime clock changed by more than 10 ms")
        return round(timestamp_ms * 1_000_000) - self.offset_ns


def _intrinsics(profile):
    value = profile.as_video_stream_profile().get_intrinsics()
    return {"width": value.width, "height": value.height,
            "fx": value.fx, "fy": value.fy, "ppx": value.ppx, "ppy": value.ppy,
            "model": str(value.model), "coeffs": list(value.coeffs)}


class CameraRig:
    def __init__(self, config, *, rs=None, clock_ns=time.monotonic_ns,
                 wall_ns=time.time_ns, startup_timeout_s=10.):
        self.config, self.rs = config, rs
        self.clock_ns, self.wall_ns = clock_ns, wall_ns
        self.startup_timeout_s = startup_timeout_s
        self.metadata, self.error = {}, None
        self._queue = Queue(90)
        self._stop, self._active = threading.Event(), threading.Event()
        self._lock, self._delivery_lock = threading.Lock(), threading.Lock()
        self._delivery_mode = "record"
        self._preview_next_ns = {}
        self._pipelines, self._threads = [], []
        self._last, self._warm = {}, {}
        self._accepted, self._delivered, self._discarded = {}, {}, {}
        self._queue_full_count = 0
        self._required = tuple((f"camera_{i}", kind) for i in range(3)
                               for kind in (("rgb", "depth") if i == 0 and config.main_depth
                                            else ("rgb",)))
        self._bridge = self._np = None
        self._started = False

    def _fail(self, error):
        with self._lock:
            self.error = self.error or str(error)
        self._stop.set()

    @property
    def ready(self):
        if not self._started or self._stop.is_set():
            return False
        now = self.clock_ns()
        with self._lock:
            ready = all(self._warm.get(key, 0) >= 3 and key in self._last
                        and -10_000_000 <= now - self._last[key][2] <= 500_000_000
                        for key in self._required)
        if self._active.is_set() and not ready:
            self._fail("Camera stream is missing or older than 0.5 s")
        return ready

    def start(self):
        if self._started or self._stop.is_set():
            raise RuntimeError("Create a new CameraRig after start/close")
        if self.rs is None:
            import pyrealsense2 as rs
            self.rs = rs
        import numpy as np
        self._np = np
        self._bridge = _ClockBridge(self.clock_ns, self.wall_ns)
        self._started = True
        deadline = time.monotonic() + self.startup_timeout_s
        try:
            context = self.rs.context()
            devices = {device.get_info(self.rs.camera_info.serial_number): device
                       for device in context.query_devices()}
            for index, serial in enumerate(self.config.cameras):
                if serial not in devices:
                    raise RuntimeError(f"Camera {serial} is not connected")
                device = devices[serial]
                for sensor in device.query_sensors():
                    if sensor.supports(self.rs.option.global_time_enabled):
                        sensor.set_option(self.rs.option.global_time_enabled, 1.)
                name, depth = f"camera_{index}", index == 0 and self.config.main_depth
                pipeline = self.rs.pipeline(context)
                settings = self.rs.config()
                settings.enable_device(serial)
                settings.enable_stream(self.rs.stream.color, 640, 480, self.rs.format.rgb8, 30)
                if depth:
                    settings.enable_stream(self.rs.stream.depth, 640, 480, self.rs.format.z16, 30)
                self._pipelines.append(pipeline)
                profile = pipeline.start(settings)
                color = profile.get_stream(self.rs.stream.color)
                info = {"serial": serial, "model": device.get_info(self.rs.camera_info.name),
                        "resolution": [640, 480], "fps": 30, "rgb_intrinsics": _intrinsics(color)}
                if depth:
                    depth_profile = profile.get_stream(self.rs.stream.depth)
                    extrinsics = depth_profile.get_extrinsics_to(color)
                    info.update(depth_intrinsics=_intrinsics(depth_profile),
                        depth_to_rgb={"rotation": list(extrinsics.rotation),
                                      "translation": list(extrinsics.translation)},
                        depth_scale=profile.get_device().first_depth_sensor().get_depth_scale())
                self.metadata[name] = info
                thread = threading.Thread(target=self._capture, args=(name, pipeline, depth),
                                          name=f"record-{name}", daemon=True)
                self._threads.append(thread)
                thread.start()
            while not self.ready:
                if self.error:
                    raise RuntimeError(self.error)
                if time.monotonic() >= deadline:
                    raise RuntimeError("Cameras did not produce three consecutive GLOBAL_TIME frames within 10 s")
                self._stop.wait(.005)
            self._active.set()
        except Exception as error:
            self._fail(error)
            self.close()
            raise RuntimeError(self.error) from error

    def _capture(self, name, pipeline, depth):
        try:
            while not self._stop.is_set():
                frames = pipeline.poll_for_frames()
                if frames:
                    self._accept(name, "rgb", frames.get_color_frame())
                    if depth:
                        self._accept(name, "depth", frames.get_depth_frame())
                if self._active.is_set():
                    self.ready  # Check all streams even when a pipeline stops delivering frames.
                self._stop.wait(.001)
        except Exception as error:
            if not self._stop.is_set():
                self._fail(f"{name}: {error}")

    def _accept(self, name, kind, frame):
        if not frame:
            return
        key = (name, kind)
        domain = frame.get_frame_timestamp_domain()
        if domain != self.rs.timestamp_domain.global_time:
            with self._lock:
                self._warm[key] = 0
                self._last.pop(key, None)
            if self._active.is_set():
                raise RuntimeError(f"{name}/{kind}: timestamp domain left GLOBAL_TIME")
            return
        source_ms = float(frame.get_timestamp())
        sequence = int(frame.get_frame_number())
        stamp = self._bridge.map(source_ms)
        with self._lock:
            previous = self._last.get(key)
            if previous is not None:
                if source_ms < previous[0] or sequence < previous[1]:
                    if self._active.is_set():
                        raise RuntimeError(
                            f"{name}/{kind}: frame time or sequence moved backwards "
                            f"(time_ms {previous[0]:.6f} -> {source_ms:.6f}, "
                            f"sequence {previous[1]} -> {sequence})")
                    # A sensor may restart while the other pipelines are still
                    # starting. No frames have been published yet: require a
                    # fresh warmup run instead of failing the whole preflight.
                    self._warm[key] = 0
                elif sequence == previous[1]:
                    return
            self._last[key] = (source_ms, sequence, stamp)
            self._warm[key] = self._warm.get(key, 0) + 1
        if not self._active.is_set():
            return
        # Episode setup and final flush may import codecs, create arrays or
        # close encoders.  They deliberately suspend delivery so those bounded
        # disk operations cannot fill this live-frame queue.  Timestamp and
        # freshness tracking above stays active while delivery is suspended.
        with self._delivery_lock:
            mode = self._delivery_mode
            if mode == "off" or mode == "preview" and kind != "rgb":
                return
            if mode == "preview":
                next_ns = self._preview_next_ns.get(key, stamp)
                if stamp < next_ns:
                    return
                self._preview_next_ns[key] = stamp + 200_000_000
            if mode not in ("record", "preview"):
                return
            image = self._np.asarray(frame.get_data()).copy()
            record = Record(f"cameras/{name}/{kind}", stamp, sequence,
                            {"source_time_ms": source_ms})
            try:
                self._queue.put_nowait((name, kind, image, record))
            except Full as error:
                with self._lock:
                    self._queue_full_count += 1
                    status = self.status_locked()
                raise RuntimeError(
                    "Camera recording queue is full: "
                    f"queued={status['queue_size']}/{status['queue_capacity']}, "
                    f"accepted={status['accepted']}, delivered={status['delivered']}, "
                    f"discarded={status['discarded']}, backlog={status['backlog']}, "
                    f"accounting_delta={status['accounting_delta']}") from error
            with self._lock:
                self._accepted[key] = self._accepted.get(key, 0) + 1

    def status_locked(self):
        keys = set(self._accepted) | set(self._delivered) | set(self._discarded)
        backlog = {key: self._accepted.get(key, 0) - self._delivered.get(key, 0)
                   - self._discarded.get(key, 0) for key in keys}
        queue_size = self._queue.qsize()
        return {
            "queue_size": queue_size, "queue_capacity": self._queue.maxsize,
            "accepted": {f"{name}/{kind}": count
                         for (name, kind), count in self._accepted.items()},
            "delivered": {f"{name}/{kind}": count
                          for (name, kind), count in self._delivered.items()},
            "discarded": {f"{name}/{kind}": count
                          for (name, kind), count in self._discarded.items()},
            "backlog": {f"{name}/{kind}": count for (name, kind), count in backlog.items()},
            "accounting_delta": queue_size - sum(backlog.values()),
            "queue_full_count": self._queue_full_count,
            "delivery_mode": self._delivery_mode,
        }

    def status(self):
        with self._lock:
            return self.status_locked()

    def _set_delivery(self, mode):
        if mode not in ("off", "preview", "record"):
            raise ValueError(f"Unknown camera delivery mode: {mode}")
        discarded = {}
        with self._delivery_lock:
            self._delivery_mode = mode
            self._preview_next_ns.clear()
            while True:
                try:
                    item = self._queue.get_nowait()
                except Empty:
                    break
                key = item[0], item[1]
                discarded[key] = discarded.get(key, 0) + 1
        if discarded:
            with self._lock:
                for key, count in discarded.items():
                    self._discarded[key] = self._discarded.get(key, 0) + count

    def suspend_delivery(self):
        """Keep cameras healthy while dropping frames during non-recording disk stalls."""
        self._set_delivery("off")

    def preview_delivery(self):
        """Deliver only three RGB streams at 5 Hz while no episode is active."""
        self._set_delivery("preview")

    def resume_delivery(self):
        self._set_delivery("record")

    def poll(self):
        self.ready
        if self.error:
            raise RuntimeError(self.error)
        result = []
        for _ in range(6):
            try:
                item = self._queue.get_nowait()
            except Empty:
                break
            result.append(item)
            key = item[0], item[1]
            with self._lock:
                self._delivered[key] = self._delivered.get(key, 0) + 1
        return result

    def close(self):
        self._stop.set()
        for pipeline in self._pipelines:
            try:
                pipeline.stop()
            except Exception as error:
                self._fail(f"Camera shutdown failed: {error}")
        self._pipelines.clear()
        for thread in self._threads:
            thread.join(timeout=1.)
            if thread.is_alive():
                self._fail("Camera capture thread did not stop")
        self._threads.clear()
