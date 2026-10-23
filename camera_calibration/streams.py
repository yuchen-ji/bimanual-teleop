"""Local RealSense lifecycle and timestamp pairing; imports drivers only on entry."""

import platform
import time
from collections import deque
from contextlib import ExitStack
from pathlib import Path

import numpy as np


def require_local_usb():
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise RuntimeError("Camera capture requires the USB-connected Linux x86_64 PC")
    if not list(Path("/dev").glob("video*")):
        raise RuntimeError(
            "No local video devices; run capture on the USB-connected PC"
        )


class CameraStreams:
    def __init__(self, serials, resolutions):
        if len(set(serials.values())) != len(serials):
            raise ValueError("Wrist and external cameras must be different devices")
        self.serials, self.resolutions = serials, resolutions
        self.pipelines, self.metadata = {}, {}
        self.cleanup = ExitStack()
        self.wall_offset = time.time_ns() - time.monotonic_ns()
        self.started_at = time.monotonic()
        self.last_frame = {}

    def __enter__(self):
        require_local_usb()
        import pyrealsense2 as rs

        self.rs = rs
        try:
            context = rs.context()
            devices = {
                d.get_info(rs.camera_info.serial_number): d
                for d in context.query_devices()
            }
            missing = set(self.serials.values()) - devices.keys()
            if missing:
                raise RuntimeError(
                    f"Missing cameras {sorted(missing)}; found {sorted(devices)}"
                )
            for name, serial in self.serials.items():
                device = devices[serial]
                for sensor in device.query_sensors():
                    if sensor.supports(rs.option.global_time_enabled):
                        sensor.set_option(rs.option.global_time_enabled, 1.0)
                pipeline, config = rs.pipeline(context), rs.config()
                config.enable_device(serial)
                width, height = self.resolutions[name]
                config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, 30)
                profile = pipeline.start(config)
                self.cleanup.callback(pipeline.stop)
                self.pipelines[name] = pipeline
                self.last_frame[name] = time.monotonic()
                lens = (
                    profile.get_stream(rs.stream.color)
                    .as_video_stream_profile()
                    .get_intrinsics()
                )
                self.metadata[name] = {
                    "serial": serial,
                    "model": device.get_info(rs.camera_info.name),
                    "resolution": [width, height],
                    "fps": 30,
                    "output_rotation_deg": 0,
                    "rgb_intrinsics": {
                        "width": lens.width,
                        "height": lens.height,
                        "fx": lens.fx,
                        "fy": lens.fy,
                        "ppx": lens.ppx,
                        "ppy": lens.ppy,
                        "model": str(lens.model),
                        "coeffs": list(lens.coeffs),
                    },
                }
            return self
        except BaseException:
            self.cleanup.close()
            raise

    def poll(self):
        if abs(time.time_ns() - time.monotonic_ns() - self.wall_offset) > 10_000_000:
            raise RuntimeError("Host clock changed; start a fresh capture")
        result = {}
        for name, pipeline in self.pipelines.items():
            frames = pipeline.poll_for_frames()
            if frames and (frame := frames.get_color_frame()):
                self.last_frame[name] = time.monotonic()
                if (
                    frame.get_frame_timestamp_domain()
                    == self.rs.timestamp_domain.global_time
                ):
                    stamp = round(frame.get_timestamp() * 1_000_000) - self.wall_offset
                    result[name] = (
                        stamp,
                        int(frame.get_frame_number()),
                        np.asarray(frame.get_data()).copy(),
                    )
                elif time.monotonic() - self.started_at > 20:
                    raise RuntimeError(f"{name} camera did not enter GLOBAL_TIME")
            if time.monotonic() - self.last_frame[name] > 10:
                raise RuntimeError(f"No {name} frames for 10 seconds")
        return result

    def __exit__(self, *exc):
        return self.cleanup.__exit__(*exc)


def paired_frame(queues, now_ns):
    """Consume one recent pair, leaving enough future feedback for stop detection."""
    candidate = None
    while queues["wrist"] and queues["wrist"][0][0] < now_ns - 220_000_000:
        candidate = queues["wrist"].popleft()
    if (
        candidate is None
        or not queues["external"]
        or now_ns - candidate[0] > 500_000_000
    ):
        return None
    external = min(queues["external"], key=lambda f: abs(f[0] - candidate[0]))
    if abs(external[0] - candidate[0]) > 25_000_000:
        return None
    while queues["external"] and queues["external"][0][0] <= external[0]:
        queues["external"].popleft()
    return candidate, external


class BoardStability:
    def __init__(self):
        self.history = deque(maxlen=16)

    def clear(self):
        self.history.clear()

    def update(self, timestamp, centers):
        self.history.append((timestamp, np.concatenate(centers)))
        while self.history and timestamp - self.history[0][0] > 600_000_000:
            self.history.popleft()
        if len(self.history) < 3 or timestamp - self.history[0][0] < 350_000_000:
            return False
        pixels = np.stack([value[1] for value in self.history])
        return bool(
            np.linalg.norm(pixels - np.median(pixels, axis=0), axis=2).max() < 0.75
        )
