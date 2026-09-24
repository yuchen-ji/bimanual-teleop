"""RealSense acquisition and timestamp checks using in-memory devices only."""

from queue import Empty, Queue
import time
from types import SimpleNamespace as NS
import unittest

import numpy as np

from bimanual_teleop.recording.camera import CameraRig, _ClockBridge
from bimanual_teleop.recording.config import RecordingConfig


class Clock:
    def __init__(self):
        self.now = 10_000_000_000
        self.offset = 1_700_000_000_000_000_000

    def mono(self):
        return self.now

    def wall(self):
        return self.now + self.offset


class Frame:
    def __init__(self, clock, sequence, *, age_ns=0, domain="global", data=None):
        self.sequence, self.domain = sequence, domain
        self.stamp = (clock.wall() - age_ns) / 1_000_000
        self.data = np.zeros((2, 3, 3), dtype=np.uint8) if data is None else data

    def get_timestamp(self):
        return self.stamp

    def get_frame_number(self):
        return self.sequence

    def get_frame_timestamp_domain(self):
        return self.domain

    def get_data(self):
        return self.data


class Profile:
    def as_video_stream_profile(self):
        return self

    def get_intrinsics(self):
        return NS(width=640, height=480, fx=600., fy=601., ppx=320., ppy=240.,
                  model="brown_conrady", coeffs=[.1, .2, .3, .4, .5])

    def get_extrinsics_to(self, _color):
        return NS(rotation=[1., 0., 0., 0., 1., 0., 0., 0., 1.], translation=[.01, 0., 0.])


class Device:
    def __init__(self, serial, supported=True):
        self.serial, self.supported, self.options = serial, supported, []

    def get_info(self, key):
        return self.serial if key == "serial" else "Fake RealSense"

    def query_sensors(self):
        return [self]

    def supports(self, _option):
        return self.supported

    def set_option(self, option, value):
        self.options.append((option, value))

    def first_depth_sensor(self):
        return self

    def get_depth_scale(self):
        return .001


class Settings:
    def __init__(self):
        self.streams = []

    def enable_device(self, serial):
        self.serial = serial

    def enable_stream(self, *args):
        self.streams.append(args)


def frameset(rgb, depth=None):
    return NS(get_color_frame=lambda: rgb, get_depth_frame=lambda: depth)


class Pipeline:
    def __init__(self, rs):
        self.rs, self.queue, self.stopped = rs, Queue(), False

    def start(self, settings):
        self.settings = settings
        depth = any(stream[0] == "depth" for stream in settings.streams)
        for sequence, domain in enumerate(self.rs.warmup_domains, 1):
            frame = Frame(self.rs.clock, sequence, age_ns=(10 - sequence) * 1_000_000,
                          domain=domain)
            self.queue.put(frameset(frame, frame if depth else None))
        return NS(get_stream=lambda _kind: Profile(),
                  get_device=lambda: self.rs.devices[settings.serial])

    def poll_for_frames(self):
        try:
            return self.queue.get_nowait()
        except Empty:
            return None

    def stop(self):
        self.stopped = True


class RealSense:
    camera_info = NS(serial_number="serial", name="name")
    option = NS(global_time_enabled="global_time")
    stream = NS(color="rgb", depth="depth")
    format = NS(rgb8="rgb8", z16="z16")
    timestamp_domain = NS(global_time="global")
    config = Settings

    def __init__(self, clock, domains=("global",) * 3):
        self.clock, self.warmup_domains, self.pipelines = clock, domains, []
        self.devices = {s: Device(s, supported=s != "c") for s in ("a", "b", "c")}

    def context(self):
        return NS(query_devices=lambda: list(self.devices.values()))

    def pipeline(self, _context):
        pipeline = Pipeline(self)
        self.pipelines.append(pipeline)
        return pipeline


class CameraTests(unittest.TestCase):
    def rig(self, *, depth=True, domains=("global",) * 3):
        clock = Clock()
        rs = RealSense(clock, domains)
        rig = CameraRig(RecordingConfig(("a", "b", "c"), main_depth=depth), rs=rs,
                        clock_ns=clock.mono, wall_ns=clock.wall, startup_timeout_s=.1)
        self.addCleanup(rig.close)
        return rig, rs, clock

    def prepared(self, *, depth=False):
        rig, rs, clock = self.rig(depth=depth)
        rig._np, rig._bridge, rig._started = np, _ClockBridge(clock.mono, clock.wall), True
        for name, kind in rig._required:
            for sequence in range(1, 4):
                rig._accept(name, kind, Frame(clock, sequence, age_ns=(4 - sequence) * 1_000_000))
        rig._active.set()
        return rig, rs, clock

    def test_only_main_camera_has_depth_and_metadata_is_calibrated(self):
        rig, rs, _clock = self.rig()
        self.assertFalse(rig.ready)
        rig.start()
        self.assertTrue(rig.ready)
        for index, pipeline in enumerate(rs.pipelines):
            self.assertEqual(pipeline.settings.serial, ("a", "b", "c")[index])
            expected = [("rgb", 640, 480, "rgb8", 30)]
            if index == 0:
                expected.append(("depth", 640, 480, "z16", 30))
            self.assertEqual(pipeline.settings.streams, expected)
            info = rig.metadata[f"camera_{index}"]
            self.assertEqual(info["rgb_intrinsics"]["fx"], 600.)
            self.assertEqual("depth_intrinsics" in info, index == 0)
        self.assertEqual(rs.devices["a"].options, [("global_time", 1.)])
        self.assertEqual(rs.devices["c"].options, [])
        self.assertEqual(rig.metadata["camera_0"]["depth_scale"], .001)
        self.assertEqual(rig.metadata["camera_0"]["depth_to_rgb"]["translation"], [.01, 0., 0.])
        self.assertEqual(rig.poll(), [])  # Warmup frames precede recording readiness.
        threads = tuple(rig._threads)
        rig.close()
        self.assertTrue(all(p.stopped for p in rs.pipelines))
        self.assertTrue(all(not t.is_alive() for t in threads))

    def test_global_warmup_resets_until_three_consecutive_frames_and_depth_is_optional(self):
        rig, rs, _clock = self.rig(depth=False,
            domains=("global", "global", "hardware", "global", "global", "global"))
        rig.start()
        self.assertTrue(rig.ready)
        self.assertEqual(set(rig._warm.values()), {3})
        self.assertTrue(all(len(p.settings.streams) == 1 for p in rs.pipelines))
        self.assertNotIn("depth_scale", rig.metadata["camera_0"])

    def test_unavailable_global_time_fails_start_and_closes_every_pipeline(self):
        rig, rs, _clock = self.rig(domains=("hardware",) * 4)
        with self.assertRaisesRegex(RuntimeError, "GLOBAL_TIME"):
            rig.start()
        self.assertIsNotNone(rig.error)
        self.assertTrue(all(p.stopped for p in rs.pipelines))
        self.assertFalse(rig.ready)

    def test_startup_rollback_restarts_warmup_without_publishing_frames(self):
        for kind in ("rgb", "depth"):
            for rollback in ("sequence", "time"):
                with self.subTest(kind=kind, rollback=rollback):
                    rig, _rs, clock = self.prepared(depth=True)
                    rig._active.clear()
                    key = ("camera_0", kind)
                    sequence = 1 if rollback == "sequence" else 4
                    age_ns = 0 if rollback == "sequence" else 2_000_000
                    rig._accept(*key, Frame(clock, sequence, age_ns=age_ns))
                    self.assertEqual(rig._warm[key], 1)
                    self.assertFalse(rig.ready)
                    for step in (1, 2):
                        clock.now += 33_000_000
                        rig._accept(*key, Frame(clock, sequence + step))
                        self.assertEqual(rig.ready, step == 2)
                    self.assertTrue(rig._queue.empty())
                    rig._active.set()
                    clock.now += 33_000_000
                    rig._accept(*key, Frame(clock, sequence + 3))
                    self.assertEqual([item[3].sequence for item in rig.poll()], [sequence + 3])
                    with self.assertRaisesRegex(RuntimeError, "backwards.*sequence"):
                        rig._accept(*key, Frame(clock, 1))

    def test_domain_change_after_individual_warmup_can_retry_before_rig_is_active(self):
        rig, _rs, clock = self.prepared(depth=True)
        rig._active.clear()
        rig._accept("camera_0", "depth", Frame(clock, 4, domain="hardware"))
        self.assertFalse(rig.ready)
        self.assertNotIn(("camera_0", "depth"), rig._last)
        for sequence in (1, 2, 3):
            clock.now += 33_000_000
            rig._accept("camera_0", "depth", Frame(clock, sequence))
        self.assertTrue(rig.ready)
        self.assertTrue(rig._queue.empty())

    def test_pipeline_start_failure_still_releases_the_pipeline(self):
        rig, rs, _clock = self.rig()
        original_factory = rs.pipeline

        def factory(context):
            pipeline = original_factory(context)
            if len(rs.pipelines) == 2:
                def fail(_settings):
                    raise RuntimeError("fake device failed during start")
                pipeline.start = fail
            return pipeline

        rs.pipeline = factory
        with self.assertRaisesRegex(RuntimeError, "fake device failed during start"):
            rig.start()
        self.assertTrue(all(p.stopped for p in rs.pipelines))

    def test_rgb_and_depth_use_their_own_timestamps_and_copy_owned_pixels(self):
        rig, _rs, clock = self.prepared(depth=True)
        rgb = Frame(clock, 4)
        depth = Frame(clock, 9, age_ns=500_000, data=np.full((2, 3), 100, dtype=np.uint16))
        rig._accept("camera_0", "rgb", rgb)
        rig._accept("camera_0", "depth", depth)
        rgb.data[:] = 77
        depth.data[:] = 88
        color_item, depth_item = rig.poll()
        self.assertTrue(np.all(color_item[2] == 0))
        self.assertTrue(np.all(depth_item[2] == 100))
        self.assertEqual(depth_item[2].dtype, np.uint16)
        for item, frame in ((color_item, rgb), (depth_item, depth)):
            record = item[3]
            self.assertEqual(record.sequence, frame.sequence)
            self.assertEqual(record.time_ns, round(frame.stamp * 1_000_000) - clock.offset)
            self.assertEqual(record.values, {"source_time_ms": frame.stamp})
            self.assertEqual(record.stream, f"cameras/{item[0]}/{item[1]}")
        self.assertLess(depth_item[3].time_ns, color_item[3].time_ns)

    def test_frame_gaps_are_kept_duplicates_ignored_and_poll_is_bounded(self):
        rig, _rs, clock = self.prepared()
        for sequence in [4, 4, 9, 10, 11, 12, 13, 14, 15]:
            rig._accept("camera_0", "rgb", Frame(clock, sequence))
        self.assertEqual([x[3].sequence for x in rig.poll()], [4, 9, 10, 11, 12, 13])
        self.assertEqual([x[3].sequence for x in rig.poll()], [14, 15])
        self.assertEqual(rig.poll(), [])

    def test_domain_change_and_time_or_sequence_rollback_fail_without_substitution(self):
        cases = ((4, {"domain": "hardware"}, "GLOBAL_TIME"),
                 (4, {"age_ns": 2_000_000}, "backwards"),
                 (2, {}, "backwards"))
        for sequence, kwargs, message in cases:
            with self.subTest(sequence=sequence, kwargs=kwargs):
                rig, _rs, clock = self.prepared()
                with self.assertRaisesRegex(RuntimeError, message):
                    rig._accept("camera_0", "rgb", Frame(clock, sequence, **kwargs))
                self.assertTrue(rig._queue.empty())

    def test_host_wallclock_jump_is_rejected(self):
        rig, _rs, clock = self.prepared()
        clock.offset += 11_000_000
        with self.assertRaisesRegex(RuntimeError, "realtime clock"):
            rig._accept("camera_0", "rgb", Frame(clock, 4))
        self.assertTrue(rig._queue.empty())

    def test_stale_stream_is_an_error_and_capture_error_reaches_poll(self):
        rig, _rs, clock = self.prepared()
        clock.now += 501_000_000
        with self.assertRaisesRegex(RuntimeError, "0.5 s"):
            rig.poll()
        self.assertFalse(rig.ready)
        rig, rs, clock = self.rig(depth=False)
        rig.start()
        rs.pipelines[0].queue.put(frameset(Frame(clock, 4, domain="hardware")))
        deadline = time.monotonic() + 1.
        while rig.error is None and time.monotonic() < deadline:
            time.sleep(.001)
        with self.assertRaisesRegex(RuntimeError, "GLOBAL_TIME"):
            rig.poll()

    def test_queue_overflow_fails_instead_of_silently_losing_frames(self):
        rig, _rs, clock = self.prepared()
        for sequence in range(4, 94):
            rig._accept("camera_0", "rgb", Frame(clock, sequence))
        with self.assertRaisesRegex(RuntimeError, r"queue is full.*queued=90/90"):
            rig._accept("camera_0", "rgb", Frame(clock, 94))
        self.assertEqual(rig.status(), {
            "queue_size": 90, "queue_capacity": 90,
            "accepted": {"camera_0/rgb": 90}, "delivered": {}, "discarded": {},
            "backlog": {"camera_0/rgb": 90}, "accounting_delta": 0,
            "queue_full_count": 1, "delivery_mode": "record",
        })

    def test_suspended_delivery_drops_backlog_but_keeps_camera_health_current(self):
        rig, _rs, clock = self.prepared()
        rig._accept("camera_0", "rgb", Frame(clock, 4))
        rig.suspend_delivery()
        self.assertEqual(rig.poll(), [])
        self.assertEqual(rig.status()["discarded"], {"camera_0/rgb": 1})
        self.assertEqual(rig.status()["backlog"], {"camera_0/rgb": 0})
        clock.now += 33_000_000
        rig._accept("camera_0", "rgb", Frame(clock, 5))
        self.assertEqual(rig.poll(), [])
        self.assertEqual(rig._last[("camera_0", "rgb")][1], 5)
        rig.resume_delivery()
        clock.now += 33_000_000
        rig._accept("camera_0", "rgb", Frame(clock, 6))
        self.assertEqual([item[3].sequence for item in rig.poll()], [6])

    def test_preview_delivery_copies_only_rgb_at_five_hz(self):
        rig, _rs, clock = self.prepared(depth=True)
        rig.preview_delivery()
        for sequence in range(4, 14):
            clock.now += 33_000_000
            rig._accept("camera_0", "rgb", Frame(clock, sequence))
            rig._accept("camera_0", "depth", Frame(clock, sequence))
        frames = rig.poll()
        self.assertEqual([(name, kind) for name, kind, _image, _record in frames],
                         [("camera_0", "rgb"), ("camera_0", "rgb")])
        self.assertEqual(rig.status()["delivery_mode"], "preview")


if __name__ == "__main__":
    unittest.main()
