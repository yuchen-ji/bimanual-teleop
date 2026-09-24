"""Online spooling stays lossless while finalization preserves the old schema."""

import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import zarr

from bimanual_teleop.recording.finalize import finalize_episode
from bimanual_teleop.recording.sink import Record
from bimanual_teleop.recording.spool import (
    NVENCVideo, RawEpisodeWriter, SharedFrameRing, _release, preflight_nvenc)
from bimanual_teleop.recording.storage import RGBVideo
from tests.test_recording_storage import _Kinematics


class RecordingSpoolTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.episode = Path(temporary.name) / "episode_000000"
        self.start = 1_000_000_000
        self.kine = _Kinematics()
        self.kine.model = type("Model", (), {"digest": "test-model"})()
        self.metadata = {
            "recording": {"main_depth": True},
            "model_sha256": "test-model",
            "units": {"joint_pos": "rad", "position": "m", "force": "N", "torque": "Nm"},
        }

    def test_frame_ring_refuses_to_overwrite_unread_image(self):
        ring = SharedFrameRing(mp.get_context("spawn"), (2, 2, 3), "u1", capacity=1)
        image = np.arange(12, dtype="u1").reshape(2, 2, 3)
        record = Record("cameras/camera_0/rgb", self.start, 7, {"source_time_ms": 1.})
        ring.put_nowait(1, image, record)
        with self.assertRaises(Exception):
            ring.put_nowait(1, image, record)
        position, generation, shared, restored = ring.peek()
        self.assertEqual((generation, restored.sequence), (1, 7))
        np.testing.assert_array_equal(shared, image)
        ring.acknowledge(position)
        self.assertEqual(ring.size(), 0)
        ring.close_queues()

    def test_release_keeps_a_private_copy_and_frees_the_only_slot(self):
        ring = SharedFrameRing(mp.get_context("spawn"), (2, 2, 3), "u1", capacity=1)
        first = np.zeros((2, 2, 3), dtype="u1")
        second = np.full((2, 2, 3), 9, dtype="u1")
        record = Record("cameras/camera_0/rgb", self.start, 7, {"source_time_ms": 1.})
        ring.put_nowait(1, first, record)
        _generation, owned, restored = _release(ring)
        self.assertEqual(restored.sequence, 7)
        ring.put_nowait(1, second, record)
        np.testing.assert_array_equal(owned, first)
        self.assertEqual(ring.processed.value, 1)
        ring.close_queues()

    def test_online_encoder_is_nvenc_and_driver_failure_has_no_cpu_fallback(self):
        container = Mock()
        stream = container.add_stream.return_value
        with patch("av.open", return_value=container):
            NVENCVideo(self.episode / "camera_0.mp4")
        container.add_stream.assert_called_once_with("h264_nvenc", rate=30)
        self.assertEqual(stream.options, {"preset": "p4", "cq": "21"})
        failed = subprocess.CompletedProcess(["nvidia-smi", "-L"], 1, "", "driver unavailable")
        with patch("bimanual_teleop.recording.spool.subprocess.run", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "NVIDIA 驱动不可用"):
                preflight_nvenc()

    def test_raw_spool_and_offline_finalize(self):
        # Fork keeps the test-only CPU encoder replacement inside worker processes.
        context = mp.get_context("fork")
        with patch("bimanual_teleop.recording.spool.NVENCVideo", RGBVideo):
            writer = RawEpisodeWriter(self.episode, self.start, self.metadata,
                                      context=context, frame_capacity=2)
            writer.prepare_rgb({f"camera_{index}": {} for index in range(3)})
            for camera_index in range(3):
                camera = f"camera_{camera_index}"
                for frame_index in range(2):
                    image = np.full((480, 640, 3), 30 + camera_index + frame_index,
                                    dtype="u1")
                    writer.write_rgb(camera, image, Record(
                        f"cameras/{camera}/rgb",
                        self.start + 1_000_000 + frame_index * 33_333_333,
                        100 + frame_index,
                        {"source_time_ms": 1000. + frame_index * 1000 / 30}))
            depth = np.arange(480 * 640, dtype="u2").reshape(480, 640)
            for frame_index in range(2):
                writer.append(Record("cameras/camera_0/depth",
                    self.start + 2_000_000 + frame_index * 33_333_333,
                    200 + frame_index,
                    {"source_time_ms": 2000. + frame_index * 1000 / 30,
                     "image": depth}))
            writer.append(Record("arms/left", self.start + 3_000_000, 1,
                {"joint_pos": (.1,) * 7, "wrench": (.2,) * 6}))
            writer.append(Record("hands/right", self.start + 4_000_000, 2,
                {"joint_pos": (.3,) * 20}))
            self.assertEqual(writer.close(self.start + 100_000_000), "captured")
        captured = json.loads((self.episode / "episode.json").read_text())
        self.assertEqual(captured["status"], "captured")
        self.assertFalse((self.episode / "raw.zarr").exists())
        with patch("bimanual_teleop.devices.tianji.model.TianjiKinematics",
                   return_value=self.kine):
            self.assertEqual(finalize_episode(self.episode), "complete")
        document = json.loads((self.episode / "episode.json").read_text())
        self.assertEqual(document["status"], "complete")
        raw = zarr.open_group(str(self.episode / "raw.zarr"), mode="r")
        self.assertEqual(raw["arms/left/joint_pos"].shape, (1, 7))
        self.assertEqual(raw["hands/right/joint_pos"].shape, (1, 20))
        self.assertEqual(raw["cameras/camera_0/depth/image"].shape, (2, 480, 640))
        np.testing.assert_array_equal(raw["cameras/camera_0/depth/image"][0], depth)
        for index in range(3):
            self.assertEqual(raw[f"cameras/camera_{index}/rgb/time_ns"].shape, (2,))


if __name__ == "__main__":
    unittest.main()
