"""Online spooling stays lossless while finalization preserves the old schema."""

import errno
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import zarr

from bimanual_teleop.recording.finalize import archive_spool, finalize_episode, finalize_recordings
from bimanual_teleop.recording.sink import Record
from bimanual_teleop.recording.spool import (
    NVENCVideo, RawEpisodeWriter, SharedFrameRing, _release, preflight_nvenc)
from bimanual_teleop.recording.storage import RGBVideo
from tests.support.recording import CountingProgress
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

    def _capture(self, episode, *, phase_markers=()):
        # Fork keeps the test-only CPU encoder replacement inside worker processes.
        context = mp.get_context("fork")
        with patch("bimanual_teleop.recording.spool.NVENCVideo", RGBVideo):
            writer = RawEpisodeWriter(episode, self.start, self.metadata,
                                      context=context, frame_capacity=2)
            writer.prepare_rgb({f"camera_{index}": {} for index in range(3)})
            for stamp in phase_markers:
                writer.add_phase_marker(stamp)
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
        return depth

    def test_raw_spool_and_offline_finalize(self):
        depth = self._capture(self.episode)
        captured = json.loads((self.episode / "episode.json").read_text())
        self.assertEqual(captured["status"], "captured")
        self.assertFalse((self.episode / "raw.zarr").exists())
        captured.pop("phase_markers")  # Older spools did not store markers.
        (self.episode / "episode.json").write_text(json.dumps(captured))
        progress = CountingProgress()
        progress.episode("episode_000000")
        with patch("bimanual_teleop.devices.tianji.model.TianjiKinematics",
                   return_value=self.kine):
            self.assertEqual(finalize_episode(self.episode, progress=progress), "complete")
        self.assertEqual(progress.totals, [8])
        self.assertEqual(progress.advanced, progress.totals)
        document = json.loads((self.episode / "episode.json").read_text())
        self.assertEqual(document["status"], "complete")
        raw = zarr.open_group(str(self.episode / "raw.zarr"), mode="r")
        self.assertEqual(raw["arms/left/joint_pos"].shape, (1, 7))
        self.assertEqual(raw["hands/right/joint_pos"].shape, (1, 20))
        self.assertEqual(raw["cameras/camera_0/depth/image"].shape, (2, 480, 640))
        np.testing.assert_array_equal(raw["cameras/camera_0/depth/image"][0], depth)
        for index in range(3):
            self.assertEqual(raw[f"cameras/camera_{index}/rgb/time_ns"].shape, (2,))

    def test_finalize_and_refinalize_preserve_captured_markers(self):
        stamps = (self.start + 500_000, self.start + 40_000_000)
        self._capture(self.episode, phase_markers=stamps)
        markers = [{"index": i, "time_ns": stamp} for i, stamp in enumerate(stamps, 1)]
        self.assertEqual(json.loads((self.episode / "episode.json").read_text())["phase_markers"], markers)
        with patch("bimanual_teleop.devices.tianji.model.TianjiKinematics", return_value=self.kine):
            finalize_episode(self.episode)
            report = finalize_recordings(self.episode, refinalize=True)
        self.assertEqual(report["complete"], 1)
        self.assertEqual(report["failed"], 0)
        document = json.loads((self.episode / "episode.json").read_text())
        self.assertEqual(document["phase_markers"], markers)
        self.assertEqual(document["schema_version"], 1)
        raw = zarr.open_group(str(self.episode / "raw.zarr"), mode="r")
        self.assertEqual(raw.attrs["phase_markers"], markers)

    def test_finalize_searches_every_depth_and_deletes_discarded(self):
        root = self.episode.parent / "recordings"
        statuses = {
            "a/episode_000000": "captured",
            "a/b/c/episode_000001": "discarded",
            "d/episode_000002": "failed",
            "d/e/episode_000003": "complete",
            "d/.episode_000004.finalizing-1": "captured",
        }
        for name, status in statuses.items():
            (root / name).mkdir(parents=True)
            (root / name / "episode.json").write_text(json.dumps({"status": status}))
        (root / "a/episode_000000/raw_spool/inner").mkdir(parents=True)
        (root / "a/episode_000000/raw_spool/inner/episode.json").write_text("{}")
        seen = []
        def fake(path, sdk_root=None, progress=None):
            seen.append(Path(path).relative_to(root).as_posix())
            if seen[-1] == "a/episode_000000":
                raise ValueError("broken spool")
            return "complete"
        with patch("bimanual_teleop.recording.finalize.finalize_episode", side_effect=fake):
            report = finalize_recordings(root)
        self.assertEqual(seen, ["a/episode_000000", "d/e/episode_000003"])
        self.assertFalse((root / "a/b/c/episode_000001").exists())
        self.assertTrue((root / "d/episode_000002").exists())
        self.assertEqual((report["complete"], report["discarded"], report["skipped"], report["failed"]),
                         (1, 1, 1, 1))
        self.assertEqual(report["errors"][0][1], "broken spool")

    def test_finalize_archives_spool_and_refinalizes_through_link(self):
        root = self.episode.parent / "recordings"
        archive = self.episode.parent / "spools"
        episode = root / "session" / "episode_000000"
        self._capture(episode)
        link, target = episode / "raw_spool", archive / "session" / "episode_000000"
        with patch("bimanual_teleop.devices.tianji.model.TianjiKinematics", return_value=self.kine):
            report = finalize_recordings(root, spool_archive=archive)
            self.assertEqual((report["complete"], report["archived"]), (1, [str(episode)]))
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.resolve(), target.resolve())
            self.assertTrue((target / "streams").is_dir())
            self.assertEqual(list(episode.glob(".*")), [])
            shutil.rmtree(episode / "raw.zarr")
            self.assertEqual(finalize_recordings(root, spool_archive=archive)["archived"], [])
            self.assertFalse((episode / "raw.zarr").exists())
            link.unlink()
            report = finalize_recordings(root, spool_archive=archive, refinalize=True)
        self.assertEqual((report["complete"], report["failed"]), (1, 0))
        self.assertEqual(link.resolve(), target.resolve())
        self.assertEqual(json.loads((episode / "episode.json").read_text())["status"], "complete")
        raw = zarr.open_group(str(episode / "raw.zarr"), mode="r")
        self.assertEqual(raw["arms/left/eef_pose"].shape, (1, 7))
        with self.assertRaisesRegex(ValueError, "输入目录内"):
            finalize_recordings(root, spool_archive=root / "spools")

    def test_archive_copies_across_filesystems(self):
        episode = self.episode.parent / "session" / "episode_000000"
        (episode / "raw_spool" / "streams").mkdir(parents=True)
        (episode / "raw_spool" / "streams" / "a.bin").write_bytes(b"12345")
        archive = self.episode.parent / "spools"
        with patch("bimanual_teleop.recording.finalize.os.rename",
                   side_effect=OSError(errno.EXDEV, "cross-device")):
            self.assertTrue(archive_spool(episode, archive))
        self.assertTrue((episode / "raw_spool").is_symlink())
        self.assertEqual((episode / "raw_spool" / "streams" / "a.bin").read_bytes(), b"12345")
        self.assertEqual(list(episode.glob(".*")), [])
        self.assertEqual(list((archive / "session").glob(".*")), [])

    def test_refinalize_without_spool_keeps_episode_complete(self):
        root = self.episode.parent / "recordings"
        episode = root / "session" / "episode_000000"
        episode.mkdir(parents=True)
        (episode / "episode.json").write_text(json.dumps({"status": "complete"}))
        report = finalize_recordings(root, spool_archive=self.episode.parent / "spools", refinalize=True)
        self.assertEqual(report["failed"], 1)
        self.assertIn("找不到 raw_spool", report["errors"][0][1])
        self.assertEqual(json.loads((episode / "episode.json").read_text())["status"], "complete")


if __name__ == "__main__":
    unittest.main()
