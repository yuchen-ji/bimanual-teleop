"""Raw episode files retain independent clocks, source samples and video frames."""

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import av
import numpy as np
import zarr

from bimanual_teleop.recording.sink import Record
from bimanual_teleop.recording.storage import EpisodeWriter, RGBVideo
from bimanual_teleop.types import Pose


class _Kinematics:
    def __init__(self):
        self.calls = []

    def fk(self, side, joints):
        self.calls.append((side, joints))
        return Pose(f"tianji_{side}_base", f"tianji_{side}_flange",
            (joints[0], joints[1], .1 if side == "left" else -.1),
            (0., 0., math.sin(joints[2] / 2), math.cos(joints[2] / 2)))


class RecordingStorageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "episode"
        self.kine = _Kinematics()
        self.start = 1_000_000_000
        self.metadata = {"units": {"joint_pos": "rad", "position": "m", "force": "N", "torque": "Nm"}}

    def writer(self):
        return EpisodeWriter(self.path, self.start, self.metadata, self.kine)

    def test_rgb_encoder_uses_realtime_single_core_settings(self):
        container = Mock()
        stream = container.add_stream.return_value
        with patch("av.open", return_value=container):
            RGBVideo(self.path / "camera_0.mp4")
        container.add_stream.assert_called_once_with("libx264", rate=30)
        self.assertEqual(stream.options, {"crf": "21", "preset": "ultrafast"})
        self.assertEqual(stream.thread_count, 2)

    def test_numeric_flush_preserves_si_values_and_fk_uses_actual_joints_only(self):
        writer = self.writer()
        self.assertEqual(json.loads((self.path / "episode.json").read_text())["status"], "recording")
        wrench = (.1, -2., 3., .04, -.05, .06)
        writer.append(Record("arms/left", self.start - 1, 90, {"joint_pos": (0.,) * 7, "wrench": wrench}))
        for index in range(130):
            joints = (.1 + index * .001, -.2, math.pi / 2, .4, .5, .6, .7)
            writer.append(Record("arms/left", self.start + index * 5_000_000, 100 + index,
                {"joint_pos": joints, "wrench": wrench}))
        desired = (1., 2., 3., 0., 0., 0., 1.)
        writer.append(Record("arm_commands/left", self.start + 100, 1,
            {"joint_pos": (0.,) * 7, "eef_pose": desired}))
        writer.append(Record("hands/right", self.start + 200, 902, {"joint_pos": (-.2,) * 20}))
        writer.close(self.start + 700_000_000)
        raw = zarr.open_group(str(self.path / "raw.zarr"), mode="r")
        measured = raw["arms/left"]
        self.assertEqual(measured["time_ns"].dtype, np.dtype("int64"))
        self.assertEqual(measured["sequence"].dtype, np.dtype("int64"))
        np.testing.assert_array_equal(measured["time_ns"][:], self.start + np.arange(130) * 5_000_000)
        np.testing.assert_array_equal(measured["sequence"][:], np.arange(100, 230))
        np.testing.assert_array_equal(measured["wrench"][:], np.tile(wrench, (130, 1)))
        np.testing.assert_allclose(measured["eef_pose"][0], (.1, -.2, .1, 0., 0., 2**-.5, 2**-.5))
        self.assertAlmostEqual(measured["joint_pos"][-1, 0], .229)
        np.testing.assert_array_equal(raw["arm_commands/left/eef_pose"][0], desired)
        np.testing.assert_array_equal(raw["hands/right/joint_pos"][0], (-.2,) * 20)
        self.assertEqual(len(self.kine.calls), 130)
        self.assertTrue(all(side == "left" for side, _ in self.kine.calls))
        document = json.loads((self.path / "episode.json").read_text())
        self.assertEqual(document["status"], "complete")
        self.assertEqual(document["counts"], {"arms/left": 130, "arm_commands/left": 1, "hands/right": 1})
        self.assertEqual(document["metadata"], self.metadata)

    def test_three_rgb_videos_encode_each_irregular_source_frame_once_and_depth_is_lossless(self):
        writer = self.writer()
        offsets = np.array([1_000_000, 35_000_000, 180_000_000, 201_000_000], dtype=np.int64)
        levels = (40, 85, 140, 210)
        source_ms = (1000.125, 1033.875, 1179.625, 1200.5)
        for camera_index in range(3):
            name = f"camera_{camera_index}"
            for index, (offset, level, source_time) in enumerate(zip(offsets, levels, source_ms)):
                rgb = np.full((480, 640, 3), level + camera_index, dtype=np.uint8)
                record = Record(f"cameras/{name}/rgb", self.start + int(offset) + camera_index,
                    50 + index, {"source_time_ms": source_time})
                writer.write_rgb(name, rgb, record)
        depth = np.arange(480 * 640, dtype=np.uint16).reshape(480, 640)
        for index in range(9):
            writer.append(Record("cameras/camera_0/depth", self.start + index * 33_333_333,
                700 + index, {"source_time_ms": 2000. + index * 1000 / 30, "image": depth}))
        writer.close(self.start + 400_000_000)
        raw = zarr.open_group(str(self.path / "raw.zarr"), mode="r")
        for camera_index in range(3):
            name = f"camera_{camera_index}"
            with av.open(str(self.path / f"{name}.mp4")) as container:
                self.assertEqual(container.streams.video[0].codec_context.name, "h264")
                frames = list(container.decode(video=0))
            self.assertEqual(len(frames), len(offsets))
            self.assertTrue(all(a.pts < b.pts for a, b in zip(frames, frames[1:])))
            means = [frame.to_ndarray(format="rgb24").mean() for frame in frames]
            np.testing.assert_allclose(means, np.asarray(levels) + camera_index, atol=3.)
            group = raw[f"cameras/{name}/rgb"]
            np.testing.assert_array_equal(group["time_ns"][:], self.start + offsets + camera_index)
            np.testing.assert_array_equal(group["sequence"][:], np.arange(50, 54))
            np.testing.assert_array_equal(group["source_time_ms"][:], source_ms)
            # The 145 ms source gap is metadata, not repeated video frames.
            self.assertGreater(np.diff(group["time_ns"][:]).max(), 100_000_000)
            self.assertLess(float(frames[-1].pts * frames[-1].time_base), .15)
        depth_array = raw["cameras/camera_0/depth/image"]
        self.assertEqual(depth_array.shape, (9, 480, 640))
        self.assertEqual(depth_array.dtype, np.dtype("uint16"))
        self.assertEqual(int(depth_array[0].max()), 65535)
        for index in range(9):
            np.testing.assert_array_equal(depth_array[index], depth)
        self.assertEqual(self.kine.calls, [])

    def test_rejected_camera_timestamp_does_not_encode_an_extra_frame(self):
        writer = self.writer()
        image = np.full((480, 640, 3), 80, dtype=np.uint8)
        valid = Record("cameras/camera_0/rgb", self.start + 10, 1, {"source_time_ms": 1.})
        writer.write_rgb("camera_0", image, valid)
        with self.assertRaisesRegex(ValueError, "Non-increasing"):
            writer.write_rgb("camera_0", image, valid)
        writer.close(self.start + 20, status="failed", reason="camera clock repeated")
        with av.open(str(self.path / "camera_0.mp4")) as container:
            frames = list(container.decode(video=0))
        raw = zarr.open_group(str(self.path / "raw.zarr"), mode="r")
        self.assertEqual(len(frames), len(raw["cameras/camera_0/rgb/time_ns"]))
        self.assertEqual(len(frames), 1)
        document = json.loads((self.path / "episode.json").read_text())
        self.assertEqual(document["status"], "failed")
        self.assertEqual(document["reason"], "camera clock repeated")

    def test_rgb_encoders_can_be_prepared_before_the_first_live_frame(self):
        writer = self.writer()
        writer.prepare_rgb(("camera_0", "camera_1", "camera_2"))
        writer.prepare_rgb(("camera_0", "camera_1", "camera_2"))
        self.assertEqual(set(writer.videos), {"camera_0", "camera_1", "camera_2"})
        image = np.full((480, 640, 3), 80, dtype=np.uint8)
        record = Record("cameras/camera_0/rgb", self.start + 1, 1,
                        {"source_time_ms": 1.})
        writer.write_rgb("camera_0", image, record)
        writer.close(self.start + 2, status="failed", reason="test")
        with av.open(str(self.path / "camera_0.mp4")) as container:
            self.assertEqual(len(list(container.decode(video=0))), 1)

    def test_encoder_or_numeric_flush_failure_marks_episode_failed(self):
        for component in ("encoder", "arrays"):
            with self.subTest(component=component), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "episode"
                writer = EpisodeWriter(path, self.start, self.metadata, self.kine)
                writer.append(Record("hands/left", self.start + 1, 1, {"joint_pos": (.1,) * 20}))
                if component == "encoder":
                    writer.videos["camera_0"] = Mock()
                    writer.videos["camera_0"].close.side_effect = OSError("encoder output failed")
                    with self.assertRaisesRegex(OSError, "encoder output failed"):
                        writer.close(self.start + 100)
                    raw = zarr.open_group(str(path / "raw.zarr"), mode="r")
                    self.assertEqual(raw["hands/left/joint_pos"].shape, (1, 20))
                else:
                    with patch.object(writer, "flush", side_effect=OSError("disk full")):
                        with self.assertRaisesRegex(OSError, "disk full"):
                            writer.close(self.start + 100)
                document = json.loads((path / "episode.json").read_text())
                self.assertEqual(document["status"], "failed")
                self.assertIn("failed" if component == "encoder" else "disk full", document["reason"])
                self.assertEqual(document["counts"], {"hands/left": 1})


if __name__ == "__main__":
    unittest.main()
