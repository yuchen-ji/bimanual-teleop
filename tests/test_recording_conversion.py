"""Strict-mode conversion tests with actual encoded video and asynchronous streams."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from bimanual_teleop.recording.convert import convert_recordings
from tests.support.recording import CountingProgress, START_NS, make_episode, write_video

STRICT = {"mode": "strict"}


@unittest.skipUnless(importlib.util.find_spec("av") and importlib.util.find_spec("zarr"),
                     "recording dependencies are not installed")
class RecordingConversionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root / "raw"
        self.output = self.root / "dataset.zarr"

    def convert(self, action_space="eef", **kwargs):
        import zarr

        kwargs.setdefault("conversion_config", STRICT)
        report = convert_recordings(self.input, self.output, action_space=action_space, **kwargs)
        return zarr.open_group(str(self.output), mode="r"), report

    def test_eef_actions_are_controller_goals_and_observations_are_interpolated(self):
        make_episode(self.input, depth=True)
        dataset, report = self.convert(include_depth=True)
        data = dataset["data"]
        np.testing.assert_allclose(data["timestamp"][:], [.010, .043, .077, .110])
        np.testing.assert_allclose(data["robot_joint"][:, 0], [.010, .043, .077, .110])
        np.testing.assert_allclose(data["robot_joint"][:, 7], [100.010, 100.043, 100.077, 100.110], atol=1e-5)
        np.testing.assert_allclose(data["robot_eef_pose"][:, 5], [.010, .043, .077, .110], atol=1e-6)
        self.assertEqual(data["action"].shape, (4, 52))
        np.testing.assert_allclose(data["action"][:, 0], [5., 5.040, 5.060, 5.100], atol=1e-6)
        np.testing.assert_allclose(data["action"][:, 6], [105., 105.040, 105.060, 105.100], atol=1e-5)
        np.testing.assert_allclose(data["action"][0, 12:32], 20 + np.arange(20))
        np.testing.assert_allclose(data["action"][0, 32:], 120 + np.arange(20))
        np.testing.assert_array_equal(data["camera_2"][1, 0, 0], [40, 100, 0])
        np.testing.assert_array_equal(data["camera_0_depth"][:, 0, 0], [100, 101, 102, 103])
        self.assertEqual(data["camera_0"].dtype, np.dtype("uint8"))
        self.assertEqual(data["robot_joint"].dtype, np.dtype("float32"))
        self.assertEqual(data["timestamp"].dtype, np.dtype("float64"))
        np.testing.assert_array_equal(dataset["meta/episode_ends"][:], [4])
        np.testing.assert_array_equal(dataset["meta/segment_ends"][:], [4])
        self.assertEqual(report["output_frames"], 4)
        self.assertEqual(dataset["meta"].attrs["quality_report"]["episodes"][0]["metadata"]["model"], "test-model")

    def test_joint_action_layout_and_failed_episodes_are_skipped(self):
        make_episode(self.input)
        failed = self.input / "episode_000001"
        failed.mkdir()
        (failed / "episode.json").write_text(json.dumps({"status": "failed"}))
        dataset, report = self.convert("joint")
        actions = dataset["data/action"]
        self.assertEqual(actions.shape, (4, 54))
        np.testing.assert_array_equal(actions[0, :7], 10 + np.arange(7))
        np.testing.assert_array_equal(actions[0, 7:14], 110 + np.arange(7))
        np.testing.assert_array_equal(actions[0, 14:34], 20 + np.arange(20))
        self.assertEqual(report["episodes"][1]["status"], "failed")

    def test_missing_camera_and_invalid_force_split_without_joining_gaps(self):
        _, raw = make_episode(self.input, main=(10, 40, 70, 100, 130, 160),
                              other={1: (10, 40, 100, 130, 160)})
        raw["arms/left/wrench"][13, 0] = np.nan  # Exactly at 130 ms.
        dataset, report = self.convert()
        np.testing.assert_allclose(dataset["data/timestamp"][:], [.01, .04, .10, .16])
        np.testing.assert_array_equal(dataset["meta/episode_ends"][:], [4])
        np.testing.assert_array_equal(dataset["meta/segment_ends"][:], [2, 3, 4])
        self.assertEqual(report["output_episodes"], 1)
        self.assertEqual(report["output_segments"], 3)
        self.assertEqual(report["episodes"][0]["invalid_reasons"]["camera_1_unmatched"], 1)
        self.assertEqual(dataset["meta"].attrs["segments"][1]["reference_frame_start"], 3)

    def test_large_main_camera_gap_starts_a_new_episode(self):
        make_episode(self.input, main=(10, 40, 120, 150))
        dataset, report = self.convert()
        np.testing.assert_array_equal(dataset["meta/episode_ends"][:], [4])
        np.testing.assert_array_equal(dataset["meta/segment_ends"][:], [2, 4])
        self.assertEqual(report["episodes"][0]["main_camera_gaps"], 1)

    def test_state_interpolation_never_bridges_more_than_50ms_or_extrapolates(self):
        make_episode(self.input, main=(0, 30, 60, 90), state=(0, 60), commands=(0, 30, 60, 90))
        dataset, _ = self.convert()
        np.testing.assert_allclose(dataset["data/timestamp"][:], [0., .06])
        np.testing.assert_array_equal(dataset["meta/episode_ends"][:], [2])
        np.testing.assert_array_equal(dataset["meta/segment_ends"][:], [1, 2])

    def test_commands_do_not_use_future_samples_or_hold_stale_values(self):
        make_episode(self.input, main=(10, 40, 80, 100), commands=(20, 100))
        dataset, _ = self.convert()
        np.testing.assert_allclose(dataset["data/timestamp"][:], [.04, .10])
        np.testing.assert_allclose(dataset["data/action"][:, 0], [5.02, 5.10], atol=1e-6)
        np.testing.assert_array_equal(dataset["meta/episode_ends"][:], [2])
        np.testing.assert_array_equal(dataset["meta/segment_ends"][:], [1, 2])

    def test_depth_gaps_are_ignored_by_default_and_checked_only_when_requested(self):
        import zarr

        _, raw = make_episode(self.input, main=(10, 40, 70, 100), depth=True)
        # Remove the depth frame near the 70 ms RGB frame.
        for key in ("time_ns", "sequence", "source_time_ms", "image"):
            array = raw[f"cameras/camera_0/depth/{key}"]
            kept = array[:][[0, 1, 3]]
            array.resize(kept.shape)
            array[:] = kept
        dataset, report = self.convert()
        self.assertNotIn("camera_0_depth", dataset["data"])
        self.assertNotIn("camera_0_depth_unmatched", report["episodes"][0]["invalid_reasons"])
        np.testing.assert_array_equal(dataset["meta/episode_ends"][:], [4])
        np.testing.assert_array_equal(dataset["meta/segment_ends"][:], [4])
        with_depth = self.root / "with_depth.zarr"
        report = convert_recordings(self.input, with_depth, action_space="eef", include_depth=True,
                                    conversion_config=STRICT)
        dataset = zarr.open_group(str(with_depth), mode="r")
        self.assertIn("camera_0_depth", dataset["data"])
        np.testing.assert_array_equal(dataset["meta/episode_ends"][:], [3])
        np.testing.assert_array_equal(dataset["meta/segment_ends"][:], [2, 3])
        self.assertEqual(report["episodes"][0]["invalid_reasons"]["camera_0_depth_unmatched"], 1)

    def test_multiple_demonstrations_keep_separate_episode_and_segment_boundaries(self):
        make_episode(self.input, main=(10, 40, 120, 150), depth=True)
        make_episode(self.input, name="episode_000001")
        progress = CountingProgress()
        dataset, report = self.convert(progress=progress)
        self.assertEqual((progress.episodes, progress.finished), (2, 2))
        self.assertEqual(progress.names, ["episode_000000", "episode_000001"])
        self.assertEqual(progress.totals, [12, 12])
        self.assertEqual(progress.advanced, progress.totals)
        np.testing.assert_array_equal(dataset["meta/episode_ends"][:], [4, 8])
        np.testing.assert_array_equal(dataset["meta/segment_ends"][:], [2, 4, 8])
        self.assertEqual(report["output_episodes"], 2)
        self.assertEqual(report["output_segments"], 3)
        self.assertEqual(dataset.attrs["schema_version"], 2)
        self.assertNotIn("camera_0_depth", dataset["data"])

    def test_including_depth_rejects_mixed_depth_recordings(self):
        make_episode(self.input, depth=True)
        make_episode(self.input, name="episode_000001")
        with self.assertRaisesRegex(ValueError, "consistently include or omit"):
            self.convert(include_depth=True)
        self.assertFalse(self.output.exists())

    def test_pose_interpolation_uses_shortest_rotation_path(self):
        _, raw = make_episode(self.input, main=(20,), state=(0, 40), commands=(0, 40))
        poses = raw["arms/left/eef_pose"][:]
        poses[:, 3:] = Rotation.from_euler("z", [170, -170], degrees=True).as_quat()
        raw["arms/left/eef_pose"][:] = poses
        dataset, _ = self.convert()
        self.assertAlmostEqual(abs(float(dataset["data/robot_eef_pose"][0, 5])), np.pi, places=6)

    def test_invalid_quaternion_is_excluded_without_repairing_it(self):
        _, raw = make_episode(self.input, main=(10, 40, 70))
        raw["arms/right/eef_pose"][4, 3:] = 0.
        dataset, report = self.convert()
        np.testing.assert_allclose(dataset["data/timestamp"][:], [.01, .07])
        self.assertEqual(report["episodes"][0]["invalid_reasons"]["right_robot_eef_pose_invalid_or_gap"], 1)

    def test_episode_bounds_are_half_open_and_reference_time_is_relative(self):
        path, _ = make_episode(self.input, main=(0, 10, 40, 70))
        manifest = path / "episode.json"
        descriptor = json.loads(manifest.read_text())
        descriptor.update(start_ns=START_NS + 10_000_000, end_ns=START_NS + 70_000_000)
        manifest.write_text(json.dumps(descriptor))
        # A command at 10 ms is necessary; a command before the episode is not reused.
        import zarr
        raw = zarr.open_group(str(path / "raw.zarr"), mode="a")
        for kind in ("arm_commands", "hand_commands"):
            for side in ("left", "right"):
                raw[f"{kind}/{side}/time_ns"][0] = START_NS + 10_000_000
        dataset, _ = self.convert()
        np.testing.assert_allclose(dataset["data/timestamp"][:], [0., .03])

    def test_frame_count_mismatch_fails_without_publishing_partial_output(self):
        path, _ = make_episode(self.input)
        write_video(path / "camera_2.mp4", 3, 2)
        with self.assertRaisesRegex(ValueError, "decoded 3 frames"):
            self.convert()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".*.converting-*")))

    def test_existing_output_is_never_overwritten(self):
        make_episode(self.input)
        self.output.mkdir()
        marker = self.output / "important.txt"
        marker.write_text("keep me")
        with self.assertRaises(FileExistsError):
            self.convert()
        self.assertEqual(marker.read_text(), "keep me")

    def test_nonmonotonic_timestamps_are_rejected(self):
        _, raw = make_episode(self.input)
        raw["hands/right/time_ns"][1] = raw["hands/right/time_ns"][0]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self.convert()
        self.assertFalse(self.output.exists())

    def test_cli_requires_action_space(self):
        from bimanual_teleop.cli.convert_recording import main
        from contextlib import redirect_stderr
        import io

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main(["--input", str(self.input), "--output", str(self.output)])
        self.assertEqual(caught.exception.code, 2)

    def test_cli_depth_is_opt_in_and_counts_demonstrations_separately(self):
        from bimanual_teleop.cli.convert_recording import main
        from contextlib import redirect_stdout
        from unittest.mock import ANY, patch
        import io

        report = {"episodes": [], "output_episodes": 1, "output_segments": 3, "output_frames": 4}
        for extra, expected in (([], False), (["--include-depth"], True)):
            with self.subTest(include_depth=expected):
                output = io.StringIO()
                with patch("bimanual_teleop.cli.convert_recording.convert_recordings", return_value=report) as convert:
                    with redirect_stdout(output):
                        result = main(["--input", str(self.input), "--output", str(self.output),
                                       "--action-space", "eef"] + extra)
                self.assertEqual(result, 0)
                convert.assert_called_once_with(self.input, self.output, action_space="eef",
                                                include_depth=expected, allow_mixed_metadata=False,
                                                dry_run=False, progress=ANY)
                self.assertIn("1 条原始演示、3 个连续片段", output.getvalue())

    def test_dry_run_reports_what_conversion_writes_without_decoding_or_writing(self):
        make_episode(self.input, main=(10, 40, 70, 100, 130, 160), other={1: (10, 40, 100, 130, 160)})
        make_episode(self.input, name="episode_000001", main=(10, 40, 120, 150))
        progress = CountingProgress()
        planned = convert_recordings(self.input, None, action_space="eef", conversion_config=STRICT,
                                     dry_run=True, progress=progress)
        self.assertFalse(self.output.exists())
        self.assertEqual(progress.totals, [])
        self.assertEqual(progress.finished, 2)
        _, report = self.convert()
        for key in ("output_episodes", "output_segments", "output_frames", "episodes"):
            self.assertEqual(planned[key], report[key], key)
        self.assertTrue(planned["dry_run"])

    def test_edge_trimming_is_reported_apart_from_interior_rejections(self):
        from bimanual_teleop.cli.convert_recording import summary_lines

        make_episode(self.input, main=(0, 30, 60, 90, 120, 150), other={1: (0, 30, 60, 120, 150)},
                     state=np.arange(5, 200, 10), commands=np.arange(0, 81, 20))
        _, report = self.convert()
        item = report["episodes"][0]
        self.assertEqual(item["edge_trimmed_frames"], {"start": 1, "end": 1})
        self.assertEqual(item["interior_invalid_frames"], 1)
        self.assertEqual(item["interior_invalid_reasons"], {"camera_1_unmatched": 1})
        self.assertEqual(item["invalid_reasons"]["left_robot_joint_invalid_or_gap"], 1)
        self.assertEqual(item["invalid_reasons"]["right_arm_command_invalid_or_stale"], 1)
        self.assertEqual(item["edge_trimmed_reasons"]["left_robot_joint_invalid_or_gap"], 1)
        self.assertEqual(item["edge_trimmed_reasons"]["right_arm_command_invalid_or_stale"], 1)
        self.assertNotIn("camera_1_unmatched", item["edge_trimmed_reasons"])
        text = "\n".join(summary_lines(report))
        self.assertIn("丢弃：首尾 1+1 帧（", text)
        self.assertIn("；中间 1 帧（camera_1 缺帧 1）", text)
        self.assertIn("写入 3 帧（利用率 50.0%）", text)

    def test_metadata_that_defines_data_must_match_unless_mixing_is_allowed(self):
        from tests.support.recording import TEST_METADATA

        for name, serial, user in (("episode_000000", "A", "alice"), ("episode_000001", "B", "bob")):
            make_episode(self.input, name=name, metadata={
                **TEST_METADATA, "cameras": {"camera_1": {"serial": serial}},
                "wuji_config": {"sdk_user_name": user, "control_hz": 120}})
        with self.assertRaisesRegex(ValueError, r"cameras\.camera_1\.serial"):
            self.convert()
        self.assertFalse(self.output.exists())
        _, report = self.convert(allow_mixed_metadata=True)
        self.assertNotIn("metadata_differences", report["episodes"][0])
        self.assertEqual(report["episodes"][1]["metadata_differences"], ["cameras.camera_1.serial"])

    def test_cli_dry_run_defaults_to_repair_and_needs_no_output(self):
        from bimanual_teleop.cli.convert_recording import main
        from contextlib import redirect_stdout
        import io

        make_episode(self.input)
        output = io.StringIO()
        with redirect_stdout(output):
            result = main(["--input", str(self.input), "--action-space", "joint", "--dry-run"])
        self.assertEqual(result, 0)
        self.assertIn("修复模式", output.getvalue())
        self.assertIn("episode_000000：参考 4 帧，写入 4 帧（100.0%）", output.getvalue())
        self.assertIn("修复：缺帧沿用前一张图 0 帧；放宽插值 0 帧", output.getvalue())
        self.assertIn("试运行：将导出 1 条原始演示", output.getvalue())
        self.assertFalse(any(self.root.glob("*.zarr")))


if __name__ == "__main__":
    unittest.main()
