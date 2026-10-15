"""Real Parquet/video exports: timing, actions, portable raw data and v3 loading."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from bimanual_teleop.recording.lerobot_export import convert_lerobot
from tests.support.recording import TEST_METADATA, make_episode, write_video


DEPENDENCIES = all(importlib.util.find_spec(name) for name in ("av", "zarr", "datasets", "pyarrow"))


@unittest.skipUnless(DEPENDENCIES, "install recording and lerobot-export dependencies")
class LeRobotExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.input = self.root / "input"
        self.output = self.root / "lerobot"

    def convert(self, **kwargs):
        kwargs.setdefault("action_space", "joint")
        kwargs.setdefault("task", "Pick and place the car seat rails")
        return convert_lerobot(self.input, self.output, **kwargs)

    def table(self):
        import pyarrow.parquet as pq

        return pq.read_table(sorted(self.output.glob("data/*/*.parquet"))).to_pydict()

    def info(self):
        return json.loads((self.output / "meta/info.json").read_text())

    def tianji(self):
        return json.loads((self.output / "meta/tianji.json").read_text())

    def test_measured_state_recorded_joint_targets_and_fixed_rate_timestamps(self):
        import av
        import pyarrow.parquet as pq

        make_episode(self.input)
        report = self.convert()
        table = self.table()
        info = self.info()
        self.assertEqual((info["codebase_version"], info["fps"], info["total_frames"]), ("v3.0", 30, 4))
        self.assertEqual(info["features"]["observation.state"]["shape"], [54])
        self.assertEqual(info["features"]["action"]["shape"], [54])
        np.testing.assert_allclose(table["timestamp"], np.arange(4) / 30, atol=1e-8)
        np.testing.assert_array_equal(table["frame_index"], np.arange(4))
        state, action = np.asarray(table["observation.state"]), np.asarray(table["action"])
        np.testing.assert_allclose(state[:, 0], [.01, .043333333, .076666666, .11], atol=1e-6)
        np.testing.assert_allclose(state[0, 14:34], np.arange(20) + .01, atol=1e-5)
        np.testing.assert_array_equal(action[0, :7], 10 + np.arange(7))
        np.testing.assert_array_equal(action[0, 7:14], 110 + np.arange(7))
        np.testing.assert_array_equal(action[0, 14:34], 20 + np.arange(20))
        self.assertEqual(report["lerobot_episodes"], 1)
        self.assertIn(b"huggingface", pq.read_schema(next(self.output.glob("data/*/*.parquet"))).metadata)
        for camera in range(3):
            key = f"observation.images.camera_{camera}"
            path = self.output / info["video_path"].format(video_key=key, chunk_index=0, file_index=0)
            with av.open(str(path)) as container:
                frames = list(container.decode(video=0))
                self.assertEqual(len(frames), 4)
                np.testing.assert_allclose(frames[0].to_ndarray(format="rgb24")[0, 0],
                                           [20, camera * 50, 0], atol=4)
        stats = json.loads((self.output / "meta/stats.json").read_text())
        np.testing.assert_allclose(stats["action"]["mean"], action.mean(axis=0))
        self.assertEqual(stats["action"]["count"], [4])

    def test_eef_actions_use_recorded_controller_pose_targets(self):
        make_episode(self.input)
        self.convert(action_space="eef")
        table = self.table()
        self.assertEqual(self.info()["features"]["action"]["shape"], [52])
        np.testing.assert_allclose(np.asarray(table["action"])[:, 0], [5, 5.04, 5.06, 5.1], atol=1e-6)
        self.assertEqual(self.info()["features"]["observation.state"]["shape"], [54])

    def test_native_raw_and_depth_are_exact_and_input_is_unchanged(self):
        import zarr

        source, _ = make_episode(self.input, depth=True)
        before = {p.relative_to(self.input): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in self.input.rglob("*") if p.is_file()}
        self.convert()
        record = self.tianji()["sources"][0]
        raw_path = self.output / record["raw_path"]
        raw = zarr.open_group(str(raw_path), mode="r")
        self.assertEqual(raw["cameras/camera_0/depth/image"].dtype, np.dtype("uint16"))
        self.assertEqual(record["native_depth_frames"], 4)
        np.testing.assert_array_equal(self.table()["source.depth_row"], [0, 1, 2, 3])
        for path in (source / "raw.zarr").rglob("*"):
            if path.is_file():
                self.assertEqual(path.read_bytes(), (raw_path / path.relative_to(source / "raw.zarr")).read_bytes())
        self.assertEqual((source / "episode.json").read_bytes(),
                         (self.output / record["episode_manifest_path"]).read_bytes())
        after = {p.relative_to(self.input): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in self.input.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        from bimanual_teleop.recording.episodes import find_episodes
        self.assertEqual(find_episodes(self.output), [])

    def test_missing_depth_does_not_discard_rgb_samples(self):
        _, raw = make_episode(self.input, depth=True)
        raw["cameras/camera_0/depth/time_ns"][2] += 25_000_000
        report = self.convert()
        self.assertEqual(report["output_frames"], 4)
        self.assertEqual(self.table()["source.depth_row"][2], -1)

    def test_depth_after_pause_uses_recording_clock_and_keeps_pause_bounds(self):
        times = 10 + np.arange(8) * 1000 / 30
        sources = times.copy()
        sources[4:] += 1000  # Wall time advanced while recording time was paused.
        make_episode(self.input, main=times, depth=True, camera_sources={i: sources for i in range(3)})
        self.convert()
        table = self.table()
        blocks = np.asarray(table["source.recording_block"])
        depth_rows = np.asarray(table["source.depth_row"])
        self.assertTrue((depth_rows[blocks == 1] >= 4).any())
        self.assertTrue((depth_rows[blocks == 0] < 4).all())
        self.assertGreater(np.asarray(table["source.time_ns"])[blocks == 1][0],
                           np.asarray(table["source.recording_time_ns"])[blocks == 1][0] + 900_000_000)

    def test_task_and_camera_roles_from_dataset_manifest(self):
        cameras = {f"camera_{i}": {"serial": str(i)} for i in range(3)}
        make_episode(self.input, metadata={**TEST_METADATA, "cameras": cameras})
        manifest = {"schema_version": 1, "format": "tianji_raw_dataset", "dataset_id": "source-session",
                    "task": "Place a rail", "cameras": {key: {**value, "role": f"role-{key}"}
                                                        for key, value in cameras.items()}}
        (self.input / "dataset.json").write_text(json.dumps(manifest))
        self.convert(task=None)
        source = self.tianji()["sources"][0]
        self.assertEqual(source["task"], "Place a rail")
        self.assertEqual(source["source_dataset_id"], "source-session")
        self.assertEqual(source["dataset_manifest"], manifest)

    def test_missing_task_requires_an_explicit_description(self):
        make_episode(self.input)
        with self.assertRaisesRegex(ValueError, "provide --task"):
            self.convert(task=None)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".*.lerobot-*")))

    def test_gaps_split_episodes_and_shards_keep_correct_video_offsets(self):
        import av
        import pyarrow.parquet as pq

        make_episode(self.input, main=(10, 40, 100, 130))
        make_episode(self.input, name="episode_000004")
        report = self.convert(conversion_config={"mode": "repair", "max_missing_camera_frames": 0},
                              max_frames_per_shard=4)
        self.assertGreater(report["lerobot_episodes"], 2)
        rows = pq.read_table(self.output / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
        table = self.table()
        for row in rows:
            start, end = row["dataset_from_index"], row["dataset_to_index"]
            np.testing.assert_allclose(table["timestamp"][start:end], np.arange(end - start) / 30, atol=1e-8)
            np.testing.assert_array_equal(table["frame_index"][start:end], np.arange(end - start))
        self.assertGreater(len(list(self.output.glob("data/*/*.parquet"))), 1)
        for path in self.output.glob("videos/observation.images.camera_0/*/*.mp4"):
            chunk, file = int(path.parent.name.split("-")[1]), int(path.stem.split("-")[1])
            selected = [row for row in rows if row["data/chunk_index"] == chunk and row["data/file_index"] == file]
            with av.open(str(path)) as container:
                self.assertEqual(len(list(container.decode(video=0))), sum(row["length"] for row in selected))
            self.assertEqual(selected[0]["videos/observation.images.camera_0/from_timestamp"], 0.)

    def test_two_tasks_share_a_shard_and_keep_distinct_source_identities(self):
        import pyarrow.parquet as pq

        cameras = {f"camera_{i}": {"serial": str(i)} for i in range(3)}
        for session, task, name in (("session-a", "Task A", "episode_000001"),
                                    ("session-b", "Task B", "episode_000004")):
            directory = self.input / session
            _, raw = make_episode(directory, name=name, metadata={**TEST_METADATA, "cameras": cameras})
            if session == "session-b":
                raw["arm_commands/left/joint_pos"][:] += 2
            (directory / "dataset.json").write_text(json.dumps({
                "schema_version": 1, "format": "tianji_raw_dataset", "dataset_id": session,
                "task": task, "cameras": cameras}))
        self.convert(task=None)
        self.assertEqual(self.info()["total_tasks"], 2)
        np.testing.assert_array_equal(self.table()["task_index"], [0] * 4 + [1] * 4)
        rows = pq.read_table(self.output / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
        self.assertEqual(rows[1]["videos/observation.images.camera_0/from_timestamp"], 4 / 30)
        self.assertEqual([s["source_episode_id"] for s in self.tianji()["sources"]],
                         ["episode_000001", "episode_000004"])
        actions = np.asarray(self.table()["action"])
        stats = json.loads((self.output / "meta/stats.json").read_text())["action"]
        np.testing.assert_allclose(stats["mean"], actions.mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(stats["std"], actions.std(axis=0), atol=1e-12)

    def test_dry_run_and_no_raw_option(self):
        make_episode(self.input, depth=True)
        report = self.convert(dry_run=True)
        self.assertFalse(self.output.exists())
        self.assertEqual(report["lerobot_episodes"], 1)
        actual = self.convert(include_raw=False)
        self.assertEqual(actual["output_frames"], report["output_frames"])
        self.assertFalse((self.output / "raw").exists())
        np.testing.assert_array_equal(self.table()["source.depth_row"], [-1] * 4)

    def test_overwrites_and_output_inside_input_are_rejected(self):
        make_episode(self.input)
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.convert()
        self.assertEqual(marker.read_text(), "keep")
        with self.assertRaisesRegex(ValueError, "outside"):
            convert_lerobot(self.input, self.input / "output", action_space="joint", task="Task")

    def test_frame_count_failure_never_publishes_partial_output(self):
        episode, _ = make_episode(self.input)
        write_video(episode / "camera_2.mp4", 3, 2)
        with self.assertRaisesRegex(ValueError, "decoded 3 frames"):
            self.convert()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".*.lerobot-*")))

    def test_strict_or_fractional_rate_is_rejected(self):
        make_episode(self.input)
        with self.assertRaisesRegex(ValueError, "fixed-rate"):
            self.convert(conversion_config={"mode": "strict"})
        with self.assertRaisesRegex(ValueError, "integer"):
            self.convert(fps=29.97)

    @unittest.skipUnless(importlib.util.find_spec("lerobot"), "optional official LeRobot reader not installed")
    def test_official_reader_loads_samples_and_clamps_temporal_windows_at_episode_boundaries(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        make_episode(self.input)
        make_episode(self.input, name="episode_000004")
        self.convert()
        dataset = LeRobotDataset("local/tianji", root=self.output, video_backend="pyav",
                                 delta_timestamps={"action": [-1 / 30, 0, 1 / 30]})
        self.assertEqual(len(dataset), 8)
        for index in (0, 3, 4, 7):
            sample = dataset[index]
            self.assertEqual(tuple(sample["observation.state"].shape), (54,))
            self.assertEqual(tuple(sample["action"].shape), (3, 54))
            self.assertEqual(tuple(sample["observation.images.camera_0"].shape), (3, 16, 16))
        np.testing.assert_array_equal(dataset[3]["action"][-1].numpy(), dataset[3]["action"][1].numpy())
        np.testing.assert_array_equal(dataset[4]["action"][0].numpy(), dataset[4]["action"][1].numpy())


if __name__ == "__main__":
    unittest.main()
