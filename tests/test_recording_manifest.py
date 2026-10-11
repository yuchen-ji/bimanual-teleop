"""Session metadata, portable episode indices and offline deletion synchronization."""

from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import yaml

from bimanual_teleop.recording.config import DatasetConfig, RecordingConfig, load_config
from bimanual_teleop.recording.dataset_manifest import (
    ensure_dataset_manifest, find_dataset_sessions, read_dataset_manifest, sync_dataset_episodes)
from bimanual_teleop.recording.episodes import find_episodes
from bimanual_teleop.recording.finalize import finalize_recordings
from bimanual_teleop.recording.storage import write_json


class DatasetConfigTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "recording.yaml"

    def load(self, dataset):
        self.path.write_text(yaml.safe_dump({"cameras": ["top", "left", "right"], "dataset": dataset}),
                             encoding="utf-8")
        return load_config(self.path)

    def test_default_and_legacy_configs_keep_the_rig_metadata(self):
        config = load_config()
        self.assertEqual(config.dataset.task, "Pick and place the car seat rails")
        self.assertEqual(config.dataset.camera_roles,
                         {"camera_0": "overhead", "camera_1": "left_arm", "camera_2": "right_wrist"})
        self.assertEqual([device.model for device in config.dataset.collection_devices],
                         ["Wuji-glove", "Quest-3s"])
        self.path.write_text("cameras: [top, left, right]\n", encoding="utf-8")
        self.assertEqual(load_config(self.path).dataset, DatasetConfig())

    def test_custom_metadata_round_trips_through_spawn_pickle(self):
        config = self.load({
            "task": "把导轨放到台面上",
            "camera_roles": {"camera_0": "top", "camera_1": "left", "camera_2": "right"},
            "collection_devices": [{"model": "test-glove", "role": "finger_motion_input", "sides": ["left"]}],
        })
        restored = pickle.loads(pickle.dumps(config))
        self.assertEqual(restored, config)
        self.assertEqual(restored.dataset.collection_devices[0].sides, ("left",))
        self.assertEqual(restored.dataset.task, "把导轨放到台面上")

    def test_invalid_dataset_fields_are_rejected(self):
        invalid = [None, [], {"unknown": "value"}, {"task": " "}, {"task": 1},
                   {"camera_roles": {"camera_0": "overhead"}},
                   {"camera_roles": {"camera_0": "top", "camera_1": "left", "camera_2": ""}},
                   {"collection_devices": []}, {"collection_devices": {}},
                   {"collection_devices": [{"model": "glove", "role": "finger"}]},
                   {"collection_devices": [{"model": "", "role": "finger", "sides": ["left"]}]},
                   {"collection_devices": [{"model": "glove", "role": "finger", "sides": "left"}]},
                   {"collection_devices": [{"model": "glove", "role": "finger", "sides": ["center"]}]},
                   {"collection_devices": [{"model": "glove", "role": "finger", "sides": ["left", "left"]}]},
                   {"collection_devices": [{"model": "vr", "role": "wrist", "sides": ["right"],
                                              "components": ["controller", "controller"]}]},
                   {"collection_devices": [{"model": "vr", "role": "wrist", "sides": ["right"],
                                              "extra": True}]}]
        for dataset in invalid:
            with self.subTest(dataset=dataset), self.assertRaises(ValueError):
                self.load(dataset)


class DatasetManifestTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.session = self.root / "20261011_120000_abcd1234"
        self.config = RecordingConfig(("00123", "00456", "00789"))
        self.cameras = {f"camera_{index}": {"serial": serial, "rgb_intrinsics": {"fx": 100.}}
                        for index, serial in enumerate(self.config.cameras)}
        self.created_at = "2026-10-11T12:00:00+08:00"

    def ensure(self, session=None, config=None, cameras=None):
        return ensure_dataset_manifest(self.session if session is None else session,
            self.config if config is None else config, self.cameras if cameras is None else cameras,
            created_at=self.created_at)

    def episode(self, name, status="captured", session=None):
        episode = (self.session if session is None else session) / name
        episode.mkdir(parents=True)
        write_json(episode / "episode.json", {"status": status, "metadata": {"calibration": "per-episode"}})
        return episode

    def test_manifest_describes_started_cameras_and_input_devices(self):
        self.ensure()
        document = read_dataset_manifest(self.session)
        self.assertEqual(document["dataset_id"], self.session.name)
        self.assertEqual(document["robot"], {"name": "Tianji"})
        self.assertIsNotNone(datetime.fromisoformat(document["created_at"]).tzinfo)
        self.assertEqual(document["cameras"]["camera_0"],
                         {"role": "overhead", "serial": "00123", "streams": ["rgb", "depth"]})
        self.assertEqual(document["cameras"]["camera_1"]["role"], "left_arm")
        self.assertEqual(document["cameras"]["camera_2"]["role"], "right_wrist")
        self.assertEqual(document["collection_devices"][0],
                         {"model": "Wuji-glove", "role": "finger_motion_input", "sides": ["left", "right"]})
        self.assertEqual(document["collection_devices"][1]["components"],
                         ["headset", "left_controller", "right_controller"])
        self.assertEqual(document["episodes"], [])
        self.assertNotIn("rgb_intrinsics", document["cameras"]["camera_0"])

    def test_disabled_depth_and_serial_mismatch(self):
        document = self.ensure(config=replace(self.config, main_depth=False))
        self.assertEqual(document["cameras"]["camera_0"]["streams"], ["rgb"])
        for cameras in ({}, {**self.cameras, "camera_1": {"serial": "another-camera"}}):
            with self.subTest(cameras=cameras), self.assertRaises(ValueError):
                self.ensure(cameras=cameras)
        self.assertEqual(read_dataset_manifest(self.session), document)

    def test_changed_task_or_device_mapping_cannot_overwrite_a_session(self):
        document = self.ensure()
        for config in (replace(self.config, dataset=replace(self.config.dataset, task="Another task")),
                       replace(self.config, main_depth=False),
                       replace(self.config, cameras=("00123", "00999", "00789"))):
            cameras = {f"camera_{index}": {"serial": serial} for index, serial in enumerate(config.cameras)}
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, "start a new"):
                self.ensure(config=config, cameras=cameras)
        self.assertEqual(read_dataset_manifest(self.session), document)

    def test_index_rebuild_is_sorted_idempotent_and_preserves_failures(self):
        self.ensure()
        for name, status in (("episode_000010", "discarded"), ("episode_000002", "failed"),
                             ("episode_000000", "captured"), (".episode_000003.finalizing-1", "complete")):
            self.episode(name, status)
        self.episode("raw_spool/inner", session=self.session / "episode_000000")
        self.assertTrue(sync_dataset_episodes(self.session))
        document = read_dataset_manifest(self.session)
        self.assertEqual(document["episodes"], [
            {"episode_id": name, "path": name}
            for name in ("episode_000000", "episode_000002", "episode_000010")])
        with patch("bimanual_teleop.recording.dataset_manifest.write_json") as write:
            self.ensure()
            sync_dataset_episodes(self.session)
            write.assert_not_called()
        self.assertEqual(document["created_at"], self.created_at)

    def test_moved_session_preserves_identity_and_relative_paths(self):
        self.ensure()
        self.episode("episode_000000")
        sync_dataset_episodes(self.session)
        moved = self.root / "renamed-upload"
        self.session.rename(moved)
        document = self.ensure(session=moved)
        self.assertTrue(sync_dataset_episodes(moved))
        self.assertEqual(document["dataset_id"], self.session.name)
        self.assertEqual(document["created_at"], self.created_at)
        self.assertTrue((moved / document["episodes"][0]["path"] / "episode.json").is_file())

    def test_failed_atomic_replace_leaves_old_index_readable_and_retryable(self):
        previous = self.ensure()
        self.episode("episode_000000")
        with patch("bimanual_teleop.recording.storage.Path.replace", side_effect=OSError("disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"):
                sync_dataset_episodes(self.session)
        self.assertEqual(read_dataset_manifest(self.session), previous)
        sync_dataset_episodes(self.session)
        self.assertEqual(read_dataset_manifest(self.session)["episodes"],
                         [{"episode_id": "episode_000000", "path": "episode_000000"}])

    def test_unsupported_manifest_is_not_rewritten(self):
        document = self.ensure()
        document["schema_version"] = 2
        write_json(self.session / "dataset.json", document)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            sync_dataset_episodes(self.session)
        self.assertEqual(json.loads((self.session / "dataset.json").read_text()), document)

    def test_legacy_sessions_still_discover_episodes_without_backfill(self):
        episode = self.episode("episode_000000", "complete")
        self.assertFalse(sync_dataset_episodes(self.session))
        self.assertEqual(find_episodes(self.root), [episode])
        report = finalize_recordings(self.root)
        self.assertEqual((report["complete"], report["failed"]), (1, 0))
        self.assertFalse((self.session / "dataset.json").exists())

    def test_finalize_refreshes_each_session_and_keeps_number_gaps(self):
        self.ensure()
        other = self.root / "other-session"
        self.ensure(session=other)
        deleted = self.episode("episode_000000", "discarded")
        self.episode("episode_000002", "complete")
        self.episode("episode_000003", "failed")
        self.episode("episode_000001", "complete", session=other)
        sync_dataset_episodes(self.session)
        sync_dataset_episodes(other)
        report = finalize_recordings(self.root)
        self.assertEqual((report["complete"], report["discarded"], report["skipped"], report["failed"]),
                         (2, 1, 1, 0))
        self.assertFalse(deleted.exists())
        self.assertEqual(read_dataset_manifest(self.session)["episodes"], [
            {"episode_id": name, "path": name} for name in ("episode_000002", "episode_000003")])
        self.assertEqual(read_dataset_manifest(other)["episodes"],
                         [{"episode_id": "episode_000001", "path": "episode_000001"}])

    def test_deleting_last_episode_can_retry_a_failed_index_update(self):
        self.ensure()
        deleted = self.episode("episode_000000", "discarded")
        sync_dataset_episodes(self.session)
        with patch("bimanual_teleop.recording.finalize.sync_dataset_episodes",
                   side_effect=OSError("index disk failure")):
            report = finalize_recordings(self.root)
        self.assertFalse(deleted.exists())
        self.assertEqual((report["discarded"], report["failed"]), (1, 1))
        self.assertIn("index disk failure", report["errors"][0][1])
        self.assertEqual(len(read_dataset_manifest(self.session)["episodes"]), 1)
        retry = finalize_recordings(self.root)
        self.assertEqual(retry["failed"], 0)
        self.assertEqual(read_dataset_manifest(self.session)["episodes"], [])

    def test_finalize_single_episode_updates_its_session(self):
        self.ensure()
        deleted = self.episode("episode_000000", "discarded")
        self.episode("episode_000001", "complete")
        sync_dataset_episodes(self.session)
        self.assertEqual(find_dataset_sessions(deleted), [self.session])
        report = finalize_recordings(deleted)
        self.assertEqual(report["discarded"], 1)
        self.assertEqual(read_dataset_manifest(self.session)["episodes"],
                         [{"episode_id": "episode_000001", "path": "episode_000001"}])


if __name__ == "__main__":
    unittest.main()
