"""Phase markers persist independently of streams and map to retained output frames."""

from pathlib import Path
from contextlib import redirect_stderr
import io
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import zarr

from bimanual_teleop.recording.config import RecordingConfig, load_config, validate_mark_key
from bimanual_teleop.recording.convert import convert_recordings
from bimanual_teleop.recording.episodes import read_manifest, read_phase_markers
from bimanual_teleop.recording.spool import RawEpisodeWriter
from bimanual_teleop.recording.storage import write_json
from bimanual_teleop.recording.ui import recording_help
from tests.support.recording import START_NS, make_episode


class MarkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_config_defaults_case_normalization_and_control_conflicts(self):
        self.assertEqual(load_config().mark_key, "b")
        path = self.root / "recording.yaml"
        path.write_text("cameras: [a, b, c]\ncontrols: {mark_key: M}\n")
        config = load_config(path)
        self.assertEqual(config.mark_key, "m")
        self.assertIn("M 边界标记", recording_help(config))
        for key in ("S", "X", "Q", "C"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "distinct"):
                RecordingConfig(("a", "b", "c"), mark_key=key)
        for key in ("", "bb", " ", "\n"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "one character"):
                RecordingConfig(("a", "b", "c"), mark_key=key)
        for key, options in (("q", {}), ("\x03", {}), ("\x04", {}),
                             ("b", {"toggle_engagement_key": "B"}), ("h", {})):
            with self.subTest(key=key, options=options), self.assertRaisesRegex(ValueError, "conflicts"):
                validate_mark_key(key, **options)
        validate_mark_key("h", home_enabled=False)

    def test_cli_rejects_conflicting_marker_key_before_camera_or_motion_start(self):
        from bimanual_teleop.cli import teleop_quest_tianji as cli

        for name in ("toggle_engagement_key", "ready_pose_key"):
            settings = cli.load_config(cli.DEFAULT_CONFIG)
            settings["controls"][name] = "b"
            output = io.StringIO()
            with self.subTest(name=name), \
                    patch.object(cli, "configure_runtime_logging"), \
                    patch.object(cli, "load_config", return_value=settings), \
                    patch("bimanual_teleop.control.hand.follow.preflight"), \
                    patch("bimanual_teleop.devices.wuji.config.load_config", return_value={}), \
                    patch("bimanual_teleop.recording.recorder.Recorder") as recorder, \
                    patch.object(cli, "confirm_motion") as confirm, \
                    patch.object(cli, "prepare_initial_pose") as prepare, \
                    patch.object(cli, "create_runtime") as create, redirect_stderr(output):
                self.assertEqual(cli.main(["--record"]), 1)
                recorder.assert_not_called()
                confirm.assert_not_called()
                prepare.assert_not_called()
                create.assert_not_called()
            self.assertIn("conflicts", output.getvalue())

    def test_online_marker_is_immediately_published_without_stream_counts(self):
        path = self.root / "episode"
        writer = RawEpisodeWriter(path, START_NS, {})
        self.addCleanup(writer.close, START_NS + 100)
        for index, stamp in enumerate((START_NS, START_NS + 1), 1):
            self.assertEqual(writer.add_phase_marker(stamp), {"index": index, "time_ns": stamp})
            self.assertEqual(read_manifest(path)["phase_markers"], writer.document["phase_markers"])
        self.assertEqual(writer.counts, {})
        self.assertEqual(writer.last_ns, {})
        with patch("bimanual_teleop.recording.storage.write_json", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                writer.add_phase_marker(START_NS + 2)
        self.assertEqual(len(writer.document["phase_markers"]), 2)
        self.assertEqual(read_manifest(path)["phase_markers"], writer.document["phase_markers"])
        writer.close(START_NS + 100)
        self.assertEqual(len(read_manifest(path)["phase_markers"]), 2)

    def test_optional_markers_are_validated_without_accepting_bad_times_or_indices(self):
        self.assertEqual(read_phase_markers({}), [])
        for markers in (None, {}, [{"index": 2, "time_ns": START_NS}],
                        [{"index": True, "time_ns": START_NS}],
                        [{"index": 1, "time_ns": float(START_NS)}],
                        [{"index": 1, "time_ns": START_NS - 1}],
                        [{"index": 1, "time_ns": START_NS + 100}],
                        [{"index": 1, "time_ns": START_NS + 50}, {"index": 2, "time_ns": START_NS + 40}]):
            with self.subTest(markers=markers), self.assertRaisesRegex(ValueError, "phase_markers"):
                read_phase_markers({"start_ns": START_NS, "end_ns": START_NS + 100, "phase_markers": markers})


class MarkerConversionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def episode(self, *, marks=(), **kwargs):
        path, raw = make_episode(self.root / "raw", **kwargs)
        document = read_manifest(path)
        document["phase_markers"] = [{"index": i, "time_ns": START_NS + round(ms * 1e6)}
                                     for i, ms in enumerate(marks, 1)]
        write_json(path / "episode.json", document)
        return path, raw

    def convert(self, name="out.zarr", **config):
        output = self.root / name
        report = convert_recordings(self.root / "raw", output, action_space="eef", conversion_config=config)
        dry = convert_recordings(self.root / "raw", action_space="eef", conversion_config=config, dry_run=True)
        self.assertEqual(dry["episodes"], report["episodes"])
        return zarr.open_group(str(output), mode="r"), report

    def test_strict_maps_exact_between_frame_and_trimmed_edge_markers(self):
        self.episode(marks=(0, 10, 11, 20, 111, 150))
        data, report = self.convert(mode="strict")
        markers = data["meta"].attrs["phase_markers"]
        self.assertEqual([m["output_boundary"] for m in markers], [0, 0, 1, 1, 4, 4])
        self.assertEqual([m["recording_time_ns"] for m in markers],
                         [START_NS + ms * 1_000_000 for ms in (0, 10, 11, 20, 111, 150)])
        self.assertEqual(markers, report["episodes"][0]["phase_markers"])
        self.assertEqual(data.attrs["schema_version"], 2)
        np.testing.assert_array_equal(data["meta/segment_ends"][:], [4])

    def test_strict_maps_to_retained_frames_after_rejection_and_short_run_filtering(self):
        _, raw = self.episode(main=(10, 40, 70, 100, 130, 160),
                              other={1: (10, 40, 100, 130, 160)}, marks=(0, 70, 95, 130, 199))
        raw["arms/left/wrench"][13, 0] = np.nan
        data, _ = self.convert(mode="strict")
        self.assertEqual([m["output_boundary"] for m in data["meta"].attrs["phase_markers"]], [0, 2, 2, 3, 4])
        data, _ = self.convert("short.zarr", mode="strict", min_segment_frames=2)
        self.assertEqual([m["output_boundary"] for m in data["meta"].attrs["phase_markers"]], [0, 2, 2, 2, 2])

    def test_repair_grid_preserves_markers_at_reused_images(self):
        times = np.arange(9) * 1000 / 30
        self.episode(main=times, other={1: times[[0, 1, 3, 4, 7, 8]]}, marks=(62, 67))
        data, _ = self.convert()
        self.assertTrue(data["meta/camera_reused"][2, 1])
        self.assertEqual([m["output_boundary"] for m in data["meta"].attrs["phase_markers"]], [2, 3])

    def test_repair_maps_gaps_and_dropped_short_runs(self):
        times = np.arange(9) * 1000 / 30
        self.episode(main=times, other={1: times[[0, 1, 5, 6, 7, 8]]}, marks=(20, 80, 200, 300))
        data, _ = self.convert()
        self.assertEqual([m["output_boundary"] for m in data["meta"].attrs["phase_markers"]], [1, 2, 3, 6])
        data, _ = self.convert("short.zarr", min_segment_frames=3)
        self.assertEqual([m["output_boundary"] for m in data["meta"].attrs["phase_markers"]], [0, 0, 1, 4])

    def test_pause_compression_maps_recording_time_instead_of_training_time(self):
        times = np.r_[np.arange(4) * 1000 / 30, 200 + np.arange(4) * 1000 / 30]
        _, raw = self.episode(main=times, marks=(180, 220))
        for camera in range(3):
            raw[f"cameras/camera_{camera}/rgb/source_time_ms"][4:] += 2000
        separate, _ = self.convert()
        merged, _ = self.convert("merged.zarr", pause_reviews={
            "episode_000000": {"4": {"merge": True, "reason": "Synthetic continuous seam"}}})
        self.assertEqual(merged["meta"].attrs["phase_markers"], separate["meta"].attrs["phase_markers"])
        self.assertEqual([m["output_boundary"] for m in merged["meta"].attrs["phase_markers"]], [4, 5])
        self.assertLess(merged["data/timestamp"][5], .220)
        self.assertEqual(len(merged["meta/segment_ends"]), 1)

    def test_merged_episodes_use_global_offsets_and_skipped_markers_remain_in_report(self):
        self.episode(marks=(40,))
        self.episode(name="episode_000001", status="failed", marks=(50,))
        rejected, _ = self.episode(name="episode_000002", marks=(50,))
        write_json(rejected / "review.json", {"decision": "reject"})
        self.episode(name="episode_000003", commands=(500,), marks=(50,))
        self.episode(name="episode_000004", marks=(50,))
        data, report = self.convert(mode="strict")
        markers = data["meta"].attrs["phase_markers"]
        self.assertEqual([m["source_episode"] for m in markers], ["episode_000000", "episode_000004"])
        self.assertEqual([m["output_boundary"] for m in markers], [1, 6])
        for item in report["episodes"][1:4]:
            self.assertIsNone(item["phase_markers"][0]["output_boundary"])
            self.assertEqual(item["phase_markers"][0]["recording_time_ns"], START_NS + 50_000_000)

    def test_historical_episode_without_markers_exports_normally(self):
        make_episode(self.root / "raw")
        data, report = self.convert(mode="strict")
        self.assertEqual(data["meta"].attrs["phase_markers"], [])
        self.assertEqual(report["episodes"][0]["phase_markers"], [])

    def test_invalid_markers_do_not_publish_output_and_dry_run_reports_error(self):
        path, _ = self.episode()
        document = read_manifest(path)
        document["phase_markers"] = [{"index": 2, "time_ns": START_NS}]
        write_json(path / "episode.json", document)
        with self.assertRaisesRegex(ValueError, "phase_markers"):
            self.convert(mode="strict")
        self.assertFalse((self.root / "out.zarr").exists())
        report = convert_recordings(self.root / "raw", action_space="eef", dry_run=True)
        self.assertIn("phase_markers", report["episodes"][0]["error"])
        self.assertFalse(list(self.root.glob(".*.converting-*")))
