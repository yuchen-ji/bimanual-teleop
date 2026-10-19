"""Quality levels in conversion reports, with actual encoded video."""

import copy
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.support.recording import TEST_METADATA, make_episode

TIMES = np.arange(30) * 1000 / 30


def _thresholds():
    from bimanual_teleop.recording.quality import load_thresholds

    thresholds = copy.deepcopy(load_thresholds())
    thresholds["episode"]["min_duration_s"] = {"warn": .5, "fail": .1}
    for limits in thresholds["streams"].values():
        limits["min_hz"] = {"warn": 40, "fail": 30}
    return thresholds


@unittest.skipUnless(importlib.util.find_spec("av") and importlib.util.find_spec("zarr"),
                     "recording dependencies are not installed")
class RecordingQualityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "raw"

    def plan(self, **kwargs):
        from bimanual_teleop.recording.convert import convert_recordings

        kwargs.setdefault("conversion_config", {"mode": "strict"})
        report = convert_recordings(self.root, action_space="eef", dry_run=True, quality_config=_thresholds(),
                                    **kwargs)
        return report, {item["source_episode"]: item for item in report["episodes"]}

    def messages(self, item):
        return [entry["message"] for entry in item["checks"] if entry["level"] != "OK"]

    def test_clean_episode_is_ok_and_dry_run_writes_nothing(self):
        from bimanual_teleop.recording.quality import summarize

        make_episode(self.root, main=TIMES)
        before = {path: path.stat().st_mtime_ns for path in self.root.rglob("*")}
        report, items = self.plan()
        item = items["episode_000000"]
        self.assertEqual(item["level"], "OK", self.messages(item))
        self.assertEqual(item["cameras"]["camera_1"]["dropped_frames"], 0)
        self.assertEqual(item["cross_camera"]["camera_2"]["unmatched_frames"], 0)
        self.assertEqual(summarize(report)["levels"], {"OK": 1})
        self.assertEqual(report["quality_thresholds"], _thresholds())
        self.assertEqual(before, {path: path.stat().st_mtime_ns for path in self.root.rglob("*")})

    def test_drops_and_gaps_are_reported_next_to_the_frames_they_cost(self):
        kept = np.delete(np.arange(30), 10)
        make_episode(self.root, main=TIMES, other={1: TIMES[kept], 2: TIMES[kept]},
                     camera_sequences={1: kept})
        _, items = self.plan()
        item = items["episode_000000"]
        self.assertEqual(item["level"], "WARN")
        camera_1, camera_2 = item["cameras"]["camera_1"], item["cameras"]["camera_2"]
        self.assertEqual((camera_1["dropped_frames"], camera_1["gaps"]), (1, 1))
        self.assertEqual((camera_2["dropped_frames"], camera_2["gaps"]), (0, 1))
        self.assertEqual(item["cross_camera"]["camera_1"]["unmatched_frames"], 1)
        self.assertIn("camera_2 断档 1 次", self.messages(item))
        self.assertEqual(item["interior_invalid_reasons"], {"camera_1_unmatched": 1, "camera_2_unmatched": 1})

    def test_pauses_are_found_once_for_both_modes_and_must_agree_across_cameras(self):
        from bimanual_teleop.recording.convert import convert_recordings

        sequence = np.r_[np.arange(15), np.arange(15, 30) + 60]
        sources = TIMES.copy()
        sources[15:] += 2000
        make_episode(self.root, main=TIMES, camera_sequences={camera: sequence for camera in range(3)},
                     camera_sources={camera: sources for camera in range(3)})
        make_episode(self.root, name="episode_000001", main=TIMES,
                     camera_sources={camera: sources for camera in range(2)})

        _, strict = self.plan()
        paused = strict["episode_000000"]
        self.assertEqual((paused["cameras"]["camera_0"]["pauses"], paused["cameras"]["camera_0"]["pause_after_rows"]),
                         (1, [15]))
        self.assertEqual(paused["cameras"]["camera_1"]["dropped_frames"], 0)
        self.assertEqual(paused["level"], "WARN")
        self.assertTrue(any("严格模式不会在暂停处切开" in message for message in self.messages(paused)))
        self.assertTrue(any("暂停数不一致" in message for message in self.messages(strict["episode_000001"])))
        self.assertEqual(strict["episode_000001"]["level"], "FAIL")

        _, repair = self.plan(conversion_config={})
        paused = repair["episode_000000"]
        self.assertEqual(paused["level"], "OK", self.messages(paused))
        self.assertEqual([pause["after_main_row"] for pause in paused["pauses"]], [15])
        mismatched = repair["episode_000001"]
        self.assertEqual(mismatched["level"], "FAIL")
        self.assertIn("camera pause counts differ", mismatched["error"])
        with self.assertRaisesRegex(ValueError, "camera pause counts differ"):
            convert_recordings(self.root, self.root.parent / "out.zarr", action_space="eef", conversion_config={})

    def test_slow_streams_statuses_and_metadata_in_one_report(self):
        stop = TIMES[-1] + 41
        make_episode(self.root, main=TIMES, stream_times={"hands/left": np.arange(0, stop, 40)})
        make_episode(self.root, name="episode_000001", main=TIMES,
                     metadata={**TEST_METADATA, "model_sha256": "other"})
        make_episode(self.root, name="episode_000002", status="failed")
        (self.root / "episode_000002" / "episode.json").write_text(
            json.dumps({"status": "failed", "reason": "遥操作退出"}))
        make_episode(self.root, name="episode_000003", status="captured")
        make_episode(self.root, name="episode_000004", main=TIMES)
        report, items = self.plan()
        slow = items["episode_000000"]
        self.assertEqual(slow["level"], "FAIL")
        self.assertIn("hands/left 26 Hz", self.messages(slow))
        self.assertIn("hands/left 最大间隔 40 ms", self.messages(slow))
        # The dry run records what would stop the conversion and keeps planning later episodes.
        mixed = items["episode_000001"]
        self.assertEqual((mixed["level"], mixed["metadata_differences"]), ("FAIL", ["model_sha256"]))
        self.assertIn("allow_mixed_metadata", mixed["error"])
        self.assertEqual(items["episode_000004"]["level"], "OK")
        self.assertEqual((items["episode_000002"]["level"], items["episode_000002"]["reason"]), ("SKIP", "遥操作退出"))
        self.assertEqual(items["episode_000003"]["level"], "WARN")

        from bimanual_teleop.recording.quality import summarize

        summary = summarize(report)
        self.assertEqual(summary["problems"]["hands/left/hz"], 1)
        self.assertEqual(list(summary["errors"]), ["episode_000001"])
        _, allowed = self.plan(allow_mixed_metadata=True)
        self.assertEqual(allowed["episode_000001"]["level"], "WARN")
        self.assertIn("model_sha256", self.messages(allowed["episode_000001"])[0])

    def test_thresholds_must_be_complete_and_ordered(self):
        from bimanual_teleop.recording.quality import load_thresholds

        thresholds = _thresholds()
        self.assertEqual(load_thresholds(thresholds), thresholds)
        broken = copy.deepcopy(thresholds)
        broken["camera"]["gaps"] = {"warn": 5, "fail": 1}
        with self.assertRaisesRegex(ValueError, r"camera\.gaps.*warn <= fail"):
            load_thresholds(broken)
        broken = copy.deepcopy(thresholds)
        del broken["streams"]["hands"]
        with self.assertRaisesRegex(ValueError, "streams"):
            load_thresholds(broken)

    def test_cli_levels_annotate_and_only_errors_fail_a_dry_run(self):
        import yaml

        from bimanual_teleop.cli.convert_recording import main

        make_episode(self.root, main=TIMES)
        config = self.root.parent / "quality.yaml"
        config.write_text(yaml.safe_dump(_thresholds()), encoding="utf-8")
        saved = self.root.parent / "report.json"
        output = io.StringIO()
        with redirect_stdout(output):
            result = main(["--input", str(self.root), "--action-space", "eef", "--dry-run",
                           "--quality-config", str(config), "--report", str(saved), "--verbose"])
        self.assertEqual(result, 0, output.getvalue())
        self.assertIn("[OK] episode_000000：参考 30 帧", output.getvalue())
        self.assertIn("camera_1 对 camera_0", output.getvalue())
        self.assertEqual(json.loads(saved.read_text(encoding="utf-8"))["episodes"][0]["level"], "OK")

        short = self.root.parent / "short"
        make_episode(short)
        output = io.StringIO()
        with redirect_stdout(output):
            result = main(["--input", str(short), "--action-space", "eef",
                           "--output", str(self.root.parent / "short.zarr")])
        self.assertEqual(result, 0, output.getvalue())
        self.assertIn("[FAIL] episode_000000：参考 4 帧，写入 4 帧", output.getvalue())
        self.assertIn("问题：时长 0.2 s", output.getvalue())

        make_episode(short, name="episode_000001", metadata={**TEST_METADATA, "model_sha256": "other"})
        output = io.StringIO()
        with redirect_stdout(output):
            result = main(["--input", str(short), "--action-space", "eef", "--dry-run"])
        self.assertEqual(result, 1)
        self.assertIn("正式转换会中止", output.getvalue())
        self.assertIn("试运行发现 1 条错误", output.getvalue())


if __name__ == "__main__":
    unittest.main()
