"""review.json decisions, review previews and the keyboard review command."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.support.recording import make_episode

TIMES = np.arange(9) * 1000 / 30


def _pause_episode(root, name="episode_000000"):
    sources = TIMES.copy()
    sources[4:] += 2000
    return make_episode(root, name=name, main=TIMES, camera_sources={camera: sources for camera in range(3)})


@unittest.skipUnless(importlib.util.find_spec("av") and importlib.util.find_spec("zarr"),
                     "recording dependencies are not installed")
class RecordingReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "raw"
        self.outputs = 0

    def convert(self, **kwargs):
        from bimanual_teleop.recording.convert import convert_recordings

        self.outputs += 1
        return convert_recordings(self.root, self.root.parent / f"out{self.outputs}.zarr",
                                  action_space="eef", **kwargs)

    def test_rejected_episodes_are_skipped_and_reported(self):
        from bimanual_teleop.cli.convert_recording import summary_lines
        from bimanual_teleop.recording.review import write_review

        make_episode(self.root)
        rejected, _ = make_episode(self.root, name="episode_000001")
        write_review(rejected, decision="reject", note="抓取失败")
        report = self.convert()
        item = report["episodes"][1]
        self.assertEqual((item["skipped"], item["review"]), ("review_reject", {"decision": "reject", "note": "抓取失败"}))
        self.assertNotIn("reference_frames", item)
        self.assertEqual((report["output_episodes"], report["output_frames"]), (1, 4))
        self.assertIn("episode_000001：审片剔除，整条跳过（抓取失败）", "\n".join(summary_lines(report)))

    def test_pause_reviews_in_review_json_match_the_config_and_win_per_seam(self):
        import zarr

        from bimanual_teleop.recording.review import write_review

        episode, _ = _pause_episode(self.root)
        merged = {"4": {"merge": True, "reason": "Synthetic seam"}}
        from_config = self.convert(conversion_config={"pause_reviews": {"episode_000000": merged}})
        write_review(episode, pause_reviews=merged)
        from_review = self.convert(conversion_config={})
        overridden = self.convert(conversion_config={"pause_reviews": {"episode_000000": {
            "4": {"merge": False, "reason": "Old conclusion"}}}})
        for report in (from_config, from_review, overridden):
            self.assertEqual(report["output_segments"], 1)
            self.assertTrue(report["episodes"][0]["pauses"][0]["merged"])
        np.testing.assert_array_equal(zarr.open_group(str(self.root.parent / "out1.zarr"))["data/timestamp"][:],
                                      zarr.open_group(str(self.root.parent / "out2.zarr"))["data/timestamp"][:])
        strict = self.convert(conversion_config={"mode": "strict"})
        self.assertNotIn("pauses", strict["episodes"][0])

    def test_invalid_review_json_stops_conversion(self):
        episode, _ = make_episode(self.root)
        for review, message in (({"decision": "maybe"}, "decision"), ({"verdict": "keep"}, "unknown"),
                                ({"pause_reviews": {"4": {"merge": True}}}, "nonempty reason")):
            with self.subTest(review=review):
                (episode / "review.json").write_text(json.dumps(review), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    self.convert()

    def test_preview_tiles_the_cameras_and_marks_missing_frames(self):
        import av
        import zarr

        from bimanual_teleop.recording.review import detect_pauses, make_preview

        episode, _ = make_episode(self.root, main=TIMES[:6], other={1: TIMES[[0, 1, 3, 4, 5]]})
        path = make_preview(episode, scale=.05)
        self.assertEqual(path, episode / "preview.mp4")
        with av.open(str(path)) as container:
            frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        self.assertEqual(len(frames), 6)
        self.assertEqual(frames[0].shape, (24, 96, 3))
        np.testing.assert_allclose(frames[2][4:20, 36:60].reshape(-1, 3).mean(axis=0), (80, 0, 0), atol=20)
        np.testing.assert_allclose(frames[1][4:20, 36:60].reshape(-1, 3).mean(axis=0), (40, 50, 0), atol=20)
        self.assertEqual(sorted(p.name for p in episode.iterdir() if p.name.startswith(".")), [])
        paused, _ = _pause_episode(self.root, "episode_000001")
        pauses = detect_pauses(zarr.open_group(str(paused / "raw.zarr"), mode="r"),
                               json.loads((paused / "episode.json").read_text()), 100)
        self.assertEqual([pause["after_main_row"] for pause in pauses], [4])
        self.assertAlmostEqual(pauses[0]["removed_wait_ms"], 2000, delta=1)

    def test_keyboard_review_records_decisions_notes_and_pause_seams(self):
        from bimanual_teleop.cli.review_recording import main
        from bimanual_teleop.recording.review import read_review, write_review

        paused, _ = _pause_episode(self.root)
        rejected, _ = make_episode(self.root, name="episode_000001")
        decided, _ = make_episode(self.root, name="episode_000002")
        write_review(decided, decision="keep")
        answers = iter(["x", "y", "动作完整", "m", "", "n", "抓取失败"])
        output = io.StringIO()
        with patch("builtins.input", lambda prompt="": next(answers)), redirect_stdout(output):
            result = main(["--input", str(self.root), "--no-play", "--scale", ".05"])
        self.assertEqual(result, 0, output.getvalue())
        self.assertEqual(next(answers, None), None)
        self.assertIn("待审 2 条", output.getvalue())
        self.assertIn("暂停接缝：主相机原始帧 4 起", output.getvalue())
        self.assertIn("本次保留 1 条，剔除 1 条，跳过 0 条", output.getvalue())
        review = read_review(paused)
        self.assertEqual((review["decision"], review["note"]), ("keep", "动作完整"))
        self.assertEqual(review["pause_reviews"], {"4": {"merge": True, "reason": "审片确认接缝两侧动作连续"}})
        self.assertEqual(read_review(rejected)["decision"], "reject")
        self.assertTrue((paused / "preview.mp4").is_file())
        self.assertFalse((decided / "preview.mp4").exists())
        report = self.convert(conversion_config={})
        self.assertEqual(report["episodes"][0]["segments"], 1)
        self.assertEqual(report["episodes"][1]["skipped"], "review_reject")

    def test_end_of_input_stops_review_without_writing(self):
        from bimanual_teleop.cli.review_recording import main

        episode, _ = make_episode(self.root)

        def closed(prompt=""):
            raise EOFError

        with patch("builtins.input", closed), redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--input", str(self.root), "--no-play", "--scale", ".05"]), 0)
        self.assertFalse((episode / "review.json").exists())

    def test_player_command_prefers_explicit_player_then_ffplay(self):
        from bimanual_teleop.cli.review_recording import player_command

        self.assertEqual(player_command("vlc --play-and-exit", Path("/a b/p.mp4"), "x"),
                         ["vlc", "--play-and-exit", "/a b/p.mp4"])
        with patch("shutil.which", lambda name: f"/usr/bin/{name}" if name == "ffplay" else None):
            self.assertEqual(player_command(None, Path("p.mp4"), "t")[:2], ["ffplay", "-autoexit"])
        with patch("shutil.which", lambda name: None):
            self.assertIsNone(player_command(None, Path("p.mp4"), "t"))


if __name__ == "__main__":
    unittest.main()
