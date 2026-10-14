import json
import tempfile
import unittest
from pathlib import Path

from bimanual_teleop.sim._dependencies import np
from bimanual_teleop.sim._dependencies import yaml
from bimanual_teleop.sim._dependencies import zarr

from bimanual_teleop.sim.export import export
from bimanual_teleop.sim.recording import SimRecorder, read_boundaries
from bimanual_teleop.sim.segments import make_scheme


def state(index):
    pose = [[index*.001, 0, 0, 0, 0, 0, 1], [0, index*.001, 0, 0, 0, 0, 1]]
    command = [[index*.001+.01, 0, 0, 0, 0, 0, 1], [0, index*.001+.02, 0, 0, 0, 0, 1]]
    return {"tick": (index+1)*8, "time_ns": round((index+1)*1e9/30), "q": [index*.001]*54,
            "eef": pose, "eef_target": command, "q_target": [index*.002]*54,
            "wrench": [[1, 2, 3, 4, 5, 6]]*2, "rail": [0, 0, 0, 0, 0, 0, 1], "command_tick": index*8}


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def record(self, pause=False):
        path = self.root / "raw/episode_000000"
        recorder = SimRecorder(path, {"backend": "transport_test", "asset_signature": "test"}, 32, 24, codec="libx264")
        for index in range(48):
            if pause and index == 30:
                recorder.pause(state(index)["tick"]); recorder.resume(state(index)["tick"])
            recorder.state(state(index))
            event = [{"event_id": "cut-1", "client_press_ns": 100, "server_receive_ns": 200}] if index == 24 else []
            recorder.frame(state(index), np.full((3, 24, 32, 3), index, dtype=np.uint8), event)
        document = recorder.close()
        self.assertEqual(document["status"], "complete", document.get("failure"))
        self.assertEqual(document["frames"], 48)
        return path, document

    def test_full_windows_cross_marker_subtask_windows_do_not(self):
        path, document = self.record()
        scheme = make_scheme(self.root / "raw")
        self.assertEqual([r["task"] for r in scheme["episodes"][0]["ranges"]], [None, None])
        scheme["episodes"][0]["ranges"][0]["task"] = "pick"
        scheme["episodes"][0]["ranges"][1]["task"] = "place"
        scheme_path = self.root / "scheme.yaml"; scheme_path.write_text(yaml.safe_dump(scheme))
        reports = export(self.root / "raw", self.root / "datasets", "both", scheme_path, allow_test_data=True)
        full = zarr.open_group(str(self.root / "datasets/full.zarr"), mode="r")
        np.testing.assert_array_equal(full["meta/segment_ends"][:], [48])
        self.assertEqual(full["data/action"].shape, (48, 52))
        np.testing.assert_allclose(full["data/action"][24, :3], [.034, 0, 0], atol=1e-6)
        for task, expected in (("pick", np.arange(24)), ("place", np.arange(24, 48))):
            data = zarr.open_group(str(self.root / f"datasets/subtasks/{task}.zarr"), mode="r")
            np.testing.assert_array_equal(data["meta/source_frame"][:], expected)
            self.assertEqual(data["meta"].attrs["segments"][0]["source_episode"], document["source_demo_id"])
        self.assertEqual(read_boundaries(path)[0]["frame_index"], 24)

    def test_pause_preserved_in_full_dataset(self):
        self.record(pause=True)
        export(self.root / "raw", self.root / "full", allow_test_data=True)
        data = zarr.open_group(str(self.root / "full/full.zarr"), mode="r")
        np.testing.assert_array_equal(data["meta/segment_ends"][:], [30, 48])

    def test_unreviewed_scheme_cannot_implicitly_label_tasks(self):
        self.record()
        path = self.root / "scheme.yaml"; path.write_text(yaml.safe_dump(make_scheme(self.root / "raw")))
        with self.assertRaisesRegex(ValueError, "task label"):
            export(self.root / "raw", self.root / "output", "both", path, allow_test_data=True)
        self.assertFalse((self.root / "output").exists())

    def test_test_data_requires_explicit_override(self):
        self.record()
        with self.assertRaisesRegex(ValueError, "not physical"):
            export(self.root / "raw", self.root / "output")
        self.assertFalse((self.root / "output").exists())

    def test_subtask_does_not_cross_pause_after_merging_ranges(self):
        self.record(pause=True)
        scheme = make_scheme(self.root / "raw")
        scheme["episodes"][0]["ranges"] = [{"start": 0, "end": 48, "task": "merged"}]
        path = self.root / "scheme.yaml"; path.write_text(yaml.safe_dump(scheme))
        export(self.root / "raw", self.root / "output", "subtasks", path, allow_test_data=True)
        data = zarr.open_group(str(self.root / "output/subtasks/merged.zarr"), mode="r")
        np.testing.assert_array_equal(data["meta/segment_ends"][:], [30, 48])


if __name__ == "__main__":
    unittest.main()
