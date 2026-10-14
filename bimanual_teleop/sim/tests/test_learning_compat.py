"""Read an exported fixture through the existing learning dataset unchanged."""

import importlib.util
import sys
import unittest
from pathlib import Path

from bimanual_teleop.sim import ROOT


class LearningCompatibilityTests(unittest.TestCase):
    def test_existing_dataset_reads_full_and_subtask_windows(self):
        project = ROOT.parents[2] / "diffusion_policy"
        if not (project / "diffusion_policy/dataset/tianji_rail_image_dataset.py").is_file():
            self.skipTest("Sibling learning project is not available")
        if importlib.util.find_spec("torch") is None:
            self.skipTest("Learning runtime is not installed in the host environment")
        from bimanual_teleop.sim.tests.test_recording import RecordingTests
        from bimanual_teleop.sim.export import export
        from bimanual_teleop.sim.segments import make_scheme
        from bimanual_teleop.sim._dependencies import yaml

        sys.path.insert(0, str(project))
        try:
            from diffusion_policy.dataset.tianji_rail_image_dataset import TianjiRailImageDataset
            fixture = RecordingTests()
            fixture.setUp()
            try:
                _, document = fixture.record()
                scheme = make_scheme(fixture.root / "raw")
                for span in scheme["episodes"][0]["ranges"]:
                    span["task"] = "selected"
                scheme_path = fixture.root / "scheme.yaml"
                scheme_path.write_text(yaml.safe_dump(scheme))
                export(fixture.root / "raw", fixture.root / "export", "both", scheme_path, allow_test_data=True)
                shape_meta = {"action": {"shape": [52]}, "obs": {
                    **{f"camera_{index}": {"shape": [3, 24, 32], "type": "rgb"} for index in range(3)},
                    "robot_joint": {"shape": [14]}, "hand_joint": {"shape": [40]}, "robot_eef_pose": {"shape": [12]}}}
                options = dict(shape_meta=shape_meta, horizon=16, n_obs_steps=2, image_shape=(3, 24, 32), resize_enabled=False)
                full = TianjiRailImageDataset(fixture.root / "export/full.zarr", **options)
                parts = TianjiRailImageDataset(fixture.root / "export/subtasks/selected.zarr", **options)
                self.assertEqual(len(full), 33)
                self.assertEqual(len(parts), 18)
                self.assertEqual(full.n_source_episodes, 1)
                self.assertEqual(parts.sources, [document["source_demo_id"]]*2)
                self.assertEqual(tuple(full[0]["action"].shape), (16, 52))
                self.assertEqual(tuple(parts[0]["obs"]["camera_0"].shape), (2, 3, 24, 32))
            finally:
                fixture.tearDown()
        finally:
            sys.path.remove(str(project))


if __name__ == "__main__":
    unittest.main()
