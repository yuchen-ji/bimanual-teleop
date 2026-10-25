import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[2])]

from bridge import Bridge
from common import write_json
from controller import PreinsertController
from reload_settings import ReloadSettings, load_settings, runtime_bundle
from task_settings import rebuild_changes


class ReloadTests(unittest.TestCase):
    def setup_run(self, directory):
        output = Path(directory)
        document = yaml.safe_load((HERE / "config.demonstrated.yaml").read_text())
        path = output / "input.yaml"
        path.write_text(yaml.safe_dump(document))
        config, task = load_settings(path)
        bridge = Bridge(config, task, output)
        reloads = ReloadSettings(path, bridge)
        runtime = runtime_bundle(bridge, path)
        source = PreinsertController(config, bridge.geometry.robot, task, output, runtime,
                                     mode="step", headless=False, perception="oracle", grasp="physics")
        return document, path, reloads, source

    def test_complete_revision_updates_pose_hand_threshold_and_planner_without_mutating_old_run(self):
        with tempfile.TemporaryDirectory() as directory:
            document, path, reloads, source = self.setup_run(directory)
            old_ready = source.ready.copy()
            old_task = copy.deepcopy(source.task)
            grasp = reloads.initial.references["grasp"].copy()
            grasp[3] += .01
            document["task"].update(initial_rail_offset_world_m=[-.035, .10, .135],
                                    hand_grasp_override_rad=grasp.tolist(), max_grasp_translation_m=.008,
                                    joint_speed_rad_s=.07)
            document["simulation"]["rail"] = {"mass_kg": .9}
            path.write_text(yaml.safe_dump(document))
            supplied = reloads.prepare("revision1")["configuration"]
            self.assertEqual(source.task, old_task)
            self.assertEqual(reloads.bridge_for({}).config["joint_speed_rad_s"], old_task["joint_speed_rad_s"])
            source.pending_configuration = supplied
            source.install_pending_settings()
            source.reset()
            self.assertEqual(source.task["max_grasp_translation_m"], .008)
            self.assertEqual(source.config["rail"]["mass_kg"], .9)
            self.assertEqual(source.runtime["revision"], "revision1")
            self.assertGreater(np.linalg.norm(source.ready-old_ready), .001)
            np.testing.assert_allclose(source.runtime["grasp"], grasp)
            active = reloads.bridge_for({"revision": source.runtime["revision"]})
            self.assertEqual(active.config["joint_speed_rad_s"], .07)
            snapshot = json.loads((source.trial_directory / "settings.json").read_text())
            self.assertEqual(snapshot["runtime"]["revision"], "revision1")
            source.close()

    def test_invalid_yaml_is_rejected_and_geometry_is_prepared_without_changing_active_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            document, path, reloads, source = self.setup_run(directory)
            original = copy.deepcopy(source.task)
            document["task"]["max_grasp_translation_m"] = -1
            path.write_text(yaml.safe_dump(document))
            with self.assertRaisesRegex(ValueError, "Invalid task"):
                reloads.prepare("bad")
            document["task"]["max_grasp_translation_m"] = .005
            document["simulation"]["robot"]["collision"] = {"finger_approximation": "convexDecomposition"}
            path.write_text(yaml.safe_dump(document))
            response = reloads.prepare("geometry")
            self.assertEqual(response["operation"], "rebuild")
            self.assertIsNot(reloads.versions["geometry"].geometry, reloads.initial.geometry)
            self.assertEqual(source.task, original)
            self.assertEqual(set(reloads.versions), {"startup", "geometry"})
            source.close()

    def test_rebuild_compares_to_active_revision_and_reverting_structure_rebuilds_again(self):
        with tempfile.TemporaryDirectory() as directory:
            document, path, reloads, source = self.setup_run(directory)
            changed = copy.deepcopy(document)
            changed["simulation"]["scene"] = {"bin_stack_count": 2}
            path.write_text(yaml.safe_dump(changed))
            self.assertEqual(reloads.prepare("changed")["operation"], "rebuild")
            self.assertEqual(reloads.prepare("unchanged", current_revision="changed")["operation"], "reset")
            path.write_text(yaml.safe_dump(document))
            self.assertEqual(reloads.prepare("reverted", current_revision="unchanged")["operation"], "rebuild")
            self.assertEqual(reloads.prepare("forced", current_revision="reverted", rebuild=True)["operation"], "rebuild")
            source.close()

    def test_rebuild_after_partial_bind_preserves_history_and_starts_waiting(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, reloads, source = self.setup_run(directory)
            source.finish("failed", "injected construction error")
            old_trial = source.trial_directory
            source.request_rebuild()  # No bound physics handles yet.
            source.service_reload()
            request = json.loads((Path(directory) / "rpc/request.json").read_text())
            self.assertTrue(request["rebuild"])
            write_json(Path(directory) / "rpc/response.json", {
                "id": request["id"], **reloads.prepare(request["id"], rebuild=True)})
            source.service_reload()
            self.assertTrue(source.scene_rebuild_requested)
            self.assertFalse(source.reset_requested)
            replacement = source.take_scene_replacement()
            source.close()
            new = replacement.factory(replacement.config, reloads.initial.geometry.robot)
            self.assertNotEqual(new.trial_directory, old_trial)
            self.assertTrue(new.waiting)
            self.assertEqual(json.loads((old_trial / "result.json").read_text())["reason"], "injected construction error")
            self.assertIn("scene_rebuild_requested", (Path(directory) / "events.jsonl").read_text())
            new.close()

    def test_wrist_calibration_file_is_reread_and_rebuilt(self):
        with tempfile.TemporaryDirectory() as directory:
            document, path, reloads, source = self.setup_run(directory)
            calibration = Path(directory) / "wrist.yaml"
            def save_wrist(x):
                calibration.write_text(yaml.safe_dump({"cameras": {"wrist": {"left": {
                    "pose_in_bracket": {"position_m": [x, 0., 0.], "quaternion_xyzw": [0., 0., 0., 1.]}}}}}))
            save_wrist(.01)
            document["simulation"]["cameras"] = {"wrist_calibration_file": str(calibration)}
            path.write_text(yaml.safe_dump(document))
            first = reloads.prepare("camera1")
            self.assertEqual(first["operation"], "rebuild")
            save_wrist(.02)
            second = reloads.prepare("camera2", current_revision="camera1")
            self.assertEqual(second["operation"], "rebuild")
            self.assertEqual(second["configuration"]["simulation"]["cameras"]["wrist"]["left"]["pose_in_bracket"]["position_m"][0], .02)
            source.close()

    def test_reload_while_paused_ignores_stale_response_and_allows_retry_after_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, source = self.setup_run(directory)
            source.bound = True
            source.request_reload()
            source.service_reload()
            request_id = source.reload_id
            self.assertEqual(source.physics_action_requested, "pause")
            write_json(Path(directory) / "rpc/response.json", {"id": "stale", "ok": True})
            source.service_reload()
            self.assertEqual(source.reload_id, request_id)
            write_json(Path(directory) / "rpc/response.json", {"id": request_id, "ok": False, "error": "bad YAML"})
            source.service_reload()
            self.assertIsNone(source.pending_configuration)
            self.assertIn("bad YAML", source.config_message)
            source.request_reset()
            self.assertTrue(source.reset_requested)
            source.close()

    def test_camera_and_collision_changes_require_scene_rebuild(self):
        for path in ("cameras", "scene", "robot"):
            with self.subTest(path=path):
                self.assertEqual(rebuild_changes({path: {"model": 1}}, {path: {"model": 2}}, {}, {}), {path+".model"})

    def test_layout_resolution_is_validated_before_starting_isaac(self):
        with tempfile.TemporaryDirectory() as directory:
            document = yaml.safe_load((HERE / "config.yaml").read_text())
            document["task"]["desktop_layout"]["resolution"] = [1280, 720]
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(document))
            with self.assertRaisesRegex(ValueError, "4:3"):
                load_settings(path)

    def test_flange_direction_reloads_through_existing_reset_without_changing_position(self):
        with tempfile.TemporaryDirectory() as directory:
            document, path, reloads, source = self.setup_run(directory)
            model = reloads.initial.geometry.robot
            document["task"]["initial_flange_z_axis_world"] = None
            path.write_text(yaml.safe_dump(document))
            original = reloads.prepare("original")["configuration"]
            source.pending_configuration = original
            source.install_pending_settings()
            before = model.flange("left", source.ready, world=True)
            document["task"]["initial_flange_z_axis_world"] = [1., 0., 0.]
            path.write_text(yaml.safe_dump(document))
            source.pending_configuration = reloads.prepare("aligned")["configuration"]
            source.install_pending_settings()
            after = model.flange("left", source.ready, world=True)
            np.testing.assert_allclose(after[:3, 2], [1, 0, 0], atol=1e-4)
            np.testing.assert_allclose(after[:3, 3], before[:3, 3], atol=1e-4)
            document["task"]["initial_flange_z_axis_world"] = [0., 0., 0.]
            path.write_text(yaml.safe_dump(document))
            with self.assertRaisesRegex(ValueError, "nonzero"):
                reloads.prepare("zero")
            self.assertEqual(source.runtime["revision"], "aligned")
            source.close()

    def test_explicit_default_reloads_even_without_new_axis_alignment_code(self):
        from unittest.mock import patch

        from common import pose
        with tempfile.TemporaryDirectory() as directory:
            _, path, reloads, source = self.setup_run(directory)
            explicit = yaml.safe_load((HERE / "config.yaml").read_text())
            path.write_text(yaml.safe_dump(explicit))
            # The open application predates configure_initial_direction. Exercise
            # its existing joint/hand/rail overrides without the new IK feature.
            with patch.object(Bridge, "configure_initial_direction", lambda self: None):
                source.pending_configuration = reloads.prepare("explicit")["configuration"]
            source.install_pending_settings()
            # Explicit pose is user-adjustable; do not pin this regression to
            # the historical wrist position or the automatic template's hand.
            np.testing.assert_allclose(source.ready[:7], np.radians(explicit["simulation"]["robot"]["ready_deg"]["left"]))
            for name in ("open", "grasp"):
                np.testing.assert_allclose(source.runtime[name], explicit["task"][f"hand_{name}_override_rad"], atol=1e-10)
            np.testing.assert_allclose(
                pose(source.task["rail_in_flange_xyz_m"], source.task["rail_in_flange_rpy_deg"]),
                pose(explicit["task"]["rail_in_flange_override"]["xyz_m"],
                     explicit["task"]["rail_in_flange_override"]["rpy_deg"]), atol=1e-10)
            source.close()


if __name__ == "__main__":
    unittest.main()
