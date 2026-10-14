"""Local action validation and recording without network/device dependencies."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from bimanual_teleop.sim import ROOT
from bimanual_teleop.sim._dependencies import np
from bimanual_teleop.sim.actions import JointTrajectory
from bimanual_teleop.sim.config import load_config, sim_path
from bimanual_teleop.sim.kinematics import RobotModel, pose_vector
from bimanual_teleop.sim.local_session import LocalSession


class TrajectoryTests(unittest.TestCase):
    def test_interpolation_limits_order_and_speed(self):
        class Model:
            names = tuple(f"joint_{i}" for i in range(54))
            lower = np.full(54, -2.)
            upper = np.full(54, 2.)

        runtime = {"joint_speed_rad_s": 1., "hand_speed_rad_s": 3.}
        document = {"joint_names": list(Model.names), "waypoints": [
            {"time_s": 0, "positions_rad": [0.]*54},
            {"time_s": 2, "positions_rad": [.5]*54}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"trajectory.json"

            def load(value):
                path.write_text(json.dumps(value))
                return JointTrajectory(path, Model(), runtime)

            trajectory = load(document)
            np.testing.assert_allclose(trajectory.sample(1), [.25]*54)
            np.testing.assert_allclose(trajectory.sample(8), [.5]*54)
            for mutation in ("order", "speed", "names", "limits", "nan"):
                invalid = copy.deepcopy(document)
                if mutation == "order": invalid["waypoints"][1]["time_s"] = 0
                elif mutation == "speed": invalid["waypoints"][1]["time_s"] = .01
                elif mutation == "names": invalid["joint_names"].reverse()
                elif mutation == "limits": invalid["waypoints"][1]["positions_rad"][0] = 4
                else: invalid["waypoints"][1]["positions_rad"][0] = float("nan")
                with self.subTest(mutation=mutation), self.assertRaises(ValueError): load(invalid)


@unittest.skipUnless((ROOT/"assets/generated/robot.urdf").is_file(), "Prepare assets first")
class LocalRecordingTests(unittest.TestCase):
    def test_marker_pause_save_and_test_data_exclusion(self):
        from bimanual_teleop.sim.export import export
        from bimanual_teleop.sim.recording import read_boundaries
        config = load_config()
        config["cameras"].update(width=32, height=24)
        model = RobotModel(sim_path(config["robot"]["urdf"]), config["robot"]["world_yaw_deg"])
        with tempfile.TemporaryDirectory() as directory:
            config["paths"]["recordings"] = str(Path(directory)/"recordings")
            session = LocalSession(config, model, record=True, codec="libx264", status_path=Path(directory)/"status.json")
            q = session.ready
            images = np.full((3, 24, 32, 3), 110, dtype=np.uint8)

            def accept(tick):
                eef = [pose_vector(model.flange(side, q)).tolist() for side in ("left", "right")]
                session.accept({"tick": tick, "time_ns": round(tick*1e9/240), "q": q.tolist(),
                    "eef": eef, "eef_target": eef, "q_target": q.tolist(), "command_tick": tick-1,
                    "wrench": [[0.]*6]*2, "rail": [*config["rail"]["initial_center"], 0, 0, 0, 1],
                    "hand_contact_force": [0., 0.]}, images)

            try:
                for tick in (8, 16, 24): accept(tick)
                session.boundary(); accept(32)
                session.toggle_recording(); accept(40)
                session.toggle_recording(); accept(48)
                session.save()
                completed = session.last_completed
                self.assertEqual(completed["status"], "complete")
                self.assertEqual(completed["frames"], 5)
                self.assertEqual(read_boundaries(completed["path"])[0]["frame_index"], 3)
                with self.assertRaises(ValueError):
                    export(Path(directory)/"recordings", Path(directory)/"learning")
            finally:
                session.close()


if __name__ == "__main__":
    unittest.main()
