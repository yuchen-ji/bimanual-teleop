"""Regression checks for the supplied, physically assembled CAD assets."""

import json
import unittest

from bimanual_teleop.sim import ROOT
from bimanual_teleop.sim._dependencies import np, Rotation
from bimanual_teleop.sim.config import load_config, sim_path
from bimanual_teleop.sim.kinematics import RobotModel
from bimanual_teleop.sim.task import TaskMonitor


@unittest.skipUnless((ROOT / "assets/generated/task/cad_manifest.json").exists(), "Prepare task CAD first")
class CADAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.manifest = json.loads((ROOT / "assets/generated/manifest.json").read_text())

    def test_runtime_and_rebuild_resources_are_complete_and_tracked(self):
        from bimanual_teleop.sim.resources import audit
        result = audit(check_git=True)
        self.assertTrue(result["passed"], result)

    def test_cad_transforms_preserve_handedness_and_outward_normals(self):
        import trimesh
        for name, item in self.manifest["cad"]["models"].items():
            self.assertAlmostEqual(float(np.linalg.det(item["cad_rotation"])), 1., msg=name)
        fixture = trimesh.load(ROOT / "assets/generated/task/fixture.obj", force="mesh")
        self.assertTrue(fixture.is_watertight)
        self.assertGreater(fixture.volume, 0., "A reversed static mesh lets bodies fall through its top")
        collision = self.manifest["cad"]["models"]["fixture"]["collision"]
        self.assertLessEqual(collision["sampled_surface_error_m"], .0001)
        reduced = trimesh.load(ROOT / "assets/generated" / collision["mesh"], force="mesh")
        self.assertGreater(reduced.volume, 0.)

    def test_new_fixed_tools_preserve_hand_orientation_and_actuated_joint_count(self):
        model = RobotModel(sim_path(self.config["robot"]["urdf"]))
        self.assertEqual(len(model.names), 54)
        zero = np.zeros(54)
        for side, prefix in (("left", "l"), ("right", "r")):
            self.assertEqual(model.by_child[prefix+"_mount"].parent, prefix+"_camera_bracket")
            self.assertEqual(model.by_child[prefix+"_camera_bracket"].parent, prefix+"_force_sensor")
            self.assertEqual(model.by_child[prefix+"_force_sensor"].parent, "TCP_Link_"+("L" if side == "left" else "R"))
            relative = np.linalg.inv(model.flange(side, zero, world=True)) @ model.link_pose(prefix+"_mount", zero)
            np.testing.assert_allclose(relative[:3, 3], [0, 0, .042], atol=1e-9)
            np.testing.assert_allclose(relative[:3, :3], Rotation.from_euler("x", np.pi).as_matrix(), atol=1e-9)

    def test_both_cameras_are_on_the_palmar_side(self):
        model = RobotModel(sim_path(self.config["robot"]["urdf"]))
        zero = np.zeros(54)
        for prefix in ("l", "r"):
            curled = zero.copy(); curled[model.names.index(prefix+"_middle_finger_mcp_flex")] = .3
            bracket = model.link_pose(prefix+"_camera_bracket", zero)
            before = model.link_pose(prefix+"_middle_finger_tip", zero)[:3, 3]
            after = model.link_pose(prefix+"_middle_finger_tip", curled)[:3, 3]
            curl = bracket[:3, :3].T @ (after-before)
            camera = np.array(self.manifest["cad"]["tool_stack"]["camera_position_in_bracket_m"])
            self.assertGreater(np.dot(camera[:2], curl[:2]), 0.)

    def test_default_rail_is_aligned_to_actual_unequal_pins(self):
        geometry = self.manifest["fixture_geometry"]
        quaternion = Rotation.from_euler("xyz", self.config["rail"]["initial_rpy_deg"], degrees=True).as_quat()
        state = {"rail": [*self.config["rail"]["initial_center"], *quaternion], "hand_contact_force": [0, 0]}
        monitor = TaskMonitor(self.config, geometry)
        for stamp in range(4):
            state["time_ns"] = stamp*1_000_000_000
            result = monitor.update(state)
        self.assertTrue(result["fixture_completed"], result)
        self.assertLess(result["pin_error_m"], .00001)
        self.assertNotEqual(*geometry["pin_radii_m"])

    def test_rail_collision_volume_and_real_holes(self):
        import trimesh
        from scipy.spatial import ConvexHull
        rail = self.manifest["rail"]
        total = 0.
        probes = np.array(rail["hole_centers_m"])+[0, 0, .0008]
        for name in rail["colliders"]:
            mesh = trimesh.load(ROOT / "assets/generated/rail" / name, force="mesh")
            total += mesh.volume
            planes = ConvexHull(mesh.vertices).equations
            for point in probes:
                self.assertFalse(np.all(planes[:, :3] @ point+planes[:, 3] < 1e-9), f"{name} fills a locating hole")
        self.assertAlmostEqual(total/rail["mesh_volume_m3"], 1., places=4)

    def test_table_narrow_end_and_two_five_layer_stacks(self):
        from bimanual_teleop.sim.bin_geometry import bin_geometry
        scene = self.config["scene"]
        crate = bin_geometry(scene)
        self.assertGreater(scene["table_size"][0], scene["table_size"][1])
        self.assertGreater(scene["table_center"][0]-scene["table_size"][0]/2, 0.)
        self.assertEqual(len(scene["bin_centers"]), 2)
        self.assertEqual(scene["bin_stack_count"], 5)
        for center in scene["bin_centers"]:
            self.assertAlmostEqual(center[2]-4*scene["bin_stack_pitch_m"], 0.)
            self.assertAlmostEqual(center[2]+crate["rim_height_m"], .83)
            self.assertGreater(abs(center[1])-crate["outer_size_m"][1]/2, scene["table_size"][1]/2)
