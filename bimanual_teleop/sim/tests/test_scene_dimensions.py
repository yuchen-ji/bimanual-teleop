"""Measure actual source surfaces after applying the scene's CAD transforms."""

import copy
import unittest

import trimesh

from bimanual_teleop.sim import ROOT
from bimanual_teleop.sim._dependencies import np
from bimanual_teleop.sim.bin_geometry import bin_geometry
from bimanual_teleop.sim.config import load_config, sim_path, validate
from bimanual_teleop.sim.kinematics import RobotModel


def section_hits(segments, axis, value, coordinate):
    a, b = segments[:, 0], segments[:, 1]
    delta = b[:, axis]-a[:, axis]
    selected = (np.abs(delta) > 1e-12) & ((a[:, axis]-value)*(b[:, axis]-value) <= 1e-14)
    fraction = (value-a[selected, axis])/delta[selected]
    points = a[selected]+fraction[:, None]*(b[selected]-a[selected])
    return np.unique(np.round(points[:, coordinate], 10))


class SceneDimensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.scene = cls.config["scene"]
        cls.geometry = bin_geometry(cls.scene)
        cls.bin = trimesh.load(ROOT/"assets/generated/task/bins.obj", force="mesh", process=False)
        cls.bin.apply_scale(cls.geometry["scale"])
        cls.table = trimesh.load(ROOT/"assets/generated/task/table.obj", force="mesh", process=False)
        cls.table.apply_translation(cls.scene["table_center"])
        cls.table_rear = cls.table.bounds[0, 0]

    def test_real_mesh_opening_and_depth(self):
        rim = self.bin.bounds[1, 2]
        for normal, axis, wanted in (([0, 1, 0], 0, .445), ([1, 0, 0], 1, .318)):
            section = trimesh.intersections.mesh_plane(self.bin, plane_normal=normal, plane_origin=[0, 0, 0])
            opening = section_hits(section, 2, rim-.0005, axis)
            positive = opening[opening > 0]
            self.assertAlmostEqual(positive.min()*2, wanted, places=8)
            floor = section_hits(section, axis, 0., 2).max()
            self.assertAlmostEqual(rim-floor, .165, places=8)

    def test_ground_and_upper_rim_of_both_five_crate_stacks(self):
        self.assertEqual(self.scene["bin_stack_count"], 5)
        for origin in self.scene["bin_centers"]:
            bottom = origin[2]-4*self.scene["bin_stack_pitch_m"]+self.bin.bounds[0, 2]
            self.assertAlmostEqual(bottom, 0., places=8)
            self.assertAlmostEqual(origin[2]+self.bin.bounds[1, 2], .83, places=8)

    def test_rear_rim_to_table_rear(self):
        top = self.bin.vertices[self.bin.vertices[:, 2] > self.bin.bounds[1, 2]-1e-8]
        for origin in self.scene["bin_centers"]:
            self.assertAlmostEqual(origin[0]+top[:, 0].min()-self.table_rear, .153, places=8)

    def test_column_front_to_table_rear(self):
        model = RobotModel(sim_path(self.config["robot"]["urdf"]))
        robot = trimesh.load(ROOT/"assets/generated/meshes/mesh_000.stl", force="mesh", process=False)
        robot.apply_transform(model.link_pose("base_link", np.zeros(54)))
        section = trimesh.intersections.mesh_plane(robot, plane_normal=[0, 0, 1], plane_origin=[0, 0, .73])
        front = section[:, :, 0].max()
        self.assertAlmostEqual(self.table_rear-front, .295, places=8)

    def test_fixture_offset_and_table_height_preserved(self):
        self.assertAlmostEqual(self.table.bounds[1, 2], .73, places=8)
        self.assertAlmostEqual(self.scene["fixture_origin"][0]-.05-self.table_rear, .13, places=8)

    def test_wide_layout_keeps_measured_x_and_z(self):
        wide = load_config(ROOT/"configs/wide_workspace.yaml")
        for narrow, broad in zip(self.scene["bin_centers"], wide["scene"]["bin_centers"]):
            np.testing.assert_allclose(np.asarray(narrow)[[0, 2]], np.asarray(broad)[[0, 2]])

    def test_invalid_inner_dimensions_rejected(self):
        for size in ([.445, .318, 0], [.445, float("nan"), .165], [.445, .318]):
            config = copy.deepcopy(self.config)
            config["scene"]["bin_inner_size"] = size
            with self.assertRaises(ValueError):
                validate(config)
