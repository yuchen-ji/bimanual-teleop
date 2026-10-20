import copy
import json
import unittest

from bimanual_teleop.sim import ROOT
from bimanual_teleop.sim._dependencies import np, Rotation
from bimanual_teleop.sim.camera_calibration import camera_window_faces, camera_world_from_rail, camera_world_from_table_plane, head_camera_local_pose, pinhole_attributes, wrist_camera_local_pose
from bimanual_teleop.sim.config import load_config, validate


class CameraCalibrationTests(unittest.TestCase):
    def test_rail_cad_origin_and_axis_conversion(self):
        # A rail at (1, 2, 3), with CAD rotated 90 degrees and origin offset,
        # observed by a camera at (4, 5, 6) with world-aligned optical axes.
        rail = {"initial_center": [1, 2, 3], "initial_rpy_deg": [0, 0, 0]}
        cad = {"cad_rotation": [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
               "origin_in_rotated_cad_m": [.25, 0, .01]}
        camera_cad = [[0, 1, 0, -3.25], [-1, 0, 0, -3], [0, 0, 1, -3.01], [0, 0, 0, 1]]
        expected = np.eye(4); expected[:3, 3] = [4, 5, 6]
        np.testing.assert_allclose(camera_world_from_rail(camera_cad, rail, cad), expected, atol=1e-12)

    def test_optical_axes_and_nonidentity_mount(self):
        parent = np.eye(4)
        parent[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        parent[:3, 3] = [1, 2, 3]
        settings = {"position_world_m": [4, 5, 6], "quaternion_xyzw": [0, 0, 0, 1]}
        position, quaternion = head_camera_local_pose(settings, parent)
        local = np.eye(4); local[:3, :3] = Rotation.from_quat(quaternion).as_matrix(); local[:3, 3] = position
        world = parent @ local
        np.testing.assert_allclose(world[:3, 3], [4, 5, 6], atol=1e-12)
        np.testing.assert_allclose(world[:3, :3] @ [0, 0, -1], [0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(world[:3, :3] @ [0, 1, 0], [0, -1, 0], atol=1e-12)

    def test_table_plane_recovers_downward_camera(self):
        rail = {"initial_center": [.5, .1, .82], "initial_rpy_deg": [0, 0, 0]}
        cad = {"cad_rotation": [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
               "origin_in_rotated_cad_m": [.25, 0, 0]}
        # Camera at (0, 0, 2), looking down; table at world Z=.73.
        observed = [[0, 1, 0, .25], [1, 0, 0, -.1], [0, 0, -1, 1.18], [0, 0, 0, 1]]
        fitted = camera_world_from_table_plane(observed, rail, cad, [0, 0, -2], 2.54, .73)
        np.testing.assert_allclose(fitted[:3, 3], [0, 0, 2], atol=1e-12)
        np.testing.assert_allclose(fitted[:3, :3], np.diag([1, -1, -1]), atol=1e-12)

    def test_offcenter_unequal_focal_lengths(self):
        p = pinhole_attributes(640, 480, {"fx": 400, "fy": 500, "cx": 330, "cy": 260})
        self.assertAlmostEqual(p["horizontal_aperture"], 32)
        self.assertAlmostEqual(p["vertical_aperture"], 19.2)
        self.assertAlmostEqual(p["horizontal_offset"], -.5)
        self.assertAlmostEqual(p["vertical_offset"], .8)

    def test_wrist_optics_are_outside_the_opaque_camera_model(self):
        import trimesh
        manifest = json.loads((ROOT / "assets/generated/manifest.json").read_text())
        tools = manifest["cad"]["tool_stack"]
        position, quaternion = wrist_camera_local_pose(tools)
        forward = Rotation.from_quat(quaternion).apply([0, 0, -1])
        np.testing.assert_allclose(forward, tools["camera_front_normal_in_bracket"], atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(position-np.array(tools["camera_position_in_bracket_m"])), .002)
        # An entire convex combination of triangle vertices remains behind this plane.
        # Thus no forward ray can see the camera body or opaque glass cover itself.
        for part in (1, 2):
            mesh = trimesh.load(ROOT / f"assets/generated/task/wrist_part_{part}.obj", force="mesh", process=False)
            self.assertLess(float(((mesh.vertices-position) @ forward).max()), -.001)

    def test_default_calibration_reproduces_archived_reference(self):
        config = load_config()
        reference = json.loads((ROOT / "configs/head_camera_reference.json").read_text())
        head = config["cameras"]["head"]
        np.testing.assert_allclose(head["position_world_m"], np.array(reference["T_world_camera_opencv"])[:3, 3], atol=1e-8)
        np.testing.assert_allclose(Rotation.from_quat(head["quaternion_xyzw"]).as_matrix(),
                                   np.array(reference["T_world_camera_opencv"])[:3, :3], atol=1e-8)

    def test_optical_window_preserves_bracket_and_camera_shell(self):
        import trimesh
        tools = json.loads((ROOT/"assets/generated/manifest.json").read_text())["cad"]["tool_stack"]
        for part in (0, 1, 2):
            mesh = trimesh.load(ROOT/f"assets/generated/task/wrist_part_{part}.obj", process=False, force="mesh")
            selected = camera_window_faces(mesh.vertices, mesh.faces, tools)
            if part == 0:
                self.assertEqual(len(selected), 0)
            elif part == 1:
                self.assertGreater(len(selected), 0)
                self.assertLess(len(selected), len(mesh.faces)*.02)
            else:
                self.assertGreater(len(selected), 0)
                self.assertLess(len(selected), len(mesh.faces))

    def test_left_wrist_intrinsics_match_the_rotated_recorded_pixels(self):
        def project(intrinsics, coefficients, xy):
            x, y = xy
            k1, k2, p1, p2, k3 = coefficients
            r2 = x*x+y*y; radial = 1+k1*r2+k2*r2*r2+k3*r2*r2*r2
            return np.array([intrinsics["fx"]*(x*radial+2*p1*x*y+p2*(r2+2*x*x))+intrinsics["cx"],
                             intrinsics["fy"]*(y*radial+p1*(r2+2*y*y)+2*p2*x*y)+intrinsics["cy"]])
        oriented = load_config()["cameras"]["wrist"]["left"]
        sensor_intrinsics = {**oriented["intrinsics"], "cx": 309.89324951171875, "cy": 248.1322021484375}
        sensor_distortion = [-.0560857392847538, .06375053524971008, .0007888228865340352,
                             .0001234443043358624, -.021498262882232666]
        for xy in ((0, 0), (.4, -.3), (-.6, .5)):
            sensor_pixel = project(sensor_intrinsics, sensor_distortion, xy)
            video_pixel = project(oriented["intrinsics"], oriented["distortion_coefficients"], -np.array(xy))
            np.testing.assert_allclose(video_pixel, [639, 479]-sensor_pixel, atol=1e-10)

    def test_invalid_camera_parameters_rejected(self):
        for key, value in (("fx", 0), ("fy", float("nan")), ("cx", 640)):
            config = copy.deepcopy(load_config()); config["cameras"]["head"]["intrinsics"][key] = value
            with self.assertRaises(ValueError):
                validate(config)
        config = load_config(); config["cameras"]["head"]["quaternion_xyzw"] = [0, 0, 0, 0]
        with self.assertRaises(ValueError):
            validate(config)


if __name__ == "__main__":
    unittest.main()
