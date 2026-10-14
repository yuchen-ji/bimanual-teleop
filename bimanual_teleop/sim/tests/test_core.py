import copy
import json
import tempfile
import unittest
from pathlib import Path

from bimanual_teleop.sim._dependencies import np
from bimanual_teleop.sim._dependencies import Rotation

from bimanual_teleop.sim import ROOT
from bimanual_teleop.sim.assets import rail_meshes
from bimanual_teleop.sim.config import load_config, sim_path
from bimanual_teleop.sim.kinematics import RobotModel
from bimanual_teleop.sim.protocol import ClockEstimate, ImageRing, InputGuard, pack, unpack


def input_frame(sequence=0, stamp=1_000_000_000, **changes):
    return {"kind": "input", "sequence": sequence, "session": "operator", "origin": "origin-0",
            "sample_ns": stamp, "send_ns": stamp, "valid": True, "fault_generation": 0,
            "controllers": {s: [0, 0, 0, 0, 0, 0, 1] for s in ("left", "right")},
            "hands": {s: [0.]*20 for s in ("left", "right")}, **changes}


class ConfigurationTests(unittest.TestCase):
    def test_lan_requires_operator_token(self):
        config = load_config(); config["network"]["bind"] = "0.0.0.0"
        from bimanual_teleop.sim.config import validate
        with self.assertRaises(ValueError):
            validate(config)

    def test_nondivisible_rates_rejected(self):
        config = load_config(); config["runtime"]["control_hz"] = 200
        from bimanual_teleop.sim.config import validate
        with self.assertRaises(ValueError):
            validate(config)


class ProtocolTests(unittest.TestCase):
    def guard(self):
        guard = InputGuard()
        guard.update_clock(ClockEstimate(5_000_000_000, 1_000_000, 6_000_000_000))
        return guard

    def test_clock_offset_and_old_input_not_retimed(self):
        estimate = ClockEstimate.from_exchange(1_000_000_000, 6_001_000_000, 6_001_100_000, 1_002_100_000)
        self.assertEqual(estimate.offset_ns, 5_000_000_000)
        guard = self.guard()
        self.assertTrue(guard.receive(input_frame(), 6_002_000_000))
        self.assertFalse(guard.receive(input_frame(1), 6_101_000_000))
        self.assertIn("expired", guard.fault)

    def test_invalid_then_valid_requires_reengage(self):
        guard = self.guard()
        guard.receive(input_frame(valid=False), 6_002_000_000)
        self.assertFalse(guard.receive(input_frame(1, 1_010_000_000), 6_012_000_000))
        self.assertTrue(guard.reengage(6_012_000_000))

    def test_hidden_tracking_fault_generation(self):
        guard = self.guard(); guard.receive(input_frame(), 6_002_000_000)
        self.assertFalse(guard.receive(input_frame(1, 1_010_000_000, fault_generation=1), 6_012_000_000))

    def test_conflated_fresh_samples_do_not_imply_device_fault(self):
        guard = self.guard(); guard.receive(input_frame(), 6_002_000_000)
        self.assertTrue(guard.receive(input_frame(30, 1_250_000_000), 6_252_000_000))
        # A real outage observed by the host still latches across fresh input.
        self.assertFalse(guard.healthy(6_351_000_000))
        self.assertFalse(guard.receive(input_frame(50, 1_360_000_000), 6_362_000_000))
        self.assertTrue(guard.reengage(6_362_000_000))

    def test_origin_and_out_of_order(self):
        guard = self.guard(); guard.receive(input_frame(), 6_002_000_000)
        self.assertFalse(guard.receive(input_frame(), 6_003_000_000))
        self.assertFalse(guard.receive(input_frame(1, 1_010_000_000, origin="origin-1"), 6_012_000_000))

    def test_protocol_version(self):
        self.assertEqual(unpack(pack({"kind": "status"}))["kind"], "status")
        import msgpack
        with self.assertRaises(ValueError):
            unpack(msgpack.packb({"version": 999}))

    def test_image_ring_ownership_and_overflow(self):
        owner = ImageRing.create(2, 16, 12)
        producer = ImageRing(**owner.descriptor())
        try:
            image = np.full(owner.shape, 17, dtype=np.uint8)
            self.assertEqual(producer.put(image), 0)
            self.assertEqual(producer.put(image), 1)
            with self.assertRaises(BufferError):
                producer.put(image)
            first = owner.take(0)
            producer.put(np.full(owner.shape, 23, dtype=np.uint8))
            self.assertTrue(np.all(first == 17))
            self.assertTrue(np.all(owner.take(1) == 17)); self.assertTrue(np.all(owner.take(2) == 23))
        finally:
            producer.close(); owner.close(owner=True)


class GeometryTests(unittest.TestCase):
    def test_real_holes_and_area(self):
        settings = load_config()["rail"]
        with tempfile.TemporaryDirectory() as directory:
            manifest = rail_meshes(directory, settings)
            radius, segments = settings["hole_diameter_m"]/2, settings["hole_segments"]
            expected = settings["size_m"][0]*settings["size_m"][1]-segments*radius**2*np.sin(2*np.pi/segments)
            self.assertAlmostEqual(manifest["cross_section_area_m2"], expected, places=12)
            import trimesh
            visual = trimesh.load(Path(directory)/"rail.obj", force="mesh")
            self.assertTrue(visual.is_watertight)
            self.assertTrue(visual.is_winding_consistent)
            self.assertAlmostEqual(visual.volume, expected*settings["size_m"][2], places=10)
            for name in manifest["colliders"]:
                mesh = trimesh.load(Path(directory)/name, force="mesh")
                self.assertTrue(mesh.is_watertight)
                lower_face = mesh.vertices[np.isclose(mesh.vertices[:, 2], -settings["size_m"][2]/2), :2]
                from scipy.spatial import Delaunay
                shape = Delaunay(lower_face)
                for center in manifest["hole_centers_m"]:
                    self.assertLess(shape.find_simplex(np.asarray(center[:2])), 0, "Collider fills a hole")

    @unittest.skipUnless((ROOT / "assets/generated/robot.urdf").exists(), "Prepare assets first")
    def test_jacobian_matches_finite_difference(self):
        config = load_config(); model = RobotModel(sim_path(config["robot"]["urdf"]))
        q = np.r_[np.deg2rad(config["robot"]["ready_deg"]["left"]), np.deg2rad(config["robot"]["ready_deg"]["right"]), np.zeros(40)]
        for side, offset in (("left", 0), ("right", 7)):
            jacobian = model.jacobian(side, q)
            for index in range(7):
                plus, minus = q.copy(), q.copy(); plus[offset+index] += 1e-6; minus[offset+index] -= 1e-6
                a, b = model.flange(side, plus, world=True), model.flange(side, minus, world=True)
                linear = (a[:3, 3]-b[:3, 3])/2e-6
                angular = (Rotation.from_matrix(a[:3, :3])*Rotation.from_matrix(b[:3, :3]).inv()).as_rotvec()/2e-6
                np.testing.assert_allclose(jacobian[:, index], np.r_[linear, angular], atol=1e-8)

    @unittest.skipUnless((ROOT / "assets/generated/robot.urdf").exists(), "Prepare assets first")
    def test_world_shoulders_and_outward_hand_mounts(self):
        config = load_config(); model = RobotModel(sim_path(config["robot"]["urdf"]))
        q = np.r_[np.deg2rad(config["robot"]["ready_deg"]["left"]), np.deg2rad(config["robot"]["ready_deg"]["right"]), np.zeros(40)]
        for side, prefix, sign in (("left", "l", 1), ("right", "r", -1)):
            np.testing.assert_allclose(model.base_pose(side)[:3, 3], [0, sign*.2005, 1.121], atol=1e-6)
            flange = model.flange(side, q, world=True)
            tip = model.link_pose(prefix+"_middle_finger_tip", q)
            local = np.linalg.inv(flange) @ np.r_[tip[:3, 3], 1]
            self.assertGreater(local[2], .15, "Fingers must extend outward from the arm flange")


if __name__ == "__main__":
    unittest.main()
