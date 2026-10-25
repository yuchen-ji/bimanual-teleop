"""Fit distal contacts with a fixed rail rotation, without editing colliders."""
from __future__ import annotations

import json
from pathlib import Path

import fcl
import numpy as np
import trimesh
from geometry_audit import hand_meshes
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from bimanual_teleop.sim.kinematics import pose_matrix

FINGERS = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")


def convex(mesh):
    hull = mesh.convex_hull
    faces = np.column_stack((np.full(len(hull.faces), 3), hull.faces)).flatten()
    return fcl.Convex(hull.vertices, len(hull.faces), faces)


class GraspFit:
    """Conservative hull envelope for planning; actual mesh/PhysX checks follow."""

    def __init__(self, model, urdf, manifest_path, state):
        self.model = model
        self.q = np.asarray(state["q"]).copy()
        self.rail_pose = pose_matrix(state["rail"])
        flange = model.base_pose("left") @ pose_matrix(state["eef"][0])
        self.correction = flange @ np.linalg.inv(model.flange("left", self.q, world=True))
        self.link_corrections = {name: pose_matrix(vector) @ np.linalg.inv(self.correction @ model.link_pose(name, self.q))
                                 for name, vector in state.get("hand_link_poses", {}).items()}
        self.meshes = hand_meshes(urdf)
        self.objects = {name: fcl.CollisionObject(convex(mesh)) for name, mesh in self.meshes}
        self.names = list(self.objects)
        self.tips = [self.names.index("l_" + f + "_tip_sensor_frame") for f in FINGERS]
        p = Path(manifest_path)
        manifest = json.loads(p.read_text())
        rail = trimesh.load(p.parent / "rail" / manifest["rail"]["visual"], force="mesh")
        self.rail_mesh = rail
        self.rail = fcl.CollisionObject(convex(rail))
        self.request = fcl.DistanceRequest(enable_nearest_points=True, enable_signed_distance=True)
        self.indices = np.array([14, 15, 16, 17] + [j for i in range(4) for j in (18+4*i, 20+4*i, 21+4*i)])
        self.last_x = None
        self.last_distances = None

    def transforms(self, q):
        positions = dict(zip(self.model.names, q))
        cache = {}

        def visit(name):
            if name not in cache:
                joint = self.model.by_child.get(name)
                if joint is None:
                    cache[name] = self.correction @ self.model.world
                else:
                    matrix = visit(joint.parent) @ joint.origin
                    if joint.kind == "revolute":
                        rotation = Rotation.from_rotvec(joint.axis * positions[joint.name]).as_matrix()
                        matrix[:3, :3] = matrix[:3, :3] @ rotation
                    cache[name] = matrix
            return cache[name]
        return {name: self.link_corrections.get(name, np.eye(4)) @ visit(name) for name in self.names}

    def distances(self, q, xyz):
        self.rail.setTransform(fcl.Transform(self.rail_pose[:3, :3], xyz))
        result = []
        for name, matrix in self.transforms(q).items():
            obj = self.objects[name]
            obj.setTransform(fcl.Transform(matrix[:3, :3], matrix[:3, 3]))
            result.append(fcl.distance(obj, self.rail, self.request, fcl.DistanceResult()))
        return np.asarray(result)

    def unpack(self, x):
        q = self.q.copy()
        q[self.indices] = x[:-3]
        xyz = self.rail_pose[:3, 3] + .01*x[-3:]
        return q, xyz

    def evaluate(self, x):
        if self.last_x is None or not np.array_equal(x, self.last_x):
            self.last_distances = self.distances(*self.unpack(x))
            self.last_x = x.copy()
        return self.last_distances

    def solve(self, initial=None):
        x0 = np.r_[self.q[self.indices], np.zeros(3)] if initial is None else np.asarray(initial)
        bounds = list(zip(self.model.lower[self.indices], self.model.upper[self.indices])) + [(-4, 4)]*3
        # Flex normally, allowing the thumb's opposing axes their full range.
        bounds[4:] = [(max(0, a), b) for a, b in bounds[4:-3]] + bounds[-3:]

        def objective(x):
            gaps = self.evaluate(x)[self.tips]
            return np.sum(((gaps-.0002)/.01)**2) + .000003*np.sum((x-x0)**2)

        result = minimize(objective, x0, method="SLSQP", bounds=bounds,
                          constraints=[{"type": "ineq", "fun": lambda x: (self.evaluate(x)-.0002)/.01}],
                          options={"maxiter": 240, "ftol": 1e-10, "eps": 2e-6})
        return self.pack(result.x, bool(result.success), result.message, result.nit)

    def pack(self, x, success, message, iterations):
        q, xyz = self.unpack(x)
        distances = self.distances(q, xyz)
        return {"q": q.tolist(), "rail_xyz_m": xyz.tolist(), "rail_quaternion_xyzw":
                Rotation.from_matrix(self.rail_pose[:3, :3]).as_quat().tolist(),
                "tip_gap_m": dict(zip(FINGERS, distances[self.tips].tolist())),
                "minimum_envelope_gap_m": float(distances.min()),
                "solver_success": success, "solver_message": message,
                "x": np.asarray(x).tolist(), "iterations": iterations,
                "planning_geometry": "Conservative convex envelopes; existing simulation colliders unchanged"}

    def plan(self):
        # Seed a wrap around opposite rail sides; fit all geometry anew against
        # the measured scene. These are starting angles, never sent directly.
        seed_path = Path(__file__).parent / "hand_profiles/locked_grasp_seed.json"
        initial = np.asarray(json.loads(seed_path.read_text())["optimizer_seed"])
        result = self.pack(initial, True, "Previously fitted seed revalidated against the current measured scene", 0)
        if result["minimum_envelope_gap_m"] < .00015 or max(result["tip_gap_m"].values()) > .0005:
            result = self.solve(initial)
        if result["minimum_envelope_gap_m"] < .00009 or max(result["tip_gap_m"].values()) > .0006:
            raise RuntimeError("No clear five-distal-contact grasp found")
        # Stop the first drive trajectory 1 mm short. Actual articulation and
        # gravity deflections are then handled by measured-surface refinement,
        # instead of driving straight to a submillimetre kinematic target.
        q = np.asarray(result["q"])
        xyz = np.asarray(result["rail_xyz_m"])
        groups = [np.arange(14, 18)] + [np.array([18+4*i, 20+4*i, 21+4*i]) for i in range(4)]
        for _ in range(12):
            for i, indices in enumerate(groups):
                gap = self.distances(q, xyz)[self.tips[i]]
                if gap >= .00095:
                    continue
                gradient = []
                for j in indices:
                    moved = q.copy(); moved[j] += .0001
                    gradient.append((self.distances(moved, xyz)[self.tips[i]]-gap)/.0001)
                gradient = np.asarray(gradient)
                scale = np.linalg.norm(gradient)
                if scale > 1e-8:
                    q[indices] += min(.03, .7*(.001-gap)/scale)*gradient/scale
                    q = np.clip(q, self.model.lower, self.model.upper)
        result = self.pack(np.r_[q[self.indices], result["x"][-3:]], True,
                           "Validated wrap with 1 mm approach clearance; final alignment uses measured surfaces", 0)
        if result["minimum_envelope_gap_m"] < .00009:
            raise RuntimeError("Approach clearance adjustment intersects hand geometry")
        target = np.asarray(result["q"])
        xyz = np.asarray(result["rail_xyz_m"])
        minimum = float("inf")
        # First translate the supported rail, then close the hand. A dense
        # sampled check supplements (does not replace) live force/mesh checks.
        for u in np.linspace(0, 1, 151):
            minimum = min(minimum, float(self.distances(self.q, (1-u)*self.rail_pose[:3, 3]+u*xyz).min()))
            minimum = min(minimum, float(self.distances((1-u)*self.q+u*target, xyz).min()))
        if minimum < 0:
            raise RuntimeError(f"Proposed grasp path intersects conservative geometry: {minimum}")
        result["sampled_path_minimum_gap_m"] = minimum
        return result
