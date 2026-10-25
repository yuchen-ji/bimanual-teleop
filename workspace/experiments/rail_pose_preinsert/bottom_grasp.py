"""Fit and verify four inward distal pads against the rail's underside."""
import json
from pathlib import Path

import fcl
import numpy as np
from grasp_fit import FINGERS, GraspFit
from scipy.optimize import minimize


class BottomGraspFit(GraspFit):
    def __init__(self, *args):
        super().__init__(*args)
        self.indices = np.arange(14, 34)
        names = [n for n in self.names if n.endswith(("_middle", "_distal", "_tip_sensor_frame"))]
        self.self_pairs = [(a, b) for i, a in enumerate(names) for b in names[i+1:]
                           if a.split("_")[1] != b.split("_")[1]]
        self.bottom_z = float(self.rail_mesh.bounds[0, 2])
        self.pad_vertices = {f: dict(self.meshes)["l_"+f+"_tip_sensor_frame"].vertices for f in FINGERS[1:]}

    def self_distances(self):
        # Call distances first to update the same actual-pose-corrected objects.
        return np.asarray([fcl.distance(self.objects[a], self.objects[b], self.request, fcl.DistanceResult())
                           for a, b in self.self_pairs])

    def bottom_metrics(self, q, xyz):
        transforms = self.transforms(q)
        result = {}
        for f, vertices in self.pad_vertices.items():
            t = transforms["l_"+f+"_tip_sensor_frame"]
            points = (vertices @ t[:3, :3].T+t[:3, 3]-xyz) @ self.rail_pose[:3, :3]
            j = int(np.argmax(points[:, 2]))
            result[f] = {"gap_m": float(self.bottom_z-points[j, 2]),
                         "rail_point_m": points[j].tolist(), "pad_point_m": vertices[j].tolist(),
                         "inward_alignment": float(-t[:3, 1] @ self.rail_pose[:3, 2])}
        return result

    def solve_bottom(self, initial=None):
        # Bent distal joints put the inward (-Y) pad surface beneath the rail.
        seed = json.loads((Path(__file__).parent / "hand_profiles/bottom_grasp_seed.json").read_text())
        x0 = np.r_[seed["hand_seed"], (np.asarray(seed["rail_xyz_m"])-self.rail_pose[:3, 3])/.01]
        if initial is not None:
            x0 = np.asarray(initial)
        bounds = list(zip(self.model.lower[self.indices], self.model.upper[self.indices])) + [(-4, 4)]*3
        for i in range(4):
            for j in (4+4*i, 6+4*i, 7+4*i):
                bounds[j] = (0, bounds[j][1])

        def objective(x):
            q, xyz = self.unpack(x)
            metrics = self.bottom_metrics(q, xyz)
            value = 20*((self.evaluate(x)[self.tips[0]]-.001)/.01)**2
            for m in metrics.values():
                value += ((m["gap_m"]-.001)/.01)**2 + .03*((1-m["inward_alignment"])/.1)**2
                value += .05*(m["rail_point_m"][1]/.01)**2
            return value+.000001*np.sum((x-x0)**2)

        def constraints(x):
            q, xyz = self.unpack(x)
            metrics = self.bottom_metrics(q, xyz)
            values = list((self.evaluate(x)-.0007)/.01) + list((self.self_distances()-.0007)/.01)
            for m in metrics.values():
                values.extend([(m["gap_m"]-.0008)/.01, m["inward_alignment"]-.95,
                               (.008-abs(m["rail_point_m"][1]))/.01, (-.004-m["pad_point_m"][1])/.01])
            return np.asarray(values)

        result = minimize(objective, x0, method="SLSQP", bounds=bounds,
                          constraints=[{"type": "ineq", "fun": constraints}],
                          options={"maxiter": 200, "ftol": 1e-10, "eps": 2e-6})
        plan = self.pack(result.x, bool(result.success), result.message, result.nit)
        plan["bottom_contact"] = self.bottom_metrics(*self.unpack(result.x))
        plan["minimum_constraint"] = float(constraints(result.x).min())
        return plan

    def plan(self):
        seed = json.loads((Path(__file__).parent / "hand_profiles/bottom_grasp_seed.json").read_text())
        x0 = np.r_[seed["hand_seed"], (np.asarray(seed["rail_xyz_m"])-self.rail_pose[:3, 3])/.01]
        plan = self.pack(x0, True, "Revalidated bottom-wrap seed", 0)
        metrics = self.bottom_metrics(*self.unpack(x0))
        if (plan["minimum_envelope_gap_m"] < .00035 or self.self_distances().min() < .0005 or
                any(m["gap_m"] < .0007 or m["gap_m"] > .0015 or m["inward_alignment"] < .94
                    or abs(m["rail_point_m"][1]) > .008 for m in metrics.values())):
            plan = self.solve_bottom(x0)
        q = np.asarray(plan["q"])
        xyz = np.asarray(plan["rail_xyz_m"])
        metrics = self.bottom_metrics(q, xyz)
        if plan["minimum_envelope_gap_m"] < .0003 or self.self_distances().min() < .0003 or any(m["gap_m"] < .0006 or m["gap_m"] > .002
                or m["inward_alignment"] < .94 or abs(m["rail_point_m"][1]) > .008 for m in metrics.values()):
            raise RuntimeError("No clear bottom-pad grasp found")
        thumb = self.q.copy(); thumb[14:18] = q[14:18]
        minimum = float("inf")
        self_minimum = float("inf")
        for u in np.linspace(0, 1, 151):
            for joints, pos in (((1-u)*self.q+u*thumb, self.rail_pose[:3, 3]),
                                (thumb, (1-u)*self.rail_pose[:3, 3]+u*xyz),
                                ((1-u)*thumb+u*q, xyz)):
                minimum = min(minimum, float(self.distances(joints, pos).min()))
                self_minimum = min(self_minimum, float(self.self_distances().min()))
        if minimum < 0 or self_minimum < 0:
            raise RuntimeError(f"Bottom grasp path intersects geometry: {minimum}")
        plan.update(bottom_contact=metrics, thumb_before_translation=True, sampled_path_minimum_gap_m=minimum, sampled_interfinger_gap_m=self_minimum)
        return plan
