"""Host-side mesh checks. Reads geometry only; never changes collision bodies."""
import json
from pathlib import Path

import numpy as np
import trimesh
from four_fingers import FINGERS
from geometry_audit import hand_meshes

from bimanual_teleop.sim.kinematics import pose_matrix


class FourFingerGeometry:
    def __init__(self, model, urdf, manifest_path):
        self.model = model
        self.all = trimesh.collision.CollisionManager()
        self.fingers = {f: trimesh.collision.CollisionManager() for f in FINGERS}
        self.distal = {f: trimesh.collision.CollisionManager() for f in ("thumb", *FINGERS)}
        self.free_segments = trimesh.collision.CollisionManager()
        self.names = []
        for name, mesh in hand_meshes(urdf):
            self.names.append(name)
            self.all.add_object(name, mesh)
            if name.endswith(("_middle", "_distal", "_tip_sensor_frame")):
                self.free_segments.add_object(name, mesh)
            for f, manager in self.fingers.items():
                if name.startswith("l_"+f+"_"):
                    manager.add_object(name, mesh)
            for f, manager in self.distal.items():
                if name in ("l_"+f+"_distal", "l_"+f+"_tip_sensor_frame"):
                    manager.add_object(name, mesh)
        path = Path(manifest_path)
        manifest = json.loads(path.read_text())
        self.rail = trimesh.load(path.parent / "rail" / manifest["rail"]["visual"], force="mesh")

    def check(self, state):
        q = np.asarray(state["q"])
        flange = self.model.base_pose("left") @ pose_matrix(state["eef"][0])
        correction = flange @ np.linalg.inv(self.model.flange("left", q, world=True))
        for name in self.names:
            measured = state.get("hand_link_poses", {}).get(name)
            transform = pose_matrix(measured) if measured is not None else correction @ self.model.link_pose(name, q)
            self.all.set_transform(name, transform)
            if name.endswith(("_middle", "_distal", "_tip_sensor_frame")):
                self.free_segments.set_transform(name, transform)
            for f, manager in self.fingers.items():
                if name.startswith("l_"+f+"_"):
                    manager.set_transform(name, transform)
            for f, manager in self.distal.items():
                if name in ("l_"+f+"_distal", "l_"+f+"_tip_sensor_frame"):
                    manager.set_transform(name, transform)
        rail_pose = pose_matrix(state["rail"])
        _, names = self.all.in_collision_single(self.rail, rail_pose, return_names=True)
        _, pairs = self.free_segments.in_collision_internal(return_names=True)
        interfinger = sorted([sorted((a, b)) for a, b in pairs if a.split("_")[1] != b.split("_")[1]])
        gaps = {f: float(manager.min_distance_single(self.rail, rail_pose)) for f, manager in self.fingers.items()}
        bottom = {}
        for f in FINGERS:
            gap, name, data = self.distal[f].min_distance_single(self.rail, rail_pose, return_name=True, return_data=True)
            rail_point = (data.point("__external")-rail_pose[:3, 3]) @ rail_pose[:3, :3]
            measured = state.get("hand_link_poses", {}).get(name)
            t = pose_matrix(measured) if measured is not None else correction @ self.model.link_pose(name, q)
            pad_point = (data.point(name)-t[:3, 3]) @ t[:3, :3]
            alignment = float(-t[:3, 1] @ rail_pose[:3, 2])
            bottom[f] = {"gap_m": float(gap), "rail_point_m": rail_point.tolist(),
                         "pad_point_m": pad_point.tolist(), "inward_alignment": alignment,
                         "face_ok": bool(name.endswith("_tip_sensor_frame") and
                                         abs(rail_point[2]-self.rail.bounds[0, 2]) < .00002 and
                                         abs(rail_point[1]) < .009 and abs(rail_point[0]) < .1 and
                                         pad_point[1] < -.004 and alignment >= .94)}
        return {"intersecting_links": sorted(names), "finger_gap_m": gaps, "interfinger_intersections": interfinger,
                "bottom_contact": bottom, "bottom_contact_face_ok": all(v["face_ok"] for v in bottom.values()),
                "distal_gap_m": {f: float(manager.min_distance_single(self.rail, rail_pose)) for f, manager in self.distal.items()},
                "time_ns": state["time_ns"], "method": "Full URDF mesh triangle intersection; " +
                ("measured PhysX link poses" if state.get("hand_link_poses") else "measured joint and flange poses")}
