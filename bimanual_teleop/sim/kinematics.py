"""URDF kinematics and bounded velocity IK, independent of the vendor SDK."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

from bimanual_teleop.sim._dependencies import np
from bimanual_teleop.sim._dependencies import Rotation


def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    result[:3, 3] = xyz
    return result


def pose_vector(matrix):
    return np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()]


def pose_matrix(vector):
    result = np.eye(4)
    result[:3, 3] = vector[:3]
    result[:3, :3] = Rotation.from_quat(vector[3:]).as_matrix()
    return result


@dataclass
class Joint:
    name: str
    parent: str
    child: str
    origin: object
    axis: object
    kind: str
    lower: float = 0
    upper: float = 0
    effort: float = 0
    velocity: float = 0


class RobotModel:
    def __init__(self, path, yaw_deg=90):
        root = ET.parse(path).getroot()
        self.collidable_links = tuple(link.get("name") for link in root.findall("link") if link.find("collision") is not None)
        self.joints = {}
        self.by_child = {}
        for element in root.findall("joint"):
            origin = element.find("origin")
            xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()] if origin is not None else [0]*3
            rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()] if origin is not None else [0]*3
            axis = element.find("axis")
            values = np.array([float(v) for v in axis.get("xyz").split()]) if axis is not None else np.array([0., 0., 1.])
            limit = element.find("limit")
            joint = Joint(element.get("name"), element.find("parent").get("link"), element.find("child").get("link"),
                          transform(xyz, rpy), values, element.get("type"),
                          *[float(limit.get(k, "0")) if limit is not None else 0 for k in ("lower", "upper", "effort", "velocity")])
            self.joints[joint.name] = joint
            self.by_child[joint.child] = joint
        self.world = transform(rpy=(0, 0, np.deg2rad(yaw_deg)))
        if "sim_world_frame" in self.joints:
            if not np.allclose(self.joints["sim_world_frame"].origin, self.world):
                raise ValueError("World yaw differs from prepared assets; regenerate sim.assets")
            self.world = np.eye(4)
        self.arm_names = {side: tuple(f"Joint{i}_{suffix}" for i in range(1, 8)) for side, suffix in (("left", "L"), ("right", "R"))}
        self.hand_names = {side: tuple(j.name for j in self.joints.values() if j.kind == "revolute" and j.name.startswith(prefix))
                           for side, prefix in (("left", "l_"), ("right", "r_"))}
        self.names = (*self.arm_names["left"], *self.arm_names["right"], *self.hand_names["left"], *self.hand_names["right"])
        if len(self.names) != 54 or any(len(names) != 20 for names in self.hand_names.values()):
            raise ValueError("Expected two 7-axis arms and two 20-axis hands")
        self.lower = np.array([self.joints[n].lower for n in self.names])
        self.upper = np.array([self.joints[n].upper for n in self.names])
        self.effort = np.array([self.joints[n].effort for n in self.names])

    def chain(self, link):
        result, seen = [], set()
        while link in self.by_child:
            if link in seen:
                raise ValueError("URDF joint graph contains a cycle")
            seen.add(link)
            joint = self.by_child[link]; result.append(joint); link = joint.parent
        return list(reversed(result))

    def link_pose(self, link, positions):
        q = dict(zip(self.names, positions))
        matrix = self.world.copy()
        for joint in self.chain(link):
            matrix = matrix @ joint.origin
            if joint.kind == "revolute":
                motion = np.eye(4)
                motion[:3, :3] = Rotation.from_rotvec(joint.axis * q.get(joint.name, 0)).as_matrix()
                matrix = matrix @ motion
        return matrix

    def base_pose(self, side):
        first = self.joints[self.arm_names[side][0]]
        return self.link_pose(first.parent, np.zeros(54)) @ first.origin

    def flange(self, side, positions, world=False):
        matrix = self.link_pose("TCP_Link_L" if side == "left" else "TCP_Link_R", positions)
        return matrix if world else np.linalg.inv(self.base_pose(side)) @ matrix

    def jacobian(self, side, positions):
        names = self.arm_names[side]
        q = dict(zip(self.names, positions))
        flange = self.flange(side, positions, world=True)
        matrix = self.world.copy(); result = []
        for joint in self.chain("TCP_Link_L" if side == "left" else "TCP_Link_R"):
            matrix = matrix @ joint.origin
            if joint.name in names:
                axis = matrix[:3, :3] @ joint.axis
                result.append(np.r_[np.cross(axis, flange[:3, 3]-matrix[:3, 3]), axis])
            if joint.kind == "revolute":
                motion = np.eye(4); motion[:3, :3] = Rotation.from_rotvec(joint.axis*q.get(joint.name, 0)).as_matrix()
                matrix = matrix @ motion
        return np.stack(result, axis=1)

    def velocity_ik(self, side, positions, target_world, dt, settings, command_positions=None):
        current = self.flange(side, positions, world=True)
        error = np.r_[target_world[:3, 3]-current[:3, 3],
                      (Rotation.from_matrix(target_world[:3, :3]) * Rotation.from_matrix(current[:3, :3]).inv()).as_rotvec()]
        velocity = error * 4
        for group, maximum in ((slice(0, 3), settings["max_linear_speed_m_s"]), (slice(3, 6), settings["max_angular_speed_rad_s"])):
            norm = np.linalg.norm(velocity[group])
            if norm > maximum:
                velocity[group] *= maximum / norm
        jac = self.jacobian(side, positions)
        weights = np.diag([1, 1, 1, .25, .25, .25])
        weighted = weights @ jac
        delta = weighted.T @ np.linalg.solve(weighted @ weighted.T + .0025*np.eye(6), weights @ velocity)
        offset = 0 if side == "left" else 7
        selected = slice(offset, offset+7)
        maximum = np.array([min(self.joints[name].velocity, settings["joint_speed_rad_s"]) for name in self.arm_names[side]])
        delta = np.clip(delta, -maximum, maximum)
        reference = positions if command_positions is None else command_positions
        return np.clip(np.asarray(reference)[selected] + delta*dt, self.lower[selected], self.upper[selected])
