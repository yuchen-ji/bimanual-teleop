"""Lula IK, RRT and time-parameterized trajectories for one moving arm."""

import json
import hashlib
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from bimanual_teleop.sim.bin_geometry import bin_geometry
from bimanual_teleop.sim.config import sim_path


def load_lula():
    try:
        import lula
    except ImportError:
        sys.path.insert(0, "/isaac-sim/exts/isaacsim.robot_motion.lula/pip_prebundle")
        import lula
    return lula


class ArmPlanner:
    def __init__(self, config, model, task, output):
        self.lula = load_lula()
        self.config, self.model, self.task = config, model, task
        self.output = Path(output)/"planning"; self.output.mkdir(parents=True, exist_ok=True)
        self.serial = 0
        self.spheres = json.loads((Path(__file__).parent/"collision_spheres.json").read_text())
        metadata = json.loads((Path(__file__).parent/"collision_spheres.metadata.json").read_text())
        root = sim_path(config["robot"]["urdf"]).parent
        manifest = json.loads((root/"manifest.json").read_text())
        if (metadata["robot_urdf_sha256"] != hashlib.sha256(sim_path(config["robot"]["urdf"]).read_bytes()).hexdigest()
                or metadata["robot_signature"] != manifest["robot_signature"]
                or metadata["fixture_sha256"] != hashlib.sha256((root/"task/fixture_collision.obj").read_bytes()).hexdigest()):
            raise ValueError("Planning geometry is stale; run build_planning_model.py in the host Conda environment")
        self.fixture_bounds = np.asarray(metadata["fixture_bounds_m"])

    def pose(self, matrix):
        return self.lula.Pose3(self.lula.Rotation3(matrix[:3, :3]), matrix[:3, 3])

    def robot(self, side, q, reference, payload=None):
        self.serial += 1
        names = self.model.arm_names[side]
        active = set(names)
        prefix, suffix = ("l_", "_L") if side == "left" else ("r_", "_R")
        collision = {link: spheres for link, spheres in self.spheres.items()
                     if link.startswith(prefix) or (link.startswith("Link") and link.endswith(suffix) and not link.startswith(("Link1_", "Link2_")))}
        descriptor = {"api_version": "1.0", "root_link": "sim_world", "cspace": list(names), "default_q": reference.tolist(),
                      "acceleration_limits": [2.]*7, "jerk_limits": [20.]*7,
                      "cspace_to_urdf_rules": [{"name": name, "rule": "fixed", "value": float(value)}
                                               for name, value in zip(self.model.names, q) if name not in active],
                      "collision_spheres": [{link: spheres} for link, spheres in collision.items()]}
        urdf = sim_path(self.config["robot"]["urdf"])
        # Planning payload is collision geometry only; no simulation attachment.
        if payload is not None:
            from scipy.spatial.transform import Rotation
            tree = ET.parse(urdf); root = tree.getroot()
            ET.SubElement(root, "link", name="planning_rail")
            joint = ET.SubElement(root, "joint", name="planning_rail_fixed", type="fixed")
            ET.SubElement(joint, "parent", link="TCP_Link_L" if side == "left" else "TCP_Link_R")
            ET.SubElement(joint, "child", link="planning_rail")
            ET.SubElement(joint, "origin", xyz=" ".join(map(str, payload[:3, 3])),
                          rpy=" ".join(map(str, Rotation.from_matrix(payload[:3, :3]).as_euler("xyz"))))
            descriptor["collision_spheres"].append({"planning_rail": [{"center": [float(x), 0., 0.], "radius": .027}
                                                                       for x in np.linspace(-.23, .23, 18)]})
            # Relative meshes must still resolve when the temporary URDF moves.
            for mesh in root.findall(".//mesh"):
                mesh.set("filename", str((urdf.parent/mesh.get("filename")).resolve()))
            urdf = self.output/f"robot_{self.serial}.urdf"; tree.write(urdf)
        path = self.output/f"{side}_{self.serial}.yaml"
        path.write_text(yaml.safe_dump(descriptor, sort_keys=False))
        return self.lula.load_robot(str(path), str(urdf))

    def solve(self, side, q, goal, reference):
        offset = 0 if side == "left" else 7
        robot = self.robot(side, q, reference)
        config = self.lula.CyclicCoordDescentIkConfig()
        config.position_tolerance = .0002; config.orientation_tolerance = .002
        config.max_num_descents = 5
        choices = []
        seeds = [reference, np.asarray(q[offset:offset+7]), (self.model.lower[offset:offset+7]+self.model.upper[offset:offset+7])/2]
        for seed in seeds:
            config.cspace_seeds = [seed]
            result = self.lula.compute_ik_ccd(robot.kinematics(), self.pose(goal),
                                              "TCP_Link_L" if side == "left" else "TCP_Link_R", config)
            answer = np.asarray(result.cspace_position)
            if result.success and np.all(answer >= self.model.lower[offset:offset+7]+.025) and np.all(answer <= self.model.upper[offset:offset+7]-.025):
                score = np.linalg.norm(answer-reference)+.3*np.linalg.norm(answer-q[offset:offset+7])
                choices.append((score, answer.copy()))
        if not choices:
            raise ValueError(f"{side} IK has no solution with joint-limit margin")
        return min(choices, key=lambda pair: pair[0])[1]

    def add_box(self, world, center, size):
        cube = self.lula.create_obstacle(self.lula.Obstacle.Type.CUBE)
        cube.set_attribute(self.lula.Obstacle.Attribute.SIDE_LENGTHS, np.asarray(size))
        handle = world.add_obstacle(cube, self.lula.Pose3.from_translation(np.asarray(center)))
        self.obstacle_names.append((handle, f"box:{np.asarray(center).round(3).tolist()}"))

    def world(self, side, q, rail=None):
        world = self.lula.create_world()
        self.obstacle_names = []
        scene = self.config["scene"]
        center = np.asarray(scene["table_center"])+[0, 0, scene["table_size"][2]-.025]
        self.add_box(world, center, [*scene["table_size"][:2], .05])
        self.add_box(world, np.asarray(scene["fixture_origin"])+self.fixture_bounds.mean(axis=0), np.diff(self.fixture_bounds, axis=0)[0])
        self.add_box(world, [0., 0., .5], [.24, .30, 1.0])
        crate = bin_geometry(scene)
        for origin in scene["bin_centers"]:
            x, y, z = origin; length, width = scene["bin_inner_size"][:2]; height = crate["rim_height_m"]
            self.add_box(world, [x, y, z+crate["floor_height_m"]/2], [length+.02, width+.02, crate["floor_height_m"]])
            for sign in (-1, 1):
                self.add_box(world, [x+sign*(length+.01)/2, y, z+height/2], [.01, width+.02, height])
                self.add_box(world, [x, y+sign*(width+.01)/2, z+height/2], [length, .01, height])
        inactive = "r_" if side == "left" else "l_"
        suffix = "_R" if side == "left" else "_L"
        for link, spheres in self.spheres.items():
            if not (link.startswith(inactive) or link.startswith("Link") and link.endswith(suffix)):
                continue
            pose = self.model.link_pose(link, q)
            for value in spheres:
                obstacle = self.lula.create_obstacle(self.lula.Obstacle.Type.SPHERE)
                obstacle.set_attribute(self.lula.Obstacle.Attribute.RADIUS, value["radius"])
                center = pose[:3, :3]@value["center"]+pose[:3, 3]
                handle = world.add_obstacle(obstacle, self.lula.Pose3.from_translation(center))
                self.obstacle_names.append((handle, link))
        if rail is not None:
            for x in np.linspace(-.23, .23, 18):
                obstacle = self.lula.create_obstacle(self.lula.Obstacle.Type.SPHERE)
                obstacle.set_attribute(self.lula.Obstacle.Attribute.RADIUS, .027)
                handle = world.add_obstacle(obstacle, self.lula.Pose3.from_translation(rail[:3, :3]@[x, 0., 0.]+rail[:3, 3]))
                self.obstacle_names.append((handle, "rail"))
        return world

    def collision_hits(self, side, q, view, payload=None):
        prefix, suffix = ("l_", "_L") if side == "left" else ("r_", "_R")
        hits = []
        for link, spheres in self.spheres.items():
            if not (link.startswith(prefix) or (link.startswith("Link") and link.endswith(suffix) and not link.startswith(("Link1_", "Link2_")))):
                continue
            matrix = self.model.link_pose(link, q)
            for sphere in spheres:
                center = matrix[:3, :3]@sphere["center"]+matrix[:3, 3]
                if view.in_collision(center, sphere["radius"]):
                    matches = [name for handle, name in self.obstacle_names if view.in_collision(handle, center, sphere["radius"])]
                    hits.append({"link": link, "obstacles": matches})
        if payload is not None:
            matrix = self.model.flange(side, q, True)@payload
            for x in np.linspace(-.23, .23, 18):
                center = matrix[:3, :3]@[x, 0., 0.]+matrix[:3, 3]
                if view.in_collision(center, .027):
                    matches = [name for handle, name in self.obstacle_names if view.in_collision(handle, center, .027)]
                    hits.append({"link": "planning_rail", "obstacles": matches})
        return hits

    def plan(self, side, q, goal, reference, label, *, payload=None, rail=None):
        start = time.monotonic(); offset = 0 if side == "left" else 7
        target = self.solve(side, q, goal, reference)
        robot = self.robot(side, q, reference, payload)
        world = self.world(side, q, rail)
        view = world.add_world_view(); view.update()
        destination = np.array(q, copy=True); destination[offset:offset+7] = target
        for name, positions in (("start", q), ("target", destination)):
            hits = self.collision_hits(side, positions, view, payload)
            if hits:
                (self.output/f"{label}_{name}_collisions.json").write_text(json.dumps(hits, indent=2))
                raise ValueError(f"Planning {name} overlaps collision model: {hits[:3]}")
        planner = self.lula.create_motion_planner(robot, view)
        planner.set_param("seed", 23); planner.set_param("step_size", .04); planner.set_param("max_iterations", 10000)
        result = planner.plan_to_cspace_target(np.asarray(q[offset:offset+7]), target, True)
        if not result.path_found:
            raise ValueError(f"No collision-free RRT path for {label}")
        waypoints = np.asarray(result.path)
        generator = self.lula.create_c_space_trajectory_generator(robot.kinematics())
        generator.set_velocity_limits(np.full(7, min(self.config["runtime"]["joint_speed_rad_s"], .6)))
        generator.set_acceleration_limits(np.full(7, 1.5)); generator.set_jerk_limits(np.full(7, 10.))
        trajectory = generator.generate_trajectory(waypoints)
        if trajectory is None:
            raise ValueError(f"Lula time parameterization failed: {label}")
        duration = trajectory.domain().upper
        # Check the interpolated trajectory, including between the RRT vertices.
        prefix, suffix = ("l_", "_L") if side == "left" else ("r_", "_R")
        active_spheres = {link: spheres for link, spheres in self.spheres.items()
                          if link.startswith(prefix) or (link.startswith("Link") and link.endswith(suffix) and not link.startswith(("Link1_", "Link2_")))}
        for stamp in np.linspace(0., duration, max(2, int(duration*60)+1)):
            values = np.asarray(trajectory.eval(float(stamp)))
            candidate = np.asarray(q).copy(); candidate[offset:offset+7] = values
            if np.any(values < self.model.lower[offset:offset+7]) or np.any(values > self.model.upper[offset:offset+7]):
                raise ValueError(f"Interpolated trajectory violates joint limits: {label}")
            for link, spheres in active_spheres.items():
                matrix = self.model.link_pose(link, candidate)
                for sphere in spheres:
                    center = matrix[:3, :3]@sphere["center"]+matrix[:3, 3]
                    if view.in_collision(center, sphere["radius"]):
                        raise ValueError(f"Interpolated trajectory collides: {label}, {link}")
            if payload is not None:
                matrix = self.model.flange(side, candidate, True)@payload
                for x in np.linspace(-.23, .23, 18):
                    if view.in_collision(matrix[:3, :3]@[x, 0., 0.]+matrix[:3, 3], .027):
                        raise ValueError(f"Interpolated payload collides: {label}")
        report = {"label": label, "side": side, "engine": "Lula RRT + Lula CSpaceTrajectoryGenerator",
                  "planning_wall_s": time.monotonic()-start, "duration_s": duration, "waypoints": waypoints.tolist(),
                  "goal_world": goal.tolist(), "posture_reference_rad": reference.tolist(),
                  "collision_model": "CAD bounding-box sphere cover, other arm and crates/table", "carried_rail": payload is not None}
        (self.output/f"{label}.json").write_text(json.dumps(report, indent=2))
        return trajectory, float(duration), target
