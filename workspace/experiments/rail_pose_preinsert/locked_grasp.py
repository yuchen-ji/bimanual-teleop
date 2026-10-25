"""Supported grasp: translate without rotation, verify distal contacts, attach."""
from __future__ import annotations

import json
import time
import uuid

import numpy as np
from common import finger_forces, opposed_grasp, pose_error, write_json

from bimanual_teleop.sim.kinematics import pose_matrix

FINGERS = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")


def distal_forces(state):
    contacts = state.get("rail_contacts", {})
    return {f: float(sum(v for name, v in contacts.items()
                        if name in ("l_"+f+"_distal", "l_"+f+"_tip_sensor_frame"))) for f in FINGERS}


class LockedGrasp:
    def __init__(self, source, state):
        self.source = source
        self.start = source.command.copy()
        self.rail_start = np.asarray(state["rail"]).copy()
        self.rail_xyz = self.rail_start[:3].copy()
        self.quaternion = self.rail_start[3:].copy()
        self.phase, self.elapsed = "plan", 0.
        self.request_id = None
        self.plan = None
        self.target = self.start.copy()
        self.stable = 0.
        self.refinements = 0
        self.last_safe = self.start.copy()
        self.attached = False
        self.geometric_since = None
        self.request("locked_grasp_plan", state)

    def request(self, stage, state):
        # Use actual PhysX link transforms for submillimetre contact checks.
        # Joint FK alone omits the small articulation constraint deflections.
        state = dict(state)
        if hasattr(self.source, "robot"):
            view = self.source.robot._articulation_view._physics_view
            names = list(view.shared_metatype.link_names)
            if names and isinstance(names[0], (list, tuple)):
                names = list(names[0])
            transforms = view.get_link_transforms()[0]
            state["hand_link_poses"] = {name: np.asarray(transforms[i]).tolist() for i, name in enumerate(names)
                                        if name == "l_wrist" or name.startswith(tuple("l_"+f+"_" for f in FINGERS))}
        self.request_id = uuid.uuid4().hex
        self.request_started = time.monotonic()
        write_json(self.source.output / "rpc/request.json", {"id": self.request_id, "stage": stage,
                   "state": state, "command": self.source.command.tolist(),
                   "trial": self.source.reset_index, "revision": self.source.runtime.get("revision", "startup")})

    def response(self):
        path = self.source.output / "rpc/response.json"
        value = json.loads(path.read_text()) if path.is_file() else {}
        if value.get("id") != self.request_id:
            if time.monotonic()-self.request_started > 60:
                self.source.finish("failed", "Grasp geometry request timed out")
            return None
        self.request_id = None
        if not value.get("ok"):
            self.source.finish("failed", value.get("error", "Grasp planner failed"))
            return None
        return value

    def step(self, state, dt):
        self.elapsed += dt
        forces = distal_forces(state)
        self.source.metrics["distal_contact_force_n"] = forces
        gripping = min(forces.values()) >= .15 and opposed_grasp(state, minimum_n=.15)
        self.stable = self.stable+dt if gripping else 0.
        if max(finger_forces(state).values()) > 4:
            self.source.command[14:34] = self.last_safe[14:34]
            self.source.finish("failed", "Grasp contact exceeded 4 N; returned to last checked hand target")
            return self.source.command.copy()
        if self.phase == "plan":
            response = self.response()
            if response:
                self.plan = response["plan"]
                self.target[14:34] = self.plan["q"][14:34]
                self.phase, self.elapsed = ("thumb_prepare" if self.plan.get("thumb_before_translation") else "translate"), 0.
                self.source.event("orientation_locked_grasp_planned", plan=self.plan)
        elif self.phase == "thumb_prepare":
            u = min(self.elapsed/3., 1.)
            u = u*u*u*(10-15*u+6*u*u)
            desired = self.start.copy()
            desired[14:18] = (1-u)*self.start[14:18]+u*self.target[14:18]
            if self.elapsed >= 3.7:
                self.start[14:18] = self.target[14:18]
                self.phase, self.elapsed = "translate", 0.
            return desired
        elif self.phase == "translate":
            u = min(self.elapsed/4., 1.)
            u = u*u*u*(10-15*u+6*u*u)
            self.rail_xyz = (1-u)*self.rail_start[:3]+u*np.asarray(self.plan["rail_xyz_m"])
            if self.elapsed >= 4.5:
                self.phase, self.elapsed = "close", 0.
        elif self.phase == "close":
            u = min(self.elapsed/12., 1.)
            u = u*u*u*(10-15*u+6*u*u)
            desired = (1-u)*self.start+u*self.target
            if self.elapsed >= 13.:
                self.phase, self.elapsed = "review", 0.
                self.request("locked_grasp_check", state)
            return desired
        elif self.phase == "review":
            response = self.response()
            if response:
                self.source.metrics["grasp_geometry"] = response["geometry"]
                if response["geometry"]["intersecting_links"] or response["geometry"].get("interfinger_intersections"):
                    self.source.finish("failed", "Distal contact check found mesh intersection")
                else:
                    geometry = response["geometry"]
                    geometric = self.source.task.get("grasp_attachment_criterion") == "distal_clearance"
                    near = max(geometry.get("distal_gap_m", {"missing": 1.}).values()) <= .0002
                    if self.source.task.get("grasp_contact_surface") == "bottom":
                        near = near and geometry.get("bottom_contact_face_ok", False)
                    if geometric and near:
                        if self.geometric_since is None:
                            self.geometric_since = geometry["time_ns"]
                    else:
                        self.geometric_since = None
                    ready = (geometry["time_ns"]-self.geometric_since >= 500_000_000
                             if geometric and self.geometric_since is not None else self.stable >= .5 and not geometric)
                    if ready:
                        self.source.pending_attachment = True
                        self.phase, self.elapsed = "attach", 0.
                        self.source.metrics["attachment_criterion"] = "distal_clearance_0.2mm" if geometric else "opposed_distal_forces"
                        self.source.event("five_distal_surfaces_verified", distal_force_n=forces, geometry=geometry)
                    elif self.refinements >= 100:
                        self.source.finish("failed", "Distal contacts did not all stabilize; rail remains supported")
                    else:
                        self.last_safe = self.source.command.copy()
                        self.target = np.asarray(response["command"])
                        self.refinements += 1
                        self.phase, self.elapsed = "refine", 0.
        elif self.phase == "refine":
            if self.elapsed >= 1.2:
                self.phase, self.elapsed = "review", 0.
                self.request("locked_grasp_check", state)
            return self.target.copy()
        elif self.phase == "attach" and self.elapsed >= 1.:
            self.attached = True
            self.source.metrics["rigid_attachment_active"] = True
            self.source.metrics["pregrasp_rotation_locked"] = True
            self.follow_start = self.source.command.copy()
            self.follow_flange = self.source.flange(state)
            self.follow_max_error = np.zeros(2)
            self.follow_max_motion = np.zeros(2)
            self.phase, self.elapsed = "follow_check", 0.
        elif self.phase == "follow_check":
            actual = np.linalg.inv(self.source.flange(state)) @ pose_matrix(state["rail"])
            error = pose_error(actual, self.source.grip)
            self.follow_max_error = np.maximum(self.follow_max_error, error)
            self.follow_max_motion = np.maximum(self.follow_max_motion, pose_error(self.follow_flange, self.source.flange(state)))
            desired = self.follow_start.copy()
            # Small out-and-back arm motion exercises translation and rotation.
            u = min(self.elapsed/6., 1.)
            amplitude = np.sin(np.pi*u)**2
            desired[0] += .008*amplitude
            desired[6] += .02*amplitude
            if self.elapsed < 7.:
                return desired
            if self.follow_max_error[0] > .001 or self.follow_max_error[1] > .01:
                self.source.finish("failed", "Attached rail did not follow the measured flange")
                return desired
            self.source.metrics["follow_check"] = {"max_relative_error_m_rad": self.follow_max_error.tolist(),
                                                   "flange_motion_m_rad": self.follow_max_motion.tolist()}
            self.phase, self.elapsed = "final_review", 0.
            self.request("locked_grasp_verify", state)
            return desired
        elif self.phase == "final_review":
            response = self.response()
            if response is None:
                return self.source.command.copy()
            self.source.metrics["final_grasp_geometry"] = response["geometry"]
            if response["geometry"]["intersecting_links"] or response["geometry"].get("interfinger_intersections"):
                self.source.finish("failed", "Post-motion grasp geometry intersects")
                return self.source.command.copy()
            if self.source.task.get("grasp_contact_surface") == "bottom" and not response["geometry"].get("bottom_contact_face_ok", False):
                self.source.finish("failed", "Post-motion contact left the bottom pad surface")
                return self.source.command.copy()
            if (self.source.task.get("grasp_attachment_criterion") == "distal_clearance" and
                    max(response["geometry"].get("distal_gap_m", {"missing": 1.}).values()) > .0002):
                self.source.finish("failed", "Post-motion distal clearance exceeds 0.2 mm")
                return self.source.command.copy()
            write_json(self.source.trial_directory / "locked_grasp_state.json", state)
            self.source.phase = "locked_grasp_review"
            self.source.snapshot_requested = True
        return self.source.command.copy()
