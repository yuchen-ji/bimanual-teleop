"""Left end grasp -> verified right middle grasp -> right-hand insertion."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from bimanual_teleop.sim.kinematics import pose_matrix, pose_vector, transform
from bimanual_teleop.sim.bin_geometry import bin_geometry
from bimanual_teleop.sim import collisions as collision_module
from bimanual_teleop.sim.recording import atomic_json
from controller import RailController, RailReplayController, interpolate_pose
from evaluate import initial_pose_valid, insertion_metrics, is_seated, pose_error
from grasping import FINGERS, grasp_axis_position, gripping, hand_contacts, make_grasp
from planning import ArmPlanner


PHASES = ("settle", "left_pregrasp", "left_approach", "left_close", "left_lift", "left_clear", "right_ready", "left_handover",
          "right_pregrasp", "right_approach", "right_close", "handover_hold", "left_release", "left_retreat",
          "right_transfer", "right_align", "right_insert", "right_release", "right_retreat", "verify")
MILESTONES = {"settle": "scene", "left_lift": "grasp", "left_clear": "extract", "left_retreat": "handover",
              "right_align": "transport", "verify": "insert"}


def scene_signature(config, task=None):
    fields = {key: config[key] for key in ("robot", "scene", "rail")}
    # A common collision-code change can alter effective geometry even when
    # the YAML values stay identical. Frozen input alone cannot restore it.
    fields["collision_code_sha256"] = hashlib.sha256(Path(collision_module.__file__).read_bytes()).hexdigest()
    if task and task.get("convex_decomposition_links"):
        fields["convex_decomposition_links"] = sorted(set(task["convex_decomposition_links"]))
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def extraction_lift_pose(config, task, start, fraction=1.):
    """Lift vertically, or tilt on the far bottom corner before pulling out."""
    def ramp(value):
        value = np.clip(value, 0., 1.)
        return value**3*(10-15*value+6*value*value)
    if "left_lift_vertical_m" in task:
        height = float(task["left_lift_vertical_m"])
        if not np.isfinite(height) or height <= 0.:
            raise ValueError("left_lift_vertical_m must be finite and positive")
        result = start.copy()
        result[2, 3] += height*ramp(fraction)
        return result
    local_pivot = np.array([-config["rail"]["size_m"][0]/2, 0., -config["rail"]["size_m"][2]/2])
    rpy = Rotation.from_matrix(start[:3, :3]).as_euler("xyz")
    rpy[1] = np.deg2rad(task["left_lift_pitch_deg"])
    target = start.copy(); target[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    result = interpolate_pose(start, target, ramp(fraction/.8))
    pivot = start[:3, :3]@local_pivot+start[:3, 3]
    pivot += np.array([-task["left_lift_pullback_m"]*ramp((fraction-.4)/.6), 0.,
                       task["left_lift_floor_clearance_m"]*ramp((fraction-.8)/.2)])
    result[:3, 3] = pivot-result[:3, :3]@local_pivot
    return result


class HandoverController(RailController):
    name = "rail_bimanual_handover"

    def __init__(self, config, model, task, output, **kwargs):
        if kwargs.get("start_phase") or kwargs.get("initial_positions") is not None:
            raise ValueError("Handover diagnostic initialization is not supported; start from the normal scene")
        self.templates = {side: make_grasp(model, task, side) for side in ("left", "right")}
        self.planner = ArmPlanner(config, model, task, output)
        super().__init__(config, model, task, output, **kwargs)
        kp, kd = np.tile(task["hand_stiffness"], 5), np.tile(task["hand_damping"], 5)
        self.joint_gain_overrides.update({name: (float(p), float(d)) for name, p, d in zip(model.hand_names["right"], kp, kd)})
        self.parent_sources = [task["demo_provenance"]]
        atomic_json(Path(output)/"grasp_templates.json", {side: {key: value.tolist() if isinstance(value, np.ndarray) else value
                                                                  for key, value in template.items()} for side, template in self.templates.items()})

    @property
    def phase(self):
        return PHASES[self.phase_index]

    def bind_simulation(self, *, world, robot, rail, robot_path, camera_paths):
        """Use the shared scene implementation, retaining explicit task geometry."""
        import omni.usd
        from bimanual_teleop.sim.collisions import configure_robot_collisions
        extra = sorted(set(self.task.get("convex_decomposition_links", [])))
        report = configure_robot_collisions(omni.usd.get_context().get_stage(), robot_path,
                                            self.model, self.config, extra)
        atomic_json(self.output/"collision_approximations.json", report["colliders"])
        atomic_json(self.output/"collision_shape_overrides.json", {
            "scope": "scene defaults plus explicit experiment overrides",
            "convex_decomposition_links": extra,
            "effective_decomposition_links": report["effective_decomposition_links"]})
        atomic_json(self.output/"hand_material_binding.json", report["hand_material_binding"])

    def reset(self):
        super().reset()
        self.entry_phase = "left_pregrasp"
        self.held = {"left": None, "right": None}
        self.right_verified = False
        self.closed = {"left": False, "right": False}
        self.preshaped = {"left": False, "right": False}
        self.loss_s = {"left": 0., "right": 0.}
        self.motion = None
        self.phase_pose = None
        self.hold_pose = None
        self.initial_rail = None
        self.arm_holds = {side: self.model.flange(side, self.ready, True) for side in ("left", "right")}
        self.previous_flange = {}
        self.flange_velocity = {side: np.zeros(6) for side in ("left", "right")}
        self.hand_motion = {side: None for side in ("left", "right")}
        self.adaptive_motion = {side: None for side in ("left", "right")}
        self.grasp_commands = {side: self.templates[side]["closed"].copy() for side in ("left", "right")}
        self.index_close_stage = 0
        self.report()

    def advance(self):
        if self.finished:
            return
        if self.waiting:
            self.waiting = False
            self.phase_index += 1
            self.elapsed = self.stable = 0.
            self.motion = self.phase_pose = self.hold_pose = None
            self.event("phase_started"); self.report()
        self.active = True

    def complete_phase(self):
        if self.motion is not None and self.last_state is not None:
            side = self.motion["side"]
            self.arm_holds[side] = self.model.flange(side, self.last_state["q"], True).copy()
        if self.phase.endswith("approach") and self.last_state is not None:
            side = self.phase.split("_")[0]
            self.arm_holds[side] = self.model.flange(side, self.last_state["q"], True).copy()
        if self.phase in ("left_lift", "left_retreat") and self.last_state is not None:
            side = "left" if self.phase == "left_lift" else "right"
            self.held[side] = np.linalg.inv(self.model.flange(side, self.last_state["q"], True))@pose_matrix(self.last_state["rail"])
            self.event("loaded_grip_verified", side=side, contacts=hand_contacts(self.last_state, side))
        self.completed_phases.append(self.phase)
        self.event("phase_passed")
        if self.capture_snapshots:
            self.snapshot_requested = True; self.snapshot_label = self.phase
        if MILESTONES.get(self.phase) == self.until:
            self.finish("success" if self.phase == "verify" else "milestone_passed")
        elif self.mode == "step":
            self.waiting = True; self.report()
        else:
            self.waiting = True
            if not self.capture_snapshots:
                self.advance()

    def save_trajectory(self):
        super().save_trajectory()
        path = self.trial/"trajectory.json"
        if path.exists():
            document = json.loads(path.read_text())
            document.update(strategy="bimanual_handover", scene_signature=scene_signature(self.config, self.task),
                            demo_provenance=self.task["demo_provenance"])
            atomic_json(path, document)

    def reference(self, side):
        stage = "left_grasp" if self.phase in ("left_pregrasp", "left_approach", "left_close") else "left_carry"
        if side == "right":
            stage = "ready" if self.phase == "right_ready" else ("right_grasp" if self.phase in ("right_pregrasp", "right_approach", "right_close", "handover_hold") else "seated")
        return np.deg2rad(self.task["arm_reference_deg"][side][stage])

    def setup(self, state):
        rail = pose_matrix(state["rail"]); phase = self.phase
        side = "left" if phase.startswith("left_") else "right"
        current = self.model.flange(side, state["q"], world=True)
        self.phase_pose = rail.copy()
        goal_object = None
        if phase.endswith("pregrasp") or phase.endswith("approach"):
            goal = rail@self.templates[side]["flange_in_rail"]
            if phase.endswith("pregrasp"):
                goal[:3, 3] += rail[:3, :3]@[0., 0., self.task.get(side+"_cartesian_approach_m", self.task["cartesian_approach_m"])]
        elif phase == "left_lift":
            goal_object = extraction_lift_pose(self.config, self.task, rail)
        elif phase == "left_clear":
            goal_object = rail.copy(); goal_object[:3, 3] = self.task["left_clearance_center_m"]
        elif phase == "left_handover":
            goal_object = transform(self.task["handover_center_m"], np.deg2rad(self.task["handover_rpy_deg"]))
        elif phase == "right_ready":
            goal = current.copy(); goal[:3, 3] = self.task["right_wait_flange_xyz_m"]
        elif phase in ("right_transfer", "right_align", "right_insert"):
            goal_object = self.goal_rail.copy()
            goal_object[2, 3] += self.task.get("right_transfer_height_m", .10) if phase == "right_transfer" else (self.task["preinsert_height_m"] if phase == "right_align" else 0.)
        elif phase in ("left_retreat", "right_retreat"):
            goal = current.copy()
            goal[:3, 3] += rail[:3, :3]@self.task.get(side+"_retreat_in_rail_m", [-.12, .03, .09])
        else:
            raise ValueError(f"No motion for phase {phase}")
        if goal_object is not None:
            grip = np.linalg.inv(current)@rail
            goal = goal_object@np.linalg.inv(grip)
        planned = phase in ("left_pregrasp", "left_clear", "right_ready", "left_handover", "right_pregrasp", "right_transfer")
        payload = np.linalg.inv(current)@rail if goal_object is not None else None
        if planned:
            trajectory, duration, target = self.planner.plan(side, state["q"], goal, self.reference(side), phase,
                                                            payload=payload, rail=rail if phase.endswith("pregrasp") or phase == "right_ready" else None)
        else:
            target = self.planner.solve(side, state["q"], goal, self.reference(side))
            trajectory = None
            position, angle = pose_error(current, goal)
            speed = self.task["insertion_speed_m_s"] if phase == "right_insert" else self.task["linear_speed_m_s"]
            duration = max(.75, 1.875*position/speed, 1.875*angle/self.task["angular_speed_rad_s"])
        self.motion = {"side": side, "start": current, "goal": goal, "object": goal_object,
                       "trajectory": trajectory, "duration": duration, "target": target, "grip": payload,
                       "tracking_bias": self.command[(0 if side == "left" else 7):(7 if side == "left" else 14)]-np.asarray(state["q"])[(0 if side == "left" else 7):(7 if side == "left" else 14)]}
        self.arm_holds[side] = goal.copy()
        self.event("motion_target", side=side, target_world=pose_vector(goal).tolist(), duration_s=duration,
                   planner="Lula RRT" if planned else "Cartesian feedback + Lula IK")

    def update_hands(self, dt, state=None):
        for side, start in (("left", 14), ("right", 34)):
            if (self.closed[side] and self.task.get(side+"_adaptive_closing", False) and state is not None
                    and not (side == "right" and self.right_verified)):
                self.update_contact_closing(side, start, state, dt)
                continue
            profile = ("release" if self.phase in (side+"_release", side+"_retreat")
                       else "preshape" if self.preshaped[side] else "open")
            target = (self.grasp_commands[side] if self.closed[side] else self.templates[side][profile]).copy()
            corners = self.task.get("left_index_close_waypoints_rad", []) if side == "left" and self.closed[side] else []
            stage = self.index_close_stage if corners else 0
            if stage < len(corners):
                target[4:8] = corners[stage]
            key = (self.closed[side], stage, profile)
            motion = self.hand_motion[side]
            if motion is None or motion["key"] != key:
                initial = self.command[start:start+20].copy()
                motion = {"key": key, "start": initial, "goal": target, "elapsed": 0.,
                          "duration": max(dt, float(np.max(abs(target-initial)))/self.task["hand_speed_rad_s"])}
                self.hand_motion[side] = motion
            motion["elapsed"] += dt
            fraction = min(1., motion["elapsed"]/motion["duration"])
            self.command[start:start+20] = (1-fraction)*motion["start"]+fraction*motion["goal"]
            if fraction == 1. and corners and stage < len(corners):
                self.index_close_stage += 1

    def update_contact_closing(self, side, start, state, dt):
        """Let fingers still clear of the rail catch up with fingers in contact."""
        motion = self.adaptive_motion[side]
        if motion is None:
            initial = self.command[start:start+20].reshape(5, 4).copy()
            goal = self.templates[side]["closed"].reshape(5, 4)
            motion = {"start": initial, "goal": goal, "progress": np.zeros(5),
                      "duration": np.maximum(1., np.max(abs(goal-initial), axis=1)/self.task["hand_speed_rad_s"])}
            self.adaptive_motion[side] = motion
        prefix = "l_" if side == "left" else "r_"
        contacts = hand_contacts(state, side)
        forces = np.array([sum(force for name, force in contacts.items() if name.startswith(prefix+finger)) for finger in FINGERS])
        targets = np.full(5, self.task["closing_finger_force_n"])
        targets[0] = np.clip(forces[1:].sum(), self.task["closing_finger_force_n"], self.task["closing_thumb_force_max_n"])
        rate = np.clip((targets-forces)/targets, -1., 1.)/motion["duration"]
        motion["progress"] = np.clip(motion["progress"]+rate*dt, 0., 1.)
        progress = motion["progress"][:, None]
        self.command[start:start+20] = ((1-progress)*motion["start"]+progress*motion["goal"]).reshape(20)

    def closing_complete(self, side, state):
        motion = self.adaptive_motion[side]
        if motion is None:
            start = 14 if side == "left" else 34
            return bool(np.max(abs(self.command[start:start+20]-self.templates[side]["closed"])) < .005)
        prefix = "l_" if side == "left" else "r_"
        contacts = hand_contacts(state, side)
        reached = [progress >= .995 or sum(force for name, force in contacts.items() if name.startswith(prefix+finger)) >= self.task["grasp_contact_min_n"]
                   for progress, finger in zip(motion["progress"], FINGERS)]
        return all(reached)

    def check_carry(self, state, dt):
        phase = self.phase
        left_needed = phase in ("left_lift", "left_clear", "right_ready", "left_handover", "right_pregrasp", "right_approach", "right_close", "handover_hold")
        right_needed = self.right_verified and phase not in ("right_release", "right_retreat", "verify")
        for side, required in (("left", left_needed), ("right", right_needed)):
            if not required:
                continue
            self.loss_s[side] = 0. if gripping(state, side, self.task["grasp_contact_min_n"]) else self.loss_s[side]+dt
            if self.loss_s[side] > self.task["carry_contact_loss_s"]:
                return self.fail(f"{side}_grip_lost")
            if self.held[side] is not None:
                relative = np.linalg.inv(self.model.flange(side, state["q"], True))@pose_matrix(state["rail"])
                _, angle = pose_error(relative, self.held[side])
                # Rotation about an end grasp moves the object's center even
                # when the grasp point has not slid through the fingers.
                anchor = np.array([self.task[f"{side}_grasp_axis_m"], 0., 0., 1.])
                distance = np.linalg.norm((relative@anchor-self.held[side]@anchor)[:3])
                loading = (side == "left" and phase == "left_lift") or (side == "right" and phase in ("left_release", "left_retreat"))
                limit = self.task["loading_grip_slip_m"] if loading else self.task["grip_slip_m"]
                if distance > limit or angle > self.task["grip_slip_rad"]:
                    return self.fail(f"{side}_grasp_slip: {distance:.4f} m, {angle:.4f} rad")
        if phase in ("left_clear", "right_ready", "left_handover", "right_pregrasp", "right_approach", "right_close", "handover_hold", "left_release", "left_retreat", "right_transfer", "right_align"):
            if max(state.get("rail_support_contacts", {}).values(), default=0.) > 1.:
                return self.fail("carried_rail_hit_environment")
        allowed = tuple(prefix+finger for prefix in ("l_", "r_") for finger in FINGERS)
        required_holds = {"left": left_needed, "right": right_needed}
        palms = {prefix+"wrist" for side, prefix in (("left", "l_"), ("right", "r_"))
                 if gripping(state, side, self.task["grasp_contact_min_n"])
                 or (required_holds[side] and self.held[side] is not None)
                 or phase in (side+"_approach", side+"_close", side+"_release")}
        illegal = {k: v for k, v in state.get("rail_contacts", {}).items()
                   if v > 1. and not k.startswith(allowed) and k not in palms}
        if illegal:
            return self.fail("rail_hit_arm: "+max(illegal, key=illegal.get))
        if phase == "left_retreat":
            touched = {name: force for name, force in hand_contacts(state, "left").items()
                       if force > self.task["grasp_contact_min_n"]}
            if touched:
                return self.fail("released_left_hand_recontact: "+max(touched, key=touched.get))
        return None

    def servo(self, side, state, target, dt, *, insertion=False):
        """Pose feedback with measured velocity damping and gravity bias integration."""
        current = self.model.flange(side, state["q"], True)
        error = np.r_[target[:3, 3]-current[:3, 3],
                      Rotation.from_matrix(target[:3, :3]@current[:3, :3].T).as_rotvec()]
        velocity = 4.*error-self.flange_velocity[side]
        for selected, maximum in ((slice(0, 3), self.task["insertion_speed_m_s"] if insertion else self.task["linear_speed_m_s"]),
                                  (slice(3, 6), .05 if insertion else self.task["angular_speed_rad_s"])):
            norm = np.linalg.norm(velocity[selected])
            if norm > maximum:
                velocity[selected] *= maximum/norm
        weights = np.diag([1., 1., 1., .25, .25, .25])
        jacobian = weights@self.model.jacobian(side, state["q"])
        delta = jacobian.T@np.linalg.solve(jacobian@jacobian.T+.0025*np.eye(6), weights@velocity)
        offset = 0 if side == "left" else 7
        self.command[offset:offset+7] += np.clip(delta, -self.config["runtime"]["joint_speed_rad_s"], self.config["runtime"]["joint_speed_rad_s"])*dt

    def step(self, state, dt):
        previous = self.command.copy()
        for side in ("left", "right"):
            current = self.model.flange(side, state["q"], True)
            last = self.previous_flange.get(side, current)
            velocity = np.r_[current[:3, 3]-last[:3, 3], Rotation.from_matrix(current[:3, :3]@last[:3, :3].T).as_rotvec()]/dt
            self.flange_velocity[side] += min(1., dt/.05)*(velocity-self.flange_velocity[side])
            self.previous_flange[side] = current
        self._step(state, dt)
        if not self.finished:
            moving = self.motion["side"] if self.motion is not None and not self.waiting else None
            for side in ("left", "right"):
                if side != moving:
                    self.servo(side, state, self.arm_holds[side], dt)
        maximum = np.r_[np.full(14, self.config["runtime"]["joint_speed_rad_s"]),
                        np.full(40, min(self.task["hand_speed_rad_s"], self.config["runtime"]["hand_speed_rad_s"]))]*dt
        self.command = np.clip(self.command, previous-maximum, previous+maximum)
        self.command = np.clip(self.command, self.model.lower, self.model.upper)
        return self.command.copy()

    def _step(self, state, dt):
        if self.waiting and self.mode == "auto" and not self.snapshot_requested and not self.finished:
            self.advance()
        if self.finished or self.waiting or not self.active:
            return self.command.copy()
        self.elapsed += dt; phase = self.phase
        if not np.isfinite(np.r_[state["q"], state["rail"]]).all():
            return self.fail("nonfinite_state")
        if self.initial_rail is None:
            self.initial_rail = pose_matrix(state["rail"])
        environment = state.get("environment_contacts", {})
        tracking = np.max(abs(np.asarray(state.get("q_target", state["q"]))[:14]-np.asarray(state["q"])[:14]))
        self.arm_tracking_bad_s = self.arm_tracking_bad_s+dt if tracking > self.task["max_arm_tracking_error_rad"] else 0.
        if self.arm_tracking_bad_s > self.task["arm_tracking_timeout_s"]:
            return self.fail("arm_tracking_error")
        if self.elapsed > .2 and max(environment.values(), default=0.) > self.task["environment_force_limit_n"]:
            return self.fail("unexpected_environment_contact: "+max(environment, key=environment.get))
        if max(state.get("robot_base_contacts", {}).values(), default=0.) > 5. or state.get("rail_robot_base_force_n", 0.) > 1.:
            return self.fail("unexpected_robot_base_contact")
        if self.elapsed > max(self.task["stage_timeout_s"], (self.motion or {}).get("duration", 0.)+12):
            return self.fail("stage_timeout")
        if self.check_carry(state, dt) is not None:
            return self.command.copy()
        self.update_hands(dt, state)
        if phase == "settle":
            stable = initial_pose_valid(state, self.config) and self.linear_speed < .005 and self.angular_speed < .04
            hands_ready = all(np.max(abs(self.command[start:start+20]-self.templates[side]["open"])) < .005
                              for side, start in (("left", 14), ("right", 34)))
            stable = stable and hands_ready
            self.stable = self.stable+dt if stable else 0.
            if self.stable >= self.task["settle_s"]:
                self.complete_phase()
            elif self.elapsed > self.task["settle_timeout_s"]:
                return self.fail("initial_rail_not_stable")
            return self.command.copy()
        if phase.endswith("close"):
            side = phase.split("_")[0]; self.closed[side] = True
            rail = pose_matrix(state["rail"])
            if self.phase_pose is None:
                self.phase_pose = rail.copy()
            displacement, rotation = pose_error(rail, self.phase_pose)
            if displacement > .03 or rotation > .25:
                return self.fail(side+"_object_moved_during_closing")
            # Hold the wrist while fingers balance their contact forces. Chasing
            # a pushed rail here creates positive feedback and moves it away.
            if self.hold_pose is None:
                self.hold_pose = self.arm_holds[side].copy()
            self.arm_holds[side] = self.hold_pose.copy()
            correct_region = abs(grasp_axis_position(self.model, state, side)-self.task[f"{side}_grasp_axis_m"]) < .045
            start = 14 if side == "left" else 34
            closed = self.closing_complete(side, state)
            contacts = hand_contacts(state, side)
            supporting = sum(sum(force for name, force in contacts.items() if name.startswith("l_"+finger)) >= self.task["grasp_contact_min_n"]
                             for finger in FINGERS[1:]) if side == "left" else 4
            distributed = side != "left" or supporting >= self.task["left_grasp_min_support_fingers"]
            self.stable = self.stable+dt if closed and correct_region and distributed and gripping(state, side, self.task["grasp_contact_min_n"]) else 0.
            if self.stable >= .4:
                self.grasp_commands[side] = self.command[start:start+20].copy()
                self.hand_motion[side] = None
                self.held[side] = np.linalg.inv(self.model.flange(side, state["q"], True))@pose_matrix(state["rail"])
                if side == "left":
                    self.initial_rail = pose_matrix(state["rail"])
                self.event("grip_verified", side=side, contacts=hand_contacts(state, side))
                self.complete_phase()
            elif self.elapsed > 8:
                self.event("grip_rejected", side=side, closing_complete=closed,
                           correct_region=correct_region, supporting_fingers=supporting,
                           opposing_contacts=gripping(state, side, self.task["grasp_contact_min_n"]),
                           contacts=contacts)
                if not distributed:
                    return self.fail(side+"_insufficient_supporting_fingers")
                return self.fail(side+"_no_opposing_contacts")
            return self.command.copy()
        if phase == "handover_hold":
            rail = pose_matrix(state["rail"])
            if self.hold_pose is None:
                self.hold_pose = rail.copy()
            position, angle = pose_error(rail, self.hold_pose)
            stable = gripping(state, "left", self.task["grasp_contact_min_n"]) and gripping(state, "right", self.task["grasp_contact_min_n"]) and position < .002 and angle < .025
            if not stable:
                self.hold_pose = rail.copy()
            self.stable = self.stable+dt if stable else 0.
            if self.stable >= self.task["handover_hold_s"]:
                # Keep the balanced position preload during load transfer.
                # Extra force from the rail's weight must not open the fingers.
                self.grasp_commands["right"] = self.command[34:54].copy()
                self.hand_motion["right"] = None
                self.right_verified = True
                self.event("handover_verified", right_grasp_axis_m=self.task["right_grasp_axis_m"])
                self.complete_phase()
            return self.command.copy()
        if phase.endswith("release"):
            side = phase.split("_")[0]
            if side == "left" and not self.right_verified:
                return self.fail("left_release_before_verified_right_grip")
            self.closed[side] = False
            self.preshaped[side] = False
            start = 14 if side == "left" else 34
            opened = np.max(abs(self.command[start:start+20]-self.templates[side]["release"])) < .005
            if side == "left" and opened and self.hold_pose is None:
                self.hold_pose = self.arm_holds[side].copy()
                self.hold_pose[:3, 3] += pose_matrix(state["rail"])[:3, :3]@[0., 0., .02]
                self.arm_holds[side] = self.hold_pose.copy()
            palm_clear = side != "left" or (self.hold_pose is not None and pose_error(self.model.flange(side, state["q"], True), self.hold_pose)[0] < .002)
            if opened and palm_clear and max(hand_contacts(state, side).values(), default=0.) <= .05:
                self.event("hand_opened", side=side); self.complete_phase()
            return self.command.copy()
        if phase == "verify":
            passed, _ = is_seated(state, self.config, self.geometry, self.task, self.linear_speed, self.angular_speed)
            self.stable = self.stable+dt if passed else 0.
            if self.stable >= self.task["seated_hold_s"]:
                self.complete_phase()
            return self.command.copy()
        if self.motion is None:
            try:
                self.setup(state)
            except (ValueError, RuntimeError) as exc:
                return self.fail(str(exc))
        motion = self.motion; side = motion["side"]; offset = 0 if side == "left" else 7
        fraction = min(1., self.elapsed/motion["duration"]); smooth = fraction**3*(10-15*fraction+6*fraction*fraction)
        vertical_lift = phase == "left_lift" and "left_lift_vertical_m" in self.task
        if motion["trajectory"] is not None and fraction < 1.:
            desired = np.asarray(motion["trajectory"].eval(self.elapsed))
            error = desired-np.asarray(state["q"])[offset:offset+7]
            motion["tracking_bias"] += np.clip(2.*error*dt, -.2*dt, .2*dt)
            self.command[offset:offset+7] = desired+motion["tracking_bias"]
        else:
            target = interpolate_pose(motion["start"], motion["goal"], smooth)
            if motion["object"] is not None and not vertical_lift:
                desired = (extraction_lift_pose(self.config, self.task, self.phase_pose, smooth) if phase == "left_lift"
                           else interpolate_pose(self.phase_pose, motion["object"], smooth))
                actual = self.model.flange(side, state["q"], True)
                measured_grip = np.linalg.inv(actual)@pose_matrix(state["rail"])
                target = desired@np.linalg.inv(measured_grip)
            self.servo(side, state, target, dt, insertion=phase == "right_insert")
        track_object = motion["object"] is not None and not vertical_lift
        actual = pose_matrix(state["rail"]) if track_object else self.model.flange(side, state["q"], True)
        position, angle = pose_error(actual, motion["object"] if track_object else motion["goal"])
        approach_tolerance = .003 if phase == "left_approach" and state.get("rail_contacts", {}).get("l_wrist", 0.) > 1. else .0007
        reached = fraction >= 1. and position < (approach_tolerance if phase.endswith("approach") else self.task["position_tolerance_m"]) and angle < .025
        if phase.endswith("pregrasp") and reached and "preshape" in self.templates[side]:
            self.preshaped[side] = True
            profile = self.templates[side]["preshape"]
            hand_start = 14 if side == "left" else 34
            reached = (np.max(abs(self.command[hand_start:hand_start+20]-profile)) < .005
                       and np.max(abs(np.asarray(state["q"])[hand_start:hand_start+20]-profile)) < .06)
        if phase.endswith("approach"):
            reached = reached and np.linalg.norm(self.flange_velocity[side][:3]) < .005 and np.linalg.norm(self.flange_velocity[side][3:]) < .04
            self.stable = self.stable+dt if reached else 0.
            reached = self.stable >= .4
        if phase == "left_lift":
            reached = reached and pose_matrix(state["rail"])[2, 3]-self.initial_rail[2, 3] >= self.task["lift_m"]-.002
            reached = reached and max(state.get("rail_support_contacts", {}).values(), default=0.) < .1
            if vertical_lift:
                rail_pose = pose_matrix(state["rail"])
                bottom = rail_pose[2, 3]-np.dot(abs(rail_pose[2, :3]), np.asarray(self.config["rail"]["size_m"])/2)
                rim = self.config["scene"]["bin_centers"][0][2]+bin_geometry(self.config["scene"])["rim_height_m"]
                reached = reached and bottom > rim+.02
        if phase in ("left_lift", "left_clear", "left_handover"):
            reached = reached and self.linear_speed < .015 and self.angular_speed < .08
        if phase == "right_insert":
            metrics = insertion_metrics(state, self.config, self.geometry)
            reached = reached and metrics["support_gap_m"] <= self.task["support_gap_tolerance_m"] and metrics["insertion_depth_m"] >= self.task["insertion_depth_min_m"]
        if phase == "right_align":
            metrics = insertion_metrics(state, self.config, self.geometry)
            clearance = self.config["rail"]["hole_diameter_m"]/2-max(self.geometry["pin_radii_m"])-.0001
            reached = reached and metrics["pin_error_m"] < clearance and metrics["axis_error_rad"] < .01
        if phase == "left_retreat":
            reached = reached and max(hand_contacts(state, "left").values(), default=0.) < .05
            filtered = self.filtered_motion if self.filtered_motion is not None else np.full(3, np.inf)
            reached = reached and filtered[0] < .01 and filtered[1] < .08
            self.stable = self.stable+dt if reached else 0.
            reached = self.stable >= self.task["handover_hold_s"]
        if phase in ("left_lift", "left_handover"):
            self.stable = self.stable+dt if reached else 0.
            reached = self.stable >= (.5 if phase == "left_handover" else 1.)
        if reached:
            self.complete_phase()
        return self.command.copy()


class HandoverReplayController(RailReplayController):
    def __init__(self, config, model, task, output, trajectory, **kwargs):
        document = json.loads(Path(trajectory).read_text())
        if document.get("scene_signature") != scene_signature(config, task):
            raise ValueError("Replay scene differs from the recorded scene; use the complete original scene configuration")
        super().__init__(config, model, task, output, trajectory, **kwargs)
        kp, kd = np.tile(task["hand_stiffness"], 5), np.tile(task["hand_damping"], 5)
        self.joint_gain_overrides.update({name: (float(p), float(d)) for name, p, d in zip(model.hand_names["right"], kp, kd)})

    def bind_simulation(self, **kwargs):
        # Replay must use the recorded material and collider overrides as well
        # as checking their scene signature.
        HandoverController.bind_simulation(self, **kwargs)


def preflight(config, task, output):
    from bimanual_teleop.sim.config import sim_path
    from bimanual_teleop.sim.kinematics import RobotModel
    model = RobotModel(sim_path(config["robot"]["urdf"]), config["robot"]["world_yaw_deg"])
    templates = {side: make_grasp(model, task, side) for side in ("left", "right")}
    q = np.r_[np.deg2rad(config["robot"]["ready_deg"]["left"]), np.deg2rad(config["robot"]["ready_deg"]["right"]),
              templates["left"]["open"], templates["right"]["open"]]
    rail = transform(config["rail"]["initial_center"], np.deg2rad(config["rail"]["initial_rpy_deg"]))
    goals, stage = {}, "planning_model"
    try:
        planner = ArmPlanner(config, model, task, output)
        for side in ("left", "right"):
            stage = side+"_grasp"
            object_pose = rail if side == "left" else transform(task["handover_center_m"], np.deg2rad(task["handover_rpy_deg"]))
            goal = object_pose@templates[side]["flange_in_rail"]
            answer = planner.solve(side, q, goal, np.deg2rad(task["arm_reference_deg"][side]["left_grasp" if side == "left" else "right_grasp"]))
            goals[side] = {"arm_deg": np.rad2deg(answer).tolist(), "flange_world": pose_vector(goal).tolist(),
                           "tip_fit_errors_m": templates[side]["tip_fit_errors_m"]}
        carried = q.copy(); carried[:7] = np.deg2rad(goals["left"]["arm_deg"])
        carried[14:34] = templates["left"]["closed"]
        lifted = extraction_lift_pose(config, task, rail)
        for stage in ("left_lift", "left_clear"):
            object_pose = lifted.copy()
            if stage == "left_clear":
                object_pose[:3, 3] = task["left_clearance_center_m"]
            goal = object_pose@templates["left"]["flange_in_rail"]
            carried[:7] = planner.solve("left", carried, goal, np.deg2rad(task["arm_reference_deg"]["left"]["left_carry"]))
            goals[stage] = {"arm_deg": np.rad2deg(carried[:7]).tolist(), "flange_world": pose_vector(goal).tolist()}
        stage = "left_pregrasp_path"
        left_goal = rail@templates["left"]["flange_in_rail"]
        left_goal[:3, 3] += rail[:3, :3]@[0., 0., task.get("left_cartesian_approach_m", task["cartesian_approach_m"])]
        planner.plan("left", q, left_goal, np.deg2rad(task["arm_reference_deg"]["left"]["left_grasp"]), "preflight_left_pregrasp", rail=rail)
        report = {"status": "preflight_passed", "reason": "nominal_grasp_extraction_ik_and_initial_path_only"}
    except (ValueError, RuntimeError) as error:
        report = {"status": "failed", "reason": str(error), "failed_stage": stage}
    report.update(scene_signature=scene_signature(config, task), grasp_goals=goals)
    atomic_json(Path(output)/"grasp_templates.json", {side: {key: value.tolist() if isinstance(value, np.ndarray) else value
                                                              for key, value in template.items()} for side, template in templates.items()})
    atomic_json(Path(output)/"preflight.json", report)
    atomic_json(Path(output)/"result.json", report)
    atomic_json(Path(output)/"summary.json", {"passed_trials": int(report["status"] == "preflight_passed"), "completed_trials": 1, "requested_trials": 1})
    return report
