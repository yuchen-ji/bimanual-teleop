"""Feedback-driven geometric baseline; no physical robot drivers or grasp attachment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from bimanual_teleop.sim import ROOT
from bimanual_teleop.sim.kinematics import pose_matrix, pose_vector, transform
from bimanual_teleop.sim.recording import atomic_json
from evaluate import initial_pose_valid, insertion_metrics, is_seated, opposing_contacts, pose_error, seated_pose, unintended_robot_contacts


PHASES = ("settle", "raise", "pregrasp", "slot", "approach", "wrap", "close", "lift", "clear", "transfer", "level", "align", "insert", "release", "retreat", "verify")
MILESTONES = {"settle": "scene", "lift": "grasp", "align": "transport", "verify": "insert"}


def interpolate_pose(start, target, fraction):
    pose = start.copy()
    pose[:3, 3] = (1-fraction)*start[:3, 3]+fraction*target[:3, 3]
    pose[:3, :3] = Slerp([0, 1], Rotation.from_matrix(np.stack([start[:3, :3], target[:3, :3]])))(fraction).as_matrix()
    return pose


class RailController:
    name = "rail_grasp_insert_script"

    def __init__(self, config, model, task, output, *, mode="step", until="insert", exit_on_complete=False, snapshots=False, start_phase=None,
                 initial_positions=None):
        self.config, self.model, self.task = config, model, task
        self.output = Path(output)
        self.mode, self.until, self.exit_on_complete = mode, until, exit_on_complete
        self.entry_phase = start_phase or "raise"
        self.diagnostic = start_phase is not None
        self.capture_snapshots = snapshots
        self.geometry = json.loads((ROOT/"assets/generated/manifest.json").read_text())["fixture_geometry"]
        kp, kd = np.asarray(task["hand_stiffness"]), np.asarray(task["hand_damping"])
        if kp.shape != (4,) or kd.shape != (4,) or not np.isfinite(np.r_[kp, kd]).all() or np.any(kp <= 0) or np.any(kd <= 0):
            raise ValueError("Hand gains require four finite positive stiffness and damping values")
        self.joint_gain_overrides = {name: (float(p), float(d)) for name, p, d in
                                     zip(model.hand_names["left"], np.tile(kp, 5), np.tile(kd, 5))}
        self.ready = np.r_[np.deg2rad(config["robot"]["ready_deg"]["left"]),
                           np.deg2rad(config["robot"]["ready_deg"]["right"]), np.zeros(40)]
        if initial_positions is not None:
            supplied = np.asarray(initial_positions, dtype=float)
            if supplied.shape != (54,) or not np.isfinite(supplied).all() or np.any(supplied < model.lower-.01) or np.any(supplied > model.upper+.01):
                raise ValueError("Invalid initial joint positions")
            self.ready = np.clip(supplied, model.lower, model.upper)
        for name in ("open_hand_rad", "wrap_hand_rad", "closed_hand_rad"):
            if np.any(np.asarray(task[name]) < model.lower[14:34]) or np.any(np.asarray(task[name]) > model.upper[14:34]):
                raise ValueError(f"{name} exceeds Hand2 joint limits")
        clearance_xy = np.asarray(task["clearance_center_xy_m"])
        if clearance_xy.shape != (2,) or not np.isfinite(clearance_xy).all():
            raise ValueError("clearance_center_xy_m requires two finite coordinates")
        pregrasp_offset = np.asarray(task["pregrasp_offset_world_m"])
        if pregrasp_offset.shape != (3,) or not np.isfinite(pregrasp_offset).all():
            raise ValueError("pregrasp_offset_world_m requires three finite coordinates")
        retreat_direction = np.asarray(task.get("retreat_direction_in_rail", [0, 1, .4]), dtype=float)
        if retreat_direction.shape != (3,) or not np.isfinite(retreat_direction).all() or np.linalg.norm(retreat_direction) < 1e-9:
            raise ValueError("retreat_direction_in_rail requires a finite nonzero XYZ vector")
        self.retreat_direction = retreat_direction/np.linalg.norm(retreat_direction)
        self.transfer_center = np.asarray(task.get("transfer_center_m", [.53, .18, 1.00]), dtype=float)
        if self.transfer_center.shape != (3,) or not np.isfinite(self.transfer_center).all():
            raise ValueError("transfer_center_m requires three finite coordinates")
        margin = float(task.get("joint_limit_margin_rad", np.deg2rad(5)))
        if not np.isfinite(margin) or margin <= 0 or np.any(self.model.upper[:7]-self.model.lower[:7] <= 2*margin):
            raise ValueError("Invalid arm joint planning margin")
        self.ik_lower, self.ik_upper = self.model.lower[:7]+margin, self.model.upper[:7]-margin
        waypoints = np.asarray(task["approach_waypoints_in_rail_m"])
        if (waypoints.ndim != 2 or waypoints.shape[1] != 3 or len(waypoints) == 0
                or not np.isfinite(waypoints).all() or np.linalg.norm(waypoints[-1]) > 1e-9):
            raise ValueError("Approach waypoints require finite XYZ offsets ending at the grasp origin")
        self.grip_template_base = transform(task["rail_in_flange_xyz_m"], np.deg2rad(task["rail_in_flange_rpy_deg"]))
        self.goal_rail = seated_pose(config, self.geometry, task["reverse_pin_order"])
        self.label = None
        self.repeat = 1
        self.parent_sources = []
        self.trial_results = []
        self.trial_index = -1
        self.trace = self.events = None
        self.last_state = None
        self.reset()

    def reset(self):
        if self.trace is not None:
            self.finish("interrupted", "scene reset")
            self.trace.close(); self.events.close()
        self.trial_index += 1
        self.trial = self.output/f"trial_{self.trial_index:03d}"
        self.trial.mkdir(parents=True, exist_ok=False)
        self.trace = (self.trial/"trace.jsonl").open("w")
        self.events = (self.trial/"events.jsonl").open("w")
        self.phase_index = 0
        self.elapsed = self.stable = 0.
        self.active = True
        self.waiting = False
        self.finished = self.should_stop = False
        self.reset_requested = False
        self.status, self.reason = "running", None
        self.stop_reason = None
        self.command = self.ready.copy()
        self.grip = self.lift_reference = self.grasp_rail = None
        self.grip_template = self.grip_template_base.copy()
        self.start_pose = self.target_pose = None
        self.start_object = self.target_object = self.motion_grip = None
        self.duration = 0.
        self.motion_started_s = 0.
        self.approach_index = 0
        self.ik_goal = None
        self.origin_time = None
        self.origin_command_tick = None
        self.last_state = None
        self.last_observed_pose = self.last_observed_time = None
        self.linear_speed = self.angular_speed = float("inf")
        self.filtered_motion = None
        self.contact_lost_s = 0.
        self.arm_tracking_bad_s = 0.
        self.level_tracking_bad_s = 0.
        self.level_hold_grip = None
        self.hold_object = None
        self.release_hand = None
        self.snapshot_requested = self.capture_snapshots
        self.snapshot_label = "initial"
        self.samples = [{"time_s": 0., "positions_rad": self.ready.tolist()}]
        self.completed_phases = []
        self.recordings = []
        self.metrics = {}
        self.report()

    @property
    def phase(self):
        return PHASES[self.phase_index]

    def event(self, kind, **values):
        entry = {"kind": kind, "phase": self.phase, "time_ns": self.last_state.get("time_ns") if self.last_state else None, **values}
        self.events.write(json.dumps(entry)+"\n"); self.events.flush()
        print("Rail experiment: "+json.dumps(entry), flush=True)

    def report(self):
        result = {"status": self.status, "reason": self.reason, "phase": self.phase, "mode": self.mode,
                  "waiting": self.waiting, "until": self.until, "trial": self.trial.name,
                  "completed_phases": self.completed_phases, "metrics": self.metrics, "test_data": True}
        result.update(diagnostic=self.diagnostic, entry_phase=self.entry_phase)
        result["recordings"] = self.recordings
        atomic_json(self.output/"result.json", result)
        atomic_json(self.trial/"result.json", result)
        if self.label is not None:
            self.label.text = f"{self.phase}: {self.status}"+(" | Next stage" if self.waiting else "")

    def finish(self, status, reason=None):
        if self.finished:
            return
        self.status, self.reason = status, reason
        if status == "failed" and self.capture_snapshots:
            self.snapshot_requested = True
            self.snapshot_label = self.phase+"_failed"
        self.finished = True
        self.trial_results.append({"trial": self.trial.name, "status": status, "reason": reason,
                                   "phase": self.phase, "metrics": self.metrics})
        passed = status in ("success", "milestone_passed", "diagnostic_passed")
        self.reset_requested = passed and len(self.trial_results) < self.repeat
        self.should_stop = self.exit_on_complete and not self.reset_requested
        atomic_json(self.output/"summary.json", {"requested_trials": self.repeat, "completed_trials": len(self.trial_results),
                    "passed_trials": sum(r["status"] in ("success", "milestone_passed", "diagnostic_passed") for r in self.trial_results),
                    "trials": self.trial_results})
        self.event("finished", status=status, reason=reason)
        self.report()
        self.save_trajectory()

    def save_trajectory(self):
        samples = self.samples
        if len(samples) < 2:
            samples = [*samples, {"time_s": 1/self.config["runtime"]["control_hz"],
                                  "positions_rad": self.ready.tolist()}]
        atomic_json(self.trial/"trajectory.json", {"joint_names": list(self.model.names), "test_data": True,
                    "waypoints": samples, "experiment_result": self.status, "source_trial": self.trial.name,
                    "diagnostic": self.diagnostic, "source_run": self.output.name,
                    "source_demo_ids": [item["source_demo_id"] for item in self.recordings],
                    "parent_sources": self.parent_sources})

    def recording_started(self, path, source_demo_id):
        self.recordings.append({"path": str(path), "source_demo_id": source_demo_id})
        self.event("recording_started", path=str(path), source_demo_id=source_demo_id)
        self.report()

    def recording_metadata(self):
        version_path = self.output/"metadata.json"
        version = json.loads(version_path.read_text()) if version_path.is_file() else {}
        return {"controller": self.name, "source_run": self.output.name, "source_trial": self.trial.name,
                "diagnostic": self.diagnostic, "entry_phase": self.entry_phase,
                "task_settings": self.task, "joint_gain_overrides": self.joint_gain_overrides,
                "software": version, "parent_sources": self.parent_sources}

    def scene_ready(self, diagnostics):
        atomic_json(self.trial/"runtime.json", diagnostics)
        self.event("runtime_ready")

    def advance(self):
        if self.finished:
            return
        if self.waiting:
            self.waiting = False
            self.phase_index = PHASES.index(self.entry_phase) if self.phase_index == 0 else self.phase_index+1
            self.elapsed = self.stable = 0.
            self.start_pose = self.target_pose = None
            self.approach_index = 0
            self.level_hold_grip = self.hold_object = None
            self.event("phase_started")
            self.report()
        self.active = True

    def automatic(self):
        self.mode = "auto"
        self.advance()

    def complete_phase(self):
        self.completed_phases.append(self.phase)
        self.event("phase_passed")
        if self.capture_snapshots:
            self.snapshot_requested = True
            self.snapshot_label = self.phase
        if MILESTONES.get(self.phase) == self.until:
            self.finish("diagnostic_passed" if self.diagnostic else ("success" if self.phase == "verify" else "milestone_passed"))
        elif self.mode == "step":
            self.waiting = True
            self.report()
        else:
            self.waiting = True
            self.advance()

    def build_ui(self, ui):
        ui.Separator()
        self.label = ui.Label(f"Rail experiment: {self.phase}")
        ui.Button("Next stage", clicked_fn=self.advance)
        ui.Button("Run automatically", clicked_fn=self.automatic)
        ui.Button("Pause task / Resume", clicked_fn=self.toggle)
        if self.capture_snapshots:
            ui.Button("Capture snapshot", clicked_fn=self.snapshot)

    def toggle(self):
        self.active = not self.active

    def snapshot(self):
        self.snapshot_requested = True
        self.snapshot_label = self.phase+"_manual"

    def observe(self, state, images=None):
        pose = pose_matrix(state["rail"])
        stamp = state["time_ns"]/1e9
        if self.last_observed_time is not None and stamp > self.last_observed_time:
            linear, angular = pose_error(pose, self.last_observed_pose)
            dt = stamp-self.last_observed_time
            self.linear_speed, self.angular_speed = linear/dt, angular/dt
            measured = np.array([self.linear_speed, self.angular_speed,
                                 sum(state.get("rail_support_contacts", {}).values())])
            alpha = -np.expm1(-dt/self.task["stability_filter_s"])
            self.filtered_motion = measured if self.filtered_motion is None else self.filtered_motion+alpha*(measured-self.filtered_motion)
        self.last_observed_time, self.last_observed_pose = stamp, pose
        self.last_state = state
        if self.origin_time is None:
            self.origin_time = stamp
            self.origin_command_tick = state.get("command_tick", state["tick"]-1)
        if state["tick"] % 2 == 0:
            self.metrics = {**insertion_metrics(state, self.config, self.geometry),
                            "rail_linear_speed_m_s": self.linear_speed, "rail_angular_speed_rad_s": self.angular_speed}
            if self.filtered_motion is not None:
                self.metrics.update(filtered_linear_speed_m_s=float(self.filtered_motion[0]),
                                    filtered_angular_speed_rad_s=float(self.filtered_motion[1]),
                                    filtered_support_force_n=float(self.filtered_motion[2]))
            row = {k: state[k] for k in ("tick", "time_ns", "q", "q_target", "eef", "rail", "rail_contacts", "environment_contacts")}
            row["rail_support_contacts"] = state.get("rail_support_contacts", {})
            row["rail_contact_vectors"] = state.get("rail_contact_vectors", {})
            row["command_tick"] = state.get("command_tick", state["tick"]-1)
            row["robot_base_contacts"] = state.get("robot_base_contacts", {})
            row["rail_robot_base_force_n"] = state.get("rail_robot_base_force_n", 0.)
            row["arm_joint_effort"] = state.get("arm_joint_effort", [])
            row["arm_drive_target"] = state.get("arm_drive_target", [])
            row["physics_time_s"] = state.get("physics_time_s")
            row.update(phase=self.phase, status=self.status, metrics=self.metrics)
            self.trace.write(json.dumps(row)+"\n")
            # Trajectory playback needs command-application time. Observation
            # time would delay each replayed control update by a control period.
            elapsed = (row["command_tick"]-self.origin_command_tick)/self.config["runtime"]["physics_hz"]
            if elapsed > self.samples[-1]["time_s"]:
                self.samples.append({"time_s": elapsed, "positions_rad": list(state["q_target"])})
        if state["tick"] % 120 == 0:
            self.trace.flush()
            self.report()
        if images is not None and self.snapshot_requested:
            from PIL import Image
            folder = self.trial/"snapshots"/self.snapshot_label
            folder.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(images):
                Image.fromarray(frame).save(folder/f"camera_{i}.png")
            atomic_json(folder/"state.json", state)
            self.snapshot_requested = False

    def solve_arm(self, state, goal):
        from scipy.optimize import least_squares
        from bimanual_teleop.sim.config import load_config
        def residual(arm):
            q = np.asarray(state["q"]).copy(); q[:7] = arm
            flange = self.model.flange("left", q, world=True)
            return np.r_[3*(flange[:3, 3]-goal[:3, 3]),
                         (Rotation.from_matrix(goal[:3, :3])*Rotation.from_matrix(flange[:3, :3]).inv()).as_rotvec()]
        nominal = np.deg2rad(load_config()["robot"]["ready_deg"]["left"])
        solution = None
        for seed in (np.asarray(state["q"])[:7], self.ready[:7], nominal, (self.model.lower[:7]+self.model.upper[:7])/2):
            candidate = least_squares(residual, np.clip(seed, self.ik_lower+1e-6, self.ik_upper-1e-6),
                                      bounds=(self.ik_lower+1e-6, self.ik_upper-1e-6), max_nfev=200)
            if solution is None or np.linalg.norm(candidate.fun) < np.linalg.norm(solution.fun):
                solution = candidate
            if np.linalg.norm(solution.fun) < .001:
                break
        return solution.x, float(np.linalg.norm(solution.fun))

    def select_grasp(self, state, rail):
        candidates = []
        from bimanual_teleop.sim.bin_geometry import bin_geometry
        rim = self.config["scene"]["bin_centers"][0][2]+bin_geometry(self.config["scene"])["rim_height_m"]
        # Yaw reversal preserves the thumb-inside / fingers-under-plate contact
        # geometry. Flipping the rail normal would invalidate these hand presets.
        for flip_z in (0, np.pi):
            template = self.grip_template_base@transform(rpy=(0, 0, flip_z))
            flange = rail@np.linalg.inv(template)
            if flange[2, 3] < rim+.035 or flange[0, 3] > rail[0, 3]:
                continue
            _, residual = self.solve_arm(state, flange)
            if residual < .008:
                candidates.append((residual, template))
        if not candidates:
            self.fail("no_feasible_grasp_template_above_box_rim")
            return
        _, self.grip_template = min(candidates, key=lambda pair: pair[0])
        self.event("grasp_template_selected", template=self.grip_template.tolist())

    def setup_motion(self, state):
        rail = pose_matrix(state["rail"])
        current = self.model.flange("left", state["q"], world=True)
        goal = current.copy()
        self.start_object = self.target_object = self.motion_grip = None
        if self.phase == "raise":
            goal[2, 3] = 1.20
        elif self.phase in ("pregrasp", "slot", "approach"):
            if self.grasp_rail is None:
                self.grasp_rail = rail.copy()
                self.select_grasp(state, rail)
                if self.finished:
                    return
            goal = self.grasp_rail@np.linalg.inv(self.grip_template)
            if self.phase in ("pregrasp", "slot"):
                goal[:3, 3] += self.grasp_rail[:3, :3]@np.asarray(self.task["approach_offset_in_rail_m"])
            else:
                goal[:3, 3] += self.grasp_rail[:3, :3]@np.asarray(self.task["approach_waypoints_in_rail_m"][self.approach_index])
            if self.phase == "pregrasp":
                goal[:3, 3] += self.task["pregrasp_offset_world_m"]
        elif self.phase == "lift":
            goal[:3, 3] += [0, 0, self.task["lift_m"]]
            self.lift_reference = rail.copy()
            self.start_object, self.target_object = rail.copy(), rail.copy()
            self.target_object[2, 3] += self.task["lift_m"]
            self.motion_grip = np.linalg.inv(current)@rail
        elif self.phase == "clear":
            target = rail.copy(); target[2, 3] = self.task["clearance_z_m"]
            target[:2, 3] = self.task["clearance_center_xy_m"]
            self.start_object, self.target_object, self.motion_grip = rail.copy(), target, np.linalg.inv(current)@rail
            goal = target@np.linalg.inv(self.motion_grip)
        elif self.phase in ("level", "transfer", "align"):
            target = self.goal_rail.copy()
            if self.phase == "level":
                target[2, 3] = self.transfer_center[2]
            elif self.phase == "transfer":
                target[:3, :3] = rail[:3, :3]
                target[:3, 3] = self.transfer_center
            else:
                target[2, 3] += self.task["preinsert_height_m"]
            self.start_object, self.target_object, self.motion_grip = rail.copy(), target, np.linalg.inv(current)@rail
            goal = target@np.linalg.inv(self.motion_grip)
        elif self.phase == "insert":
            self.start_object, self.target_object, self.motion_grip = rail.copy(), self.goal_rail.copy(), np.linalg.inv(current)@rail
            goal = self.goal_rail@np.linalg.inv(self.motion_grip)
        elif self.phase == "retreat":
            # Move across the rail width with a small rise. The flange's finger
            # direction is mostly along the rail in this grasp, so following it
            # backward would lift the fingers through the underside of the plate.
            goal[:3, 3] += rail[:3, :3]@self.retreat_direction*self.task["retreat_m"]
        self.start_pose, self.target_pose = current, goal
        arm, residual = self.solve_arm(state, goal)
        if residual > .01:
            self.fail(f"ik_not_solved: residual {residual:.4f}")
            return
        self.ik_goal = arm
        self.start_arm = self.command[:7].copy()
        position, angle = pose_error(current, goal)
        speed = self.task["insertion_speed_m_s"] if self.phase == "insert" else self.task["linear_speed_m_s"]
        # Quintic interpolation peaks at 1.875 times its average speed.
        self.duration = max(1., 1.875*position/speed, 1.875*angle/self.task["angular_speed_rad_s"])
        if self.start_object is not None:
            object_distance, object_angle = pose_error(self.start_object, self.target_object)
            radius = float(np.linalg.norm(self.motion_grip[:3, 3]))
            # The flange travels an arc when the rail rotates about its center.
            # Its endpoint chord alone underestimates the required travel time.
            self.duration = max(self.duration, 1.875*(object_distance+object_angle*radius)/speed)
        self.duration = max(self.duration, 1.875*float(np.max(np.abs(self.ik_goal-self.start_arm)))/self.config["runtime"]["joint_speed_rad_s"])
        self.motion_started_s = self.elapsed
        self.event("motion_target", duration_s=self.duration, target_world=pose_vector(goal).tolist())

    def fail(self, reason):
        self.finish("failed", reason)
        return self.command.copy()

    def generate_release_hand(self, state):
        """Open the primary fingertips away from the actual rail plate after rolling."""
        from scipy.optimize import least_squares
        q = np.asarray(state["q"], dtype=float).copy()
        inverse_object = np.linalg.inv(pose_matrix(state["rail"]))
        result = np.asarray(self.task["open_hand_rad"], dtype=float).copy()
        goals = {}
        for start, link, direction in ((14, "l_thumb_tip", 1), (18, "l_index_finger_tip", -1), (22, "l_middle_finger_tip", -1)):
            initial = q[start:start+4].copy()
            point = (inverse_object@self.model.link_pose(link, q))[:3, 3]
            if start == 14:
                point[1] = np.clip(point[1], -.004, .004)
            best = None
            for clearance in (self.task.get("release_clearance_m", .035), .028):
                goal = point.copy(); goal[2] = direction*clearance
                def residual(values):
                    q[start:start+4] = values
                    actual = (inverse_object@self.model.link_pose(link, q))[:3, 3]
                    return np.r_[30*(actual-goal), .001*(values-initial)]
                for seed in (initial, result[start-14:start-10], np.zeros(4)):
                    lower, upper = self.model.lower[start:start+4]+1e-6, self.model.upper[start:start+4]-1e-6
                    candidate = least_squares(residual, np.clip(seed, lower, upper), bounds=(lower, upper), max_nfev=300)
                    error = float(np.linalg.norm(candidate.fun[:3])/30)
                    if best is None or error < best[0]:
                        best = error, candidate.x.copy(), goal.copy()
                    if error < .0005:
                        break
                if best[0] < .002:
                    break
            if best[0] >= .002:
                self.fail("release_hand_ik_not_solved: "+link)
                return None
            result[start-14:start-10] = best[1]
            q[start:start+4] = best[1]
            goals[link] = best[2].tolist()
        self.event("release_hand_generated", positions_rad=result.tolist(), goals_in_rail_m=goals)
        return result

    def release_exit_ready(self, state):
        """Permit lateral withdrawal after unclamping, including light knuckle contact."""
        contacts = state.get("rail_contacts", {})
        proximal = {"l_index_finger_proximal", "l_middle_finger_proximal"}
        if max((force for name, force in contacts.items() if name not in proximal), default=0.) > .05:
            return False
        if max(contacts.values(), default=0.) > self.task["environment_force_limit_n"]:
            return False
        inverse_object = np.linalg.inv(pose_matrix(state["rail"]))
        for link, direction in (("l_thumb_tip", 1), ("l_index_finger_tip", -1), ("l_middle_finger_tip", -1)):
            point = (inverse_object@self.model.link_pose(link, state["q"]))[:3, 3]
            if direction*point[2] < .024:
                return False
        return True

    def step(self, state, dt):
        if self.finished or self.waiting or not self.active:
            return self.command.copy()
        self.elapsed += dt
        phase = self.phase
        if not np.isfinite(state["q"]).all() or not np.isfinite(state["rail"]).all():
            return self.fail("nonfinite_state")
        tracking = np.abs(np.asarray(state.get("q_target", state["q"]))[:14]-np.asarray(state["q"])[:14])
        self.arm_tracking_bad_s = self.arm_tracking_bad_s+dt if tracking.max() > self.task.get("max_arm_tracking_error_rad", .25) else 0.
        if self.arm_tracking_bad_s > self.task.get("arm_tracking_timeout_s", .25):
            return self.fail("arm_tracking_error: "+self.model.names[int(np.argmax(tracking))])
        environment = state.get("environment_contacts", {})
        if self.elapsed > .2 and max(environment.values(), default=0) > self.task["environment_force_limit_n"]:
            return self.fail("unexpected_environment_contact: "+max(environment, key=environment.get))
        if phase == "settle":
            stable = (initial_pose_valid(state, self.config) and max(state.get("rail_contacts", {}).values(), default=0) < .05
                      and self.linear_speed < self.task["stable_linear_speed_m_s"]
                      and self.angular_speed < self.task["stable_angular_speed_rad_s"])
            self.stable = self.stable+dt if stable else 0.
            if self.stable >= self.task["settle_s"]:
                self.complete_phase()
            elif self.elapsed > self.task["settle_timeout_s"]:
                return self.fail("initial_rail_not_stable")
            return self.command.copy()
        if self.elapsed > max(self.task["stage_timeout_s"], self.duration+10):
            return self.fail("stage_timeout")
        if phase == "lift" and self.elapsed > self.duration+5:
            return self.fail("lift_not_stable_or_rail_still_supported")
        if phase in ("lift", "clear", "level", "transfer", "align", "insert"):
            bad = unintended_robot_contacts(state, self.task["finger_contact_min_n"])
            if bad:
                return self.fail("rail_contact_outside_grasp: "+max(bad, key=bad.get))
        if self.grip is not None and phase in ("clear", "level", "transfer", "align", "insert"):
            self.contact_lost_s = 0. if opposing_contacts(state, self.task["finger_contact_min_n"]) else self.contact_lost_s+dt
            if self.contact_lost_s > .1:
                return self.fail("lost_opposing_finger_contacts")
            actual_grip = np.linalg.inv(self.model.flange("left", state["q"], world=True))@pose_matrix(state["rail"])
            slip, angle = pose_error(actual_grip, self.grip)
            if slip > self.task["grip_slip_m"] or (phase != "level" and angle > self.task["grip_slip_rad"]):
                return self.fail(f"grasp_slip: {slip:.4f} m, {angle:.4f} rad")
            if phase != "insert" and max(state.get("rail_support_contacts", {}).values(), default=0) > .5:
                return self.fail("carried_rail_hit_environment")
        if phase == "release" and self.release_hand is None:
            self.release_hand = self.generate_release_hand(state)
            if self.finished:
                return self.command.copy()
        desired_hand = self.task["closed_hand_rad"] if phase in ("close", "lift", "clear", "level", "transfer", "align", "insert") else self.task["open_hand_rad"]
        if phase in ("release", "retreat", "verify") and self.release_hand is not None:
            desired_hand = self.release_hand
        if phase in ("pregrasp", "slot", "approach", "wrap"):
            desired_hand = self.task["wrap_hand_rad"]
        max_hand = self.task["hand_speed_rad_s"]*dt
        self.command[14:34] += np.clip(np.asarray(desired_hand)-self.command[14:34], -max_hand, max_hand)
        if phase == "wrap":
            reached = np.max(np.abs(self.command[14:34]-desired_hand)) < 1e-5
            reached = reached and np.max(np.abs(np.asarray(state["q"])[18:26]-np.asarray(desired_hand)[4:12])) < .2
            self.stable = self.stable+dt if reached else 0.
            if self.stable >= .2:
                self.complete_phase()
            elif self.elapsed > 8:
                return self.fail("finger_wrap_blocked")
            return self.command.copy()
        if phase == "close":
            if np.max(np.abs(self.command[14:34]-desired_hand)) < 1e-5:
                self.stable = self.stable+dt if opposing_contacts(state, self.task["finger_contact_min_n"]) else 0.
                if self.stable >= .5:
                    self.complete_phase()
                elif self.elapsed > 6:
                    return self.fail("no_opposing_finger_contacts")
            return self.command.copy()
        if phase == "release":
            if np.max(np.abs(self.command[14:34]-desired_hand)) < 1e-5 and self.release_exit_ready(state):
                self.event("fingers_unclamped", remaining_contacts=state.get("rail_contacts", {}))
                self.complete_phase()
            return self.command.copy()
        if phase == "verify":
            passed, _ = is_seated(state, self.config, self.geometry, self.task, self.linear_speed, self.angular_speed)
            self.stable = self.stable+dt if passed else 0.
            if self.stable >= self.task["seated_hold_s"]:
                self.complete_phase()
            return self.command.copy()
        if self.target_pose is None:
            self.setup_motion(state)
            if self.finished:
                return self.command.copy()
        fraction = np.clip((self.elapsed-self.motion_started_s)/self.duration, 0, 1)
        smooth = fraction**3*(10-15*fraction+6*fraction*fraction)
        if self.start_object is not None:
            desired_object = interpolate_pose(self.start_object, self.target_object, smooth)
            if phase == "level":
                tracking_position, tracking_angle = pose_error(pose_matrix(state["rail"]), desired_object)
                bad = tracking_position > .01 or tracking_angle > self.task.get("level_tracking_tolerance_rad", .2)
                self.level_tracking_bad_s = self.level_tracking_bad_s+dt if bad else 0.
                if self.level_tracking_bad_s > .25:
                    return self.fail("reorientation_tracking_error")
            if phase == "lift":
                target = desired_object@np.linalg.inv(self.motion_grip)
            else:
                # Contact compliance changes the object/hand transform under
                # gravity. Servo the actual object along its planned path.
                measured_grip = np.linalg.inv(self.model.flange("left", state["q"], world=True))@pose_matrix(state["rail"])
                target = desired_object@np.linalg.inv(measured_grip)
        else:
            target = interpolate_pose(self.start_pose, self.target_pose, smooth)
        settings = {**self.config["runtime"],
                    "max_linear_speed_m_s": self.task["insertion_speed_m_s"] if phase == "insert" else self.task["linear_speed_m_s"],
                    "max_angular_speed_rad_s": min(self.task["angular_speed_rad_s"], .05) if phase == "insert" else self.task["angular_speed_rad_s"]}
        self.command[:7] = self.model.velocity_ik("left", state["q"], target, dt, settings, command_positions=self.command)
        if phase != "insert":
            jacobian = self.model.jacobian("left", state["q"])
            nullspace = np.eye(7)-jacobian.T@np.linalg.solve(jacobian@jacobian.T+1e-5*np.eye(6), jacobian)
            bias = .6*nullspace@(self.ik_goal-np.asarray(state["q"][:7]))
            previous = np.asarray(state["q_target"][:7])
            maximum = self.config["runtime"]["joint_speed_rad_s"]*dt
            self.command[:7] = previous+np.clip(self.command[:7]+bias*dt-previous, -maximum, maximum)
            self.command[:7] = np.clip(self.command[:7], self.model.lower[:7], self.model.upper[:7])
        if phase == "pregrasp" and fraction < 1:
            # A large initial wrist rotation uses a feasible joint solution; subsequent
            # short approach/insertion paths retain Cartesian feedback control.
            self.command[:7] = (1-smooth)*self.start_arm+smooth*self.ik_goal
        actual = self.model.flange("left", state["q"], world=True)
        object_control = self.target_object is not None and phase != "lift"
        position, angle = pose_error(pose_matrix(state["rail"]), self.target_object) if object_control else pose_error(actual, self.target_pose)
        position_tolerance = min(self.task["position_tolerance_m"], .00075) if phase == "approach" else self.task["position_tolerance_m"]
        rotation_tolerance = min(self.task["rotation_tolerance_rad"], .01) if phase in ("approach", "level", "align") else self.task["rotation_tolerance_rad"]
        reached = fraction >= 1 and position < position_tolerance and angle < rotation_tolerance
        if phase == "align":
            metrics = insertion_metrics(state, self.config, self.geometry)
            clearance = self.config["rail"]["hole_diameter_m"]/2-max(self.geometry["pin_radii_m"])-.0001
            reached = reached and metrics["pin_error_m"] < clearance and metrics["axis_error_rad"] < .01
        if phase in ("lift", "level", "align") and not reached:
            self.stable = 0.
            if phase in ("level", "align"):
                self.level_hold_grip = self.hold_object = None
        if reached:
            if phase == "approach" and self.approach_index+1 < len(self.task["approach_waypoints_in_rail_m"]):
                self.event("approach_waypoint_passed", index=self.approach_index)
                self.approach_index += 1
                self.start_pose = self.target_pose = None
                return self.command.copy()
            if phase in ("level", "align"):
                actual_object = pose_matrix(state["rail"])
                actual_grip = np.linalg.inv(actual)@actual_object
                if self.level_hold_grip is None or self.hold_object is None:
                    self.level_hold_grip, self.hold_object = actual_grip.copy(), actual_object.copy()
                object_position, object_angle = pose_error(actual_object, self.hold_object)
                grip_position, grip_angle = pose_error(actual_grip, self.level_hold_grip)
                stable = object_position <= .001 and object_angle <= .005 and grip_position <= .001 and grip_angle <= .01
                if not stable:
                    self.level_hold_grip, self.hold_object = actual_grip.copy(), actual_object.copy()
                self.stable = self.stable+dt if stable else 0.
                hold = self.task.get("level_hold_s", 1.) if phase == "level" else .2
                if self.stable < hold:
                    return self.command.copy()
                if phase == "level":
                    self.grip[:3, :3] = actual_grip[:3, :3]
                    self.event("grasp_direction_refreshed", actual_grip=actual_grip.tolist())
            if phase == "lift":
                moved = pose_matrix(state["rail"])[2, 3]-self.lift_reference[2, 3]
                filtered = self.filtered_motion if self.filtered_motion is not None else np.full(3, float("inf"))
                stable = (moved > self.task["lift_m"]-self.task["position_tolerance_m"] and opposing_contacts(state, self.task["finger_contact_min_n"])
                          and filtered[0] < .01 and filtered[1] < self.task["stable_angular_speed_rad_s"]
                          and filtered[2] < .1 and sum(state.get("rail_support_contacts", {}).values()) < .5
                          and self.linear_speed < .05 and self.angular_speed < .25)
                self.stable = self.stable+dt if stable else 0.
                if self.stable < self.task["lift_hold_s"]:
                    return self.command.copy()
                self.grip = np.linalg.inv(actual)@pose_matrix(state["rail"])
            if phase == "insert":
                metrics = insertion_metrics(state, self.config, self.geometry)
                if metrics["support_gap_m"] > self.task["support_gap_tolerance_m"] or metrics["insertion_depth_m"] < self.task["insertion_depth_min_m"]:
                    return self.command.copy()
            self.complete_phase()
        return self.command.copy()

    def close(self, error=None):
        if not self.finished:
            self.finish("failed" if error else "interrupted", error or self.stop_reason or "window closed or physics step limit reached")
        # Include the last observations, even when finish() ran just before the final physics step.
        self.save_trajectory()
        self.report()
        self.trace.close(); self.events.close()


class RailReplayController(RailController):
    """Replay reviewed targets with the same experiment gains and initial scene."""

    name = "rail_grasp_insert_replay"

    def __init__(self, config, model, task, output, trajectory, **kwargs):
        from bimanual_teleop.sim.actions import JointTrajectory
        self.program = JointTrajectory(trajectory, model, config["runtime"])
        document = json.loads(Path(trajectory).read_text())
        if document.get("diagnostic"):
            raise ValueError("Replay requires a trajectory recorded from the normal initial scene")
        ready = np.r_[np.deg2rad(config["robot"]["ready_deg"]["left"]),
                      np.deg2rad(config["robot"]["ready_deg"]["right"]), np.zeros(40)]
        if not np.allclose(self.program.sample(0), ready, atol=1e-6):
            raise ValueError("Replay first waypoint must match this experiment's ready pose")
        super().__init__(config, model, task, output, **kwargs)
        self.parent_sources = [{"source_run": document.get("source_run"), "source_trial": document.get("source_trial"),
                                "source_demo_ids": document.get("source_demo_ids", [])}, *document.get("parent_sources", [])]
        self.diagnostic = True
        self.entry_phase = "replay"
        self.event("replay_loaded", source_run=document.get("source_run"), source_trial=document.get("source_trial"))
        self.report()

    @property
    def phase(self):
        return "replay"

    def step(self, state, dt):
        if self.finished or not self.active:
            return self.command.copy()
        self.elapsed += dt
        self.command = self.program.sample(self.elapsed).copy()
        if self.elapsed >= self.program.times[-1]:
            if self.capture_snapshots:
                self.snapshot_requested = True
                self.snapshot_label = "replay_complete"
            # Completing target playback does not certify physical task success.
            self.finish("diagnostic_passed", "trajectory_replayed_only")
        return self.command.copy()

    def build_ui(self, ui):
        ui.Separator()
        self.label = ui.Label("Rail trajectory replay")
        ui.Button("Pause replay / Resume", clicked_fn=self.toggle)
        if self.capture_snapshots:
            ui.Button("Capture snapshot", clicked_fn=self.snapshot)
