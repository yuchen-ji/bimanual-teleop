"""Check the existing hand command under gravity, without a rail attachment."""
import numpy as np
from common import finger_forces, pose_error, write_json

from bimanual_teleop.sim.kinematics import pose_matrix


class UnsupportedHold:
    def __init__(self, source, state, duration_s=5.):
        self.source = source
        self.command = source.command.copy()
        self.initial = state
        self.reference = np.linalg.inv(source.flange(state)) @ pose_matrix(state["rail"])
        self.duration_s = duration_s
        self.maximum = np.zeros(2)
        self.peak_forces = dict.fromkeys(finger_forces(state), 0.)
        self.samples = []

    def step(self, state, dt):
        elapsed = (state["time_ns"]-self.initial["time_ns"])/1e9
        relative = np.linalg.inv(self.source.flange(state)) @ pose_matrix(state["rail"])
        error = np.asarray(pose_error(relative, self.reference))
        self.maximum = np.maximum(self.maximum, error)
        forces = finger_forces(state)
        for name, force in forces.items():
            self.peak_forces[name] = max(self.peak_forces[name], force)
        self.samples.append((elapsed, relative))
        recent = [v for t, v in self.samples if elapsed-t <= 1.]
        settled = np.max([pose_error(v, relative) for v in recent], axis=0)
        environment = max(state.get("rail_support_contacts", {}).values(), default=0.)
        metrics = {"elapsed_s": elapsed, "planned_duration_s": self.duration_s,
                   "relative_displacement_m_rad": error.tolist(), "maximum_displacement_m_rad": self.maximum.tolist(),
                   "world_vertical_displacement_m": float(state["rail"][2]-self.initial["rail"][2]),
                   "last_second_motion_m_rad": settled.tolist(), "finger_force_n": forces,
                   "peak_finger_force_n": self.peak_forces.copy(), "rail_environment_force_n": environment,
                   "fixed_constraint_enabled": bool(self.source.joint.GetJointEnabledAttr().Get()),
                   "rail_kinematic": bool(self.source.body.GetKinematicEnabledAttr().Get()),
                   "hand_command_unchanged": bool(np.array_equal(self.source.command, self.command)),
                   "simulation_only": True, "transport_grasp_validated": False}
        self.source.metrics["unsupported_hold"] = metrics
        reason = None
        if metrics["fixed_constraint_enabled"] or metrics["rail_kinematic"] or self.source.support_active:
            reason = "Unsupported test invalid: rail support is still enabled"
        elif not metrics["hand_command_unchanged"]:
            reason = "Unsupported test invalid: hand or arm command changed"
        elif environment > .05:
            reason = "Rail touched an external support"
        elif error[0] > .002 or error[1] > np.deg2rad(3.):
            reason = f"Rail slipped after releasing support: {1000*error[0]:.3f} mm, {np.degrees(error[1]):.3f} degrees"
        elif elapsed >= self.duration_s and (settled[0] > .0005 or settled[1] > np.deg2rad(.5)):
            reason = "Rail did not settle during the unsupported static hold"
        if reason is not None or elapsed >= self.duration_s:
            actual = dict(state)
            view = self.source.robot._articulation_view._physics_view
            names = list(view.shared_metatype.link_names)
            if names and isinstance(names[0], (tuple, list)):
                names = list(names[0])
            poses = view.get_link_transforms()[0]
            actual["hand_link_poses"] = {n: np.asarray(poses[i]).tolist() for i, n in enumerate(names)
                                         if n == "l_wrist" or n.startswith(("l_thumb_", "l_index_finger_", "l_middle_finger_", "l_ring_finger_", "l_pinky_"))}
            write_json(self.source.output / "unsupported_hold_state.json", actual)
            write_json(self.source.output / "unsupported_hold_result.json", {"passed": reason is None, "reason": reason, **metrics})
            self.source.finish("failed" if reason else "unsupported_static_hold_passed", reason or "Five-second unsupported static hold; transport not tested")
            self.source.pause_physics_requested = True
        return self.command.copy()
