"""Validate experiment YAML settings."""
from __future__ import annotations

import numpy as np

TIMING_FIELDS = {
    "hand_transition_s": "Close time (s)",
    "settle_s": "Settle time (s)",
    "joint_speed_rad_s": "Arm speed (rad/s)",
    "joint_acceleration_rad_s2": "Arm acceleration (rad/s2)",
}


def positive_settings(values):
    result = {}
    for key, label in TIMING_FIELDS.items():
        if key in values:
            value = float(values[key])
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{label} must be finite and positive")
            result[key] = value
    return result


def finite_vector(value, count, label):
    result = np.asarray(value, dtype=float)
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError(f"{label} requires {count} finite numbers")
    return result


def apply_overrides(task, references, model):
    """Validate YAML overrides before changing any active values."""
    edits = positive_settings(task)
    grasp = references["grasp"]
    opened = references.get("open")
    if "four_finger_flexion_limit_rad" in task:
        limits = finite_vector(task["four_finger_flexion_limit_rad"], 3, "Four-finger flexion limits")
        for i in range(4):
            indices = np.array([18+4*i, 20+4*i, 21+4*i])
            if np.any(limits < model.lower[indices]) or np.any(limits > model.upper[indices]):
                raise ValueError("Four-finger flexion limits exceed the URDF joint limits")
    if "hand_grasp_override_rad" in task:
        grasp = finite_vector(task["hand_grasp_override_rad"], 20, "Closed hand")
        if np.any(grasp < model.lower[14:34]) or np.any(grasp > model.upper[14:34]):
            raise ValueError("Closed hand exceeds the URDF joint limits")
    if "hand_open_override_rad" in task:
        opened = finite_vector(task["hand_open_override_rad"], 20, "Open hand")
        if np.any(opened < model.lower[14:34]) or np.any(opened > model.upper[14:34]):
            raise ValueError("Open hand exceeds the URDF joint limits")
    if "rail_in_flange_override" in task:
        value = task["rail_in_flange_override"]
        edits["rail_in_flange_xyz_m"] = finite_vector(value["xyz_m"], 3, "Rail position").tolist()
        edits["rail_in_flange_rpy_deg"] = finite_vector(value["rpy_deg"], 3, "Rail rotation").tolist()
    if task.get("until", "preinsert") not in ("four_fingers", "grasp", "preinsert"):
        raise ValueError("Stop stage must be four_fingers, grasp or preinsert")
    task.update(edits)
    result = {**references, "grasp": np.asarray(grasp).copy()}
    if opened is not None:
        result["open"] = np.asarray(opened).copy()
    return result


HOT_SIMULATION = {
    "robot.ready_deg.left", "robot.ready_deg.right", "robot.arm_stiffness", "robot.arm_damping",
    "rail.mass_kg", "rail.mass_is_assumption", "rail.friction", "rail.finger_friction",
    "rail.contact_offset_m", "rail.rest_offset_m", "runtime.joint_speed_rad_s", "runtime.hand_speed_rad_s",
}


def changed_paths(before, after, prefix=""):
    if isinstance(before, dict) and isinstance(after, dict):
        result = []
        for key in sorted(before.keys() | after.keys()):
            path = f"{prefix}.{key}" if prefix else key
            result.extend(changed_paths(before.get(key), after.get(key), path))
        return result
    return [] if before == after else [prefix]


def rebuild_changes(before, after, old_task, new_task):
    structural = set(changed_paths(before, after)) - HOT_SIMULATION
    for key in ("convex_decomposition_links",):
        if old_task.get(key) != new_task.get(key):
            structural.add("task." + key)
    return structural


def ready_joints(config, runtime, model):
    ready = np.r_[np.radians(finite_vector(config["robot"]["ready_deg"]["left"], 7, "Left arm")),
                  np.radians(finite_vector(config["robot"]["ready_deg"]["right"], 7, "Right arm")),
                  finite_vector(runtime["open"], 20, "Open hand"), np.zeros(20)]
    if np.any(ready < model.lower) or np.any(ready > model.upper):
        raise ValueError("Initial pose exceeds joint limits")
    return ready


def validate_task(task):
    from desktop_layout import layout_settings
    layout_settings(task.get("desktop_layout"))
    initial_rail = task.get("initial_rail_pose_world")
    if initial_rail is not None:
        finite_vector(initial_rail["xyz_m"], 3, "Initial rail world position")
        finite_vector(initial_rail["rpy_deg"], 3, "Initial rail world rotation")
    if task.get("initial_flange_z_axis_world") is not None:
        value = finite_vector(task["initial_flange_z_axis_world"], 3, "Initial flange Z direction")
        if np.linalg.norm(value) < 1e-10:
            raise ValueError("Initial flange Z direction must be nonzero")
    if task["end_stage"] != "preinsert" or task["preinsert_offset_m"] != .02:
        raise ValueError("This experiment stops at a 20 mm preinsert offset")
    template = task.get("grasp_template", "legacy_geometry")
    if template not in ("demonstrated_power_wrap", "legacy_geometry"):
        raise ValueError("Unknown grasp_template")
    if (template == "legacy_geometry" and "rail_in_flange_override" not in task
            and not all(key in task for key in ("rail_in_flange_xyz_m", "rail_in_flange_rpy_deg"))):
        raise ValueError("Legacy grasp requires a complete rail-in-flange pose")
    for key, value in task.items():
        if (key.endswith(("_s", "_rad", "_m", "_rad_s", "_rad_s2", "_n")) and not isinstance(value, (list, dict))
                and (isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0)):
            raise ValueError(f"Invalid task setting: {key}")
    for key in ("initial_rail_offset_world_m", "rail_in_flange_xyz_m", "rail_in_flange_rpy_deg"):
        if key in task:
            finite_vector(task[key], 3, key)
    if task.get("until", "preinsert") not in ("four_fingers", "grasp", "preinsert"):
        raise ValueError("Stop stage must be four_fingers, grasp or preinsert")
    if task.get("until") == "four_fingers" and initial_rail is None:
        raise ValueError("Four-finger closure requires initial_rail_pose_world at the hand")
    count = task.get("minimum_support_fingers", 4)
    if type(count) is not int or not 1 <= count <= 4:
        raise ValueError("minimum_support_fingers must be 1..4")
    for key, count in (("left_hand_stiffness", 20), ("left_hand_damping", 20)):
        if key in task and np.any(finite_vector(task[key], count, key) <= 0):
            raise ValueError(f"{key} must be positive")


