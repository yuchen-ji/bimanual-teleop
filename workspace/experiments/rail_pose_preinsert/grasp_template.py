"""A whole-hand reference from a synchronized demonstration, without wrist flipping."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

FINGERS = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")


def fit_demonstrated_wrap(model, reference, task):
    candidates = {item["label"]: item for item in reference["candidates"]}
    held = candidates["left_hold_primary"]
    measured = np.asarray(held["measured_position_rad"])
    q = np.zeros(54)
    q[:7] = reference["measured_arm_rad"]
    q[14:34] = measured
    F = model.flange("left", q, world=True)
    inverse_F = np.linalg.inv(F)
    # The recording contains joints, not a calibrated object-to-flange pose.
    # For a middle grasp align the rail with the MCP row, not the fingertip
    # diagonal of the original end grasp (which runs back into the forearm).
    knuckles = np.array([(inverse_F @ model.link_pose(f"l_{name}_proximal", q))[:3, 3]
                         for name in FINGERS[1:]])
    local_tips = np.array([(inverse_F @ model.link_pose(f"l_{name}_tip", q))[:3, 3]
                          for name in FINGERS])
    sign = float(task.get("wrap_long_axis_sign", -1))
    if sign not in (-1, 1):
        raise ValueError("wrap_long_axis_sign must be -1 or 1")
    x_F = sign*(knuckles[0]-knuckles[-1]); x_F /= np.linalg.norm(x_F)
    palm_inward = local_tips[1:].mean(axis=0)-knuckles.mean(axis=0)
    z_F = -palm_inward + np.dot(palm_inward, x_F)*x_F
    z_F /= np.linalg.norm(z_F)
    y_F = np.cross(z_F, x_F)
    S_F = np.eye(4)
    S_F[:3, :3] = np.column_stack([x_F, y_F, z_F]).T
    points = np.array([(S_F @ inverse_F @ model.link_pose(f"l_{name}_tip", q))[:3, 3] for name in FINGERS])
    # Centre the four-finger support span at the rail middle, retain its measured
    # longitudinal spread, and place the channel between opposing fingers.
    anchor = points[1:].mean(axis=0)
    anchor[1] = .5*(points[0, 1] + points[1:, 1].mean())
    S_F[:3, 3] = -anchor
    S_F[2, 3] += task["wrap_finger_tip_yz_m"][1]
    targets = points + S_F[:3, 3]
    targets[0, 1:] = task["wrap_thumb_tip_yz_m"]
    targets[1:, 1:] = task["wrap_finger_tip_yz_m"]
    thumb_side = 1. if points[0, 1] >= points[1:, 1].mean() else -1.
    targets[0, 1] = thumb_side*abs(task["wrap_thumb_tip_yz_m"][0])
    targets[1:, 1] = -thumb_side*abs(task["wrap_finger_tip_yz_m"][0])
    fitted, errors = measured.copy(), {}
    maximum = float(task["wrap_max_joint_adjustment_rad"])
    for index, name in enumerate(FINGERS):
        selected = slice(14+4*index, 18+4*index)
        seed = measured[4*index:4*index+4]
        lower = np.maximum(model.lower[selected]+1e-5, seed-maximum)
        upper = np.minimum(model.upper[selected]-1e-5, seed+maximum)
        def residual(values, selected=selected, name=name, index=index, seed=seed):
            q[selected] = values
            position = (S_F @ inverse_F @ model.link_pose(f"l_{name}_tip", q))[:3, 3]
            return np.r_[100*(position-targets[index]), .015*(values-seed)]
        result = least_squares(residual, np.clip(seed, lower, upper), bounds=(lower, upper), max_nfev=150)
        fitted[4*index:4*index+4] = result.x
        errors[name] = float(np.linalg.norm(result.fun[:3])/100)
    return {
        "schema": "demonstrated_left_power_wrap_v1",
        "source_episode": reference["source_episode"],
        "source_manifest_sha256": reference["source_manifest_sha256"],
        "reference_timestamp_ns": reference["reference_timestamp_ns"],
        "reference_relative_time_s": reference["reference_relative_time_s"],
        "reference_rail_orientation_source": reference["reference_rail_orientation_source"],
        "orientation_method": "Recorded MCP row along rail; palm above the supported rail; source arm is an IK seed only",
        "T_S_F": S_F.tolist(), "T_F_S": np.linalg.inv(S_F).tolist(),
        "constructed_R_S_F": S_F[:3, :3].tolist(), "extra_wrist_roll_deg": 0.,
        "approximate_reference_rail_rpy_used_for_transform": False,
        "long_axis_direction_sign": sign,
        "thumb_side_in_rail": thumb_side,
        "open": candidates["open_reference"]["command_position_rad"],
        "measured_reference": measured.tolist(), "grasp": fitted.tolist(),
        "joint_adjustment_rad": (fitted-measured).tolist(),
        "tip_goal_in_rail_m": dict(zip(FINGERS, targets.tolist())), "tip_fit_error_m": errors,
        "validation": "Five-finger geometric adaptation, not a physical grip certification",
    }


def load_reference():
    return json.loads((Path(__file__).parent / "hand_profiles/human_left_wrap.json").read_text())
