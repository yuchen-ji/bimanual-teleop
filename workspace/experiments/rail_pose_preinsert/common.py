"""Pure experiment geometry and trajectory sampling; no device imports."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def pose(xyz=(0, 0, 0), rpy_deg=(0, 0, 0)):
    T = np.eye(4)
    T[:3, 3] = xyz
    T[:3, :3] = Rotation.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    return T


def pose_error(a, b):
    return [float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
            float(Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).magnitude())]


def finger_forces(state):
    forces = state.get("rail_contacts", {})
    return {finger: float(sum(value for name, value in forces.items() if name.startswith("l_"+finger+"_")))
            for finger in ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")}


def opposed_grasp(state, minimum_n=.5, minimum_support_fingers=4):
    vectors = state.get("rail_contact_vectors", {})
    forces = state.get("rail_contacts", {})
    thumb = [name for name in forces if name.startswith("l_thumb_")]
    fingers = [name for name in forces if name.startswith(("l_index_finger_", "l_middle_finger_", "l_ring_finger_", "l_pinky_"))]
    a = sum((np.asarray(vectors.get(name, [0, 0, 0])) for name in thumb), np.zeros(3))
    b = sum((np.asarray(vectors.get(name, [0, 0, 0])) for name in fingers), np.zeros(3))
    scale = np.linalg.norm(a)*np.linalg.norm(b)
    supporting = sum(value >= minimum_n for finger, value in finger_forces(state).items() if finger != "thumb")
    return bool(supporting >= minimum_support_fingers and sum(forces[n] for n in thumb) >= minimum_n
                and sum(forces[n] for n in fingers) >= minimum_n
                and scale > 1e-10 and np.dot(a, b) < -.05*scale)


def perception_errors(frames, records, S_O, centre):
    if len(frames) != len(records):
        raise ValueError("Perception/ground-truth frame count differs")
    result = []
    for frame, record in zip(frames, records):
        if frame["state"]["time_ns"] != record["timestamp_ns"]:
            raise ValueError("Perception and ground truth refer to different simulation times")
        truth = np.linalg.inv(frame["T_W_C"]) @ np.asarray(frame["T_W_S"]) @ S_O
        estimated = np.asarray(record["T_C_CAD"])
        origin_error, angle = pose_error(estimated, truth)
        result.append({"cad_origin_error_m": origin_error, "rotation_error_rad": angle,
                       "centre_error_m": float(np.linalg.norm((estimated @ centre - truth @ centre)[:3]))})
    return result


def cad_to_sim(manifest):
    T = np.eye(4)
    T[:3, :3] = manifest["rail"]["cad_rotation"]
    T[:3, 3] = -np.asarray(manifest["rail"]["origin_in_rotated_cad_m"])
    return T


def seated_rail(config, manifest):
    g = manifest["fixture_geometry"]
    T = pose(np.asarray(config["scene"]["fixture_origin"]) + g["seated_center_in_fixture_m"],
             g["seated_rpy_deg"])
    T[2, 3] -= .0004  # Remove the visual/reset pose's nominal support gap.
    return T


def sample_path(plan, elapsed):
    points = np.asarray(plan["joint_waypoints_rad"], dtype=float)
    durations = np.asarray(plan["segment_durations_s"], dtype=float)
    if elapsed <= 0:
        return points[0].copy()
    if elapsed >= durations.sum():
        return points[-1].copy()
    times = np.r_[0., np.cumsum(durations)]
    i = int(np.searchsorted(times, elapsed, side="right")) - 1
    u = (elapsed - times[i]) / durations[i]
    return points[i] + (10*u**3 - 15*u**4 + 6*u**5) * (points[i+1] - points[i])
