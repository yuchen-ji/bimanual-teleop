"""Extract synchronized measured grasp references without opening device drivers."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from bimanual_teleop.devices.tianji.model import TianjiKinematics
from bimanual_teleop.recording.schema import STREAM_FIELDS
from bimanual_teleop.sim.config import ROOT
from bimanual_teleop.sim.kinematics import RobotModel, pose_matrix, pose_vector


def read_stream(episode, stream, manifest):
    dtype = np.dtype([("time_ns", "<i8"), ("sequence", "<i8")] +
                     [(name, "<f8", (size,)) for name, size in STREAM_FIELDS[stream]])
    path = episode / "raw_spool/streams" / (stream.replace("/", "__") + ".bin")
    if path.stat().st_size % dtype.itemsize:
        raise ValueError(f"Truncated stream: {stream}")
    data = np.fromfile(path, dtype=dtype)
    if len(data) != manifest["counts"][stream] or not len(data):
        raise ValueError(f"Count mismatch: {stream}")
    if not np.all(np.diff(data["time_ns"]) > 0) or not np.isfinite(data["joint_pos"]).all():
        raise ValueError(f"Invalid samples: {stream}")
    return data


def extract(episode, window):
    episode = episode.resolve()
    raw = (episode / "episode.json").read_bytes()
    manifest = json.loads(raw)
    start = manifest["start_ns"]
    streams = {s: read_stream(episode, s, manifest) for s in
               ("hands/left", "arms/left", "hand_commands/left", "arm_commands/left")}
    hand = streams["hands/left"]
    selected = hand[(hand["time_ns"] >= start + window[0]*1e9) &
                    (hand["time_ns"] <= start + window[1]*1e9)]
    if not len(selected):
        raise ValueError("Reference window has no measured hand samples")
    median = np.median(selected["joint_pos"], axis=0)
    # Keep an actual measured pose, with synchronized arm, not an artificial
    # per-joint average assembled from different postures.
    representative = selected[np.argmin(np.linalg.norm(selected["joint_pos"]-median, axis=1))]
    stamp = int(representative["time_ns"])
    records = {s: a[np.argmin(abs(a["time_ns"]-stamp))] for s, a in streams.items()}
    ages = {s: abs(int(r["time_ns"])-stamp)/1e6 for s, r in records.items()}
    if max(ages.values()) > 25:
        raise ValueError(f"Unsynchronized reference: {ages}")
    kine = TianjiKinematics()
    if kine.model.digest != manifest["metadata"]["model_sha256"]:
        raise ValueError("Recording and offline FK model hashes differ")
    arm = records["arms/left"]["joint_pos"]
    pose = kine.fk("left", tuple(arm))
    base_vector = np.r_[pose.position_m, pose.orientation_xyzw]
    model = RobotModel(ROOT / "assets/generated/robot.urdf")
    q = np.zeros(54)
    q[:7], q[14:34] = arm, representative["joint_pos"]
    fk = model.flange("left", q)
    expected = pose_matrix(base_vector)
    # The SDK base is not the URDF Joint1 origin. Derive the convention
    # transform independently at zero joints, then check it at the sample.
    zero = kine.fk("left", (0.,)*7)
    sdk_zero = pose_matrix(np.r_[zero.position_m, zero.orientation_xyzw])
    sim_base_from_record_base = model.flange("left", np.zeros(54)) @ np.linalg.inv(sdk_zero)
    mapped = sim_base_from_record_base @ expected
    position_error = float(np.linalg.norm(fk[:3, 3]-mapped[:3, 3]))
    rotation_error = float(Rotation.from_matrix(fk[:3, :3].T@mapped[:3, :3]).magnitude())
    if position_error > .0001 or rotation_error > .001:
        raise ValueError(f"Simulation and recorded FK mismatch: {position_error}, {rotation_error}")
    world = model.base_pose("left") @ mapped
    measured = representative["joint_pos"]
    command = records["hand_commands/left"]["joint_pos"]
    result = {
        "schema": "recorded_measured_left_grasp_v1",
        "source_episode": str(episode),
        "source_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "source_model_sha256": kine.model.digest,
        "reference_window_s": list(window),
        "reference_timestamp_ns": stamp,
        "reference_relative_time_s": (stamp-start)/1e9,
        "synchronization_difference_ms": ages,
        "selection": "Measured sample nearest the window median; video shows lifted rail",
        "joint_names": list(model.hand_names["left"]),
        "measured_position_rad": measured.tolist(),
        "command_position_rad": command.tolist(),
        "measured_position_deg": np.rad2deg(measured).reshape(5, 4).tolist(),
        "command_minus_measured_deg": np.rad2deg(command-measured).reshape(5, 4).tolist(),
        "window_joint_peak_to_peak_deg": np.rad2deg(np.ptp(selected["joint_pos"], axis=0)).tolist(),
        "measured_arm_rad": arm.tolist(),
        "measured_arm_deg": np.rad2deg(arm).tolist(),
        "flange_in_left_base_xyz_quaternion_xyzw": base_vector.tolist(),
        "flange_in_left_base_rpy_xyz_deg": Rotation.from_matrix(expected[:3, :3]).as_euler("xyz", degrees=True).tolist(),
        "flange_in_sim_world_xyz_quaternion_xyzw": pose_vector(world).tolist(),
        "T_W_F": world.tolist(),
        "T_sim_base_record_base": sim_base_from_record_base.tolist(),
        "base_mapping_source": "Zero-joint model convention alignment, checked at reference joints; not measured extrinsic calibration",
        "fk_verification": {"translation_error_m": position_error, "rotation_error_rad": rotation_error},
        "outside_sim_hand_limits": [n for n, v, lo, hi in zip(model.hand_names["left"], measured,
                                      model.lower[14:34], model.upper[14:34]) if v < lo or v > hi],
        "rail_in_flange_measured": False,
        "limitation": "Object pose/contact forces are not recorded. This is a posture reference, not a simulation grip certificate.",
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--window", type=float, nargs=2, default=(50., 56.))
    args = parser.parse_args()
    result = extract(args.episode, args.window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
