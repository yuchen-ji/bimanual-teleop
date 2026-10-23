"""Five-finger grasp templates fitted from measured demonstration postures."""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

FINGERS = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")


def make_grasp(model, task, side):
    """Return flange-in-rail transform and open/closed five-finger targets.

    Demonstrated joints supply the seed. A task may specify a geometric wrist
    orientation and preshape; fingertip IK then adapts the rail cross-section.
    """
    offset = 14 if side == "left" else 34
    arm_offset = 0 if side == "left" else 7
    prefix = "l" if side == "left" else "r"
    q = np.zeros(54)
    q[arm_offset:arm_offset+7] = np.deg2rad(task["arm_reference_deg"][side]["left_grasp" if side == "left" else "right_grasp"])
    initial = np.clip(task["hand_reference_rad"][side]["closed"], model.lower[offset:offset+20]+1e-5, model.upper[offset:offset+20]-1e-5)
    q[offset:offset+20] = initial
    flange = model.flange(side, q, world=True)
    rail_rotation = Rotation.from_euler("xyz", task.get("left_reference_rail_rpy_deg", [0, -26, 180]), degrees=True).as_matrix() if side == "left" else Rotation.from_euler("z", 90, degrees=True).as_matrix()
    rotation = rail_rotation.T@flange[:3, :3]
    if side in task.get("grasp_rotation_in_rail_rpy_deg", {}):
        rotation = Rotation.from_euler("xyz", task["grasp_rotation_in_rail_rpy_deg"][side], degrees=True).as_matrix()
    local_tips = np.array([(np.linalg.inv(flange)@model.link_pose(f"{prefix}_{finger}_tip", q))[:3, 3] for finger in FINGERS])
    tips = local_tips@rotation.T
    # Keep the longitudinal finger spread observed in the demonstration.
    anchor = tips.mean(axis=0)
    anchor[1] = .5*(tips[0, 1]+tips[1:, 1].mean())
    pose = np.eye(4); pose[:3, :3] = rotation
    pose[:3, 3] = np.array([task[f"{side}_grasp_axis_m"], 0., 0.])-anchor
    pose[:3, 3] += task.get("grasp_offsets_in_rail_m", {}).get(side, task.get("grasp_offset_in_rail_m", [0., 0., 0.]))
    thumb_sign = 1. if tips[0, 1] > tips[1:, 1].mean() else -1.
    coordinates = tips-anchor
    coordinates[:, 0] += task[f"{side}_grasp_axis_m"]
    # Pose of each fingertip in rail coordinates depends only on that finger.
    profiles, errors = {}, {}
    base = model.flange(side, q, world=True)
    for label in (("closed", "preshape") if side+"_preshape_tip_yz_m" in task else ("closed",)):
        if label == "closed" and side in task.get("closed_hand_profiles_rad", {}):
            values = np.asarray(task["closed_hand_profiles_rad"][side], dtype=float)
            if values.shape != (20,) or not np.isfinite(values).all() or np.any(values < model.lower[offset:offset+20]) or np.any(values > model.upper[offset:offset+20]):
                raise ValueError("Closed profile must contain 20 finite angles within hand joint limits")
            profiles[label] = values.copy()
            continue
        target = np.array(initial)
        for index, finger in enumerate(FINGERS):
            selected = slice(offset+index*4, offset+(index+1)*4)
            goal = coordinates[index].copy()
            # Spread closing contacts along the rail without changing the
            # collision-free preshape used for reaching into the bin.
            if label == "closed":
                goal[0] += task.get("tip_axis_offsets_m", {}).get(side, [0.]*5)[index]
            # The four fingers hook below the base plate, as in the measured
            # demonstration. Side-only pinching cannot carry the 0.8 kg rail
            # within the Hand2 thumb torque limit.
            points = task[side+"_preshape_tip_yz_m"] if label == "preshape" else task.get(f"{side}_wrap_tip_yz_m", task["wrap_tip_yz_m"])
            distance, height = points["thumb" if index == 0 else "fingers"]
            goal[1] = (thumb_sign if index == 0 else -thumb_sign)*distance
            goal[2] = height
            seed = initial[index*4:index*4+4]
            def residual(values):
                q[selected] = values
                point = (pose@np.linalg.inv(base)@model.link_pose(f"{prefix}_{finger}_tip", q))[:3, 3]
                return np.r_[40*(point-goal), .015*(values-seed)]
            best = None
            for candidate_seed in (seed, np.asarray(task["hand_reference_rad"][side]["open"])[index*4:index*4+4]):
                result = least_squares(residual, np.clip(candidate_seed, model.lower[selected]+1e-5, model.upper[selected]-1e-5),
                                       bounds=(model.lower[selected]+1e-5, model.upper[selected]-1e-5), max_nfev=100)
                if best is None or np.linalg.norm(result.fun[:3]) < np.linalg.norm(best.fun[:3]):
                    best = result
            target[index*4:index*4+4] = best.x
            errors[f"{label}_{finger}"] = float(np.linalg.norm(best.fun[:3])/40)
        profiles[label] = target
    # Opening is the measured demonstration gesture. Solving an artificial
    # fingertip clearance can produce a curled, joint-limit hand "opening".
    profiles["open"] = np.clip(task["hand_reference_rad"][side]["open"],
                               model.lower[offset:offset+20], model.upper[offset:offset+20])
    if side in task.get("preshape_hand_rad", {}):
        values = np.asarray(task["preshape_hand_rad"][side], dtype=float)
        if values.shape != (20,) or not np.isfinite(values).all() or np.any(values < model.lower[offset:offset+20]) or np.any(values > model.upper[offset:offset+20]):
            raise ValueError("Preshape must contain 20 finite angles within hand joint limits")
        profiles["preshape"] = values.copy()
    # Release is relative to the adapted grasp, so a demonstration's open
    # thumb posture is not necessarily clear of the inserted rail.
    values = np.asarray(task.get("release_hand_rad", {}).get(side, profiles["open"]), dtype=float)
    if values.shape != (20,) or not np.isfinite(values).all() or np.any(values < model.lower[offset:offset+20]) or np.any(values > model.upper[offset:offset+20]):
        raise ValueError("Release profile must contain 20 finite angles within hand joint limits")
    profiles["release"] = values.copy()
    return {"flange_in_rail": pose, **profiles, "tip_fit_errors_m": errors}


def hand_contacts(state, side):
    prefix = "l_" if side == "left" else "r_"
    return {name: value for name, value in state.get("rail_contacts", {}).items() if name.startswith(prefix)}


def gripping(state, side, threshold):
    prefix = "l_" if side == "left" else "r_"
    vectors = state.get("rail_contact_vectors", {})
    contacts = hand_contacts(state, side)
    thumb = sum(v for k, v in contacts.items() if k.startswith(prefix+"thumb"))
    other = sum(v for k, v in contacts.items() if k.startswith(tuple(prefix+n for n in FINGERS[1:])))
    a = sum((np.asarray(v) for k, v in vectors.items() if k.startswith(prefix+"thumb")), np.zeros(3))
    b = sum((np.asarray(v) for k, v in vectors.items() if k.startswith(tuple(prefix+n for n in FINGERS[1:]))), np.zeros(3))
    if thumb < threshold or other < threshold:
        return False
    pairs = [(a, b)]
    # A power grasp can oppose the fingers with the palm after loading. Both
    # the thumb and other fingers must still have real contact; a palm rest or
    # contact with a camera/mount is not sufficient.
    if contacts.get(prefix+"wrist", 0.) >= threshold:
        palm = np.asarray(vectors.get(prefix+"wrist", np.zeros(3)))
        pairs.append((palm, a+b))
    return bool(any(np.linalg.norm(x)*np.linalg.norm(y) > 1e-8
                    and np.dot(x, y) < -.05*np.linalg.norm(x)*np.linalg.norm(y) for x, y in pairs))


def grasp_axis_position(model, state, side):
    from bimanual_teleop.sim.kinematics import pose_matrix
    prefix = "l" if side == "left" else "r"
    inverse = np.linalg.inv(pose_matrix(state["rail"]))
    points = [(inverse@model.link_pose(f"{prefix}_{finger}_tip", state["q"]))[0, 3] for finger in FINGERS]
    return float(np.mean(points))
