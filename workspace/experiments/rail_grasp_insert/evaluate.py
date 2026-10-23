"""Geometry and physical completion checks for this experiment only."""

import numpy as np
from scipy.spatial.transform import Rotation
from bimanual_teleop.sim.kinematics import pose_matrix
from bimanual_teleop.sim.bin_geometry import bin_geometry


def pose_error(actual, expected):
    return (float(np.linalg.norm(actual[:3, 3]-expected[:3, 3])),
            float((Rotation.from_matrix(actual[:3, :3])*Rotation.from_matrix(expected[:3, :3]).inv()).magnitude()))


def seated_pose(config, geometry, reverse=False):
    holes = np.asarray(geometry["rail_hole_bottoms_m"])
    pins = np.asarray(geometry["pin_centers_m"])+np.asarray(config["scene"]["fixture_origin"])
    if reverse:
        pins = pins[::-1]
    a, b = holes[1, :2]-holes[0, :2], pins[1, :2]-pins[0, :2]
    yaw = np.arctan2(b[1], b[0])-np.arctan2(a[1], a[0])
    target = np.eye(4)
    target[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    target[:3, 3] = pins.mean(axis=0)-target[:3, :3]@holes.mean(axis=0)
    return target


def insertion_metrics(state, config, geometry):
    rail = pose_matrix(state["rail"])
    holes = np.asarray(geometry["rail_hole_bottoms_m"])@rail[:3, :3].T+rail[:3, 3]
    fixture = np.asarray(config["scene"]["fixture_origin"])
    pins = np.asarray(geometry["pin_centers_m"])+fixture
    radial = min(float(np.linalg.norm(holes[:, :2]-order[:, :2], axis=1).max()) for order in (pins, pins[::-1]))
    return {"pin_error_m": radial,
            "axis_error_rad": float(np.arccos(np.clip(rail[2, 2], -1, 1))),
            "support_gap_m": float(abs(holes[:, 2].mean()-fixture[2]-geometry["support_height_m"])),
            "insertion_depth_m": float(fixture[2]+geometry["pin_top_height_m"]-holes[:, 2].max())}


def opposing_contacts(state, threshold):
    contacts = state.get("rail_contacts", {})
    thumb = sum(v for k, v in contacts.items() if k.startswith("l_thumb"))
    fingers = sum(v for k, v in contacts.items() if k.startswith(("l_index", "l_middle")))
    vectors = state.get("rail_contact_vectors", {})
    thumb_vector = sum((np.asarray(v) for k, v in vectors.items() if k.startswith("l_thumb")), np.zeros(3))
    finger_vector = sum((np.asarray(v) for k, v in vectors.items() if k.startswith(("l_index", "l_middle"))), np.zeros(3))
    product = np.linalg.norm(thumb_vector)*np.linalg.norm(finger_vector)
    return bool(thumb >= threshold and fingers >= threshold and product > 1e-10
                and np.dot(thumb_vector, finger_vector) < -.1*product)


def unintended_robot_contacts(state, finger_threshold):
    # l_wrist is the Hand2 palm/base mesh, parent of all five fingers. Permit
    # its auxiliary contact only while the primary opposing fingers still grip.
    palm_allowed = opposing_contacts(state, finger_threshold)
    return {name: force for name, force in state.get("rail_contacts", {}).items()
            if force > .5 and not name.startswith(("l_thumb", "l_index", "l_middle", "l_ring", "l_pinky"))
            and not (name == "l_wrist" and palm_allowed)}


def initial_pose_valid(state, config):
    """Require support at the input bin's robot-facing, X-negative narrow rim."""
    rail = pose_matrix(state["rail"])
    half = config["rail"]["size_m"][0]/2
    ends = np.array([[-half, 0, 0], [half, 0, 0]])@rail[:3, :3].T+rail[:3, 3]
    low, high = ends[np.argmin(ends[:, 2])], ends[np.argmax(ends[:, 2])]
    box = np.asarray(config["scene"]["bin_centers"][0])
    size = np.asarray(config["scene"]["bin_inner_size"])
    crate = bin_geometry(config["scene"])
    floor = box[2]+crate["floor_height_m"]
    near_x = box[0]-size[0]/2
    # The higher end must project out over the near narrow edge. A rail leaning
    # over either Y-facing long edge or the far X edge cannot pass this gate.
    if not (high[0] < near_x < low[0]) or abs(high[1]-low[1]) > .05:
        return False
    fraction = (near_x-low[0])/(high[0]-low[0])
    rim_crossing = low+fraction*(high-low)
    rim_z = box[2]+crate["rim_height_m"]
    half_thickness_z = config["rail"]["size_m"][2]/2*abs(rail[2, 2])
    return bool(np.all(np.abs(low[:2]-box[:2]) < size[:2]/2+.02)
                and abs(low[2]-floor) < .025 and high[2] > rim_z
                and abs(rim_crossing[1]-box[1]) < size[1]/2-.02
                and abs(rim_crossing[2]-half_thickness_z-rim_z) < .025
                and .15 < abs(rail[2, 0]) < .9)


def is_seated(state, config, geometry, task, linear_speed, angular_speed):
    metrics = insertion_metrics(state, config, geometry)
    clearance = config["rail"]["hole_diameter_m"]/2-max(geometry["pin_radii_m"])-.0001
    released = max(state.get("hand_contact_force", [float("inf")])) <= .05
    released = released and max(state.get("rail_contacts", {}).values(), default=0) <= .05
    passed = (metrics["pin_error_m"] <= clearance and metrics["axis_error_rad"] <= np.deg2rad(1)
              and metrics["support_gap_m"] <= task["support_gap_tolerance_m"]
              and metrics["insertion_depth_m"] >= task["insertion_depth_min_m"]
              and linear_speed <= task["stable_linear_speed_m_s"]
              and angular_speed <= task["stable_angular_speed_rad_s"] and released)
    return bool(passed), metrics
