"""Simulation-only configuration; paths are relative to the sim directory."""

from __future__ import annotations

import copy
import math
from pathlib import Path

from bimanual_teleop.sim._dependencies import yaml

from . import ROOT


def load_config(path=None):
    default = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    if path is not None:
        supplied = yaml.safe_load(Path(path).read_text())
        if not isinstance(supplied, dict):
            raise ValueError("Simulation configuration must be a mapping")
        _merge(default, supplied)
    calibration_file = default["cameras"]["wrist_calibration_file"]
    if calibration_file is not None and not isinstance(calibration_file, str):
        raise ValueError("Camera wrist_calibration_file must be a path string or null")
    if calibration_file:
        supplied = yaml.safe_load(sim_path(calibration_file).read_text())
        if (not isinstance(supplied, dict) or set(supplied) != {"cameras"}
                or not isinstance(supplied["cameras"], dict) or set(supplied["cameras"]) != {"wrist"}):
            raise ValueError("Wrist calibration file must contain only cameras.wrist settings")
        _merge(default, supplied)
    validate(default)
    return default


def _merge(target, supplied):
    for key, value in supplied.items():
        if key not in target:
            raise ValueError(f"Unknown simulation setting: {key}")
        if isinstance(target[key], dict) and isinstance(value, dict):
            _merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def validate(config):
    if config["schema_version"] != 1:
        raise ValueError("Unsupported simulation configuration version")
    runtime = config["runtime"]
    for key in ("physics_hz", "control_hz", "camera_hz", "preview_hz", "frame_capacity"):
        value = runtime[key]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if runtime["physics_hz"] % runtime["control_hz"] or runtime["physics_hz"] % runtime["camera_hz"]:
        raise ValueError("Physics rate must be divisible by control and camera rates")
    if runtime["camera_hz"] != 30:
        raise ValueError("Dataset compatibility currently requires 30 Hz cameras")
    if config["cameras"]["main_depth"]:
        raise ValueError("Depth recording is not implemented in this RGB collection version")
    cameras = config["cameras"]
    for key in ("width", "height"):
        if type(cameras[key]) is not int or cameras[key] <= 0:
            raise ValueError(f"Camera {key} must be a positive integer")
    if type(cameras["preview_scale"]) is not int or not 1 <= cameras["preview_scale"] <= 4:
        raise ValueError("Camera preview_scale must be an integer from 1 to 4")
    if cameras["wrist_calibration_file"] is not None and not isinstance(cameras["wrist_calibration_file"], str):
        raise ValueError("Camera wrist_calibration_file must be a path string or null")
    for side in ("left", "right"):
        pose = cameras["wrist"][side]["pose_in_bracket"]
        if pose is not None:
            if not isinstance(pose, dict) or set(pose) != {"position_m", "quaternion_xyzw"}:
                raise ValueError(f"Invalid {side} wrist optical pose fields")
            for key, length in (("position_m", 3), ("quaternion_xyzw", 4)):
                if len(pose[key]) != length or not all(math.isfinite(value) for value in pose[key]):
                    raise ValueError(f"Invalid {side} wrist {key}")
            if not math.isclose(sum(value*value for value in pose["quaternion_xyzw"]), 1., abs_tol=1e-5):
                raise ValueError(f"{side} wrist optical quaternion must have unit norm")
    head = cameras["head"]
    if head is not None:
        for key, size in (("position_world_m", 3), ("quaternion_xyzw", 4)):
            if len(head[key]) != size or not all(math.isfinite(v) for v in head[key]):
                raise ValueError(f"Head camera {key} requires {size} finite values")
        if not math.isclose(sum(v*v for v in head["quaternion_xyzw"]), 1., abs_tol=1e-5):
            raise ValueError("Head camera quaternion must have unit length")
    for name, settings in (("Head", head), ("Left wrist", cameras["wrist"]["left"]),
                           ("Right wrist", cameras["wrist"]["right"])):
        if settings is None:
            continue
        intrinsics = settings["intrinsics"]
        if not all(math.isfinite(intrinsics[key]) for key in ("fx", "fy", "cx", "cy")):
            raise ValueError(f"{name} camera intrinsics must be finite")
        if intrinsics["fx"] <= 0 or intrinsics["fy"] <= 0:
            raise ValueError(f"{name} camera focal lengths must be positive")
        if not 0 <= intrinsics["cx"] < cameras["width"] or not 0 <= intrinsics["cy"] < cameras["height"]:
            raise ValueError(f"{name} camera principal point must lie within its image")
        distortion = settings["distortion_coefficients"]
        if len(distortion) not in (0, 5, 8, 12) or not all(math.isfinite(v) for v in distortion):
            raise ValueError(f"{name} camera distortion requires 0, 5, 8 or 12 finite OpenCV coefficients")
    for key in ("input_timeout_ms", "command_timeout_ms", "engine_feedback_timeout_ms",
                "max_clock_uncertainty_ms", "max_linear_speed_m_s",
                "max_angular_speed_rad_s", "joint_speed_rad_s", "hand_speed_rad_s"):
        if not math.isfinite(runtime[key]) or runtime[key] <= 0:
            raise ValueError(f"Invalid runtime setting: {key}")
    rail = config["rail"]
    scene = config["scene"]
    if scene["asset_profile"] != "rail_task_cad_v1":
        raise ValueError("Unsupported scene asset profile")
    if type(scene["bin_stack_count"]) is not int or scene["bin_stack_count"] <= 0:
        raise ValueError("Box stack count must be a positive integer")
    if not math.isfinite(scene["bin_stack_pitch_m"]) or scene["bin_stack_pitch_m"] <= 0:
        raise ValueError("Box stack pitch must be finite and positive")
    appearance = scene["appearance"]
    for key in ("wood_linear_rgb", "frame_linear_rgb", "mat_linear_rgb", "bins_linear_rgb", "floor_linear_rgb"):
        color = appearance[key]
        if len(color) != 3 or not all(math.isfinite(v) and 0 <= v <= 1 for v in color):
            raise ValueError(f"Scene {key} requires three linear RGB values in [0, 1]")
    for key in ("table_mat", "textures"):
        if type(appearance[key]) is not bool:
            raise ValueError(f"Scene {key} must be a boolean")
    for key in ("ambient_intensity", "key_intensity"):
        if not math.isfinite(appearance[key]) or appearance[key] < 0:
            raise ValueError(f"Scene {key} must be finite and nonnegative")
    for name in ("initial_center", "initial_rpy_deg"):
        if len(rail[name]) != 3 or not all(math.isfinite(value) for value in rail[name]):
            raise ValueError(f"Rail {name} requires three finite values")
    length, width, height = rail["size_m"]
    diameter, spacing = rail["hole_diameter_m"], rail["hole_spacing_m"]
    if not all(math.isfinite(v) and v > 0 for v in (length, width, height, diameter, spacing, rail["mass_kg"])):
        raise ValueError("Rail dimensions and mass must be finite and positive")
    if not rail["pin_diameter_m"] < diameter < width or spacing + diameter >= length:
        raise ValueError("Holes must clear pins and fit within the rail")
    if type(rail["hole_segments"]) is not int or rail["hole_segments"] < 16:
        raise ValueError("At least 16 segments per hole are required")
    ports = [config["network"][key] for key in ("input_port", "event_port", "preview_port", "engine_port")]
    if len(set(ports)) != len(ports) or any(type(p) is not int or not 1024 <= p <= 65535 for p in ports):
        raise ValueError("Network ports must be distinct integers in [1024, 65535]")
    if config["network"]["bind"] not in ("127.0.0.1", "localhost") and not config["network"]["token"]:
        raise ValueError("Set network.token before exposing operator ports on a LAN")
    keys = config["operator"]["keys"]
    if set(keys.values()) != {"toggle", "pause", "boundary", "save", "discard", "home", "reset", "quit"}:
        raise ValueError("Each simulation operation needs a key mapping")


def sim_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path
