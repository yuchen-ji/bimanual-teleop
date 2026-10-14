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
    for key in ("input_timeout_ms", "command_timeout_ms", "engine_feedback_timeout_ms",
                "max_clock_uncertainty_ms", "max_linear_speed_m_s",
                "max_angular_speed_rad_s", "joint_speed_rad_s", "hand_speed_rad_s"):
        if not math.isfinite(runtime[key]) or runtime[key] <= 0:
            raise ValueError(f"Invalid runtime setting: {key}")
    rail = config["rail"]
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
