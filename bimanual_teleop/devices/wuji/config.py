"""Device addresses, SDK users and hand-control configuration."""

from __future__ import annotations

import math
from pathlib import Path

from bimanual_teleop.paths import PROJECT_ROOT
from bimanual_teleop.common.config import load_yaml_config

DEFAULT_CONFIG = PROJECT_ROOT / "configs/wuji_teleop.yaml"


def sdk_user_name(config, *, user_name=None):
    if "sdk_user_id" in config:
        raise ValueError("请将 Wuji 配置中的 sdk_user_id 改为 sdk_user_name，并填写用户名")
    selected = config.get("sdk_user_name", "") if user_name is None else user_name
    if not isinstance(selected, str) or (selected and not selected.strip()):
        raise ValueError("sdk_user_name must be a nonblank string (or empty for default)")
    return selected


def glove_settings(config_path, side, address=None, *, user_name=None):
    config = load_yaml_config(config_path)
    selected = address or config["devices"][side]["glove"]
    user = {"user_name": sdk_user_name(config, user_name=user_name)}
    if not isinstance(selected, str) or not selected:
        raise ValueError(f"missing {side} glove address")
    return selected, user


def add_glove_arguments(parser, *, include_sdk_user=True):
    parser.add_argument("--wuji-config", "--config", dest="config", type=Path, default=DEFAULT_CONFIG,
                        help="Wuji 配置；默认 configs/wuji_teleop.yaml")
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--address", help="override the selected glove address")
    if include_sdk_user:
        parser.add_argument("--user-name", help="按已有 SDK 用户名选择用户；默认从配置文件读取")


def load_config(path):
    config = load_yaml_config(path)
    validate_control_config(config)
    config.setdefault("profile_id", "wuji-hand2")
    return config


def validate_control_config(config):
    if not isinstance(config, dict):
        raise ValueError("Wuji configuration must be an object")
    if "profile_id" in config and not config["profile_id"]:
        raise ValueError("Wuji profile_id must be nonempty when supplied")
    if config.get("mode", "mit") != "mit":
        raise ValueError("Hand2 control mode must be mit")
    sdk_user_name(config)
    for key, default in (("control_hz", 120), ("transition_s", .75),
                         ("glove_timeout_s", .25), ("hand_timeout_s", .5)):
        value = config.get(key, default)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    feedback_hz = config.get("feedback_hz", 200)
    if (isinstance(feedback_hz, bool) or not isinstance(feedback_hz, int)
            or not 1 <= feedback_hz <= 1000):
        raise ValueError("feedback_hz must be an integer in [1, 1000]")
    parameters = config.get("parameters", {})
    for key in ("kp", "kd", "current_limit_a"):
        value = parameters.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    devices = config.get("devices", {})
    for side in ("left", "right"):
        for kind in ("glove", "hand"):
            address = devices.get(side, {}).get(kind)
            if not isinstance(address, str) or not address:
                raise ValueError(f"devices.{side}.{kind} requires an address")
