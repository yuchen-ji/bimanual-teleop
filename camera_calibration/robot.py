"""Offline FK in the selected arm's native base frame; no robot connection."""

import numpy as np
from scipy.spatial.transform import Rotation

from bimanual_teleop.devices.tianji.model import TianjiKinematics
from .geometry import transform


def load_model(model_path=None, expected_hash=None, sdk_root=None):
    model = TianjiKinematics(sdk_root, model_path)
    if expected_hash is not None and model.model.digest != expected_hash:
        raise ValueError("Robot model changed since capture; use the original --model")
    return model


def base_flange(model, side, joints):
    q = np.asarray(joints, dtype=float)
    if q.shape != (7,) or not np.isfinite(q).all():
        raise ValueError("Expected seven finite measured joint angles in radians")
    pose = model.fk(side, tuple(q))
    return transform(Rotation.from_quat(pose.orientation_xyzw).as_matrix(), pose.position_m)
