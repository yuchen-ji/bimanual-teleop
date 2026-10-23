"""Offline project adapter. No camera or robot connection is created here."""

import numpy as np

from .storage import digest


def load_model(expected_hash=None):
    from bimanual_teleop.sim.config import load_config, sim_path
    from bimanual_teleop.sim.kinematics import RobotModel

    config = load_config()
    urdf = sim_path(config["robot"]["urdf"])
    actual = digest(urdf)
    if expected_hash and expected_hash != actual:
        raise ValueError(
            "Robot URDF changed since wrist calibration; recalibrate the wrist"
        )
    return RobotModel(urdf, config["robot"]["world_yaw_deg"]), actual


def world_flange(model, side, joints):
    q = np.asarray(joints, dtype=float)
    if q.shape != (7,) or not np.isfinite(q).all():
        raise ValueError("Expected seven finite measured joint angles in radians")
    offset = 0 if side == "left" else 7
    if np.any(q < model.lower[offset : offset + 7] - 0.01) or np.any(
        q > model.upper[offset : offset + 7] + 0.01
    ):
        raise ValueError("Measured joints exceed URDF limits")
    positions = np.zeros(len(model.names))
    positions[offset : offset + 7] = q
    return model.flange(side, positions, world=True)
