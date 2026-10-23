"""Rigid transforms: T_A_B maps B coordinates into A, translations in meters."""

import numpy as np
from scipy.spatial.transform import Rotation


def transform(rotation, translation):
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = np.asarray(translation).reshape(3)
    return result


def average_pose(poses):
    return transform(
        Rotation.from_matrix(np.asarray(poses)[:, :3, :3]).mean().as_matrix(),
        np.median(np.asarray(poses)[:, :3, 3], axis=0),
    )


def pose_errors(estimated, reference):
    translation = np.linalg.norm(estimated[:3, 3] - reference[:3, 3])
    angle = np.rad2deg(
        Rotation.from_matrix(reference[:3, :3].T @ estimated[:3, :3]).magnitude()
    )
    return float(translation), float(angle)


def checked_pose(value):
    pose = np.asarray(value, dtype=float)
    if (
        pose.shape != (4, 4)
        or not np.isfinite(pose).all()
        or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8)
        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-6)
    ):
        raise ValueError("Expected a finite, rigid T_A_B transform")
    return pose


def pack_pose(pose):
    return np.r_[Rotation.from_matrix(pose[:3, :3]).as_rotvec(), pose[:3, 3]]


def unpack_pose(values):
    return transform(Rotation.from_rotvec(values[:3]).as_matrix(), values[3:6])
