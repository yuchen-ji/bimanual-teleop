"""Wrist hand-eye fitting with robust refinement and held-out validation."""

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .geometry import average_pose, pose_errors, transform


def motion_diversity(grippers):
    vectors = [
        Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).as_rotvec()
        for i, a in enumerate(grippers)
        for b in grippers[i + 1 :]
    ]
    singular = np.linalg.svd(np.asarray(vectors), compute_uv=False)
    max_angle = float(np.rad2deg(np.linalg.norm(vectors, axis=1).max()))
    if max_angle < 20 or singular[1] < singular[0] * 0.1:
        raise ValueError(
            "Insufficient nonparallel wrist rotations: vary pitch, yaw and roll, not just position"
        )
    return {
        "rotation_singular_values": singular.tolist(),
        "max_relative_rotation_deg": max_angle,
    }


def _fit(grippers, boards):
    candidates = []
    for method in (
        cv2.CALIB_HAND_EYE_PARK,
        cv2.CALIB_HAND_EYE_TSAI,
        cv2.CALIB_HAND_EYE_HORAUD,
    ):
        try:
            rotation, translation = cv2.calibrateHandEye(
                [g[:3, :3] for g in grippers],
                [g[:3, 3] for g in grippers],
                [b[:3, :3] for b in boards],
                [b[:3, 3] for b in boards],
                method=method,
            )
            handeye = transform(rotation, translation)
            if (
                not np.isfinite(handeye).all()
                or abs(np.linalg.det(rotation) - 1) > 0.01
            ):
                continue
            targets = [g @ handeye @ b for g, b in zip(grippers, boards)]
            target = average_pose(targets)
            score = np.median(
                [sum(pose_errors(t, target) * np.array([1000.0, 1.0])) for t in targets]
            )
            candidates.append((score, handeye, target))
        except (cv2.error, ValueError):
            continue
    if not candidates:
        raise ValueError(
            "Hand-eye fitting failed; check board ordering and wrist motion"
        )
    _, initial_x, initial_b = min(candidates, key=lambda item: item[0])

    def pack(pose):
        return np.r_[Rotation.from_matrix(pose[:3, :3]).as_rotvec(), pose[:3, 3]]

    def unpack(values):
        return transform(Rotation.from_rotvec(values[:3]).as_matrix(), values[3:])

    def residual(values):
        x, b = unpack(values[:6]), unpack(values[6:])
        result = []
        for g, c in zip(grippers, boards):
            estimate = g @ x @ c
            result.extend((estimate[:3, 3] - b[:3, 3]) / 0.002)
            result.extend(
                Rotation.from_matrix(b[:3, :3].T @ estimate[:3, :3]).as_rotvec()
                / np.deg2rad(1)
            )
        return result

    solution = least_squares(
        residual, np.r_[pack(initial_x), pack(initial_b)], loss="soft_l1", max_nfev=300
    )
    if not solution.success:
        raise ValueError("Hand-eye refinement did not converge")
    return unpack(solution.x[:6]), unpack(solution.x[6:])


def solve_handeye(grippers, boards, *, minimum_samples=15):
    if len(grippers) != len(boards) or len(grippers) < minimum_samples:
        raise ValueError(
            f"At least {minimum_samples} valid, distinct stationary poses are required"
        )
    grippers, boards = np.asarray(grippers), np.asarray(boards)
    diversity = motion_diversity(grippers)
    held_out = np.arange(len(grippers))[::5]
    train = np.setdiff1d(np.arange(len(grippers)), held_out)
    motion_diversity(grippers[train])
    handeye, target = _fit(grippers[train], boards[train])
    errors = np.array(
        [pose_errors(g @ handeye @ c, target) for g, c in zip(grippers, boards)]
    )
    bad_train = train[(errors[train, 0] > 0.015) | (errors[train, 1] > 3.0)]
    if len(bad_train):
        if len(bad_train) > len(train) * 0.2:
            raise ValueError(
                "More than 20% inconsistent board observations; keep the board fixed and repeat"
            )
        train = np.setdiff1d(train, bad_train)
        motion_diversity(grippers[train])
        handeye, target = _fit(grippers[train], boards[train])
        errors = np.array(
            [pose_errors(g @ handeye @ c, target) for g, c in zip(grippers, boards)]
        )
    passed = (
        np.linalg.norm(handeye[:3, 3]) < 0.3
        and np.max(errors[held_out, 0]) < 0.01
        and np.max(errors[held_out, 1]) < 2.0
        and np.median(errors[held_out, 0]) < 0.005
    )
    report = {
        **diversity,
        "passed": bool(passed),
        "train_indices": train.tolist(),
        "held_out_indices": held_out.tolist(),
        "rejected_train_indices": bad_train.tolist(),
        "translation_errors_m": errors[:, 0].tolist(),
        "rotation_errors_deg": errors[:, 1].tolist(),
        "held_out_max_translation_m": float(errors[held_out, 0].max()),
        "held_out_max_rotation_deg": float(errors[held_out, 1].max()),
    }
    return handeye, target, report
