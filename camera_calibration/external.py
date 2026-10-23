"""Fixed-wrist-reference calibration using two-camera bundle adjustment.

The held-out board pose is recovered ONLY from wrist pixels. External validation
pixels never initialize or refine the camera or the validation board pose.
"""

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from .board import (
    CircleBoard,
    camera_matrix,
    estimate_board_pose,
    opencv_distortion,
    projected_points,
)
from .geometry import average_pose, checked_pose, pack_pose, pose_errors, unpack_pose


@dataclass(frozen=True)
class Limits:
    minimum_samples: int = 15
    validation_median_px: float = 1.5
    validation_max_px: float = 3.0
    validation_median_m: float = 0.005
    validation_max_m: float = 0.010
    validation_max_deg: float = 1.0
    max_rejected_fraction: float = 0.2
    training_outlier_px: float = 2.0

    def __post_init__(self):
        if type(self.minimum_samples) is not int or self.minimum_samples < 15:
            raise ValueError("At least 15 independent observations are required")
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Calibration limits must be finite and positive")
        if self.max_rejected_fraction > 0.2:
            raise ValueError("Reject at most 20% of training observations")


@dataclass
class View:
    world_wrist: np.ndarray
    wrist_pixels: np.ndarray
    external_pixels: np.ndarray
    group: int


def validation_group(group):
    """Assign the split before any detection or filtering, including failed views."""
    return group % 5 == 0


def independent_view(wrist_board, external_board, previous):
    for old_wrist, old_external in previous:
        w = pose_errors(wrist_board, old_wrist)
        c = pose_errors(external_board, old_external)
        if max(w[0], c[0]) < 0.02 and max(w[1], c[1]) < 5:
            return False
    return True


def view_coverage(poses):
    if len(poses) < 3:
        raise ValueError(
            "At least three different board views are needed to evaluate coverage"
        )
    vectors = np.array(
        [
            Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).as_rotvec()
            for i, a in enumerate(poses)
            for b in poses[i + 1 :]
        ]
    )
    singular = np.linalg.svd(vectors, compute_uv=False)
    angle = float(np.rad2deg(np.linalg.norm(vectors, axis=1).max()))
    if angle < 15 or singular[1] < max(singular[0] * 0.1, 1e-5):
        raise ValueError(
            "Insufficient board-view coverage: vary tilt about two axes by 15-40 degrees"
        )
    return {
        "max_relative_rotation_deg": angle,
        "rotation_singular_values": singular.tolist(),
    }


def _fit(views, prepared, indices, initial, lenses, objects):
    # Each nuisance board pose couples the two observations, with the wrist FK
    # and hand-eye transform held fixed. A shared camera is the only other pose.
    initial_boards = [
        average_pose([views[i].world_wrist @ prepared[i][0], initial @ prepared[i][1]])
        for i in indices
    ]
    values = np.concatenate(
        [pack_pose(initial), *[pack_pose(p) for p in initial_boards]]
    )
    wrist_from_world = {i: np.linalg.inv(views[i].world_wrist) for i in indices}

    def residual(parameters):
        camera_from_world = np.linalg.inv(unpack_pose(parameters[:6]))
        rows = []
        for j, i in enumerate(indices):
            board = unpack_pose(parameters[6 + j * 6 : 12 + j * 6])
            for pose, pixels, lens in (
                (wrist_from_world[i] @ board, views[i].wrist_pixels, lenses[0]),
                (camera_from_world @ board, views[i].external_pixels, lenses[1]),
            ):
                # Keep residuals defined during optimizer trial steps; final poses
                # are separately checked for positive depth.
                predicted = cv2.projectPoints(
                    objects,
                    Rotation.from_matrix(pose[:3, :3]).as_rotvec(),
                    pose[:3, 3],
                    *lens,
                )[0][:, 0]
                rows.extend((predicted - pixels).ravel())
        return np.asarray(rows)

    rows_per_view = len(objects) * 4
    sparsity = lil_matrix((rows_per_view * len(indices), len(values)), dtype=int)
    for j in range(len(indices)):
        rows = slice(j * rows_per_view, (j + 1) * rows_per_view)
        sparsity[rows, :6] = 1
        sparsity[rows, 6 + j * 6 : 12 + j * 6] = 1
    fit = least_squares(
        residual,
        values,
        jac_sparsity=sparsity.tocsr(),
        x_scale="jac",
        loss="soft_l1",
        f_scale=0.5,
        max_nfev=300,
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-8,
    )
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError("Two-camera refinement did not converge")
    camera = unpack_pose(fit.x[:6])
    boards = {
        i: unpack_pose(fit.x[6 + j * 6 : 12 + j * 6]) for j, i in enumerate(indices)
    }
    errors = {}
    for i, board in boards.items():
        pixels = [
            projected_points(objects, wrist_from_world[i] @ board, *lenses[0])
            - views[i].wrist_pixels,
            projected_points(objects, np.linalg.inv(camera) @ board, *lenses[1])
            - views[i].external_pixels,
        ]
        errors[i] = float(np.sqrt(np.mean(np.sum(np.concatenate(pixels) ** 2, axis=1))))
    return camera, errors, fit


def _uncertainty(fit):
    """Local conditional covariance, excluding wrist, FK and intrinsic uncertainty."""
    jacobian = fit.jac.toarray() if hasattr(fit.jac, "toarray") else fit.jac
    normal = jacobian.T @ jacobian
    information = (
        normal[:6, :6]
        - normal[:6, 6:] @ np.linalg.pinv(normal[6:, 6:]) @ normal[6:, :6]
    )
    information = (information + information.T) / 2
    scale = np.diag([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])
    eigenvalues = np.linalg.eigvalsh(scale @ information @ scale)
    if eigenvalues[0] <= eigenvalues[-1] * 1e-10:
        raise ValueError(
            "Camera pose is poorly constrained; add more varied board views"
        )
    variance = float(np.dot(fit.fun, fit.fun) / max(1, len(fit.fun) - len(fit.x)))
    std = np.sqrt(np.maximum(np.diag(np.linalg.inv(information)) * variance, 0))
    return {
        "rotation_parameter_std_deg": np.rad2deg(std[:3]).tolist(),
        "translation_std_m": std[3:].tolist(),
        "information_condition": float(eigenvalues[-1] / eigenvalues[0]),
        "scope": "local linearization, fixed wrist/FK/intrinsics; excludes systematic and absolute error",
    }


def solve_external(views, wrist_camera, external_camera, board=None, limits=None):
    board, limits = board or CircleBoard(), limits or Limits()
    if (
        len(views) < limits.minimum_samples
        or len({v.group for v in views}) != len(views)
        or any(type(v.group) is not int or v.group < 0 for v in views)
    ):
        raise ValueError(
            "At least 15 separately numbered independent groups are required"
        )
    validation = [i for i, v in enumerate(views) if validation_group(v.group)]
    training = [i for i, v in enumerate(views) if not validation_group(v.group)]
    if len(validation) < 3 or len(training) < 10:
        raise ValueError("Need at least three held-out groups and ten training groups")
    prepared, rejected = {}, {}
    for i, view in enumerate(views):
        try:
            checked_pose(view.world_wrist)
            for pixels, meta in (
                (view.wrist_pixels, wrist_camera),
                (view.external_pixels, external_camera),
            ):
                if np.asarray(pixels).shape != (20, 2) or not np.isfinite(pixels).all():
                    raise ValueError("Expected all 20 finite circle centers")
                if np.any(pixels < 3) or np.any(
                    pixels > np.asarray(meta["resolution"]) - 4
                ):
                    raise ValueError("Leave an image margin around the complete board")
            wrist_board, _ = estimate_board_pose(view.wrist_pixels, wrist_camera, board)
            external_board, _ = estimate_board_pose(
                view.external_pixels, external_camera, board
            )
            prepared[i] = (wrist_board, external_board)
        except (ValueError, cv2.error) as error:
            if i in validation:
                raise ValueError(
                    f"Held-out group {view.group} failed: {error}"
                ) from error
            rejected[i] = str(error)
    train = [i for i in training if i in prepared]
    previous = []
    # Validation does not influence training coverage, duplicate handling or initialization.
    for i in train.copy():
        if not independent_view(*prepared[i], previous):
            rejected[i] = "duplicate board view"
            train.remove(i)
        else:
            previous.append(prepared[i])

    def check_training():
        if (
            len(rejected) > len(training) * limits.max_rejected_fraction
            or len(train) < 10
        ):
            raise ValueError(
                "More than 20% invalid/inconsistent training groups; repeat capture"
            )
        if len(train) + len(validation) < limits.minimum_samples:
            raise ValueError(
                "At least 15 valid independent groups must remain after rejection"
            )
        return view_coverage([prepared[i][0] for i in train])

    coverage = check_training()
    # Duplicate validation frames cannot count as independent evidence.
    for i in validation:
        others = [prepared[j] for j in prepared if j != i]
        if not independent_view(*prepared[i], others):
            raise ValueError(f"Held-out group {views[i].group} repeats another view")
    candidates = {
        i: views[i].world_wrist @ prepared[i][0] @ np.linalg.inv(prepared[i][1])
        for i in prepared
    }
    seed = min(
        train,
        key=lambda i: np.median(
            [
                sum(np.array(pose_errors(candidates[i], candidates[j])) * [1000.0, 1.0])
                for j in train
            ]
        ),
    )
    initial = candidates[seed]
    chain_errors = np.array([pose_errors(candidates[i], initial) for i in train])
    center = np.median(chain_errors, axis=0)
    spread = np.median(np.abs(chain_errors - center), axis=0)
    chain_limits = np.maximum([0.025, 2.0], center + 4 * 1.4826 * spread)
    for i, errors in zip(train.copy(), chain_errors):
        if np.any(errors > chain_limits):
            rejected[i] = "inconsistent training coordinate chain"
            train.remove(i)
    coverage = check_training()
    objects = board.points()
    lenses = [
        (camera_matrix(meta), opencv_distortion(meta)[0])
        for meta in (wrist_camera, external_camera)
    ]
    camera, fit_errors, fit = _fit(views, prepared, train, initial, lenses, objects)
    median = float(np.median(list(fit_errors.values())))
    mad = float(np.median(np.abs(np.asarray(list(fit_errors.values())) - median)))
    threshold = max(limits.training_outlier_px, median + 4 * 1.4826 * mad)
    bad = [i for i in train if fit_errors[i] > threshold]
    if bad:
        rejected.update(
            {i: f"training reprojection RMS {fit_errors[i]:.3f} px" for i in bad}
        )
        train = [i for i in train if i not in rejected]
        coverage = check_training()
        camera, fit_errors, fit = _fit(views, prepared, train, camera, lenses, objects)
    uncertainty = _uncertainty(fit)

    def evaluate(pose):
        metrics = []
        for i, view in enumerate(views):
            if i not in prepared:
                continue
            predicted = np.linalg.inv(pose) @ view.world_wrist @ prepared[i][0]
            try:
                pixels = projected_points(objects, predicted, *lenses[1])
            except ValueError as error:
                if i not in rejected:
                    raise
                metrics.append(
                    {
                        "group": view.group,
                        "validation": False,
                        "rejected": True,
                        "error": str(error),
                    }
                )
                continue
            rms = float(
                np.sqrt(np.mean(np.sum((pixels - view.external_pixels) ** 2, axis=1)))
            )
            distance, angle = pose_errors(candidates[i], pose)
            metrics.append(
                {
                    "group": view.group,
                    "validation": i in validation,
                    "rejected": i in rejected,
                    "external_rms_px": rms,
                    "translation_error_m": distance,
                    "rotation_error_deg": angle,
                }
            )
        return metrics

    before, after = evaluate(initial), evaluate(camera)
    held = [m for m in after if m["validation"]]
    summary = {
        "median_reprojection_px": float(
            np.median([m["external_rms_px"] for m in held])
        ),
        "max_reprojection_px": max(m["external_rms_px"] for m in held),
        "median_translation_m": float(
            np.median([m["translation_error_m"] for m in held])
        ),
        "max_translation_m": max(m["translation_error_m"] for m in held),
        "max_rotation_deg": max(m["rotation_error_deg"] for m in held),
    }
    passed = (
        summary["median_reprojection_px"] <= limits.validation_median_px
        and summary["max_reprojection_px"] <= limits.validation_max_px
        and summary["median_translation_m"] <= limits.validation_median_m
        and summary["max_translation_m"] <= limits.validation_max_m
        and summary["max_rotation_deg"] <= limits.validation_max_deg
    )
    return camera, {
        "passed": bool(passed),
        "limits": asdict(limits),
        "coverage": coverage,
        "train_groups": [views[i].group for i in train],
        "validation_groups": [views[i].group for i in validation],
        "rejected_train_groups": {
            str(views[i].group): reason for i, reason in rejected.items()
        },
        "before": before,
        "after": after,
        "validation": summary,
        "training_joint_rms_px": float(
            np.sqrt(np.mean(np.square(list(fit_errors.values()))))
        ),
        "uncertainty": uncertainty,
    }
