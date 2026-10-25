"""Eye-to-hand solve: T_base_camera @ T_camera_board = T_base_flange @ T_flange_board.

Only training groups initialize/refine the two constant transforms. Every fifth
group is held out before detection or filtering, including invalid observations.
See https://docs.opencv.org/4.12.0/d9/d0c/group__calib3d.html (calibrateHandEye).
"""

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .board import CircleBoard, camera_matrix, estimate_board_pose, opencv_distortion, projected_points
from .geometry import average_pose, checked_pose, pack_pose, pose_errors, transform, unpack_pose


@dataclass(frozen=True)
class Limits:
    minimum_samples: int = 15
    validation_median_px: float = 1.5
    validation_max_px: float = 3.0
    validation_median_m: float = .005
    validation_max_m: float = .010
    validation_max_deg: float = 1.0
    max_rejected_fraction: float = .2
    training_outlier_px: float = 2.0

    def __post_init__(self):
        if type(self.minimum_samples) is not int or self.minimum_samples < 15:
            raise ValueError("At least 15 independent observations are required")
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Calibration limits must be finite and positive")
        if self.max_rejected_fraction > .2:
            raise ValueError("Reject at most 20% of training observations")


@dataclass
class View:
    base_flange: np.ndarray
    external_pixels: np.ndarray
    group: int


def validation_group(group):
    return group % 5 == 0


def independent_view(pose, previous):
    return all(distance >= .01 or angle >= 3
               for distance, angle in (pose_errors(pose, old) for old in previous))


def motion_diversity(poses):
    if len(poses) < 3:
        raise ValueError("At least three flange poses are needed to evaluate coverage")
    vectors = np.array([Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).as_rotvec()
                        for i, a in enumerate(poses) for b in poses[i + 1:]])
    singular = np.linalg.svd(vectors, compute_uv=False)
    angle = float(np.rad2deg(np.linalg.norm(vectors, axis=1).max()))
    if angle < 20 or singular[1] < max(singular[0] * .1, 1e-5):
        raise ValueError("Insufficient nonparallel flange rotations: vary tilt about two axes by 20-40 degrees")
    return {"max_relative_rotation_deg": angle, "rotation_singular_values": singular.tolist()}


def _initial(views, prepared, indices):
    # Invert robot FK for eye-to-hand. inv(G_i) @ X @ C_i = Y is constant.
    grippers = [np.linalg.inv(views[i].base_flange) for i in indices]
    boards = [prepared[i] for i in indices]
    candidates = []
    for method in (cv2.CALIB_HAND_EYE_PARK, cv2.CALIB_HAND_EYE_TSAI, cv2.CALIB_HAND_EYE_HORAUD):
        try:
            rotation, translation = cv2.calibrateHandEye(
                [g[:3, :3] for g in grippers], [g[:3, 3] for g in grippers],
                [c[:3, :3] for c in boards], [c[:3, 3] for c in boards], method=method)
            camera = checked_pose(transform(rotation, translation))
            mounts = [g @ camera @ c for g, c in zip(grippers, boards)]
            mount = average_pose(mounts)
            score = np.median([d / .002 + a for d, a in (pose_errors(p, mount) for p in mounts)])
            candidates.append((score, camera, mount))
        except (ValueError, cv2.error):
            continue
    if not candidates:
        raise ValueError("Eye-to-hand initialization failed; check board ordering and motion coverage")
    return min(candidates, key=lambda item: item[0])[1:]


def _fit(views, indices, initial, objects, lens):
    def residual(values):
        camera_inverse = np.linalg.inv(unpack_pose(values[:6]))
        mount = unpack_pose(values[6:])
        errors = []
        for i in indices:
            pose = camera_inverse @ views[i].base_flange @ mount
            pixels = cv2.projectPoints(objects, Rotation.from_matrix(pose[:3, :3]).as_rotvec(),
                                       pose[:3, 3], *lens)[0][:, 0]
            errors.extend((pixels - views[i].external_pixels).ravel())
        return np.asarray(errors)

    fit = least_squares(residual, np.r_[pack_pose(initial[0]), pack_pose(initial[1])],
                        loss="soft_l1", f_scale=.5, x_scale="jac", max_nfev=400,
                        ftol=1e-10, xtol=1e-10, gtol=1e-8)
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError("Eye-to-hand refinement did not converge")
    return unpack_pose(fit.x[:6]), unpack_pose(fit.x[6:]), fit


def _uncertainty(fit):
    # Normalize translation parameter units for a meaningful rank check.
    scale = np.diag([1., 1., 1., .1, .1, .1] * 2)
    singular = np.linalg.svd(fit.jac @ scale, compute_uv=False)
    if singular[-1] <= singular[0] * 1e-6:
        raise ValueError("Camera/mount poses are poorly constrained; add varied flange rotations")
    variance = float(fit.fun @ fit.fun / max(1, len(fit.fun) - 12))
    covariance = np.linalg.inv(fit.jac.T @ fit.jac) * variance
    std = np.sqrt(np.maximum(np.diag(covariance), 0))
    return {"camera_rotation_parameter_std_deg": np.rad2deg(std[:3]).tolist(),
            "camera_translation_std_m": std[3:6].tolist(),
            "information_condition": float((singular[0] / singular[-1]) ** 2),
            "scope": "local linearization, fixed FK/intrinsics; excludes systematic and absolute error"}


def solve_external(views, external_camera, board=None, limits=None):
    board, limits = board or CircleBoard(), limits or Limits()
    if (len(views) < limits.minimum_samples or len({v.group for v in views}) != len(views)
            or any(type(v.group) is not int or v.group < 0 for v in views)):
        raise ValueError("At least 15 separately numbered independent groups are required")
    validation = [i for i, view in enumerate(views) if validation_group(view.group)]
    training = [i for i, view in enumerate(views) if not validation_group(view.group)]
    if len(validation) < 3 or len(training) < 10:
        raise ValueError("Need at least three held-out groups and ten training groups")
    prepared, pnp_quality, rejected = {}, {}, {}
    for i, view in enumerate(views):
        try:
            checked_pose(view.base_flange)
            pixels = np.asarray(view.external_pixels)
            if pixels.shape != (20, 2) or not np.isfinite(pixels).all():
                raise ValueError("Expected all 20 finite circle centers")
            if np.any(pixels < 5) or np.any(pixels > np.asarray(external_camera["resolution"]) - 6):
                raise ValueError("Leave an image margin around the complete board")
            prepared[i], pnp_quality[i] = estimate_board_pose(pixels, external_camera, board)
        except (ValueError, cv2.error) as error:
            if i in validation:
                raise ValueError(f"Held-out group {view.group} failed: {error}") from error
            rejected[i] = str(error)
    train, previous = [], []
    for i in training:
        if i not in prepared:
            continue
        if not independent_view(views[i].base_flange, previous):
            rejected[i] = "duplicate flange pose"
        else:
            train.append(i)
            previous.append(views[i].base_flange)

    def check_training():
        if len(train) < 10 or len(rejected) > len(training) * limits.max_rejected_fraction:
            details = "; ".join(f"group {views[i].group}: {reason}" for i, reason in rejected.items())
            raise ValueError(
                f"Training preparation failed: {len(train)}/{len(training)} usable groups, "
                f"{len(rejected)}/{len(training)} excluded (limit 20%, minimum 10 usable). {details}"
            )
        return motion_diversity([views[i].base_flange for i in train])

    coverage = check_training()
    objects = board.points()
    lens = (camera_matrix(external_camera), opencv_distortion(external_camera)[0])
    camera, mount, fit = _fit(views, train, _initial(views, prepared, train), objects, lens)

    def metrics(i):
        predicted_pose = np.linalg.inv(camera) @ views[i].base_flange @ mount
        pixels = projected_points(objects, predicted_pose, *lens)
        rms = float(np.sqrt(np.mean(np.sum((pixels - views[i].external_pixels) ** 2, axis=1))))
        distance, angle = pose_errors(predicted_pose, prepared[i])
        return rms, distance, angle

    inconsistent = {}
    for i in train:
        try:
            rms, distance, angle = metrics(i)
            if rms <= limits.training_outlier_px and distance <= .015 and angle <= 3:
                continue
            reason = f"inconsistent training observation ({rms:.3f} px, {distance * 1000:.2f} mm, {angle:.3f} deg)"
        except ValueError as error:
            reason = str(error)
        inconsistent[i] = reason
    filtered = [i for i in train if i not in inconsistent]
    training_passed = (len(filtered) >= 10 and
                      len(rejected) + len(inconsistent) <= len(training) * limits.max_rejected_fraction)
    # An excessive rejection count means the joint fit itself is inconsistent.
    # Preserve its candidate and residuals instead of throwing away the evidence.
    # Held-out observations never influence either the fit or this decision.
    if inconsistent and training_passed:
        rejected.update(inconsistent)
        train = filtered
        coverage = check_training()
        camera, mount, fit = _fit(views, train, (camera, mount), objects, lens)
    errors = {str(views[i].group): metrics(i) for i in train + validation}
    held = np.array([errors[str(views[i].group)] for i in validation])
    summary = {"median_px": float(np.median(held[:, 0])), "max_px": float(held[:, 0].max()),
               "median_m": float(np.median(held[:, 1])), "max_m": float(held[:, 1].max()),
               "max_deg": float(held[:, 2].max())}
    failures = []
    if not training_passed:
        failures.append(
            f"Training consistency failed: {len(inconsistent)}/{len(training)} groups exceed "
            f"the fit limits ({limits.training_outlier_px:g} px, 15 mm, 3 deg); "
            f"{len(rejected)} groups already excluded, at most "
            f"{int(len(training) * limits.max_rejected_fraction)} exclusions allowed. "
            "Candidate retained with all initially usable training groups."
        )
    for name, value in summary.items():
        threshold = getattr(limits, "validation_" + name)
        if value > threshold:
            failures.append(f"Independent validation {name}={value:.6g} exceeds {threshold:g}")
    passed = training_passed and not failures
    return camera, mount, {"passed": bool(passed), "coverage": coverage,
                           "train_groups": [views[i].group for i in train],
                           "validation_groups": [views[i].group for i in validation],
                           "rejected_train_groups": {str(views[i].group): reason for i, reason in rejected.items()},
                           "initial_inconsistent_train_groups": {str(views[i].group): reason for i, reason in inconsistent.items()},
                           "training_consistency_passed": bool(training_passed), "failure_reasons": failures,
                           "per_group_pnp": {str(views[i].group): value for i, value in pnp_quality.items()},
                           "per_group_rms_px_m_deg": errors, "validation": summary,
                           "uncertainty": _uncertainty(fit)}
