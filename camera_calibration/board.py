"""Twenty-circle board detection and planar pose estimation with native lens checks."""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import transform


@dataclass(frozen=True)
class CircleBoard:
    columns: int = 4
    rows: int = 5
    same_row_spacing_m: float = 0.100

    def points(self):
        if (
            self.columns != 4
            or self.rows != 5
            or not np.isfinite(self.same_row_spacing_m)
            or self.same_row_spacing_m <= 0
        ):
            raise ValueError(
                "This calibration uses the confirmed 4 x 5 asymmetric board"
            )
        step = self.same_row_spacing_m / 2
        return np.array(
            [
                [(2 * j + i % 2) * step, i * step, 0.0]
                for i in range(self.rows)
                for j in range(self.columns)
            ],
            dtype=np.float64,
        )


def detect_board(rgb, board=None):
    board = board or CircleBoard()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    parameters = cv2.SimpleBlobDetector_Params()
    parameters.filterByColor = True
    parameters.blobColor = 0
    parameters.minThreshold = 10
    parameters.maxThreshold = 180
    parameters.thresholdStep = 10
    parameters.minDistBetweenBlobs = max(10, max(gray.shape) * 0.013)
    parameters.filterByArea = True
    parameters.minArea = max(12, gray.size * 0.00015)
    parameters.maxArea = gray.size * 0.12
    parameters.filterByCircularity = True
    parameters.minCircularity = 0.5
    parameters.filterByConvexity = True
    parameters.minConvexity = 0.75
    parameters.filterByInertia = True
    parameters.minInertiaRatio = 0.2
    detector = cv2.SimpleBlobDetector_create(parameters)
    for extra in (cv2.CALIB_CB_CLUSTERING, 0):
        found, centers = cv2.findCirclesGrid(
            255 - gray,
            (board.columns, board.rows),
            flags=cv2.CALIB_CB_ASYMMETRIC_GRID | extra,
            blobDetector=detector,
        )
        if found:
            return _canonical_centers(centers, board)
    # At oblique angles the real circles are thin ellipses. Extract conic centers
    # explicitly so background glare cannot dominate the relaxed blob detector.
    for threshold in (150, 110, 80, 180, 200, 220):
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if (
                len(contour) < 6
                or not max(12, gray.size * 0.0001) < area < gray.size * 0.01
            ):
                continue
            (x, y), (a, b), angle = cv2.fitEllipse(contour)
            if min(a, b) < 4 or min(a, b) / max(a, b) < 0.1:
                continue
            if not 0.85 < area / (np.pi * a * b / 4) < 1.1:
                continue
            theta = np.deg2rad(angle)
            rotation = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )
            normalized = ((contour[:, 0] - [x, y]) @ rotation) / [a / 2, b / 2]
            deviation = np.abs(np.linalg.norm(normalized, axis=1) - 1)
            if np.quantile(deviation, 0.95) > max(0.15, 1.4 / min(a, b)):
                continue
            candidates.append([x, y])
        if len(candidates) < board.columns * board.rows:
            continue
        # This overload needs explicit parameters to interpret the input as
        # Point2f candidates rather than an 8-bit detector image in OpenCV 4.11.
        found, centers = cv2.findCirclesGrid(
            np.float32(candidates).reshape(-1, 1, 2),
            (board.columns, board.rows),
            flags=cv2.CALIB_CB_ASYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING,
            blobDetector=None,
            parameters=cv2.CirclesGridFinderParameters(),
        )
        if found:
            return _canonical_centers(centers, board)
    return None


def _canonical_centers(centers, board):
    grid = centers.reshape(board.rows, board.columns, 2)
    # An odd-row asymmetric board has a row-reflection ambiguity. Fix image
    # winding; a front-facing board preserves it under camera roll.
    along = grid[0, 1] - grid[0, 0]
    down = grid[1, 0] - grid[0, 0]
    if along[0] * down[1] - along[1] * down[0] < 0:
        grid = grid[::-1]
    return grid.reshape(-1, 2).astype(np.float64)


def camera_matrix(metadata):
    intrinsics = metadata["rgb_intrinsics"]
    if (
        not np.isfinite([intrinsics[k] for k in ("fx", "fy", "ppx", "ppy")]).all()
        or min(intrinsics["fx"], intrinsics["fy"]) <= 0
        or metadata["resolution"] != [intrinsics["width"], intrinsics["height"]]
        or len(intrinsics["coeffs"]) not in (4, 5, 8, 12, 14)
        or not np.isfinite(intrinsics["coeffs"]).all()
    ):
        raise ValueError(
            "Camera intrinsics must be finite and match the raw image resolution"
        )
    return np.array(
        [
            [intrinsics["fx"], 0, intrinsics["ppx"]],
            [0, intrinsics["fy"], intrinsics["ppy"]],
            [0, 0, 1.0],
        ],
        dtype=float,
    )


def sensor_rays(pixels, metadata):
    """Use librealsense's pure deprojection math; never create a camera context here."""
    intrinsics = metadata["rgb_intrinsics"]
    model = intrinsics["model"].split(".")[-1]
    if model == "opencv_brown":
        normalized = cv2.undistortPoints(
            np.asarray(pixels).reshape(-1, 1, 2),
            camera_matrix(metadata),
            np.asarray(intrinsics["coeffs"]),
        )[:, 0]
        return np.c_[normalized, np.ones(len(normalized))]
    import pyrealsense2 as rs

    camera_matrix(metadata)
    if model not in (
        "none",
        "brown_conrady",
        "inverse_brown_conrady",
        "modified_brown_conrady",
    ):
        raise ValueError(f"Unsupported RealSense lens model: {model}")
    value = rs.intrinsics()
    value.width, value.height = intrinsics["width"], intrinsics["height"]
    value.fx, value.fy = intrinsics["fx"], intrinsics["fy"]
    value.ppx, value.ppy = intrinsics["ppx"], intrinsics["ppy"]
    # SDK inverse-Brown iteration inverts the modified-Brown forward projection;
    # direct SDK deprojection of the modified model asserts. The OpenCV mapping
    # is still required to pass the <0.1 px check below before calibration.
    value.model = getattr(
        rs.distortion,
        "inverse_brown_conrady" if model == "modified_brown_conrady" else model,
    )
    value.coeffs = intrinsics["coeffs"]
    return np.asarray(
        [rs.rs2_deproject_pixel_to_point(value, p.tolist(), 1.0) for p in pixels]
    )


def opencv_distortion(metadata):
    """Check the native-to-OpenCV mapping before using the lens in PnP or Isaac."""
    width, height = metadata["resolution"]
    pixels = np.array(
        [
            (x, y)
            for y in np.linspace(2, height - 3, 9)
            for x in np.linspace(2, width - 3, 11)
        ]
    )
    rays = sensor_rays(pixels, metadata)
    coeff = np.asarray(metadata["rgb_intrinsics"]["coeffs"], dtype=float)
    predicted = cv2.projectPoints(
        rays, np.zeros(3), np.zeros(3), camera_matrix(metadata), coeff
    )[0][:, 0]
    error = float(np.linalg.norm(predicted - pixels, axis=1).max())
    if error > 0.1:
        raise ValueError(
            f"RealSense/OpenCV lens conversion differs by {error:.3f} px; do not apply it to Isaac"
        )
    return coeff, error


def projected_points(points, camera_board, K, distortion):
    rotated = points @ camera_board[:3, :3].T + camera_board[:3, 3]
    if np.any(rotated[:, 2] <= 0):
        raise ValueError("Calibration puts board points behind the camera")
    return cv2.projectPoints(
        points,
        Rotation.from_matrix(camera_board[:3, :3]).as_rotvec(),
        camera_board[:3, 3],
        K,
        np.asarray(distortion, dtype=float),
    )[0][:, 0]


def estimate_board_pose(pixels, metadata, board=None):
    board = board or CircleBoard()
    rays = sensor_rays(pixels, metadata)
    normalized = np.ascontiguousarray(rays[:, :2]).reshape(-1, 1, 2)
    points = board.points()
    result = cv2.solvePnPGeneric(
        points, normalized, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE
    )
    if not result[0]:
        raise ValueError("Planar PnP failed")
    distortion, lens_error = opencv_distortion(metadata)
    candidates = []
    for rotation, translation in zip(result[1], result[2]):
        rotation, translation = cv2.solvePnPRefineLM(
            points, normalized, np.eye(3), None, rotation.copy(), translation.copy()
        )
        pose = transform(cv2.Rodrigues(rotation)[0], translation)
        try:
            projection = projected_points(
                points, pose, camera_matrix(metadata), distortion
            )
        except ValueError:
            continue
        rms = float(np.sqrt(np.mean(np.sum((projection - pixels) ** 2, axis=1))))
        candidates.append((rms, pose))
    if not candidates:
        raise ValueError("Board pose is behind the camera")
    candidates.sort(key=lambda item: item[0])
    rms, pose = candidates[0]
    if rms > 1.0:
        raise ValueError(f"Board reprojection RMS is {rms:.2f} px (limit 1 px)")
    if len(candidates) > 1 and candidates[1][0] < max(0.25, rms * 1.5):
        difference = Rotation.from_matrix(
            pose[:3, :3].T @ candidates[1][1][:3, :3]
        ).magnitude()
        if difference > np.deg2rad(5):
            raise ValueError("Planar pose is ambiguous; add some board tilt")
    return pose, {"pnp_rms_px": rms, "lens_conversion_max_error_px": lens_error}
