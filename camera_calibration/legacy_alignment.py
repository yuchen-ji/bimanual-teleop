"""Historical rail/table alignment; not a measured robot-camera calibration."""

import numpy as np
from scipy.spatial.transform import Rotation


def camera_world_from_rail(T_camera_cad, rail, cad):
    """Recover the RGB optical frame from a known seated rail and original CAD pose."""
    world_rail = np.eye(4)
    world_rail[:3, :3] = Rotation.from_euler(
        "xyz", rail["initial_rpy_deg"], degrees=True
    ).as_matrix()
    world_rail[:3, 3] = rail["initial_center"]
    rail_cad = np.eye(4)
    rail_cad[:3, :3] = cad["cad_rotation"]
    rail_cad[:3, 3] = -np.asarray(cad["origin_in_rotated_cad_m"])
    camera_cad = np.asarray(T_camera_cad, dtype=float).copy()
    # Project the estimated rotation onto SO(3) before inverting it.
    camera_cad[:3, :3] = Rotation.from_matrix(camera_cad[:3, :3]).as_matrix()
    return world_rail @ rail_cad @ np.linalg.inv(camera_cad)


def camera_world_from_table_plane(
    T_camera_cad, rail, cad, normal_camera, plane_offset, table_height
):
    """Refine a thin rail estimate using a measured horizontal tabletop plane in camera space."""
    normal = np.asarray(normal_camera, dtype=float)
    magnitude = np.linalg.norm(normal)
    normal = normal / magnitude
    plane_offset = plane_offset / magnitude
    rotation = Rotation.from_matrix(np.asarray(T_camera_cad)[:3, :3]).as_matrix()
    long_camera = rotation[:, 1] - normal * np.dot(normal, rotation[:, 1])
    long_camera /= np.linalg.norm(long_camera)
    long_world = Rotation.from_euler(
        "xyz", rail["initial_rpy_deg"], degrees=True
    ).as_matrix()[:, 0]
    up_world = np.array([0.0, 0.0, 1.0])
    basis_world = np.column_stack(
        (long_world, np.cross(up_world, long_world), up_world)
    )
    basis_camera = np.column_stack((long_camera, np.cross(normal, long_camera), normal))
    world_camera_rotation = (basis_camera @ basis_world.T).T
    original_center = np.asarray(cad["cad_rotation"]).T @ np.asarray(
        cad["origin_in_rotated_cad_m"]
    )
    center_camera = rotation @ original_center + np.asarray(T_camera_cad)[:3, 3]
    ray = center_camera / center_camera[2]
    height = table_height + plane_offset
    denominator = float(normal @ ray)
    if abs(denominator) < 1e-8:
        raise ValueError("Rail center ray is parallel to the measured table plane")
    center_camera = ray * ((rail["initial_center"][2] - height) / denominator)
    world_camera = np.eye(4)
    world_camera[:3, :3] = world_camera_rotation
    world_camera[:3, 3] = (
        np.asarray(rail["initial_center"]) - world_camera_rotation @ center_camera
    )
    return world_camera
