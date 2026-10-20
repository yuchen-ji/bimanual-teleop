"""Camera extrinsics and pinhole projection, using OpenCV optical coordinates."""

from ._dependencies import np, Rotation


def camera_world_from_rail(T_camera_cad, rail, cad):
    """Recover the RGB optical frame from a known seated rail and original CAD pose."""
    world_rail = np.eye(4)
    world_rail[:3, :3] = Rotation.from_euler("xyz", rail["initial_rpy_deg"], degrees=True).as_matrix()
    world_rail[:3, 3] = rail["initial_center"]
    rail_cad = np.eye(4)
    rail_cad[:3, :3] = cad["cad_rotation"]
    rail_cad[:3, 3] = -np.asarray(cad["origin_in_rotated_cad_m"])
    camera_cad = np.asarray(T_camera_cad, dtype=float).copy()
    # Project the estimated rotation onto SO(3) before inverting it.
    camera_cad[:3, :3] = Rotation.from_matrix(camera_cad[:3, :3]).as_matrix()
    return world_rail @ rail_cad @ np.linalg.inv(camera_cad)


def camera_world_from_table_plane(T_camera_cad, rail, cad, normal_camera, plane_offset, table_height):
    """Refine a thin rail estimate using a measured horizontal tabletop plane in camera space."""
    normal = np.asarray(normal_camera, dtype=float)
    magnitude = np.linalg.norm(normal)
    normal = normal / magnitude
    plane_offset = plane_offset / magnitude
    rotation = Rotation.from_matrix(np.asarray(T_camera_cad)[:3, :3]).as_matrix()
    long_camera = rotation[:, 1] - normal * np.dot(normal, rotation[:, 1])
    long_camera /= np.linalg.norm(long_camera)
    long_world = Rotation.from_euler("xyz", rail["initial_rpy_deg"], degrees=True).as_matrix()[:, 0]
    up_world = np.array([0., 0., 1.])
    basis_world = np.column_stack((long_world, np.cross(up_world, long_world), up_world))
    basis_camera = np.column_stack((long_camera, np.cross(normal, long_camera), normal))
    world_camera_rotation = (basis_camera @ basis_world.T).T
    original_center = np.asarray(cad["cad_rotation"]).T @ np.asarray(cad["origin_in_rotated_cad_m"])
    center_camera = rotation @ original_center + np.asarray(T_camera_cad)[:3, 3]
    ray = center_camera / center_camera[2]
    height = table_height + plane_offset
    denominator = float(normal @ ray)
    if abs(denominator) < 1e-8:
        raise ValueError("Rail center ray is parallel to the measured table plane")
    center_camera = ray * ((rail["initial_center"][2] - height) / denominator)
    world_camera = np.eye(4)
    world_camera[:3, :3] = world_camera_rotation
    world_camera[:3, 3] = np.asarray(rail["initial_center"]) - world_camera_rotation @ center_camera
    return world_camera


def head_camera_local_pose(settings, parent_world):
    """Convert a world OpenCV camera pose to a parent-relative USD camera pose."""
    world_camera = np.eye(4)
    world_camera[:3, :3] = Rotation.from_quat(settings["quaternion_xyzw"]).as_matrix()
    world_camera[:3, 3] = settings["position_world_m"]
    # OpenCV: +X right, +Y down, +Z forward; USD: +X right, +Y up, -Z forward.
    world_camera[:3, :3] = world_camera[:3, :3] @ np.diag([1., -1., -1.])
    local = np.linalg.inv(parent_world) @ world_camera
    return local[:3, 3], Rotation.from_matrix(local[:3, :3]).as_quat()


def wrist_camera_local_pose(tool_stack, measured_pose=None):
    """Use the supplied camera's front glass plane, keeping its bracket assembly fixed."""
    if measured_pose is not None:
        rotation = Rotation.from_quat(measured_pose["quaternion_xyzw"]).as_matrix() @ np.diag([1., -1., -1.])
        return np.array(measured_pose["position_m"]), Rotation.from_matrix(rotation).as_quat()
    forward = np.asarray(tool_stack["camera_front_normal_in_bracket"], dtype=float)
    forward /= np.linalg.norm(forward)
    position = np.asarray(tool_stack["camera_position_in_bracket_m"]) + forward * tool_stack["camera_optical_clearance_m"]
    # The CAD front glass points 15 degrees towards bracket +Y from +Z.
    # Keep the fingers towards the top of the delivered image.
    right = np.cross(forward, [0., 0., 1.]); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rotation = np.column_stack((right, up, -forward))
    return position, Rotation.from_matrix(rotation).as_quat()


def camera_window_faces(points, triangles, tool_stack):
    """Locate the CAD's closed front-window caps in bracket coordinates, retaining the shell."""
    normal = np.asarray(tool_stack["camera_front_normal_in_bracket"], dtype=float)
    normal /= np.linalg.norm(normal)
    plane = np.dot(normal, tool_stack["camera_position_in_bracket_m"])
    vertices = np.asarray(points)[np.asarray(triangles)]
    normals = np.cross(vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0])
    alignment = np.abs(normals @ normal)/np.maximum(np.linalg.norm(normals, axis=1), 1e-30)
    distance = np.abs(vertices @ normal-plane).max(axis=1)
    return np.flatnonzero((alignment > .999) & (distance < .0002))


def pinhole_attributes(width, height, intrinsics):
    """USD aperture values that reproduce fx/fy/cx/cy, including an off-center principal point."""
    fx, fy, cx, cy = (intrinsics[key] for key in ("fx", "fy", "cx", "cy"))
    focal = 20.
    return {
        "focal_length": focal,
        "horizontal_aperture": focal * width / fx,
        "vertical_aperture": focal * height / fy,
        "horizontal_offset": focal * (width / 2 - cx) / fx,
        "vertical_offset": focal * (cy - height / 2) / fy,
    }


def set_pinhole_intrinsics(camera, width, height, intrinsics, distortion=()):
    from pxr import Gf
    parameters = pinhole_attributes(width, height, intrinsics)
    camera.CreateFocalLengthAttr(parameters["focal_length"])
    camera.CreateHorizontalApertureAttr(parameters["horizontal_aperture"])
    camera.CreateVerticalApertureAttr(parameters["vertical_aperture"])
    camera.CreateHorizontalApertureOffsetAttr(parameters["horizontal_offset"])
    camera.CreateVerticalApertureOffsetAttr(parameters["vertical_offset"])
    # RTX uses the native OpenCV schema to preserve unequal fx/fy and the principal point.
    prim = camera.GetPrim()
    if not prim.ApplyAPI("OmniLensDistortionOpenCvPinholeAPI"):
        raise RuntimeError("Isaac Sim OpenCV pinhole lens schema is unavailable")
    prim.GetAttribute("omni:lensdistortion:model").Set("opencvPinhole")
    prefix = "omni:lensdistortion:opencvPinhole:"
    for key, value in intrinsics.items():
        prim.GetAttribute(prefix + key).Set(float(value))
    prim.GetAttribute(prefix + "imageSize").Set(Gf.Vec2i(width, height))
    names = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4")
    for key, value in zip(names, [*distortion, *([0.] * (12 - len(distortion)))]):
        prim.GetAttribute(prefix + key).Set(float(value))
