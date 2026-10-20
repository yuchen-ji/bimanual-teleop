"""Isaac USD scene authoring; imported only after SimulationApp starts."""

from __future__ import annotations

import json
from pathlib import Path

from bimanual_teleop.sim._dependencies import np
from bimanual_teleop.sim._dependencies import Rotation

from . import ROOT


def _mesh(stage, path, filename, collision=False, approximation="none"):
    import trimesh
    from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema
    mesh = trimesh.load(str(filename), force="mesh", process=False)
    prim = UsdGeom.Mesh.Define(stage, path)
    prim.CreatePointsAttr([Gf.Vec3f(*p) for p in mesh.vertices])
    prim.CreateFaceVertexCountsAttr([3]*len(mesh.faces))
    prim.CreateFaceVertexIndicesAttr(mesh.faces.reshape(-1).tolist())
    prim.CreateDoubleSidedAttr(True)
    if collision:
        UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(prim.GetPrim()).CreateApproximationAttr(approximation)
        collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim.GetPrim())
        collision_api.CreateContactOffsetAttr(.0002)
        collision_api.CreateRestOffsetAttr(0.)
        if approximation == "convexHull":
            PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim.GetPrim()).CreateMinThicknessAttr(.00001)
    return prim


def _material(stage, path, color, friction=.5, metallic=0., roughness=.45):
    from pxr import Gf, UsdShade, UsdPhysics, Sdf
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path+"/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(friction); physics.CreateDynamicFrictionAttr(friction)
    physics.CreateRestitutionAttr(0.)
    return material


def _bind(prim, material):
    from pxr import UsdShade
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim() if hasattr(prim, "GetPrim") else prim).Bind(material)


def _bind_wrist_windows(stage, robot_path, tools):
    """Make only the solid CAD front-window caps optically clear; retain collision meshes."""
    from pxr import Sdf, Usd, UsdGeom, UsdShade
    from .camera_calibration import camera_window_faces
    material = _material(stage, "/World/Materials/WristOpticalWindow", (0., 0., 0.), roughness=0.)
    shader = stage.GetPrimAtPath("/World/Materials/WristOpticalWindow/Shader")
    shader.CreateAttribute("inputs:opacity", Sdf.ValueTypeNames.Float).Set(0.)
    shader.CreateAttribute("inputs:ior", Sdf.ValueTypeNames.Float).Set(1.)
    cache = UsdGeom.XformCache()
    for prefix in ("l", "r"):
        bracket = stage.GetPrimAtPath(robot_path+f"/{prefix}_camera_bracket")
        visuals = stage.GetPrimAtPath(str(bracket.GetPath())+"/visuals")
        # URDF visual assets may be instances. Author face bindings on these two
        # camera visual copies, leaving shared prototypes and collision assets alone.
        for _ in range(4):
            instances = [prim.GetPath() for prim in Usd.PrimRange(visuals) if prim.IsInstance()]
            if not instances:
                break
            for path in instances:
                stage.GetPrimAtPath(path).SetInstanceable(False)
        parent = np.array(cache.GetLocalToWorldTransform(bracket)).T
        selected_count = 0
        diagnostics = []
        for prim in Usd.PrimRange(visuals):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            if not np.all(np.asarray(mesh.GetFaceVertexCountsAttr().Get()) == 3):
                raise RuntimeError("Wrist optical-window authoring expects the supplied triangle meshes")
            points = np.asarray(mesh.GetPointsAttr().Get())
            local = np.linalg.inv(parent) @ np.array(cache.GetLocalToWorldTransform(prim)).T
            points = points @ local[:3, :3].T + local[:3, 3]
            faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
            selected = camera_window_faces(points, faces, tools)
            diagnostics.append({"path": str(prim.GetPath()), "bounds": [points.min(axis=0).tolist(), points.max(axis=0).tolist()], "faces": len(selected)})
            if len(selected):
                binding = UsdShade.MaterialBindingAPI.Apply(prim)
                subset = binding.CreateMaterialBindSubset("OpticalWindow", selected.tolist())
                _bind(subset, material); selected_count += len(selected)
        if not selected_count:
            raise RuntimeError(f"No front-window faces found for {prefix} wrist camera: {diagnostics}")
        print(f"{prefix} wrist: {selected_count} front-window faces use the clear optical material", flush=True)


def _bind_pin_tops(mesh, geometry, material):
    from pxr import UsdShade
    points = np.asarray(mesh.GetPointsAttr().Get())
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    above_shoulder = points[:, 2] > geometry["support_height_m"] + .00001
    binding = UsdShade.MaterialBindingAPI(mesh.GetPrim())
    for index, (center, radius) in enumerate(zip(geometry["pin_centers_m"], geometry["pin_radii_m"])):
        # The supplied pins are slightly tilted, so their caps do not have constant Z.
        near_axis = np.linalg.norm(points[:, :2] - np.asarray(center[:2]), axis=1) <= radius + .0001
        candidates = np.flatnonzero(np.all((near_axis & above_shoulder)[faces], axis=1))
        triangles = points[faces[candidates]]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        top_faces = candidates[normals[:, 2] > .99 * np.linalg.norm(normals, axis=1)]
        if not len(top_faces):
            raise ValueError(f"No top faces found for fixture locating pin {index}")
        subset = binding.CreateMaterialBindSubset(f"PinTop_{index}", top_faces.tolist())
        _bind(subset, material)


def _box(stage, path, position, size, material, collision=True):
    from pxr import Gf, UsdGeom, UsdPhysics
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.)
    cube.AddTranslateOp().Set(Gf.Vec3d(*position)); cube.AddScaleOp().Set(Gf.Vec3f(*size))
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    _bind(cube, material)
    return cube


def _bind_tabletop(mesh, material):
    """The supplied CAD has a 65 mm top above the metal frame at Z=0.665 m."""
    from pxr import UsdShade
    points = np.asarray(mesh.GetPointsAttr().Get())
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    top_faces = np.flatnonzero(np.all(points[faces, 2] >= .665 - 1e-5, axis=1))
    subset = UsdShade.MaterialBindingAPI(mesh.GetPrim()).CreateMaterialBindSubset("WoodTop", top_faces.tolist())
    _bind(subset, material)


def look_at(position, target, up=(0, 0, 1)):
    # USD cameras look along -Z, with +Y upwards.
    forward = np.array(target)-np.array(position); forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(forward, (0, 1, 0))
    right /= np.linalg.norm(right)
    vertical = np.cross(right, forward)
    return Rotation.from_matrix(np.stack([right, vertical, -forward], axis=1)).as_quat()


def build_scene(world, config, model, capture=True):
    import omni.kit.commands
    import omni.usd
    from isaacsim.core.utils.extensions import enable_extension
    from isaacsim.core.prims import SingleArticulation
    from pxr import Gf, UsdGeom, UsdLux, UsdPhysics, PhysxSchema
    import omni.replicator.core as rep

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    enable_extension("isaacsim.asset.importer.urdf")
    ok, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not ok:
        raise RuntimeError("URDF importer could not initialize")
    import_config.set_merge_fixed_joints(False)
    import_config.set_fix_base(True)
    import_config.set_self_collision(True)
    import_config.set_import_inertia_tensor(True)
    import_config.set_distance_scale(1.)
    import_config.set_default_drive_strength(config["robot"]["arm_stiffness"])
    import_config.set_default_position_drive_damping(config["robot"]["arm_damping"])
    # Include meshes relative to the generated URDF, independent of host/container paths.
    ok, robot_path = omni.kit.commands.execute("URDFParseAndImportFile", urdf_path=str(ROOT / config["robot"]["urdf"]),
                                               import_config=import_config)
    if not ok:
        raise RuntimeError("Robot URDF import failed")
    quaternion = Rotation.from_matrix(model.world[:3, :3]).as_quat()
    robot = world.scene.add(SingleArticulation(prim_path=robot_path, name="bimanual_robot",
                                              orientation=quaternion[[3, 0, 1, 2]]))
    # Millimetre-clearance pin insertion and finger contacts need offsets
    # substantially smaller than the importer defaults.
    fingers = _material(stage, "/World/Materials/Fingers", (.6, .6, .6), config["rail"]["finger_friction"])
    from pxr import Usd, UsdShade
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path)):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            # Gains and effort limits are torque based. An acceleration drive
            # multiplies these gains by inertia, which makes tiny finger links
            # unable to oppose gravity with the configured Nm/rad gains.
            UsdPhysics.DriveAPI.Apply(prim, "angular").CreateTypeAttr("force")
    for descendant in model.by_child:
        moving = 0
        for joint in reversed(model.chain(descendant)):
            moving += joint.kind == "revolute"
            if moving > 2:
                break
            ancestor = stage.GetPrimAtPath(robot_path+"/"+joint.parent.replace(" ", "_"))
            child = stage.GetPrimAtPath(robot_path+"/"+descendant.replace(" ", "_"))
            if ancestor.IsValid() and child.IsValid():
                # CAD shells around a serial joint and fixed mounting plates
                # intentionally overlap. Keep collisions across fingers, arms,
                # objects and more distant links.
                UsdPhysics.FilteredPairsAPI.Apply(ancestor).CreateFilteredPairsRel().AddTarget(child.GetPath())
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path)):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            link = str(prim.GetPath()).removeprefix(robot_path+"/").split("/")[0]
            hand = link.startswith(("l_", "r_"))
            collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision.CreateContactOffsetAttr(config["rail"]["contact_offset_m"] if hand else .002)
            collision.CreateRestOffsetAttr(config["rail"]["rest_offset_m"])
            if link == "base_link" and prim.IsA(UsdGeom.Mesh):
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexDecomposition")
                decomposition = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
                decomposition.CreateMaxConvexHullsAttr(64)
                decomposition.CreateHullVertexLimitAttr(64)
                decomposition.CreateVoxelResolutionAttr(500000)
            if link.endswith(("_force_sensor", "_camera_bracket")) and prim.IsA(UsdGeom.Mesh):
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexDecomposition")
                decomposition = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
                decomposition.CreateMaxConvexHullsAttr(32)
                decomposition.CreateHullVertexLimitAttr(64)
                decomposition.CreateVoxelResolutionAttr(500000)
            if hand:
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(fingers, materialPurpose="physics")
    appearance = config["scene"]["appearance"]
    # A plain indoor floor avoids the blue grid and outdoor light rig in the default asset.
    floor = world.scene.add_ground_plane(size=20., color=np.asarray(appearance["floor_linear_rgb"]))
    floor_material = _material(stage, "/World/Materials/Floor", appearance["floor_linear_rgb"], roughness=.8)
    _bind(floor.prim, floor_material)
    manifest = json.loads((ROOT / "assets/generated/manifest.json").read_text())
    if any(config["cameras"]["wrist"][side]["pose_in_bracket"] is not None for side in ("left", "right")):
        _bind_wrist_windows(stage, robot_path, manifest["cad"]["tool_stack"])
    geometry = manifest["fixture_geometry"]
    matte = _material(stage, "/World/Materials/Table", appearance["wood_linear_rgb"], roughness=.6)
    frame_material = _material(stage, "/World/Materials/TableFrame", appearance["frame_linear_rgb"], metallic=.3, roughness=.6)
    mat_material = _material(stage, "/World/Materials/TableMat", appearance["mat_linear_rgb"], roughness=.9)
    metal = _material(stage, "/World/Materials/Rail", (.55, .58, .62), config["rail"]["friction"], .75)
    fixture_material = _material(stage, "/World/Materials/Fixture", (.05, .45, .55), config["rail"]["friction"], .5)
    pin_top_material = _material(stage, "/World/Materials/PinTops", (.005, .005, .005), config["rail"]["friction"])
    camera_housing_material = _material(stage, "/World/Materials/CameraHousing", (.2, .23, .27), config["rail"]["friction"], .5)
    bin_material = _material(stage, "/World/Materials/Bins", appearance["bins_linear_rgb"], roughness=.7)
    from .materials import connect_basecolor, planar_uvs
    if appearance["textures"]:
        connect_basecolor(stage, matte, "wood", appearance["wood_linear_rgb"])
        connect_basecolor(stage, bin_material, "plastic", appearance["bins_linear_rgb"])
    cad_scene = config["scene"]["asset_profile"] == "rail_task_cad_v1"
    if cad_scene:
        table = UsdGeom.Xform.Define(stage, "/World/Table")
        table.AddTranslateOp().Set(Gf.Vec3d(*config["scene"]["table_center"]))
        table_mesh = _mesh(stage, "/World/Table/Mesh", ROOT / "assets/generated/task/table.obj", True)
        if appearance["textures"]:
            planar_uvs(table_mesh, (.67, 1.))
        _bind(table_mesh, frame_material)
        _bind_tabletop(table_mesh, matte)
        if appearance["table_mat"]:
            # Appearance only: this thin overlay does not add a collider or change contact heights.
            near = config["scene"]["fixture_origin"][0] + .055
            far = config["scene"]["table_center"][0] + config["scene"]["table_size"][0]/2 - .02
            if far > near:
                center = config["scene"]["table_center"]
                _box(stage, "/World/Table/Mat", [(near+far)/2-center[0], 0, .7306],
                     [far-near, config["scene"]["table_size"][1]*.77, .001], mat_material, collision=False)
    else:
        _box(stage, "/World/Table", config["scene"]["table_center"], config["scene"]["table_size"], matte)
    fixture = UsdGeom.Xform.Define(stage, "/World/Fixture")
    fixture.AddTranslateOp().Set(Gf.Vec3d(*config["scene"]["fixture_origin"]))
    reduced = manifest.get("cad", {}).get("models", {}).get("fixture", {}).get("collision") if cad_scene else None
    fixture_mesh = _mesh(stage, "/World/Fixture/Mesh", ROOT / "assets/generated/task/fixture.obj", collision=not bool(reduced))
    _bind(fixture_mesh, fixture_material)
    _bind_pin_tops(fixture_mesh, geometry, pin_top_material)
    if reduced:
        collider = _mesh(stage, "/World/Fixture/Collision", ROOT / "assets/generated" / reduced["mesh"], collision=True)
        collider.CreateVisibilityAttr("invisible")
    length, width, height = config["scene"]["bin_inner_size"]
    wall = config["scene"]["bin_wall"]
    for index, center in enumerate(config["scene"]["bin_centers"]):
        center = np.asarray(center)
        base = f"/World/Bin_{index}"
        if cad_scene:
            for layer in range(config["scene"]["bin_stack_count"]):
                box = UsdGeom.Xform.Define(stage, base+f"/Layer_{layer+1}")
                position = center.copy()
                position[2] -= (config["scene"]["bin_stack_count"]-1-layer)*config["scene"]["bin_stack_pitch_m"]
                box.AddTranslateOp().Set(Gf.Vec3d(*position))
                bin_mesh = _mesh(stage, base+f"/Layer_{layer+1}/Mesh", ROOT / "assets/generated/task/bins.obj", True)
                if appearance["textures"]:
                    planar_uvs(bin_mesh, (.18, .18))
                _bind(bin_mesh, bin_material)
            continue
        _box(stage, base+"/Floor", center+[0, 0, wall/2], [length+2*wall, width+2*wall, wall], bin_material)
        for sign in (-1, 1):
            _box(stage, base+f"/X_{sign+1}", center+[sign*(length+wall)/2, 0, height/2], [wall, width+2*wall, height], bin_material)
            _box(stage, base+f"/Y_{sign+1}", center+[0, sign*(width+wall)/2, height/2], [length, wall, height], bin_material)
    rail = UsdGeom.Xform.Define(stage, "/World/Rail")
    rail.AddTranslateOp().Set(Gf.Vec3d(*config["rail"]["initial_center"]))
    rail_quaternion = Rotation.from_euler("xyz", config["rail"]["initial_rpy_deg"], degrees=True).as_quat()
    rail.AddOrientOp().Set(Gf.Quatf(float(rail_quaternion[3]), Gf.Vec3f(*rail_quaternion[:3])))
    UsdPhysics.RigidBodyAPI.Apply(rail.GetPrim())
    body = PhysxSchema.PhysxRigidBodyAPI.Apply(rail.GetPrim()); body.CreateEnableCCDAttr(True)
    body.CreateSolverPositionIterationCountAttr(12)
    body.CreateSolverVelocityIterationCountAttr(4)
    mass = UsdPhysics.MassAPI.Apply(rail.GetPrim()); mass.CreateMassAttr(config["rail"]["mass_kg"])
    visual = _mesh(stage, "/World/Rail/Visual", ROOT / "assets/generated/rail/rail.obj")
    _bind(visual, metal)
    for index, name in enumerate(manifest["rail"]["colliders"]):
        collider = _mesh(stage, f"/World/Rail/Collision_{index}", ROOT / "assets/generated/rail" / name, True, "convexHull")
        collider.CreateVisibilityAttr("invisible"); _bind(collider, metal)
    dome = UsdLux.DomeLight.Define(stage, "/World/Lighting/Dome")
    dome.CreateIntensityAttr(appearance["ambient_intensity"])
    dome.CreateColorAttr(Gf.Vec3f(1., .97, .93))
    key = UsdLux.DistantLight.Define(stage, "/World/Lighting/Key")
    key.CreateIntensityAttr(appearance["key_intensity"])
    key.CreateAngleAttr(12.)
    key.AddRotateXYZOp().Set(Gf.Vec3f(-15, 10, 25))
    width, height = config["cameras"]["width"], config["cameras"]["height"]
    zero = np.zeros(54)
    camera_paths, annotators = [], []
    for index in range(3):
        if index == 0:
            parent = robot_path + "/Head_depth_Link"
            parent_pose = model.link_pose("Head depth_Link", zero)
            if config["cameras"]["head"] is not None:
                from .camera_calibration import head_camera_local_pose
                position, orientation = head_camera_local_pose(config["cameras"]["head"], parent_pose)
            else:
                position_world = parent_pose[:3, 3]
                target_world = np.array(config["scene"]["fixture_origin"])+[0, 0, .08]
                position = np.zeros(3)
                world_orientation = Rotation.from_quat(look_at(position_world, target_world)).as_matrix()
                orientation = Rotation.from_matrix(parent_pose[:3, :3].T @ world_orientation).as_quat()
        else:
            prefix = "l" if index == 1 else "r"
            parent_link = f"{prefix}_camera_bracket"
            parent = robot_path + "/" + parent_link
            from .camera_calibration import wrist_camera_local_pose
            side = "left" if index == 1 else "right"
            position, orientation = wrist_camera_local_pose(manifest["cad"]["tool_stack"], config["cameras"]["wrist"][side]["pose_in_bracket"])
        if not stage.GetPrimAtPath(parent).IsValid():
            raise RuntimeError(f"Imported camera mounting link was not found: {parent}")
        path = parent+f"/sim_camera_{index}"
        camera = UsdGeom.Camera.Define(stage, path)
        camera.AddTranslateOp().Set(Gf.Vec3d(*position[:3]))
        camera.AddOrientOp().Set(Gf.Quatf(float(orientation[3]), Gf.Vec3f(*orientation[:3])))
        camera.CreateFStopAttr(0.)  # No depth-of-field blur in the learning cameras.
        lens = (config["cameras"]["head"] if index == 0 else config["cameras"]["wrist"]["left" if index == 1 else "right"])
        if lens is not None:
            from .camera_calibration import set_pinhole_intrinsics
            set_pinhole_intrinsics(camera, width, height, lens["intrinsics"], lens["distortion_coefficients"])
        else:
            focal_length = 20.
            aperture = 2*focal_length*np.tan(np.deg2rad(config["cameras"]["horizontal_fov_deg"])/2)
            camera.CreateFocalLengthAttr(focal_length)
            camera.CreateHorizontalApertureAttr(float(aperture)); camera.CreateVerticalApertureAttr(float(aperture*height/width))
        camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 10.))
        if index == 0 or not cad_scene:
            _box(stage, path+"/Housing", (0, 0, .014), (.09, .025, .025), camera_housing_material)
        camera_paths.append(path)
        if capture:
            product = rep.create.render_product(path, (width, height))
            annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cuda")
            annotator.attach(product); annotators.append(annotator)
    return robot, annotators, geometry, robot_path, camera_paths


def rgb_frames(annotators):
    images = []; buffers = []; device = None
    for annotator in annotators:
        data = annotator.get_data()
        if hasattr(data, "device") and data.device.is_cuda:
            import warp as wp
            buffer = getattr(annotator, "_sim_rgb_host", None)
            if buffer is None or buffer.shape != data.shape:
                buffer = wp.empty(data.shape, dtype=data.dtype, device="cpu", pinned=True)
                annotator._sim_rgb_host = buffer
            wp.copy(buffer, data)
            buffers.append(buffer); device = data.device
        else:
            buffers.append(data)
    if device is not None:
        wp.synchronize_device(device)
    for buffer in buffers:
        values = buffer.numpy() if hasattr(buffer, "numpy") else np.asarray(buffer)
        images.append(values[..., :3].copy())
    return images
