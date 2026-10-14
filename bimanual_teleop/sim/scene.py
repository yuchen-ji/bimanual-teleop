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


def _material(stage, path, color, friction=.5, metallic=0.):
    from pxr import Gf, UsdShade, UsdPhysics, Sdf
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path+"/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.45)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(friction); physics.CreateDynamicFrictionAttr(friction)
    physics.CreateRestitutionAttr(0.)
    return material


def _bind(prim, material):
    from pxr import UsdShade
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim() if hasattr(prim, "GetPrim") else prim).Bind(material)


def _box(stage, path, position, size, material):
    from pxr import Gf, UsdGeom, UsdPhysics
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.)
    cube.AddTranslateOp().Set(Gf.Vec3d(*position)); cube.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim()); _bind(cube, material)
    return cube


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
            if hand:
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(fingers, materialPurpose="physics")
    world.scene.add_default_ground_plane()
    manifest = json.loads((ROOT / "assets/generated/manifest.json").read_text())
    geometry = manifest["fixture_geometry"]
    matte = _material(stage, "/World/Materials/Table", (.43, .36, .25))
    metal = _material(stage, "/World/Materials/Rail", (.55, .58, .62), config["rail"]["friction"], .75)
    fixture_material = _material(stage, "/World/Materials/Fixture", (.2, .23, .27), config["rail"]["friction"], .5)
    bin_material = _material(stage, "/World/Materials/Bins", (.12, .28, .48))
    _box(stage, "/World/Table", config["scene"]["table_center"], config["scene"]["table_size"], matte)
    fixture = UsdGeom.Xform.Define(stage, "/World/Fixture")
    fixture.AddTranslateOp().Set(Gf.Vec3d(*config["scene"]["fixture_origin"]))
    _bind(_mesh(stage, "/World/Fixture/Mesh", ROOT / "assets/generated/fixture.obj", collision=True), fixture_material)
    length, width, height = config["scene"]["bin_inner_size"]
    wall = config["scene"]["bin_wall"]
    for index, center in enumerate(config["scene"]["bin_centers"]):
        center = np.asarray(center)
        base = f"/World/Bin_{index}"
        _box(stage, base+"/Floor", center+[0, 0, wall/2], [length+2*wall, width+2*wall, wall], bin_material)
        for sign in (-1, 1):
            _box(stage, base+f"/X_{sign+1}", center+[sign*(length+wall)/2, 0, height/2], [wall, width+2*wall, height], bin_material)
            _box(stage, base+f"/Y_{sign+1}", center+[0, sign*(width+wall)/2, height/2], [length, wall, height], bin_material)
    rail = UsdGeom.Xform.Define(stage, "/World/Rail")
    rail.AddTranslateOp().Set(Gf.Vec3d(*config["rail"]["initial_center"]))
    rail.AddOrientOp().Set(Gf.Quatf(np.cos(np.pi/4), Gf.Vec3f(0, 0, np.sin(np.pi/4))))
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
    dome = UsdLux.DomeLight.Define(stage, "/World/Lighting/Dome"); dome.CreateIntensityAttr(900.)
    sun = UsdLux.DistantLight.Define(stage, "/World/Lighting/Sun"); sun.CreateIntensityAttr(2200.)
    sun.AddRotateXYZOp().Set(Gf.Vec3f(-35, 20, 25))
    width, height = config["cameras"]["width"], config["cameras"]["height"]
    zero = np.zeros(54)
    camera_paths, annotators = [], []
    for index in range(3):
        if index == 0:
            parent = robot_path + "/Head_depth_Link"
            position_world = model.link_pose("Head depth_Link", zero)[:3, 3]
            target_world = np.array(config["scene"]["fixture_origin"])+[0, 0, .08]
            parent_pose = model.link_pose("Head depth_Link", zero)
            position = np.linalg.inv(parent_pose) @ np.r_[position_world, 1.]
            world_orientation = Rotation.from_quat(look_at(position_world, target_world)).as_matrix()
            orientation = Rotation.from_matrix(parent_pose[:3, :3].T @ world_orientation).as_quat()
        else:
            side, prefix = ("left", "l") if index == 1 else ("right", "r")
            parent = robot_path + f"/{prefix}_mount"
            mount_pose = model.link_pose(f"{prefix}_mount", zero)
            tips = [model.link_pose(f"{prefix}_{finger}_tip", zero)[:3, 3]
                    for finger in ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")]
            # Generate one rigid mounting transform from open-hand landmarks.
            tip_center = (np.linalg.inv(mount_pose) @ np.r_[np.mean(tips, axis=0), 1])[:3]
            along = tip_center / max(np.linalg.norm(tip_center), 1e-8)
            normal = np.cross(along, np.array([1., 0., 0.]))
            if np.linalg.norm(normal) < .1:
                normal = np.cross(along, np.array([0., 1., 0.]))
            normal /= np.linalg.norm(normal)
            # Choose the palmar side from positive finger flexion, avoiding
            # an arbitrary camera on the back of a mirrored hand.
            curled = zero.copy()
            for finger in ("index_finger", "middle_finger", "ring_finger", "pinky"):
                curled[model.names.index(f"{prefix}_{finger}_mcp_flex")] = .3
            curled_tips = [model.link_pose(f"{prefix}_{finger}_tip", curled)[:3, 3]
                           for finger in ("index_finger", "middle_finger", "ring_finger", "pinky")]
            open_tips = [model.link_pose(f"{prefix}_{finger}_tip", zero)[:3, 3]
                         for finger in ("index_finger", "middle_finger", "ring_finger", "pinky")]
            curl_direction = mount_pose[:3, :3].T @ (np.mean(curled_tips, axis=0)-np.mean(open_tips, axis=0))
            if np.dot(normal, curl_direction) < 0:
                normal = -normal
            position = tip_center*.5 + normal*.12 - along*.06
            orientation = look_at(position, tip_center*.65, up=along)
        if not stage.GetPrimAtPath(parent).IsValid():
            raise RuntimeError(f"Imported camera mounting link was not found: {parent}")
        path = parent+f"/sim_camera_{index}"
        camera = UsdGeom.Camera.Define(stage, path)
        camera.AddTranslateOp().Set(Gf.Vec3d(*position[:3]))
        camera.AddOrientOp().Set(Gf.Quatf(float(orientation[3]), Gf.Vec3f(*orientation[:3])))
        focal_length = 20.
        aperture = 2*focal_length*np.tan(np.deg2rad(config["cameras"]["horizontal_fov_deg"])/2)
        camera.CreateFocalLengthAttr(focal_length)
        camera.CreateHorizontalApertureAttr(float(aperture)); camera.CreateVerticalApertureAttr(float(aperture*height/width))
        camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 10.))
        housing = _box(stage, path+"/Housing", (0, 0, .014), (.09, .025, .025), fixture_material)
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
