"""USD visual palettes, portable texture graphs and metric UVs for the supplied CAD."""

import json

from . import ROOT
from ._dependencies import np


def apply_robot_appearance(stage, robot_path, appearance):
    """Override CAD debug colors on visual prims, preserving physics and measured optics."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    palette = {}
    for name, key, metallic, roughness in (
        ("RobotShellWhite", "shell_linear_rgb", 0., .28),
        ("RobotTrim", "trim_linear_rgb", 0., .48),
        ("RobotHandDark", "hand_linear_rgb", 0., .75),
        ("RobotMetal", "metal_linear_rgb", .65, .32),
    ):
        material = UsdShade.Material.Define(stage, "/World/Materials/"+name)
        shader = UsdShade.Shader.Define(stage, str(material.GetPath())+"/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*appearance[key]))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        palette[name] = material

    for link in stage.GetPrimAtPath(robot_path).GetChildren():
        name = link.GetName(); material = None
        if name.startswith("Link") and name.endswith(("_L", "_R")):
            material = palette["RobotShellWhite"]
        elif name.startswith("TCP_Link_") or name.endswith("_force_sensor"):
            material = palette["RobotMetal"]
        elif name.startswith(("l_", "r_")) and not name.endswith("_camera_bracket"):
            material = palette["RobotHandDark"]
        visuals = stage.GetPrimAtPath(str(link.GetPath())+"/visuals")
        if material is not None and visuals.IsValid():
            # This also overrides materials inside visual instances, without
            # rebinding any collision/physics prim or the camera's clear window.
            UsdShade.MaterialBindingAPI.Apply(visuals).Bind(
                material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)

    # The supplied base STL merges the dark pedestal/neck, white torso and head
    # camera. Color its existing faces in base-link coordinates; add no geometry.
    base = stage.GetPrimAtPath(robot_path+"/base_link")
    visuals = stage.GetPrimAtPath(str(base.GetPath())+"/visuals")
    for _ in range(4):
        instances = [p.GetPath() for p in Usd.PrimRange(visuals) if p.IsInstance()]
        if not instances:
            break
        for path in instances:
            stage.GetPrimAtPath(path).SetInstanceable(False)
    # The importer can put the source blue binding on ancestors and full-mesh
    # subsets. Clear those visual bindings before authoring the torso partition.
    UsdShade.MaterialBindingAPI.Apply(base).UnbindAllBindings()
    old_subsets = []
    for prim in Usd.PrimRange(visuals):
        # Some importer bindings predate the applied API schema, so HasAPI()
        # alone does not reveal all source-color relationships.
        UsdShade.MaterialBindingAPI.Apply(prim).UnbindAllBindings()
        if prim.IsA(UsdGeom.Subset) and UsdGeom.Subset(prim).GetFamilyNameAttr().Get() == "materialBind":
            old_subsets.append(prim.GetPath())
    for path in old_subsets:
        stage.RemovePrim(path)
    cache = UsdGeom.XformCache(); parent = np.array(cache.GetLocalToWorldTransform(base)).T
    for prim in Usd.PrimRange(visuals):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get())
        local = np.linalg.inv(parent) @ np.array(cache.GetLocalToWorldTransform(prim)).T
        points = points @ local[:3, :3].T+local[:3, 3]
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        centers = points[faces].mean(axis=1); x, y, z = centers.T
        torso = (np.abs(x) < .125) & (y > -.078) & (z >= .978) & (z <= 1.237)
        camera_case = (z > 1.24) & (y > -.108) & (y < -.03) & ~torso
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(palette["RobotTrim"], bindingStrength=UsdShade.Tokens.weakerThanDescendants)
        for name, selected, material in (("WhiteTorso", torso, "RobotShellWhite"),
                                         ("HeadCameraCase", camera_case, "RobotMetal")):
            indices = np.flatnonzero(selected)
            if len(indices):
                subset = binding.CreateMaterialBindSubset(name, indices.tolist())
                UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(palette[material])


def planar_uvs(mesh, metres_per_repeat):
    """Face-varying box projection avoids stretching CAD triangle sides."""
    from pxr import Gf, Sdf, UsdGeom

    points = np.asarray(mesh.GetPointsAttr().Get())
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    triangles = points[faces]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    dominant = np.abs(normals).argmax(axis=1)
    uv = np.empty((len(faces), 3, 2))
    for axis, coordinates in ((0, (1, 2)), (1, (0, 2)), (2, (1, 0))):
        select = dominant == axis
        uv[select] = triangles[select][:, :, coordinates] / np.asarray(metres_per_repeat)
    attribute = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                               UsdGeom.Tokens.faceVarying)
    attribute.Set([Gf.Vec2f(*value) for value in uv.reshape(-1, 2)])


def connect_basecolor(stage, material, texture_name, color):
    from pxr import Gf, Sdf, UsdShade

    directory = ROOT / "assets/textures"
    description = json.loads((directory / "manifest.json").read_text())["textures"][texture_name]
    filename = directory / description["file"]
    if not filename.is_file():
        raise FileNotFoundError(f"Required simulation texture is missing: {filename}")
    path = str(material.GetPath())
    reader = UsdShade.Shader.Define(stage, path+"/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    texture = UsdShade.Shader.Define(stage, path+"/BaseColor")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(filename)))
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    # Preserve the existing configurable mean material color after sRGB decoding.
    tint = np.asarray(color) / np.asarray(description["mean_linear_rgb"])
    texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*tint, 1.))
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader = UsdShade.Shader.Get(stage, path+"/Shader")
    shader.GetInput("diffuseColor").ConnectToSource(texture.ConnectableAPI(), "rgb")
