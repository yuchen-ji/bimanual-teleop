"""Portable USD texture graphs and metric UVs for the supplied CAD surfaces."""

import json

from . import ROOT
from ._dependencies import np


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
