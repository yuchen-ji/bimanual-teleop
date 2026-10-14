"""Copy and prepare task CAD; optional CAD libraries are used only offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from . import ROOT
from .assets import digest, write_obj

SOURCE_NAMES = {
    "table": "table.STEP", "bins": "EU塑料转运箱.STEP",
    "fixture": "导轨台长.step", "rail": "导轨长.SLDPRT",
    "force_sensor": "模拟六维力传感器.step",
    "wrist_assembly": "腕部相机及其支架.step",
}


def read_step(path):
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise ValueError(f"Cannot read STEP: {path}")
    reader.TransferRoots()
    return reader.OneShape()


def shape_mesh(shape):
    import numpy as np
    import trimesh
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRep import BRep_Tool
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopoDS import TopoDS
    from OCP.TopLoc import TopLoc_Location
    BRepMesh_IncrementalMesh(shape, .08, False, .10, True).Perform()
    vertices, triangles = [], []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current()); location = TopLoc_Location()
        mesh = BRep_Tool.Triangulation_s(face, location)
        if mesh is not None:
            start = len(vertices)
            for i in range(1, mesh.NbNodes()+1):
                p = mesh.Node(i).Transformed(location.Transformation())
                vertices.append([p.X()/1000, p.Y()/1000, p.Z()/1000])
            for i in range(1, mesh.NbTriangles()+1):
                a, b, c = mesh.Triangle(i).Get()
                if face.Orientation() == TopAbs_REVERSED:
                    b, c = c, b
                triangles.append([start+a-1, start+b-1, start+c-1])
        explorer.Next()
    return trimesh.Trimesh(np.asarray(vertices), np.asarray(triangles), process=True)


def _export(mesh, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_obj(path, mesh.vertices, mesh.faces)
    return {"mesh": str(path.relative_to(ROOT / "assets/generated")),
            "bounds_m": mesh.bounds.tolist(), "vertices": len(mesh.vertices),
            "triangles": len(mesh.faces), "sha256": digest(path)}


def fixture_collision(mesh):
    """Reduce CAD tessellation density with a measured 0.1 mm surface error limit."""
    import numpy as np
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    # Sample upper locating surfaces more densely than the large flat base.
    samples = np.vstack([mesh.vertices[::max(1, len(mesh.vertices)//5000)],
                         mesh.vertices[mesh.vertices[:, 2] > .075][::5]])
    for count in (60000, 120000, 240000):
        reduced = mesh.simplify_quadric_decimation(face_count=count, aggression=3)
        points = vtk.vtkPoints(); points.SetData(numpy_to_vtk(reduced.vertices, deep=True))
        cells = vtk.vtkCellArray()
        packed = np.c_[np.full(len(reduced.faces), 3), reduced.faces].astype(np.int64).ravel()
        cells.SetCells(len(reduced.faces), numpy_to_vtkIdTypeArray(packed, deep=True))
        poly = vtk.vtkPolyData(); poly.SetPoints(points); poly.SetPolys(cells)
        distance = vtk.vtkImplicitPolyDataDistance(); distance.SetInput(poly)
        error = max(abs(distance.EvaluateFunction(point)) for point in samples)
        if error <= .0001:
            return reduced, {"sampled_surface_error_m": float(error), "surface_samples": len(samples),
                             "error_limit_m": .0001, "method": "quadric decimation; dense upper-surface distance check"}
    raise ValueError("Reduced fixture collision exceeds 0.1 mm; inspect the new CAD")


def _convex_polygons(triangles):
    """Merge adjacent planar triangles only when their exact union is convex."""
    import numpy as np
    from scipy.spatial import ConvexHull
    points, lookup, groups, areas = [], {}, {}, {}
    for index, triangle in enumerate(triangles):
        ids = []
        for point in triangle:
            key = tuple(np.round(point, 9))
            if key not in lookup:
                lookup[key] = len(points); points.append(point)
            ids.append(lookup[key])
        area = abs(np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0]))/2
        if area > 1e-14:
            groups[index], areas[index] = set(ids), area
    points = np.asarray(points)
    changed = True
    while changed:
        changed = False
        membership = {}
        for index, ids in groups.items():
            for point in ids:
                membership.setdefault(point, set()).add(index)
        for index in list(groups):
            if index not in groups:
                continue
            candidates = set.union(*(membership[p] for p in groups[index]))-{index}
            for other in sorted(candidates):
                if other not in groups or len(groups[index] & groups[other]) < 2:
                    continue
                combined = groups[index] | groups[other]
                hull = ConvexHull(points[sorted(combined)])
                if len(hull.vertices) <= 30 and abs(hull.volume-areas[index]-areas[other]) < 1e-12:
                    groups[index] = combined; areas[index] += areas.pop(other); del groups[other]
                    changed = True
                    break
    return [points[sorted(ids)][ConvexHull(points[sorted(ids)]).vertices] for ids in groups.values()]


def prepare_rail(cad):
    """Use the file's complete display mesh, retaining the pierced sheet collision geometry."""
    import numpy as np
    import trimesh
    import sldkit
    path = ROOT / "assets/source/rail_task" / SOURCE_NAMES["rail"]
    decoded = sldkit.decode_geometry_file(path, profile="desktop").to_dict()
    geometry = decoded.get("geometry")
    if not geometry or geometry["length_unit"] != "millimeter":
        raise ValueError("The long rail has no supported millimetre geometry")
    meshes = [trimesh.Trimesh(np.array(item["vertices"])/1000, item["triangles"], process=False)
              for item in geometry["model"]["tessellations"] if item["triangles"]]
    mesh = trimesh.util.concatenate(meshes)
    mesh.merge_vertices(digits_vertex=7); mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight or not mesh.is_winding_consistent or len(mesh.split(only_watertight=False)) != 1:
        raise ValueError("Long rail display geometry is incomplete; export it to STEP in SolidWorks")
    # CAD +Y is the rail's length; keep local rail X as length, local Z as height.
    rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    origin = (mesh.vertices @ rotation.T).min(axis=0)/2 + (mesh.vertices @ rotation.T).max(axis=0)/2
    triangles = mesh.triangles.copy(); bottom = mesh.bounds[0, 2]
    web_top = -.015251373733383002
    if abs(bottom+.0169667517638392) > 1e-6 or abs(mesh.extents[1]-.480) > 1e-6:
        raise ValueError("Long rail CAD has changed; inspect its sheet and cross-section before rebuilding colliders")
    # Web top triangles contain every real hole. Extruding them through sheet
    # thickness leaves those holes open, unlike a single convex hull.
    web = triangles[np.all(np.isclose(triangles[:, :, 2], web_top, atol=1e-7), axis=1)]
    cap = triangles[np.all(np.isclose(triangles[:, :, 1], 0., atol=1e-7), axis=1)]
    web_min, web_max = web[:, :, 0].min(), web[:, :, 0].max()

    def clip(polygon, axis, bound, sign):
        result = []
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            da, db = sign*(a[axis]-bound), sign*(b[axis]-bound)
            if da >= -1e-12:
                result.append(a)
            if (da > 1e-12 and db < -1e-12) or (da < -1e-12 and db > 1e-12):
                result.append(a+(b-a)*da/(da-db))
        return np.asarray(result)

    wings = []
    for triangle in cap[:, :, [0, 2]]:
        polygons = [clip(triangle, 1, web_top, 1)]
        below = clip(triangle, 1, web_top, -1)
        if len(below):
            polygons.extend([clip(below, 0, web_min, -1), clip(below, 0, web_max, 1)])
        for polygon in polygons:
            for i in range(1, len(polygon)-1):
                wings.append([polygon[0], polygon[i], polygon[i+1]])
    colliders = []
    directory = ROOT / "assets/generated/rail"; directory.mkdir(parents=True, exist_ok=True)
    for planes, kind in ((web[:, :, :2], "web"), (np.asarray(wings), "section")):
        for polygon in _convex_polygons(planes):
            if kind == "web":
                vertices = np.vstack([np.c_[polygon, np.full(len(polygon), z)] for z in (bottom, web_top)])
            else:
                vertices = np.vstack([np.c_[polygon[:, 0], np.full(len(polygon), y), polygon[:, 1]] for y in (0., mesh.extents[1])])
            piece = trimesh.Trimesh(vertices, process=False).convex_hull
            piece.vertices = piece.vertices @ rotation.T-origin
            filename = f"cad_collision_{len(colliders):03d}.obj"
            _export(piece, directory / filename); colliders.append(filename)
    mesh.vertices = mesh.vertices @ rotation.T-origin
    data = _export(mesh, directory / "rail.obj")
    # Exact hole axes taken from the source B-rep: Y=102 and 367 mm.
    hole_centers_cad = np.array([[0., .102, bottom], [-.00006525182861356, .367, bottom]])
    holes = hole_centers_cad @ rotation.T-origin
    data.update(visual="rail.obj", colliders=colliders, hole_centers_m=holes.tolist(),
                collider_sha256={name: digest(directory/name) for name in colliders},
                hole_diameters_m=[.0089, .008], cad_rotation=rotation.tolist(),
                origin_in_rotated_cad_m=origin.tolist(), size_m=mesh.extents.tolist(),
                display_mesh_watertight=True, mesh_volume_m3=float(mesh.volume),
                decoder="sldkit 0.1.0 (MIT), commit c43b5ef752a0e58535b7f8f744f7d7c68e3c0e90",
                decode_status=decoded["status"], fidelity_losses=geometry["fidelity"]["losses"])
    cad["models"]["rail"] = data
    collision_volume = sum(trimesh.load(directory/name, force="mesh").volume for name in colliders)
    data["collision_volume_m3"] = float(collision_volume)
    if abs(collision_volume/mesh.volume-1) > 1e-4:
        raise ValueError("Rail collision pieces do not reproduce the source mesh volume")
    return data


def fixture_geometry(cad, rail):
    import numpy as np
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    shape = read_step(ROOT / "assets/source/rail_task" / SOURCE_NAMES["fixture"])
    ex = TopExp_Explorer(shape, TopAbs_FACE); pins = []
    rotation = np.array(cad["models"]["fixture"]["cad_rotation"])
    origin = np.array(cad["models"]["fixture"]["origin_in_rotated_cad_m"])
    support_y = .13987
    while ex.More():
        face = TopoDS.Face_s(ex.Current()); surface = BRepAdaptor_Surface(face)
        if surface.GetType() == GeomAbs_Cylinder:
            cylinder = surface.Cylinder(); radius = cylinder.Radius()/1000
            if .003 < radius < .0035:
                point = cylinder.Location(); axis = cylinder.Axis().Direction()
                point = np.array([point.X(), point.Y(), point.Z()])/1000
                axis = np.array([axis.X(), axis.Y(), axis.Z()])
                point += axis*((support_y-point[1])/axis[1])
                bounds = Bnd_Box(); BRepBndLib.Add_s(face, bounds)
                pins.append((rotation@point-origin, radius, bounds.Get()[4]/1000-origin[2]))
        ex.Next()
    pins.sort(key=lambda item: item[0][1])
    if len(pins) != 2:
        raise ValueError("Expected two locating pin cylinders in long fixture CAD")
    # Fit XY rotation+translation to actual hole and pin axes, without scaling CAD.
    holes = np.array(rail["hole_centers_m"]); centers = np.array([p[0] for p in pins])
    a = holes[1, :2]-holes[0, :2]; b = centers[1, :2]-centers[0, :2]
    yaw = np.arctan2(b[1], b[0])-np.arctan2(a[1], a[0])
    planar = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    xy = centers[:, :2].mean(axis=0)-planar@holes[:, :2].mean(axis=0)
    support = support_y-origin[2]
    center = [*xy, support-holes[:, 2].mean()+.0004]
    return {"pin_centers_m": centers.tolist(), "pin_radii_m": [p[1] for p in pins],
            "support_height_m": float(support), "pin_top_height_m": min(p[2] for p in pins),
            "hole_spacing_m": float(np.linalg.norm(b)), "rail_hole_bottoms_m": holes.tolist(),
            "seated_center_in_fixture_m": center, "seated_rpy_deg": [0., 0., float(np.rad2deg(yaw))]}


def prepare_steps(source):
    import numpy as np
    raw = ROOT / "assets/source/rail_task"; raw.mkdir(parents=True, exist_ok=True)
    generated = ROOT / "assets/generated/task"; generated.mkdir(parents=True, exist_ok=True)
    result = {"sources": {}, "models": {}, "profile": "rail_task_cad_v1"}
    # Table and fixture CAD use +Y up; the robot and USD use +Z up.
    upright = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    fixture_axes = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    for key, filename in SOURCE_NAMES.items():
        original = source / filename
        destination = raw / filename
        if original.resolve() != destination.resolve() and (not destination.exists() or digest(original) != digest(destination)):
            if destination.exists():
                destination.unlink()
            shutil.copy2(original, destination)
        result["sources"][key] = {"filename": filename, "sha256": digest(destination)}
        if destination.suffix.lower() not in (".step", ".stp"):
            continue
        shape = read_step(destination)
        mesh = shape_mesh(shape)
        original_bounds = mesh.bounds.copy()
        rotation = fixture_axes if key == "fixture" else upright if key == "table" else np.eye(3)
        if not np.isclose(np.linalg.det(rotation), 1.):
            raise ValueError(f"CAD transform must preserve orientation: {key}")
        mesh.vertices = mesh.vertices @ rotation.T
        if key == "force_sensor":
            origin = np.array([0., 0., .8705])
        elif key == "wrist_assembly":
            # CAD plate is 2 mm thick. Input is its lower face; output is upper face.
            origin = np.array([.01527271, -.03463566, .090790])
        else:
            origin = mesh.bounds.mean(axis=0)
            origin[2] = mesh.bounds[0, 2]
        mesh.vertices -= origin
        data = ({"bounds_m": mesh.bounds.tolist(), "vertices": len(mesh.vertices), "triangles": len(mesh.faces)}
                if key == "wrist_assembly" else _export(mesh, generated / f"{key}.obj"))
        if key == "fixture" and len(mesh.faces) > 100000:
            collision, accuracy = fixture_collision(mesh)
            data["collision"] = {**_export(collision, generated / "fixture_collision.obj"), **accuracy}
        data.update(cad_bounds_m=original_bounds.tolist(), cad_rotation=rotation.tolist(),
                    origin_in_rotated_cad_m=origin.tolist())
        result["models"][key] = data
        if key == "wrist_assembly":
            from OCP.TopExp import TopExp_Explorer
            from OCP.TopAbs import TopAbs_SOLID
            explorer = TopExp_Explorer(shape, TopAbs_SOLID); parts = []
            while explorer.More():
                part = shape_mesh(explorer.Current()); part.vertices -= origin
                parts.append(part); explorer.Next()
            # CAD contains one bracket, the camera body, and its thin front cover.
            parts.sort(key=lambda part: part.extents[2], reverse=True)
            data["parts"] = []
            for index, part in enumerate(parts):
                item = _export(part, generated / f"wrist_part_{index}.obj")
                item["color"] = ([.35, .38, .42] if index == 0 else [.06, .07, .08] if index == 1 else [.15, .18, .20])
                data["parts"].append(item)
    result["tool_stack"] = {"sensor_length_m": .040, "bracket_plate_thickness_m": .002,
                            "sensor_mass_kg": .2304, "bracket_camera_mass_kg": .20,
                            "mass_is_assumption": True,
                            "camera_position_in_bracket_m": [.0012793, .0592937, -.035071]}
    (generated / "cad_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def install_tool_stack(urdf, cad, config):
    """Idempotently insert the two fixed tools while retaining all 54 actuated DOFs."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    root = ET.parse(urdf).getroot()
    tools = cad["tool_stack"]
    for side, prefix, flange in (("left", "l", "L"), ("right", "r", "R")):
        for element in list(root):
            if element.get("name", "").startswith(f"sim_{side}_tool_") or element.tag == "link" and element.get("name") in (f"{prefix}_force_sensor", f"{prefix}_camera_bracket"):
                root.remove(element)
        for component, mass in (("force_sensor", tools["sensor_mass_kg"]), ("camera_bracket", tools["bracket_camera_mass_kg"])):
            link = ET.SubElement(root, "link", name=f"{prefix}_{component}")
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "origin", xyz="0 0 .02" if component == "force_sensor" else "0 .02 -.02")
            ET.SubElement(inertial, "mass", value=str(mass))
            ET.SubElement(inertial, "inertia", ixx="0.00015", ixy="0", ixz="0", iyy="0.00015", iyz="0", izz="0.00015")
            pieces = ([{"mesh": "task/force_sensor.obj", "color": [.2, .3, .45]}] if component == "force_sensor"
                      else cad["models"]["wrist_assembly"]["parts"])
            for piece in pieces:
                for kind in ("visual", "collision"):
                    item = ET.SubElement(link, kind)
                    geometry = ET.SubElement(item, "geometry")
                    ET.SubElement(geometry, "mesh", filename=piece["mesh"])
                    if kind == "visual":
                        material = ET.SubElement(item, "material", name=f"sim_{prefix}_{component}_{Path(piece['mesh']).stem}")
                        ET.SubElement(material, "color", rgba=" ".join(map(str, [*piece["color"], 1.])))
        for name, parent, child, z in (
            ("sensor", f"TCP_Link_{flange}", f"{prefix}_force_sensor", 0.),
            ("bracket", f"{prefix}_force_sensor", f"{prefix}_camera_bracket", tools["sensor_length_m"]),
        ):
            joint = ET.SubElement(root, "joint", name=f"sim_{side}_tool_{name}", type="fixed")
            ET.SubElement(joint, "parent", link=parent); ET.SubElement(joint, "child", link=child)
            roll = "0 0 3.141592653589793" if side == "right" and name == "bracket" else "0 0 0"
            ET.SubElement(joint, "origin", xyz=f"0 0 {z}", rpy=roll)
        hand_joint = root.find(f"joint[@name='sim_{side}_hand_mount']")
        hand_joint.find("parent").set("link", f"{prefix}_camera_bracket")
        hand_joint.find("origin").set("xyz", f"0 0 {tools['bracket_plate_thickness_m']}")
        mount_rotation = Rotation.from_euler("xyz", config["robot"]["hand_mount_rpy"][side])
        if side == "right":
            # Rotate the camera to the mirrored palm, preserving the hand's own
            # flange-relative Rx(pi) orientation and all existing joint signs.
            mount_rotation = Rotation.from_euler("z", -np.pi)*mount_rotation
        hand_joint.find("origin").set("rpy", " ".join(map(str, mount_rotation.as_euler("xyz"))))
    ET.indent(root)
    ET.ElementTree(root).write(urdf, encoding="utf-8", xml_declaration=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "assets/source/rail_task")
    parser.add_argument("--steps-only", action="store_true")
    args = parser.parse_args()
    cad = prepare_steps(args.source)
    if not args.steps_only:
        rail = prepare_rail(cad)
        cad["fixture_geometry"] = fixture_geometry(cad, rail)
        install(cad)
    print(json.dumps({key: value["bounds_m"] for key, value in cad["models"].items()}, ensure_ascii=False, indent=2))


def install(cad, config=None):
    if config is None:
        from .config import load_config
        config = load_config()
    generated = ROOT / "assets/generated"
    manifest_path = generated / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    install_tool_stack(generated / "robot.urdf", cad, config)
    manifest.update(cad=cad, rail=cad["models"]["rail"], fixture_geometry=cad["fixture_geometry"],
                    robot_urdf_sha256=digest(generated / "robot.urdf"),
                    camera_visual="supplied wrist camera/bracket CAD; optical aim initialized towards palm, pending calibration")
    sources = json.dumps({"sources": cad["sources"], "models": cad["models"]}, sort_keys=True)
    manifest["robot_signature"] = hashlib.sha256((manifest["robot_urdf_sha256"]+manifest["robot"]["sha256"]+manifest["hands"]["sha256"]+sources).encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    (generated / "task/cad_manifest.json").write_text(json.dumps(cad, ensure_ascii=False, indent=2))
    return manifest


if __name__ == "__main__":
    main()
