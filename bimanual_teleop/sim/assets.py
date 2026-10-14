"""Prepare the supplied models without changing their original attachments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath

from bimanual_teleop.sim._dependencies import np

from . import ROOT
from .config import load_config

ATTACHMENTS = {
    "robot": "/home/jiyuchen/.codex/attachments/753942fe-50c3-4aa4-8836-d3b9fe65f4b0/Pro-m6-696-夹爪.rar",
    "hands": "/home/jiyuchen/.codex/attachments/b5a1bced-7883-4072-b620-c65d0d691c87/hand2_beta2.zip",
    "fixture": "/home/jiyuchen/.codex/attachments/0dbac7b9-ff5b-43c6-9d2a-4d9824aeee99/导轨台2.step",
    "camera": "/home/jiyuchen/.codex/attachments/63d95243-3a81-4f57-9033-c6a3d561e3cf/D435i_Solid.SLDPRT",
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _safe_names(names):
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe archive member: {name}")


def fixture_geometry(path):
    """Read the paired outward 6 mm pin surfaces and their common CAD axis."""
    text = Path(path).read_text(encoding="utf-8")
    entities = dict(re.findall(r"#(\d+)\s*=\s*(.*?);", text, re.S))
    refs = lambda value: re.findall(r"#(\d+)", value)
    vector = lambda key: np.array([float(v) for v in entities[key].split("(")[-1].rstrip(")").split(",")])
    cylinders = []
    for key, value in entities.items():
        if value.startswith("CYLINDRICAL_SURFACE") and abs(float(value.rsplit(",", 1)[1].rstrip(")")) - .003) < 1e-9:
            outside = any(v.startswith("ADVANCED_FACE") and f"#{key},.T." in v for v in entities.values())
            if outside:
                placement = refs(entities[refs(value)[0]])
                cylinders.append((vector(placement[0]), vector(placement[1])))
    if len(cylinders) != 2:
        raise ValueError("Expected exactly two outward 6 mm locating pins; inspect the new fixture CAD")
    cylinders.sort(key=lambda item: item[0][0])
    points = np.array([point for point, _axis in cylinders])
    up = sum(axis for _point, axis in cylinders)
    up /= np.linalg.norm(up)
    across = points[1] - points[0]
    across -= up * np.dot(across, up)
    across /= np.linalg.norm(across)
    forward = np.cross(across, up)
    # New fixture coordinates are front/left/up. The long rail axis is local Y.
    basis = np.stack([forward, across, up], axis=0)
    all_points = np.array([vector(key) for key, value in entities.items() if value.startswith("CARTESIAN_POINT")])
    transformed = all_points @ basis.T
    center = points.mean(axis=0)
    origin = basis @ center
    origin[2] = transformed[:, 2].min()
    pin_centers = points @ basis.T - origin
    return {"cad_to_fixture_rotation": basis.tolist(), "cad_origin_in_fixture_axes": origin.tolist(),
            "pin_centers_m": pin_centers.tolist(), "pin_diameter_m": .006,
            "hole_spacing_m": float(np.linalg.norm((points[1] - points[0]) @ basis[:2].T)),
            "support_height_m": float(pin_centers[:, 2].mean()),
            "pin_top_height_m": float((transformed - origin)[:, 2].max())}


def write_obj(path, vertices, faces):
    with Path(path).open("w", encoding="utf-8") as output:
        for vertex in vertices:
            output.write("v " + " ".join(f"{v:.10g}" for v in vertex) + "\n")
        for face in faces:
            output.write("f " + " ".join(str(int(i) + 1) for i in face) + "\n")


def rail_meshes(directory, settings):
    """Extruded triangulation produces convex colliders whose union has real holes."""
    import mapbox_earcut

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    length, width, height = settings["size_m"]
    radius, spacing, segments = settings["hole_diameter_m"] / 2, settings["hole_spacing_m"], settings["hole_segments"]
    contour = [[-length/2, -width/2], [length/2, -width/2], [length/2, width/2], [-length/2, width/2]]
    for center in (-spacing/2, spacing/2):
        contour.extend([[center + radius * math.cos(-2*math.pi*i/segments),
                         radius * math.sin(-2*math.pi*i/segments)] for i in range(segments)])
    points = np.asarray(contour, dtype=np.float64)
    triangles = mapbox_earcut.triangulate_float64(points, np.array([4, 4+segments, 4+2*segments], dtype=np.uint32)).reshape(-1, 3)
    n = len(points)
    vertices = np.vstack([np.c_[points, np.full(n, -height/2)], np.c_[points, np.full(n, height/2)]])
    faces = [list(triangle[::-1]) for triangle in triangles] + [list(triangle + n) for triangle in triangles]
    from collections import Counter
    directed = [(int(a), int(b)) for tri in triangles for a, b in zip(tri, np.roll(tri, -1))]
    edges = Counter(tuple(sorted(edge)) for edge in directed)
    orientations = {tuple(sorted(edge)): edge for edge in directed}
    for edge, count in edges.items():
        if count == 1:
            a, b = orientations[edge]
            faces.extend([[a, b, b+n], [a, b+n, a+n]])
    write_obj(directory / "rail.obj", vertices, faces)
    prism_faces = [[2, 1, 0], [3, 4, 5], [0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [2, 0, 3], [2, 3, 5]]
    colliders = []
    for index, triangle in enumerate(triangles):
        selected = vertices[np.r_[triangle, triangle+n]]
        name = f"collision_{index:03d}.obj"
        write_obj(directory / name, selected, prism_faces)
        colliders.append(name)
    return {"visual": "rail.obj", "colliders": colliders, "hole_centers_m": [[-spacing/2, 0, 0], [spacing/2, 0, 0]],
            "cross_section_area_m2": float(sum(abs(np.cross(points[t[1]]-points[t[0]], points[t[2]]-points[t[0]]))/2 for t in triangles))}


def step_mesh(source, destination, geometry):
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRep import BRep_Tool
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopoDS import TopoDS
    from OCP.TopLoc import TopLoc_Location

    reader = STEPControl_Reader()
    if reader.ReadFile(str(source)) != IFSelect_RetDone:
        raise ValueError("Could not read fixture STEP")
    reader.TransferRoots()
    shape = reader.OneShape()
    # OCP convention is millimetres even when the STEP declares metres.
    BRepMesh_IncrementalMesh(shape, .15, False, .15, True).Perform()
    vertices, triangles = [], []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    basis, origin = np.array(geometry["cad_to_fixture_rotation"]), np.array(geometry["cad_origin_in_fixture_axes"])
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        location = TopLoc_Location()
        mesh = BRep_Tool.Triangulation_s(face, location)
        if mesh is not None:
            start = len(vertices)
            for i in range(1, mesh.NbNodes()+1):
                p = mesh.Node(i).Transformed(location.Transformation())
                vertices.append((basis @ (np.array([p.X(), p.Y(), p.Z()]) / 1000) - origin).tolist())
            for i in range(1, mesh.NbTriangles()+1):
                a, b, c = mesh.Triangle(i).Get()
                if face.Orientation() == TopAbs_REVERSED:
                    b, c = c, b
                triangles.append([start+a-1, start+b-1, start+c-1])
        explorer.Next()
    if not vertices:
        raise ValueError("Fixture STEP did not contain any tessellatable faces")
    vertices = np.asarray(vertices)
    # STEP surface-placement reference points are not shape extrema. Ground the
    # actual tessellated base, rather than using unrelated construction origins.
    base_shift = float(vertices[:, 2].min())
    vertices[:, 2] -= base_shift
    geometry["cad_origin_in_fixture_axes"][2] += base_shift
    for center in geometry["pin_centers_m"]:
        center[2] -= base_shift
    geometry["support_height_m"] -= base_shift
    geometry["pin_top_height_m"] -= base_shift
    write_obj(destination, vertices, triangles)


def _clean_robot(source, hands, destination, config):
    root = ET.parse(source).getroot()
    # Bake the canonical world orientation into a fixed URDF joint. Rotating
    # the imported outer USD Xform can disagree with its fixed world anchor.
    root.insert(0, ET.Element("link", name="sim_world"))
    world_joint = ET.SubElement(root, "joint", name="sim_world_frame", type="fixed")
    ET.SubElement(world_joint, "parent", link="sim_world")
    ET.SubElement(world_joint, "child", link="base_link")
    ET.SubElement(world_joint, "origin", xyz="0 0 0", rpy=f"0 0 {math.radians(config['robot']['world_yaw_deg'])}")
    discarded = {f"Link8_{side}" for side in ("L", "R")}
    # The branches below the flange belong to the supplied old grippers.
    changed = True
    while changed:
        changed = False
        for joint in root.findall("joint"):
            if joint.find("parent").get("link") in discarded and joint.find("child").get("link") not in discarded:
                discarded.add(joint.find("child").get("link")); changed = True
    for element in list(root):
        if element.tag == "link" and element.get("name") in discarded or element.tag == "joint" and element.find("child").get("link") in discarded:
            root.remove(element)
    robot_dir = source.parent.parent
    for link in root.findall("link"):
        for kind in ("visual", "collision"):
            for item in list(link.findall(kind)):
                mesh = item.find("geometry/mesh")
                if mesh is None:
                    continue
                path = robot_dir / mesh.get("filename").split("/", 3)[-1]
                if not path.is_file():
                    if link.get("name") != "TCP_Link_R":
                        raise FileNotFoundError(path)
                    link.remove(item)
                else:
                    mesh.set("filename", os.path.relpath(path, destination.parent))
    for side, prefix, flange in (("left", "l", "L"), ("right", "r", "R")):
        urdf_path = hands / "body/urdf" / f"{side}_with_mount.urdf"
        hand = ET.parse(urdf_path).getroot()
        for mesh in hand.findall(".//mesh"):
            path = (urdf_path.parent / mesh.get("filename")).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            mesh.set("filename", os.path.relpath(path, destination.parent))
        mount = ET.SubElement(root, "joint", name=f"sim_{side}_hand_mount", type="fixed")
        ET.SubElement(mount, "parent", link=f"TCP_Link_{flange}")
        ET.SubElement(mount, "child", link=f"{prefix}_mount")
        ET.SubElement(mount, "origin", xyz="0 0 0", rpy=" ".join(str(v) for v in config["robot"]["hand_mount_rpy"][side]))
        root.extend(list(hand))
    # The importer also uses mesh stems as USD prim names. Spaces in CAD
    # filenames otherwise produce null prims even when link names are repaired.
    mesh_dir = destination.parent / "meshes"
    mesh_dir.mkdir(exist_ok=True)
    copied = {}
    for mesh in root.findall(".//mesh"):
        source_path = (destination.parent / mesh.get("filename")).resolve()
        if source_path not in copied:
            name = f"mesh_{len(copied):03d}{source_path.suffix.lower()}"
            shutil.copy2(source_path, mesh_dir / name)
            copied[source_path] = name
        mesh.set("filename", "meshes/"+copied[source_path])
    repaired_frames = []
    for link in root.findall("link"):
        inertia = link.find("inertial")
        mass = inertia.find("mass") if inertia is not None else None
        if mass is None or float(mass.get("value")) <= 0:
            # TCP and fingertip coordinate frames have no physical mass in the
            # supplied URDF. PhysX requires positive mass when fixed links are
            # retained for cameras and force measurement. Use an explicit,
            # documented 1 g frame approximation rather than negative defaults.
            if inertia is not None:
                link.remove(inertia)
            inertia = ET.SubElement(link, "inertial")
            ET.SubElement(inertia, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(inertia, "mass", value="0.001")
            ET.SubElement(inertia, "inertia", ixx="1e-7", ixy="0", ixz="0", iyy="1e-7", iyz="0", izz="1e-7")
            repaired_frames.append(link.get("name"))
    for joint in root.findall("joint"):
        if joint.get("type") == "fixed":
            axis = joint.find("axis")
            if axis is not None:
                joint.remove(axis)
    root.set("name", "bimanual_hand2_sim")
    ET.indent(root)
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
    return {"mesh_sources": {name: os.path.relpath(path, ROOT) for path, name in copied.items()},
            "frame_mass_approximation_kg": .001, "frame_mass_repaired_links": repaired_frames}


def prepare(config, sources=None):
    sources = {key: Path(value) for key, value in (sources or ATTACHMENTS).items()}
    manifest = {key: {"filename": p.name, "sha256": digest(p)} for key, p in sources.items()}
    source_dir, generated = ROOT / "assets/source", ROOT / "assets/generated"
    source_dir.mkdir(parents=True, exist_ok=True); generated.mkdir(parents=True, exist_ok=True)
    archive_tool = shutil.which("bsdtar")
    if archive_tool is None and Path("/opt/miniconda3/bin/bsdtar").is_file():
        archive_tool = "/opt/miniconda3/bin/bsdtar"
    if archive_tool is None:
        raise RuntimeError("RAR extraction requires bsdtar; install it or add it to PATH")
    names = subprocess.check_output([archive_tool, "-tf", str(sources["robot"])], text=True).splitlines()
    _safe_names(names)
    subprocess.run([archive_tool, "-xf", str(sources["robot"]), "-C", str(source_dir)], check=True)
    with zipfile.ZipFile(sources["hands"]) as archive:
        _safe_names(archive.namelist())
        archive.extractall(source_dir)
    for key in ("fixture", "camera"):
        shutil.copy2(sources[key], source_dir / sources[key].name)
    robot = source_dir / "Pro-m6-696-夹爪/Pro-m6-696/urdf/Pro-m6-696.urdf"
    repairs = _clean_robot(robot, source_dir / "hand2_beta2", generated / "robot.urdf", config)
    geometry = fixture_geometry(sources["fixture"])
    if abs(geometry["hole_spacing_m"] - config["rail"]["hole_spacing_m"]) > .001:
        raise ValueError("Configured hole spacing does not match fixture pins")
    step_mesh(sources["fixture"], generated / "fixture.obj", geometry)
    rail = rail_meshes(generated / "rail", config["rail"])
    manifest.update(fixture_geometry=geometry, rail=rail, **repairs, robot_urdf_sha256=digest(generated / "robot.urdf"),
                    camera_visual="simplified 90x25x25 mm housing; SLDPRT retained as mounting reference")
    manifest["robot_signature"] = hashlib.sha256((manifest["robot_urdf_sha256"]+manifest["robot"]["sha256"]+manifest["hands"]["sha256"]).encode()).hexdigest()
    (generated / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Prepare supplied Tianji/Hand2/fixture assets")
    parser.add_argument("--config")
    for name in ATTACHMENTS:
        parser.add_argument(f"--{name}", default=ATTACHMENTS[name])
    args = parser.parse_args()
    result = prepare(load_config(args.config), {name: getattr(args, name) for name in ATTACHMENTS})
    print(json.dumps({"robot": result["robot_urdf_sha256"], "fixture": result["fixture_geometry"],
                      "colliders": len(result["rail"]["colliders"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
