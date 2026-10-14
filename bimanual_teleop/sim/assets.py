"""Rebuild simulation models from the repository's curated source resources."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from bimanual_teleop.sim._dependencies import np

from . import ROOT
from .config import load_config

ROBOT_SOURCE = ROOT / "assets/source/robot/urdf/robot.urdf"
HAND_SOURCE = ROOT / "assets/source/hand2"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


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


def prepare(config, task_models=None):
    """Rebuild from the repository's curated sources, without external archives."""
    from .resources import source_components, tree_digest
    from .cad_assets import prepare_steps, prepare_rail, fixture_geometry, install
    if config["scene"]["asset_profile"] != "rail_task_cad_v1":
        raise ValueError("Only the tracked rail_task_cad_v1 asset profile can be rebuilt")
    generated = ROOT / "assets/generated"
    generated.mkdir(parents=True, exist_ok=True)
    components = source_components()
    manifest = {name: {"sha256": tree_digest(files), "sha256_kind": "source_tree",
                       "files": [str(path.relative_to(ROOT)) for path in sorted(files)]}
                for name, files in components.items()}
    repairs = _clean_robot(ROBOT_SOURCE, HAND_SOURCE, generated / "robot.urdf", config)
    manifest.update(**repairs, robot_urdf_sha256=digest(generated / "robot.urdf"))
    (generated / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    cad = prepare_steps(Path(task_models) if task_models else ROOT / "assets/source/rail_task")
    rail = prepare_rail(cad)
    cad["fixture_geometry"] = fixture_geometry(cad, rail)
    return install(cad, config)


def main():
    parser = argparse.ArgumentParser(description="Rebuild simulation assets from Git-tracked source models")
    parser.add_argument("--config")
    parser.add_argument("--task-models", help="Optional replacement CAD folder; default is the tracked source/rail_task")
    args = parser.parse_args()
    result = prepare(load_config(args.config), args.task_models)
    print(json.dumps({"robot": result["robot_urdf_sha256"], "fixture": result["fixture_geometry"],
                      "colliders": len(result["rail"]["colliders"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
