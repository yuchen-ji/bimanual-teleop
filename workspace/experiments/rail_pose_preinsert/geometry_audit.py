"""Offline solid-mesh audit of a grasp; never connects to a device."""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from common import write_json
from scipy.spatial.transform import Rotation


def hand_meshes(urdf):
    """Use URDF collision meshes and origins, without filling their concavities."""
    urdf = Path(urdf)
    result = []
    for link in ET.parse(urdf).getroot().findall("link"):
        name = link.get("name")
        if name != "l_wrist" and not name.startswith(("l_thumb_", "l_index_finger_", "l_middle_finger_", "l_ring_finger_", "l_pinky_")):
            continue
        for collision in link.findall("collision"):
            element = collision.find("geometry/mesh")
            mesh = trimesh.load(urdf.parent / element.get("filename"), force="mesh")
            mesh.apply_scale([float(v) for v in element.get("scale", "1 1 1").split()])
            T = np.eye(4)
            origin = collision.find("origin")
            if origin is not None:
                T[:3, 3] = [float(v) for v in origin.get("xyz", "0 0 0").split()]
                T[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in origin.get("rpy", "0 0 0").split()]).as_matrix()
            mesh.apply_transform(T)
            result.append((name, mesh))
    return result


def audit_grasp(model, urdf, manifest_path, q, T_S_F):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    rail = trimesh.load(manifest_path.parent / "rail" / manifest["rail"]["visual"], force="mesh")
    if not rail.is_watertight:
        raise ValueError("Signed solid-mesh distances require a watertight rail")
    F_W = np.linalg.inv(model.flange("left", q, world=True))
    links = []
    for name, mesh in hand_meshes(urdf):
        T = np.asarray(T_S_F) @ F_W @ model.link_pose(name, q)
        # Deterministic vertex + triangle-centre sampling. This measures actual
        # solid intrusion at those points, not a complete penetration bound.
        points = np.vstack([mesh.vertices, mesh.triangles_center])
        points = points @ T[:3, :3].T + T[:3, 3]
        # Only points inside the rail AABB can lie inside its solid. Limit the
        # audit deterministically; report a lower bound, never absence of all
        # intersections, from this sampled calculation.
        relevant = points[np.all((points >= rail.bounds[0]) & (points <= rail.bounds[1]), axis=1)]
        if len(relevant) > 3000:
            relevant = relevant[np.linspace(0, len(relevant)-1, 3000).astype(int)]
        distance = trimesh.proximity.signed_distance(rail, relevant) if len(relevant) else np.array([-np.inf])
        links.append({"link": name, "sample_count": len(points),
                      "solid_test_sample_count": len(relevant),
                      "samples_inside_rail_over_0_2mm": int(np.count_nonzero(distance > .0002)),
                      "maximum_sampled_intrusion_m": float(max(0., distance.max())),
                      "hand_mesh_extent_m": mesh.extents.tolist()})
    data = manifest["rail"]
    return {"schema": "rail_hand_solid_mesh_audit_v1", "test_data": True,
            "rail_extent_m": rail.extents.tolist(), "rail_watertight": bool(rail.is_watertight),
            "rail_collision_piece_count": len(data["colliders"]),
            "rail_visual_volume_m3": data["mesh_volume_m3"],
            "rail_collision_volume_m3": data["collision_volume_m3"],
            "rail_collision_relative_volume_error": abs(data["collision_volume_m3"]/data["mesh_volume_m3"]-1),
            "links": links,
            "maximum_sampled_intrusion_m": max(v["maximum_sampled_intrusion_m"] for v in links),
            "method": "URDF hand collision mesh vertices and triangle centres against watertight visual rail; positive signed distance is inside solid",
            "limitation": "Sampled solid intrusion lower bound; not a PhysX cooked-hull or continuous contact certificate"}


def main():
    import argparse

    from bimanual_teleop.sim.config import ROOT
    from bimanual_teleop.sim.kinematics import RobotModel, pose_matrix
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--capture", default="grasp_review_00")
    args = parser.parse_args()
    frame = json.loads((args.output / "captures/trial_000" / args.capture / "state.json").read_text())
    state = frame.get("state", frame)
    urdf = ROOT / "assets/generated/robot.urdf"
    model = RobotModel(urdf)
    actual_F = model.base_pose("left") @ pose_matrix(state["eef"][0])
    actual_S = pose_matrix(state["rail"])
    report = audit_grasp(model, urdf, ROOT / "assets/generated/manifest.json", np.asarray(state["q"]),
                         np.linalg.inv(actual_S) @ actual_F)
    write_json(args.output / "geometry_audit.json", report)
    print(json.dumps({"maximum_sampled_intrusion_m": report["maximum_sampled_intrusion_m"],
                      "penetrating_links": [v for v in report["links"] if v["maximum_sampled_intrusion_m"] > .0002]}, indent=2))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    main()
