"""Build conservative sphere covers from the actual URDF collision meshes."""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


def build(project):
    directory = Path(__file__).resolve().parent
    generated = Path(project)/"bimanual_teleop/sim/assets/generated"
    path = generated/"robot.urdf"
    result = {}
    for link in ET.parse(path).getroot().findall("link"):
        name = link.get("name"); spheres = []
        if name == "base_link":
            continue
        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                continue
            bounds = trimesh.load(path.parent/mesh.get("filename"), force="mesh").bounds.copy()
            bounds *= np.array([float(x) for x in mesh.get("scale", "1 1 1").split()])
            origin = collision.find("origin")
            xyz = np.array([float(x) for x in origin.get("xyz", "0 0 0").split()]) if origin is not None else np.zeros(3)
            rotation = Rotation.from_euler("xyz", [float(x) for x in origin.get("rpy", "0 0 0").split()]).as_matrix() if origin is not None else np.eye(3)
            dimensions = bounds[1]-bounds[0]; axis = int(np.argmax(dimensions))
            count = max(1, int(np.ceil(dimensions[axis]/.025)))
            radius = float(np.linalg.norm(np.r_[np.delete(dimensions, axis)/2, dimensions[axis]/count/2])+.001)
            for index in range(count):
                center = bounds.mean(axis=0); center[axis] = bounds[0, axis]+dimensions[axis]*(index+.5)/count
                spheres.append({"center": (rotation@center+xyz).tolist(), "radius": radius})
        if spheres:
            result[name] = spheres
    fixture = generated/"task/fixture_collision.obj"
    metadata = {"robot_urdf_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
                "fixture_bounds_m": trimesh.load(fixture, force="mesh").bounds.tolist(),
                "robot_signature": json.loads((generated/"manifest.json").read_text())["robot_signature"]}
    (directory/"collision_spheres.json").write_text(json.dumps(result, separators=(",", ":"))+"\n")
    (directory/"collision_spheres.metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(f"Built {sum(map(len, result.values()))} spheres for {len(result)} links")


if __name__ == "__main__":
    build(Path(__file__).resolve().parents[3])
