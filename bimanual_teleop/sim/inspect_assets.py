"""Render a lightweight asset-layout preview without Isaac or robot devices."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from bimanual_teleop.sim._dependencies import np
from bimanual_teleop.sim._dependencies import trimesh

from . import ROOT
from .config import load_config, sim_path
from .kinematics import RobotModel, transform


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    parser = argparse.ArgumentParser(description="Render geometry layout for asset inspection, not physics validation")
    parser.add_argument("--output", default=str(ROOT / "data/asset-preview.png"))
    args = parser.parse_args()
    config = load_config(); path = sim_path(config["robot"]["urdf"])
    model = RobotModel(path)
    q = np.r_[np.deg2rad(config["robot"]["ready_deg"]["left"]), np.deg2rad(config["robot"]["ready_deg"]["right"]), np.zeros(40)]
    figure = plt.figure(figsize=(12, 9), dpi=140)
    axes = figure.add_subplot(111, projection="3d")

    def add(mesh, matrix, color, maximum=1500):
        if len(mesh.faces) > maximum:
            # Vertex clustering preserves visible surfaces for this inspection
            # image. The simulator always imports the original full mesh.
            cell = max(np.ptp(mesh.vertices, axis=0)) / 80
            for _ in range(20):
                _, inverse = np.unique(np.round(mesh.vertices / cell).astype(np.int64), axis=0, return_inverse=True)
                counts = np.bincount(inverse)
                vertices = np.stack([np.bincount(inverse, weights=mesh.vertices[:, axis])/counts for axis in range(3)], axis=1)
                faces = inverse[mesh.faces]
                valid = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
                faces = faces[valid]
                _, unique = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
                faces = faces[unique]
                if len(faces) <= maximum * 2:
                    break
                cell *= 1.3
            mesh = trimesh.Trimesh(vertices, faces, process=False)
        vertices = trimesh.transform_points(mesh.vertices, matrix)
        faces = mesh.faces
        collection = Poly3DCollection(vertices[faces], facecolor=color, edgecolor="none", alpha=.85)
        axes.add_collection3d(collection)

    for link in ET.parse(path).getroot().findall("link"):
        matrix = model.link_pose(link.get("name"), q)
        for visual in link.findall("visual"):
            asset = visual.find("geometry/mesh")
            if asset is None:
                continue
            origin = visual.find("origin")
            offset = transform([float(v) for v in origin.get("xyz", "0 0 0").split()],
                               [float(v) for v in origin.get("rpy", "0 0 0").split()]) if origin is not None else np.eye(4)
            mesh = trimesh.load((path.parent / asset.get("filename")).resolve(), force="mesh")
            color = "#d1d5db" if link.get("name").startswith(("l_", "r_")) else "#5686ac"
            add(mesh, matrix @ offset, color)
    add(trimesh.creation.box(config["scene"]["table_size"]), transform(config["scene"]["table_center"]), "#b49c7a")
    add(trimesh.load(ROOT / "assets/generated/fixture.obj", force="mesh"), transform(config["scene"]["fixture_origin"]), "#68717a", 6000)
    add(trimesh.load(ROOT / "assets/generated/rail/rail.obj", force="mesh"), transform(config["rail"]["initial_center"], (0, 0, np.pi/2)), "#f59e0b", 5000)
    for index, center in enumerate(config["scene"]["bin_centers"]):
        length, width, height = config["scene"]["bin_inner_size"]
        wall = config["scene"]["bin_wall"]
        for sign in (-1, 1):
            add(trimesh.creation.box((wall, width, height)), transform(np.asarray(center)+[sign*length/2, 0, height/2]), "#3d7191")
            add(trimesh.creation.box((length, wall, height)), transform(np.asarray(center)+[0, sign*width/2, height/2]), "#3d7191")
        axes.text(center[0], center[1], center[2]+height+.04, "Raw parts" if index == 0 else "Finished parts")
    axes.set(xlim=(-.3, 1.1), ylim=(-.8, .8), zlim=(0, 1.5), xlabel="Front X (m)", ylabel="Left Y (m)", zlabel="Up Z (m)")
    axes.set_box_aspect((1.4, 1.6, 1.5)); axes.view_init(elev=24, azim=-135)
    axes.set_title("Prepared assets — layout inspection only")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight"); plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
