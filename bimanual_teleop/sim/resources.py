"""Audit the complete source/runtime model set, with optional explicit pruning."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from . import ROOT


def _inside_assets(path):
    path = path.resolve()
    path.relative_to((ROOT / "assets").resolve())
    return path


def mesh_files(urdf):
    files = set()
    for item in ET.parse(urdf).getroot().findall(".//mesh"):
        filename = item.get("filename")
        path = (urdf.parent.parent / filename.split("/", 3)[-1] if filename.startswith("package://")
                else urdf.parent / filename)
        files.add(_inside_assets(path))
    return files


def source_components():
    from .assets import ROBOT_SOURCE, HAND_SOURCE
    hands = [HAND_SOURCE / "body/urdf" / f"{side}_with_mount.urdf" for side in ("left", "right")]
    return {"robot": {ROBOT_SOURCE, *mesh_files(ROBOT_SOURCE)},
            "hands": {*hands, *(path for hand in hands for path in mesh_files(hand))}}


def tree_digest(files):
    from .assets import digest
    checksum = hashlib.sha256()
    for path in sorted(files):
        checksum.update(str(path.relative_to(ROOT)).encode())
        checksum.update(b"\0"+digest(path).encode()+b"\n")
    return checksum.hexdigest()


def required_files():
    from .cad_assets import SOURCE_NAMES
    source = ROOT / "assets/source"
    generated = ROOT / "assets/generated"
    result = {source / "provenance.json", generated / "manifest.json", generated / "task/cad_manifest.json",
              generated / "robot.urdf"}
    for files in source_components().values():
        result.update(files)
    result.update(source / "rail_task" / name for name in SOURCE_NAMES.values())
    result.update(mesh_files(generated / "robot.urdf"))
    manifest = json.loads((generated / "manifest.json").read_text())

    def meshes(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "mesh" and isinstance(child, str):
                    result.add(_inside_assets(generated / child))
                else:
                    meshes(child)
        elif isinstance(value, list):
            for child in value:
                meshes(child)

    meshes(manifest["cad"]["models"])
    result.update(generated / "rail" / name for name in manifest["rail"]["colliders"])
    return {_inside_assets(path) for path in result}


def audit(*, check_git=False):
    expected = required_files()
    actual = {path.resolve() for folder in ("source", "generated")
              for path in (ROOT / "assets" / folder).rglob("*") if path.is_file()}
    missing = sorted(str(path.relative_to(ROOT)) for path in expected if not path.is_file())
    unused = sorted(str(path.relative_to(ROOT)) for path in actual-expected)
    report = {"passed": not missing and not unused, "required_files": len(expected),
              "required_bytes": sum(path.stat().st_size for path in expected if path.is_file()),
              "missing": missing, "unused": unused}
    if check_git:
        project = ROOT.parents[1]
        tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=project).decode().split("\0"))
        report["not_tracked"] = sorted(str(path.relative_to(project)) for path in expected
                                       if str(path.relative_to(project)) not in tracked)
        report["passed"] &= not report["not_tracked"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-git", action="store_true")
    parser.add_argument("--prune", action="store_true", help="Delete unreferenced files within assets/source and assets/generated")
    args = parser.parse_args()
    report = audit(check_git=args.check_git)
    if args.prune:
        if report["missing"]:
            raise ValueError("Required resources are missing; refusing to prune an incomplete model set")
        removed = report["unused"]
        removed_bytes = sum((ROOT / name).stat().st_size for name in removed)
        for name in removed:
            (ROOT / name).unlink()
        for folder in sorted((ROOT / "assets").rglob("*"), key=lambda path: len(path.parts), reverse=True):
            if folder.is_dir() and not any(folder.iterdir()):
                folder.rmdir()
        report = {**audit(check_git=args.check_git), "removed_files": len(removed), "removed_bytes": removed_bytes}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
