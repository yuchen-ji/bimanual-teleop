"""Read-only checks for the simulation runtime, assets and optional devices."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

from . import ROOT
from .config import load_config


def check():
    results = {}
    for name in ("numpy", "scipy", "yaml", "zmq", "msgpack", "av", "zarr", "trimesh", "mapbox_earcut", "OCP"):
        results[name] = importlib.util.find_spec(name) is not None
    for key, command in (("gpu", ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
                         ("docker", ["docker", "info", "--format", "{{.ServerVersion}}"]),
                         ("adb", ["adb", "devices", "-l"])):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            results[key] = {"ok": result.returncode == 0, "detail": (result.stdout or result.stderr).strip()[:1000]}
        except (OSError, subprocess.TimeoutExpired) as error:
            results[key] = {"ok": False, "detail": str(error)}
    results["apptainer"] = {"executable": str(ROOT / ".runtime/apptainer/bin/apptainer"),
                            "available": (ROOT / ".runtime/apptainer/bin/apptainer").is_file(),
                            "image_available": (ROOT / ".runtime/isaac-sim-5.1.0.sif").is_file()}
    results["assets"] = (ROOT / "assets/generated/manifest.json").is_file()
    load_config()
    return results


def main():
    argparse.ArgumentParser(description="Inspect simulation requirements without connecting actuators").parse_args()
    print(json.dumps(check(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
