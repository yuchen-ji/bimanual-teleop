"""Build a small standalone local sampler without copying the whole project."""

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def build(output):
    project = Path(__file__).resolve().parents[3]
    package = "wrist-calibration-local"
    sources = [
        "bimanual_teleop/__init__.py", "bimanual_teleop/devices/__init__.py",
        "bimanual_teleop/devices/tianji/__init__.py", "bimanual_teleop/devices/tianji/sdk.py",
        "bimanual_teleop/vendor/tianji/LICENSE", "bimanual_teleop/vendor/tianji/README.md",
        "bimanual_teleop/vendor/tianji/manifest.json",
        "bimanual_teleop/vendor/tianji/SDK_PYTHON/fx_robot.py",
        "bimanual_teleop/vendor/tianji/SDK_PYTHON/libMarvinSDK.so",
        "bimanual_teleop/sim/__init__.py",
    ]
    sources += [str(path.relative_to(project)) for path in sorted(Path(__file__).parent.glob("*.py"))
                if path.name in ("__init__.py", "__main__.py", "collect.py", "core.py")]
    sources += ["bimanual_teleop/sim/docs/wrist-handeye.md", "bimanual_teleop/sim/docs/wrist-handeye-manual-template.csv"]
    # The local bundle only provides collect/detect. Offline solve/apply run in the server project.
    environment = {"name": "bimanual-teleop", "channels": ["defaults"],
        "dependencies": ["python=3.10", "pip", "tk", {"pip": [
            "numpy==1.26.4", "scipy>=1.10,<2", "matplotlib>=3.10,<4", "Pillow>=10,<13",
            "opencv-python-headless==4.11.0.86", "pyrealsense2==2.58.4.10922"]}],
        "variables": {"PIP_USER": "false", "PYTHONNOUSERSITE": "1"}}
    # JSON is valid YAML and avoids needing PyYAML in the portable bootstrap.
    output = Path(output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"scope": "standalone local read-only wrist sampler; no project install required", "files": {}}
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for name in sources:
            raw = (project/name).read_bytes()
            manifest["files"][name] = hashlib.sha256(raw).hexdigest()
            archive.writestr(f"{package}/{name}", raw)
        archive.writestr(f"{package}/environment.yml", json.dumps(environment, indent=2)+"\n")
        archive.writestr(f"{package}/README.txt",
            "Ubuntu 22.04 x86_64 local USB PC. No full bimanual-teleop checkout needed.\n"
            "First setup: conda env create -f environment.yml\n"
            "Existing environment: conda env update -n bimanual-teleop -f environment.yml\n"
            "conda activate bimanual-teleop\n"
            "python -m bimanual_teleop.sim.wrist_calibration collect --samples 20 --output capture_real\n"
            "Robot address defaults to 192.168.1.190; override with --robot-ip if necessary.\n"
            "Keep board fixed, show all 20 circles, vary wrist rotations, hold still 1-2 s.\n"
            "Pose changes are made by the operator with the pendant/drag mode.\n"
            "Sampler only connects, polls measured joints and disconnects; it issues no motion/state commands.\n"
            "Q/window close/Ctrl+C ends capture and saves it. Upload the entire capture_real directory.\n"
            "Full Chinese instructions: bimanual_teleop/sim/docs/wrist-handeye.md\n")
        archive.writestr(f"{package}/bundle-manifest.json", json.dumps(manifest, indent=2)+"\n")
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(build(parser.parse_args().output))
