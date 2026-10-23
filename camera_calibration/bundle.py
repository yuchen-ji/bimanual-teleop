"""Build the standalone USB-PC sampler, including its wrist reference and docs."""

import json
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera_calibration.storage import ROOT, SIM_ROOT, digest


def build(output):
    project = ROOT.parent
    prefix = "camera-calibration-local"
    sources = [
        "bimanual_teleop/__init__.py",
        "bimanual_teleop/devices/__init__.py",
        "bimanual_teleop/devices/tianji/__init__.py",
        "bimanual_teleop/devices/tianji/sdk.py",
        "bimanual_teleop/vendor/tianji/LICENSE",
        "bimanual_teleop/vendor/tianji/README.md",
        "bimanual_teleop/vendor/tianji/manifest.json",
        "bimanual_teleop/vendor/tianji/SDK_PYTHON/fx_robot.py",
        "bimanual_teleop/vendor/tianji/SDK_PYTHON/libMarvinSDK.so",
    ]
    sources += [str(p.relative_to(project)) for p in sorted(ROOT.glob("*.py"))]
    sources += [
        str(p.relative_to(project))
        for p in sorted((ROOT / "docs").glob("*"))
        if p.is_file()
    ]
    sources += ["camera_calibration/README.md", "camera_calibration/requirements.txt"]
    environment = {
        "name": "bimanual-teleop",
        "channels": ["defaults"],
        "dependencies": [
            "python=3.10",
            "pip",
            "tk",
            {
                "pip": [
                    "numpy==1.26.4",
                    "scipy>=1.10,<2",
                    "matplotlib>=3.10,<4",
                    "Pillow>=10,<13",
                    "PyYAML>=6,<7",
                    "opencv-python-headless==4.11.0.86",
                    "pyrealsense2==2.58.4.10922",
                ]
            },
        ],
        "variables": {"PIP_USER": "false", "PYTHONNOUSERSITE": "1"},
    }
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "scope": "local read-only wrist/external capture; solve/export run in full project",
        "files": {},
    }
    with ZipFile(output, "x", ZIP_DEFLATED) as archive:
        for name in sources:
            archive.writestr(f"{prefix}/{name}", (project / name).read_bytes())
            manifest["files"][name] = digest(project / name)
        reference = SIM_ROOT / "configs/wrist_camera_calibration_reference.json"
        archive.writestr(f"{prefix}/wrist-result.json", reference.read_bytes())
        manifest["files"]["wrist-result.json"] = digest(reference)
        archive.writestr(
            f"{prefix}/environment.yml", json.dumps(environment, indent=2) + "\n"
        )
        archive.writestr(
            f"{prefix}/README.txt",
            "Local Linux x86_64 USB PC.\n"
            "conda env create -f environment.yml\nconda activate bimanual-teleop\n"
            "Wrist: python camera_calibration/run.py wrist collect --output capture_wrist\n"
            "External: python camera_calibration/run.py external collect --wrist-result wrist-result.json "
            "--side left --output capture_external\n"
            "The sampler reads images/measured joints. Move the robot using the pendant.\n"
            "Upload the whole capture folder and wrist-result.json; run solve/export in the full project.\n"
            "Chinese instructions: camera_calibration/README.md\n",
        )
        archive.writestr(
            f"{prefix}/bundle-manifest.json", json.dumps(manifest, indent=2) + "\n"
        )
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(build(parser.parse_args().output))
