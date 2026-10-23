"""Small file helpers shared by capture and offline workflows."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SIM_ROOT = ROOT.parent / "bimanual_teleop/sim"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_wrist_reference(path, side):
    from .geometry import checked_pose

    report = json.loads(Path(path).read_text())
    if report.get("synthetic") or side not in ("left", "right"):
        raise ValueError("A measured left/right wrist calibration is required")
    item = report["calibrations"][side]
    if not item.get("passed") or not report.get("urdf_sha256"):
        raise ValueError(
            "Wrist reference must have passed validation and identify its URDF"
        )
    checked_pose(item["T_flange_sensor_opencv"])
    return report, item
