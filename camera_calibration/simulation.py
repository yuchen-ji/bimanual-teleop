"""Explicit adapters for applying results to simulation, separate from solvers."""

import json
from pathlib import Path

from .storage import SIM_ROOT, digest


def apply_result(result_file):
    from bimanual_teleop.sim._dependencies import yaml
    from bimanual_teleop.sim.config import load_config, sim_path, validate

    result_file = Path(result_file).resolve()
    report = json.loads(result_file.read_text())
    if (
        report.get("schema") != "wrist_handeye_result_v1"
        or not report.get("passed")
        or report.get("synthetic")
    ):
        raise ValueError(
            "Only a measured calibration that passed held-out checks can be applied"
        )
    config = load_config()
    if digest(sim_path(config["robot"]["urdf"])) != report["urdf_sha256"]:
        raise ValueError(
            "Robot URDF changed since calibration; solve again before applying"
        )
    updates = {side: item["settings"] for side, item in report["calibrations"].items()}
    if not updates or set(updates) - {"left", "right"}:
        raise ValueError("Invalid wrist side in calibration")
    for side, value in updates.items():
        if not report["calibrations"][side]["passed"]:
            raise ValueError(f"Calibration failed for {side}")
        config["cameras"]["wrist"][side].update(value)
    validate(config)
    destination = SIM_ROOT / "configs/wrist_camera_calibrated.yaml"
    existing = {}
    if destination.exists():
        existing = yaml.safe_load(destination.read_text())["cameras"]["wrist"]
    existing.update(updates)
    payload = {"cameras": {"wrist": existing}}
    default_file = SIM_ROOT / "configs/default.yaml"
    text = default_file.read_text()
    before = "  wrist_calibration_file: null"
    after = "  wrist_calibration_file: configs/wrist_camera_calibrated.yaml"
    if before not in text and after not in text:
        raise ValueError(
            "Unexpected default calibration setting; refusing to rewrite unrelated configuration"
        )
    destination.write_text(
        "# Measured wrist hand-eye calibration; OpenCV optical coordinates in bracket frame.\n"
        + yaml.safe_dump(payload, sort_keys=False)
    )
    reference = {
        "source_result_sha256": digest(result_file),
        "urdf_sha256": report["urdf_sha256"],
        "board": report["board"],
        "capture_provenance": report.get("capture_provenance", {}),
        "calibrations": {
            side: {
                key: value
                for key, value in item.items()
                if key not in ("samples", "reference")
            }
            for side, item in report["calibrations"].items()
        },
    }
    (SIM_ROOT / "configs/wrist_camera_calibration_reference.json").write_text(
        json.dumps(reference, indent=2) + "\n"
    )
    default_file.write_text(text.replace(before, after, 1))
    load_config()  # Ensure the default launch resolves and validates the actual saved file.
    return destination
