"""Raw-image workflow, frame conventions, export and portable CLI regression."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image

from camera_calibration.board import (
    CircleBoard,
    camera_matrix,
    estimate_board_pose,
    projected_points,
)
from camera_calibration.bundle import build
from camera_calibration.external_workflow import export_result, solve_session
from camera_calibration.geometry import pose_errors
from camera_calibration.robot import load_model, world_flange
from camera_calibration.storage import ROOT, SIM_ROOT, digest, write_json
from camera_calibration.tests.test_external import scene


def render_board(pose, metadata):
    image = np.full((1440, 1920, 3), 30, dtype=np.uint8)
    theta = np.linspace(0, 2 * np.pi, 80)
    for point in CircleBoard().points():
        circle = (
            point
            + np.c_[0.018 * np.cos(theta), 0.018 * np.sin(theta), np.zeros(len(theta))]
        )
        pixels = projected_points(
            circle, pose, camera_matrix(metadata), metadata["rgb_intrinsics"]["coeffs"]
        )
        cv2.fillPoly(image, [np.round(pixels * 3).astype(np.int32)], (220, 220, 220))
    return cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)


class WorkflowTests(unittest.TestCase):
    def test_raw_images_fk_solve_and_frames(self):
        model, urdf_hash = load_model()
        measured = json.loads(
            (SIM_ROOT / "configs/wrist_camera_calibration_reference.json").read_text()
        )
        handeye = np.array(measured["calibrations"]["left"]["T_flange_sensor_opencv"])
        q = np.zeros(7)
        anchor = world_flange(model, "left", q) @ handeye
        views, wrist, external, local_camera = scene()
        wrist["output_rotation_deg"] = 180  # Source images are still raw sensor pixels.
        reference = {
            "board": asdict(CircleBoard()),
            "urdf_sha256": urdf_hash,
            "calibrations": {
                "left": {
                    "passed": True,
                    "camera": wrist,
                    "T_flange_sensor_opencv": handeye.tolist(),
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "capture"
            source.mkdir()
            reference_path = root / "wrist.json"
            write_json(reference_path, reference)
            session = {
                "schema": "external_camera_capture_v1",
                "status": "captured",
                "stationary_only": True,
                "images": "raw RGB sensor orientation",
                "side": "left",
                "synthetic": True,
                "board": reference["board"],
                "cameras": {"wrist": wrist, "external": external},
                "wrist_calibration_sha256": digest(reference_path),
            }
            write_json(source / "session.json", session)
            records = []
            for view in views:
                wrist_board, _ = estimate_board_pose(view.wrist_pixels, wrist)
                local_board = view.world_wrist @ wrist_board
                row = {
                    "group": view.group,
                    "joints_rad": q.tolist(),
                    "joint_window_span_rad": [0.0] * 7,
                    "time_error_ns": 0,
                    "pair_time_error_ns": 0,
                    "wrist_time_ns": (view.group + 1) * 1_000_000_000,
                    "external_time_ns": (view.group + 1) * 1_000_000_000,
                    "joint_time_ns": (view.group + 1) * 1_000_000_000,
                }
                for name, pose, meta in (
                    ("wrist", local_board, wrist),
                    ("external", np.linalg.inv(local_camera) @ local_board, external),
                ):
                    path = source / f"{name}_{view.group:04d}.png"
                    Image.fromarray(render_board(pose, meta)).save(path)
                    row[f"{name}_image"] = path.name
                    row[f"{name}_sha256"] = digest(path)
                records.append(row)
            (source / "samples.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in records)
            )
            report = solve_session(source, reference_path, root / "result")
            self.assertTrue(report["passed"], report["errors"])
            truth = anchor @ local_camera
            distance, angle = pose_errors(
                np.array(report["T_world_camera_opencv"]), truth
            )
            self.assertLess(distance, 0.004)
            self.assertLess(angle, 0.2)
            np.testing.assert_allclose(
                np.array(report["T_world_robot_base"]) @ report["T_base_camera_opencv"],
                report["T_world_camera_opencv"],
                atol=1e-10,
            )
            self.assertEqual(len(list((root / "result").glob("validation_*.png"))), 5)
            with self.assertRaisesRegex(ValueError, "measured"):
                export_result(root / "result/result.json", root / "synthetic.yaml")
            # The following measured-export branch uses this isolated synthetic
            # fixture only; it never changes the real project's configuration.
            report["synthetic"] = False
            write_json(root / "export_fixture.json", report)
            snapshot = digest(SIM_ROOT / "configs/default.yaml")
            export_result(root / "export_fixture.json", root / "external.json")
            export_result(
                root / "export_fixture.json", root / "head.yaml", simulation=True
            )
            from bimanual_teleop.sim.config import load_config

            config = load_config(root / "head.yaml")
            np.testing.assert_allclose(
                config["cameras"]["head"]["position_world_m"], truth[:3, 3], atol=0.004
            )
            self.assertEqual(snapshot, digest(SIM_ROOT / "configs/default.yaml"))
            with self.assertRaises(FileExistsError):
                export_result(root / "export_fixture.json", root / "external.json")
            # A bad held-out image cannot disappear before the split.
            (source / records[0]["external_image"]).write_bytes(b"modified")
            failed = solve_session(source, reference_path, root / "bad")
            self.assertFalse(failed["passed"])
            self.assertIn("Held-out group 0", failed["errors"][0])
            self.assertTrue((root / "bad/result.json").is_file())
            with self.assertRaises(ValueError):
                export_result(root / "bad/result.json", root / "bad.yaml")

    def test_mismatched_reference_and_failed_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(
                root / "result.json",
                {"schema": "external_camera_result_v1", "passed": False},
            )
            with self.assertRaises(ValueError):
                export_result(root / "result.json", root / "result.yaml")
            source = root / "capture"
            source.mkdir()
            write_json(source / "session.json", {})
            result = solve_session(source, "missing.json", root / "diagnostic")
            self.assertFalse(result["passed"])
            self.assertTrue((root / "diagnostic/report.md").is_file())

    def test_new_legacy_and_portable_entrypoints(self):
        project = ROOT.parent
        for script, arguments in (
            ("camera_calibration/run.py", ["--help"]),
            ("camera_calibration/run.py", ["external", "collect", "--help"]),
            ("bimanual_teleop/sim/wrist_calibration/__main__.py", ["--help"]),
            ("bimanual_teleop/sim/wrist_calibration/__main__.py", ["detect", "--help"]),
        ):
            result = subprocess.run(
                [sys.executable, str(project / script), *arguments],
                cwd=project.parent,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = build(root / "local.zip")
            with ZipFile(archive) as bundle:
                bundle.extractall(root)
            local = root / "camera-calibration-local"
            manifest = json.loads((local / "bundle-manifest.json").read_text())
            for path, expected in manifest["files"].items():
                self.assertEqual(digest(local / path), expected)
            env = dict(os.environ)
            env.pop("PYTHONPATH", None)
            result = subprocess.run(
                [
                    sys.executable,
                    "camera_calibration/run.py",
                    "external",
                    "collect",
                    "--help",
                ],
                cwd=local,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import camera_calibration.external_collect; "
                        "from bimanual_teleop.devices.tianji.sdk import ControlSDK"
                    ),
                ],
                cwd=local,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
