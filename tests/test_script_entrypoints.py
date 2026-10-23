"""Direct-file entrypoints preserve arguments and need no current-directory setup."""

import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bimanual_teleop.sim import run as simulation

PROJECT = Path(__file__).resolve().parents[1]


class ScriptEntrypointTests(unittest.TestCase):
    def test_simulation_dispatch_preserves_arguments_return_and_original_argv(self):
        original, called = sys.argv, []
        module = SimpleNamespace(main=lambda: called.append(sys.argv.copy()) or 7)
        with patch.object(
            simulation.importlib, "import_module", return_value=module
        ) as load:
            self.assertEqual(
                simulation.main(["engine", "--standalone", "--steps", "5"]), 7
            )
        load.assert_called_once_with("bimanual_teleop.sim.engine")
        self.assertEqual(called[0][1:], ["--standalone", "--steps", "5"])
        self.assertIs(sys.argv, original)

    def test_default_dispatch_and_unknown_tool(self):
        module = SimpleNamespace(main=lambda: None)
        with patch.object(
            simulation.importlib, "import_module", return_value=module
        ) as load:
            simulation.main(["--headless", "--steps", "3"])
        load.assert_called_once_with("bimanual_teleop.sim.__main__")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            simulation.main(["unknown-tool"])
        self.assertEqual(raised.exception.code, 2)

    def test_all_simulation_tool_help_from_outside_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            for tool in simulation.TOOLS:
                with self.subTest(tool=tool):
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(PROJECT / "bimanual_teleop/sim/run.py"),
                            tool,
                            "--help",
                        ],
                        cwd=directory,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)

    def test_calibration_tests_and_preview_help_from_outside_checkout(self):
        scripts = (
            "camera_calibration/run.py",
            "camera_calibration/bundle.py",
            "scripts/run_tests.py",
            "bimanual_teleop/visualization/realsense.py",
            "tests/test_recording_golden.py",
        )
        with tempfile.TemporaryDirectory() as directory:
            for script in scripts:
                with self.subTest(script=script):
                    result = subprocess.run(
                        [sys.executable, str(PROJECT / script), "--help"],
                        cwd=directory,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)

    def test_dependency_installer_uses_selected_interpreter(self):
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT / "bimanual_teleop/sim/docker/install_packages.py"),
                "--version",
            ],
            cwd=PROJECT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(Path(sys.prefix)), result.stdout)


if __name__ == "__main__":
    unittest.main()
