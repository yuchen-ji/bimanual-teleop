import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from common import write_json
from perception_service import serve
from vision_worker import PersistentVision


class VisionLifetimeTests(unittest.TestCase):
    def test_null_interpreter_uses_named_environment_command(self):
        # The live config permits null; it must never become Popen's executable.
        sys.path.insert(0, str(HERE.parents[3] / "diffusion_policy/workspace/rail_pose_insert"))
        with tempfile.TemporaryDirectory() as directory:
            worker = PersistentVision(directory, {"foundationpose_python": None,
                "foundationpose_env": "foundationpose", "foundationpose_root": "/model", "mesh": "/rail.obj"})
            with patch("rail_preinsert.perception.python_command", return_value=["conda", "run", "-n", "foundationpose", "python"]), \
                    patch("vision_worker.subprocess.Popen") as popen, \
                    patch.object(worker, "wait_json", return_value={"ready": True}):
                worker.start(lambda: False)
                self.assertEqual(popen.call_args.args[0][:5], ["conda", "run", "-n", "foundationpose", "python"])
                worker.close()

    def test_measurement_failure_does_not_replace_engine_and_next_request_still_works(self):
        class Engine:
            def __init__(self):
                self.resets = 0
            def reset(self):
                self.resets += 1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = Engine()
            calls = []
            responses = []
            request = {"id": "bad", "command": "measure", "scene": "unused", "output": str(root / "bad"), "samples": 1}
            write_json(root / "request.json", request)
            def measure(supplied, scene, destination, samples):
                self.assertIs(supplied, engine)
                calls.append(destination.name)
                if destination.name == "bad":
                    raise ValueError("invalid depth")
                return {"records": []}
            def next_request():
                path = root / "response.json"
                if not path.is_file():
                    return
                value = json.loads(path.read_text())
                responses.append(value)
                write_json(root / "request.json", {**request, "id": "good", "output": str(root / "good")})
            def alive():
                next_request()
                return len(responses) < 2
            serve(root, engine, measure, parent_alive=alive)
            self.assertEqual(calls, ["bad", "good"])
            self.assertEqual(engine.resets, 1)
            self.assertFalse(responses[0]["ok"])
            self.assertTrue(responses[1]["ok"])
            self.assertEqual(responses[0]["worker_pid"], responses[1]["worker_pid"])

    def test_cancellation_closes_worker_instead_of_accepting_stale_response(self):
        class Process:
            def poll(self):
                return None
        with tempfile.TemporaryDirectory() as directory:
            worker = PersistentVision(directory, {}, timeout_s=.01)
            worker.process, worker.directory = Process(), Path(directory)
            write_json(worker.directory / "response.json", {"id": "old", "ok": True})
            with patch.object(worker, "close") as close, self.assertRaisesRegex(RuntimeError, "cancelled"):
                worker.wait_json("response.json", lambda: True, "new")
            close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
