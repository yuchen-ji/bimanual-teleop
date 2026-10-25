"""FoundationPose process for offline simulated RGB-D only; no device drivers."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def serve(directory, engine, measure_scene, *, parent_alive=lambda: True):
    """Keep one model, resetting tracking inside each independent measurement."""
    import json

    from common import write_json
    previous = None
    while parent_alive():
        path = directory / "request.json"
        request = json.loads(path.read_text()) if path.is_file() else {}
        if request.get("id") in (None, previous):
            time.sleep(.03)
            continue
        previous = request["id"]
        if request["command"] == "shutdown":
            break
        started = time.monotonic()
        try:
            if request["command"] != "measure" or not 1 <= request["samples"] <= 10:
                raise ValueError("Invalid offline measurement request")
            destination = Path(request["output"])
            destination.mkdir(parents=True, exist_ok=False)
            observation = measure_scene(engine, Path(request["scene"]), destination, request["samples"])
            observation.update(test_data=True, worker_pid=os.getpid(), engine_load_count=1)
            write_json(destination / "observation.json", observation)
            response = {"ok": True, "observation": observation}
        except Exception as error:  # noqa: BLE001 -- A failed measurement must not unload the model.
            import traceback
            traceback.print_exc()
            engine.reset()
            response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        write_json(directory / "response.json", {
            "id": previous, "worker_pid": os.getpid(), "elapsed_s": time.monotonic()-started, **response})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundationpose-root", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--serve", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args()
    root = args.foundationpose_root.resolve()
    # Reuse only the established offline adapter and model; never construct Camera.
    live = Path(__file__).resolve().parents[4] / "diffusion_policy/workspace/rail_pose_insert"
    sys.path[:0] = [str(live), str(root / "workspace/src"), str(root)]
    import json

    from common import write_json
    from perception_worker import SceneCamera, measure
    from rail_pose.engine import PoseEngine

    directory = args.serve.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        engine = PoseEngine(args.mesh.resolve(), json.loads((root / "workspace/configs/default.json").read_text()),
                            directory / "debug")
        write_json(directory / "ready.json", {"ready": True, "worker_pid": os.getpid(),
                   "engine_load_count": 1, "load_s": time.monotonic()-started})
        def parent_alive():
            try:
                os.kill(args.parent_pid, 0)
            except ProcessLookupError:
                return False
            return True
        def measure_scene(model, scene, destination, samples):
            camera = SceneCamera(scene, 0)
            try:
                return measure(camera, model, destination / "measurement", samples)
            finally:
                camera.close()
        serve(directory, engine, measure_scene, parent_alive=parent_alive)
    except Exception as error:
        write_json(directory / "ready.json", {"ready": False, "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
