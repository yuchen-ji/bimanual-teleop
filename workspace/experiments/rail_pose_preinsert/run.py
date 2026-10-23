"""Launch an isolated simulation; host performs perception and the existing live planner."""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(PROJECT))
from common import pose, seated_rail, write_json


def settings(path):
    import yaml

    from bimanual_teleop.sim.config import _merge, load_config, validate
    document = yaml.safe_load(Path(path).read_text())
    config = load_config()
    _merge(config, document["simulation"])
    validate(config)
    task = document["task"]
    if task["end_stage"] != "preinsert" or task["preinsert_offset_m"] != .02:
        raise ValueError("This experiment stops at a 20 mm preinsert offset")
    for key, value in task.items():
        if (key.endswith(("_s", "_rad", "_m", "_rad_s", "_rad_s2", "_n")) and not isinstance(value, list)
                and (not isinstance(value, (int, float)) or not __import__("math").isfinite(value) or value <= 0)):
            raise ValueError(f"Invalid task setting: {key}")
    return config, task


def preflight(bridge, config, task):
    import numpy as np
    from rail_preinsert.core import matrix
    q = np.r_[np.radians(config["robot"]["ready_deg"]["left"]),
              np.radians(config["robot"]["ready_deg"]["right"]), bridge.references["grasp"], np.zeros(20)]
    F_S = pose(task["rail_in_flange_xyz_m"], task["rail_in_flange_rpy_deg"])
    W_F = np.linalg.inv(bridge.T_B_W) @ matrix(bridge.kinematics.fk("left", tuple(q[:7])))
    W_C = pose(config["cameras"]["head"]["position_world_m"])
    from scipy.spatial.transform import Rotation
    W_C[:3, :3] = Rotation.from_quat(config["cameras"]["head"]["quaternion_xyzw"]).as_matrix()
    manifest = json.loads(Path(bridge.config["manifest"]).read_text())
    def request(stage, W_S):
        frames = [{"T_W_C": W_C.tolist(), "T_W_S": W_S.tolist(),
                   "state": {"q": q.tolist(), "time_ns": t}} for t in [10_000_000, 20_000_000, 30_000_000]]
        return {"stage": stage, "frames": frames,
                "left_joint_history": [[f["state"]["time_ns"], q[:7].tolist()] for f in frames]}
    bridge.handle(request("teach", seated_rail(config, manifest)))
    result = bridge.handle(request("measure", W_F @ F_S))
    write_json(bridge.output / "result.json", {"status": "preflight_passed", "test_data": True,
               "physical_execution": False, "perception": "oracle", "end_stage": "preinsert",
               "duration_s": result["plan"]["duration_s"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--perception", choices=("oracle", "foundationpose"), default="oracle")
    parser.add_argument("--grasp", choices=("fixed", "physics"), default="physics")
    parser.add_argument("--mode", choices=("step", "auto"), default="step")
    parser.add_argument("--until", choices=("grasp", "preinsert"), default="preinsert")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--display")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inside-container", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.headless and args.mode != "auto":
        parser.error("Use --mode auto with --headless")
    if args.preflight and args.perception != "oracle":
        parser.error("FoundationPose requires rendered RGB-D; use a physics run")
    if args.display:
        os.environ["DISPLAY"] = args.display
    if args.inside_container:
        import yaml
        from controller import PreinsertController

        from bimanual_teleop.sim.engine import run
        effective = yaml.safe_load((args.output / "effective.yaml").read_text())
        task = effective.pop("experiment")
        runtime = json.loads((args.output / "runtime.json").read_text())
        effective["paths"]["recordings"] = str(args.output / "recordings")
        run(effective, None, gui=not args.headless, standalone=True, record=args.record,
            action_source_factory=lambda config, model: PreinsertController(config, model, task, args.output,
                runtime, mode=args.mode, headless=args.headless, perception=args.perception, grasp=args.grasp))
        return

    if not args.preflight and not args.headless and not os.environ.get("DISPLAY"):
        parser.error("Use a desktop terminal, --display, or --headless --mode auto")
    config, task = settings(args.config)
    task["until"] = args.until
    output_root = PROJECT / "workspace/outputs/rail_pose_preinsert"
    output = (args.output or output_root / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])).resolve()
    if not output.is_relative_to((PROJECT / "workspace/outputs").resolve()):
        parser.error("Output must be inside this project's workspace/outputs")
    output.mkdir(parents=True, exist_ok=False)
    print(f"Experiment output: {output}", flush=True)
    from bridge import Bridge
    bridge = Bridge(config, task, output, args.perception)
    manifest = json.loads(Path(bridge.config["manifest"]).read_text())
    goal = seated_rail(config, manifest)
    from scipy.spatial.transform import Rotation
    config["rail"]["initial_center"] = goal[:3, 3].tolist()
    config["rail"]["initial_rpy_deg"] = Rotation.from_matrix(goal[:3, :3]).as_euler("xyz", degrees=True).tolist()
    import yaml
    effective = copy.deepcopy(config); effective["experiment"] = task
    effective["network"]["token"] = ""
    (output / "effective.yaml").write_text(yaml.safe_dump(effective, allow_unicode=True))
    (output / "input.yaml").write_bytes(args.config.read_bytes())
    (output / "hand_profiles.json").write_bytes(Path(bridge.config["hand_reference"]).read_bytes())
    write_json(output / "runtime.json", {"open": bridge.references["open"].tolist(),
               "grasp": bridge.references["grasp"].tolist(), "T_B_W": bridge.T_B_W.tolist(),
               "T_S_CAD": bridge.S_O.tolist(), "goal_rail": goal.tolist()})
    from rail_preinsert.core import digest
    write_json(output / "metadata.json", {"test_data": True, "simulation_only": True,
               "grasp": args.grasp, "perception": args.perception, "end_stage": "preinsert",
               "real_grasp_validated": False, "real_control_calibrated": False,
               "hand_reference": bridge.config["hand_reference"],
               "file_sha256": {str(p): digest(p) for p in [*HERE.glob("*.py"), args.config,
                    Path(bridge.config["hand_reference"]),
                    Path(bridge.config["urdf"]), Path(bridge.config["mesh"]),
                    PROJECT / "bimanual_teleop/sim/engine.py"]}})
    try:
        if args.preflight:
            preflight(bridge, config, task)
            print("preflight_passed: coordinate chain and current-scene collision planning", flush=True)
            return
        mapped = Path("/workspace/bimanual-teleop") / output.relative_to(PROJECT)
        command = [str(PROJECT / "bimanual_teleop/sim/docker/exec-apptainer.sh"), "/isaac-sim/python.sh",
                   "/workspace/bimanual-teleop/workspace/experiments/rail_pose_preinsert/run.py",
                   "--inside-container", "--output", str(mapped), "--mode", args.mode,
                   "--perception", args.perception, "--grasp", args.grasp, "--until", args.until]
        if args.headless:
            command.append("--headless")
        if args.record:
            command.append("--record")
        env = dict(os.environ, BIMANUAL_SIM_GUI="0" if args.headless else "1",
                   APPTAINERENV_OPENBLAS_NUM_THREADS="1", APPTAINERENV_OMP_NUM_THREADS="1")
        with (output / "runtime.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            previous = None
            try:
                while process.poll() is None:
                    request_path = output / "rpc/request.json"
                    if request_path.is_file():
                        request = json.loads(request_path.read_text())
                        if request["id"] != previous:
                            previous = request["id"]
                            print(f"Simulation request: {request['stage']} ({args.perception})", flush=True)
                            try:
                                observation = None
                                if args.perception == "foundationpose":
                                    from vision_worker import infer_rendered
                                    observation = infer_rendered(request, output, bridge.config,
                                                                 cancelled=lambda: process.poll() is not None)
                                response = bridge.handle(request, observation)
                            except Exception as error:  # noqa: BLE001 -- Report RPC failures so the simulator stops safely.
                                response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
                            write_json(output / "rpc/response.json", {"id": previous, **response})
                    time.sleep(.03)
            finally:
                if process.poll() is None:
                    import signal
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=5)
        result_path = output / "result.json"
        result = json.loads(result_path.read_text()) if result_path.is_file() else {}
        accepted = ("simulation_preinsert_passed", "simulation_grasp_passed", "grasp_preview_complete")
        if result.get("status") not in accepted:
            if result.get("status") not in ("failed", "interrupted"):
                result.update(status="interrupted", reason="Simulation exited without final acceptance")
                write_json(result_path, result)
            raise RuntimeError(f"{result.get('reason')}; see {output / 'runtime.log'}")
        print(f"{result['status']} ({args.perception}, {args.grasp}); no insertion executed", flush=True)
    except BaseException as error:
        path = output / "result.json"
        result = json.loads(path.read_text()) if path.is_file() else {}
        if result.get("status") in (None, "running"):
            result.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                          reason=f"{type(error).__name__}: {error}", test_data=True)
            write_json(path, result)
        raise


if __name__ == "__main__":
    main()
