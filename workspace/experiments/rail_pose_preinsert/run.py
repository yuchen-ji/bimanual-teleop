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
from host_rpc import WorkerLease


class ExperimentStopped(RuntimeError):
    """A recorded experiment check stopped the run; details are in its output."""


def settings(path):
    from reload_settings import load_settings
    return load_settings(path)


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
    parser.add_argument("--grasp", choices=("fixed", "physics", "contact_fixed"))
    parser.add_argument("--mode", choices=("step", "auto"), default="step")
    parser.add_argument("--until", choices=("four_fingers", "grasp", "preinsert"), help="Override the config's stop stage (default: preinsert)")
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
            keep_gui_on_error=not args.headless,
            action_source_factory=lambda config, model: PreinsertController(config, model, task, args.output,
                runtime, mode=args.mode, headless=args.headless, perception=args.perception, grasp=args.grasp))
        return

    if not args.preflight and not args.headless and not os.environ.get("DISPLAY"):
        parser.error("Use a desktop terminal, --display, or --headless --mode auto")
    config, task = settings(args.config)
    until_override = args.until
    args.until = args.until or task.get("until", "preinsert")
    task["until"] = args.until
    locked_grasp = task.get("orientation_locked_grasp") and args.until == "grasp"
    args.grasp = args.grasp or ("contact_fixed" if locked_grasp else "physics")
    if locked_grasp and args.grasp != "contact_fixed":
        parser.error("Orientation-locked grasp uses contact_fixed: physical closure followed by a fixed constraint")
    if args.grasp == "contact_fixed" and not locked_grasp:
        parser.error("contact_fixed requires orientation_locked_grasp and --until grasp")
    from task_settings import validate_task
    validate_task(task)
    if args.until == "four_fingers" and (args.grasp != "physics" or args.preflight):
        parser.error("Four-finger closure requires physical finger contacts; omit --preflight and use --grasp physics")
    output_root = PROJECT / "workspace/outputs/rail_pose_preinsert"
    output = (args.output or output_root / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])).resolve()
    if not output.is_relative_to((PROJECT / "workspace/outputs").resolve()):
        parser.error("Output must be inside this project's workspace/outputs")
    output.mkdir(parents=True, exist_ok=False)
    print(f"Experiment output: {output}", flush=True)
    from bridge import Bridge
    bridge = Bridge(config, task, output, args.perception)
    from reload_settings import ReloadSettings, runtime_bundle
    reloads = ReloadSettings(args.config, bridge, until_override)
    runtime = runtime_bundle(bridge, args.config.resolve())
    import yaml
    effective = copy.deepcopy(config); effective["experiment"] = task
    effective["network"]["token"] = ""
    (output / "effective.yaml").write_text(yaml.safe_dump(effective, allow_unicode=True))
    (output / "input.yaml").write_bytes(args.config.read_bytes())
    (output / "hand_profiles.json").write_bytes(Path(bridge.config["hand_reference"]).read_bytes())
    write_json(output / "runtime.json", runtime)
    from rail_preinsert.core import digest
    write_json(output / "metadata.json", {"test_data": True, "simulation_only": True,
               "grasp": args.grasp, "perception": args.perception, "end_stage": args.until,
               "real_grasp_validated": False, "real_control_calibrated": False,
               "hand_reference": bridge.config["hand_reference"],
               "file_sha256": {str(p): digest(p) for p in [*HERE.glob("*.py"), args.config,
                    Path(bridge.config["hand_reference"]),
                    Path(bridge.config["urdf"]), Path(bridge.config["mesh"]),
                    PROJECT / "bimanual_teleop/sim/engine.py"]}})
    vision = None
    vision_signature = None
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
        with (output / "runtime.log").open("w") as log, WorkerLease(output) as worker:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            previous = None
            try:
                while process.poll() is None:
                    request_path = output / "rpc/request.json"
                    if request_path.is_file():
                        request = json.loads(request_path.read_text())
                        if request["id"] != previous:
                            previous = request["id"]
                            worker.state = {"status": "busy", "request_id": previous}
                            print(f"Simulation request: {request['stage']} ({args.perception})", flush=True)
                            try:
                                if request["stage"] == "reload":
                                    response = reloads.prepare(request["id"], current_revision=request.get("revision", "startup"),
                                                               rebuild=request.get("rebuild", False))
                                    if response.get("operation") == "rebuild" and vision is not None:
                                        # A cached mesh/worker config must not survive a CAD change.
                                        active = reloads.bridge_for({"revision": request["id"]})
                                        from rail_preinsert.core import digest
                                        signature = (active.config["mesh"], digest(active.config["mesh"]),
                                                     active.config["foundationpose_root"], active.config.get("foundationpose_python"),
                                                     active.config.get("foundationpose_env"))
                                        if signature != vision_signature:
                                            vision.close()
                                            vision = None
                                elif request["stage"] == "four_fingers_check":
                                    from four_finger_geometry import FourFingerGeometry
                                    active = reloads.bridge_for(request)
                                    if not hasattr(active, "four_finger_geometry"):
                                        active.four_finger_geometry = FourFingerGeometry(active.geometry.robot,
                                            active.config["urdf"], active.config["manifest"])
                                    geometry = active.four_finger_geometry.check(request["state"])
                                    write_json(output / "trials" / f"trial_{request['trial']:03d}" /
                                               "four_fingers_geometry" / f"{request['id']}.json",
                                               {"state": request["state"], "geometry": geometry})
                                    response = {"ok": True, "geometry": geometry}
                                elif request["stage"].startswith("locked_grasp_"):
                                    from locked_grasp_host import handle
                                    active = reloads.bridge_for(request)
                                    response = handle(active, request)
                                    write_json(output / "trials" / f"trial_{request['trial']:03d}" / "locked_grasp" /
                                               f"{request['id']}.json", {"request": request, "response": response})
                                else:
                                    active = reloads.bridge_for(request)
                                    observation = None
                                    if args.perception == "foundationpose":
                                        from vision_worker import (
                                            PersistentVision,
                                            infer_rendered,
                                        )
                                        if vision is None:
                                            vision = PersistentVision(output, active.config)
                                            from rail_preinsert.core import digest
                                            vision_signature = (active.config["mesh"], digest(active.config["mesh"]),
                                                                active.config["foundationpose_root"], active.config.get("foundationpose_python"),
                                                                active.config.get("foundationpose_env"))
                                        observation = infer_rendered(request, output, active.config, worker=vision,
                                                                     cancelled=lambda: process.poll() is not None,
                                                                     simulation=active.simulation)
                                    response = active.handle(request, observation)
                            except Exception as error:  # noqa: BLE001 -- Report RPC failures so the simulator stops safely.
                                response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
                            write_json(output / "rpc/response.json", {"id": previous, **response})
                            worker.state = {"status": "idle", "request_id": None}
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
        accepted = ("simulation_preinsert_passed", "simulation_grasp_passed", "grasp_preview_complete",
                    "supported_four_finger_contact", "four_finger_preview_complete", "constrained_grasp_complete")
        if result.get("status") not in accepted:
            if result.get("status") not in ("failed", "interrupted"):
                result.update(status="interrupted", reason="Simulation exited without final acceptance")
                write_json(result_path, result)
            raise ExperimentStopped(f"Experiment {result.get('status')} during {result.get('phase', 'startup')}: "
                                    f"{result.get('reason')}\nResult: {result_path}\nLog: {output / 'runtime.log'}")
        print(f"{result['status']} ({args.perception}, {args.grasp}); no insertion executed", flush=True)
    except BaseException as error:
        path = output / "result.json"
        result = json.loads(path.read_text()) if path.is_file() else {}
        if result.get("status") in (None, "running"):
            result.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                          reason=f"{type(error).__name__}: {error}", test_data=True)
            write_json(path, result)
        raise
    finally:
        if vision is not None:
            vision.close()


if __name__ == "__main__":
    try:
        main()
    except ExperimentStopped as error:
        print(str(error), file=sys.stderr, flush=True)
        sys.exit(1)
