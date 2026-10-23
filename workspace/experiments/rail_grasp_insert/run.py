"""Run one isolated rail experiment on the host or inside the Isaac container."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
from datetime import datetime

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(PROJECT))


def read_runtime_result(output, returncode):
    """Do not leave a crashed or interrupted child reported as still running."""
    path = output/"result.json"
    result = json.loads(path.read_text()) if path.is_file() else {
        "status": "failed", "reason": "runtime_exited_before_startup", "runtime_returncode": returncode, "test_data": True}
    if not path.is_file():
        path.write_text(json.dumps(result, indent=2))
    if result["status"] == "running":
        (output/"unfinished-result.json").write_text(json.dumps(result, indent=2))
        result.update(status="interrupted", reason="runtime_exited_without_final_report", runtime_returncode=returncode)
        path.write_text(json.dumps(result, indent=2))
        if result.get("trial"):
            (output/result["trial"]/"result.json").write_text(json.dumps(result, indent=2))
    return result


def load_experiment(path):
    import yaml
    from bimanual_teleop.sim.config import load_config, _merge, validate
    document = yaml.safe_load(Path(path).read_text())
    if not isinstance(document, dict) or set(document) != {"simulation", "task"}:
        raise ValueError("Experiment configuration requires simulation and task sections")
    config = load_config()
    _merge(config, document["simulation"])
    validate(config)
    task = copy.deepcopy(document["task"])
    import numpy as np
    for name in ("open_hand_rad", "wrap_hand_rad", "closed_hand_rad"):
        q = np.asarray(task[name], dtype=float)
        if q.shape != (20,) or not np.isfinite(q).all():
            raise ValueError(f"{name} must contain 20 finite joint angles")
    for name, value in task.items():
        if name.endswith(("_s", "_m", "_rad", "_n", "_m_s", "_rad_s")) and not isinstance(value, (list, dict)):
            if not isinstance(value, (int, float)) or not np.isfinite(value) or (value <= 0 and name not in ("left_grasp_axis_m", "right_grasp_axis_m")):
                raise ValueError(f"{name} must be finite and positive")
    return config, task


def main():
    parser = argparse.ArgumentParser(description="Left end grasp, right middle handover and rail insertion")
    parser.add_argument("--config", type=Path, default=HERE/"handover.yaml")
    parser.add_argument("--mode", choices=("step", "auto"), default="step")
    parser.add_argument("--until", choices=("scene", "grasp", "extract", "handover", "transport", "insert"), default="insert")
    parser.add_argument("--preflight", action="store_true", help="Check hand templates and Lula IK without starting physics")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--snapshots", action="store_true", help="Save three camera PNGs at phase boundaries")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1, help="Consecutive trials with an in-process scene reset")
    parser.add_argument("--display")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--initial-state", type=Path, help="Diagnostic reset to an open-hand phase snapshot")
    parser.add_argument("--replay", type=Path, help="Replay a normal-start trajectory with the experiment's scene and joint gains")
    parser.add_argument("--start-phase", choices=("slot", "approach", "wrap", "close"), help="Diagnostic start after settling; requires --initial-state")
    parser.add_argument("--inside-container", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.headless and args.mode != "auto":
        parser.error("Headless runs require --mode auto")
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    if args.repeat < 1 or (args.repeat > 1 and args.mode != "auto"):
        parser.error("--repeat must be positive; repeated runs require --mode auto")
    if args.display:
        os.environ["DISPLAY"] = args.display
    if args.replay and (args.mode != "auto" or args.repeat != 1 or args.initial_state or args.start_phase or args.until != "insert"):
        parser.error("Replay requires --mode auto, one playback, and no --until/diagnostic-start overrides")
    config, task = load_experiment(args.config)
    bimanual = task.get("strategy") == "bimanual_handover"
    if (args.preflight or args.until in ("extract", "handover")) and not bimanual:
        parser.error("Preflight and extract/handover milestones require handover.yaml")
    if bimanual and (args.initial_state or args.start_phase):
        parser.error("Bimanual runs start from the normal scene; diagnostic state injection is unavailable")
    initial_positions = None
    if bool(args.initial_state) != bool(args.start_phase):
        parser.error("--initial-state and --start-phase must be specified together")
    if args.initial_state:
        import numpy as np
        from scipy.spatial.transform import Rotation
        state = json.loads(args.initial_state.read_text())
        q, rail = np.asarray(state["q"]), np.asarray(state["rail"])
        if q.shape != (54,) or rail.shape != (7,) or not np.isfinite(q).all() or not np.isfinite(rail).all():
            parser.error("Initial state requires finite q[54] and rail[7]")
        neutral = np.max(np.abs(q[14:26])) < .15
        preshaped = np.max(np.abs(q[14:26]-np.asarray(task["open_hand_rad"][:12]))) < .15
        wrapped = np.max(np.abs(q[14:26]-np.asarray(task["wrap_hand_rad"][:12]))) < .15
        if not (neutral or preshaped or wrapped) or max(state.get("rail_contacts", {}).values(), default=0) > .1:
            parser.error("Diagnostic starts require a neutral/preshaped hand without rail contact")
        initial_positions = q
        config["robot"]["ready_deg"] = {"left": np.rad2deg(q[:7]).tolist(), "right": np.rad2deg(q[7:14]).tolist()}
        config["rail"]["initial_center"] = rail[:3].tolist()
        config["rail"]["initial_rpy_deg"] = Rotation.from_quat(rail[3:]).as_euler("xyz", degrees=True).tolist()
    if not args.inside_container:
        if not args.preflight and not args.headless and not os.environ.get("DISPLAY"):
            parser.error("Run in a Remmina terminal, use --display :10, or use --headless --mode auto")
        overlay = PROJECT/"bimanual_teleop/sim/.runtime/isaac-overlay.img"
        if overlay.is_file() and shutil.which("fuser"):
            probe = subprocess.run(["fuser", str(overlay)], capture_output=True, text=True)
            if probe.returncode == 0:
                parser.error("Isaac is already running. Close the existing simulation window before starting this task or its preflight.")
        if args.replay and not bimanual:
            source_metadata = args.replay.parent.parent/"metadata.json"
            current_digest = hashlib.sha256((PROJECT/"bimanual_teleop/sim/configs/default.yaml").read_bytes()).hexdigest()
            if source_metadata.is_file() and json.loads(source_metadata.read_text()).get("default_yaml_sha256") != current_digest:
                parser.error("The saved trajectory uses an older scene. Its old input.yaml does not restore the complete scene; use the current bimanual task.")
        output_root = PROJECT/"workspace/outputs"
        output = (args.output or output_root/"rail_grasp_insert"/
                  (datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:6])).resolve()
        if not output.is_relative_to(output_root.resolve()):
            parser.error("--output must be inside this project's workspace/outputs")
        output.mkdir(parents=True, exist_ok=False)
        # Snapshot the exact input, so paths outside the project remain usable in the container.
        (output/"input.yaml").write_bytes(args.config.read_bytes())
        if args.initial_state:
            (output/"initial-state.json").write_bytes(args.initial_state.read_bytes())
        if args.replay:
            (output/"replay-input.json").write_bytes(args.replay.read_bytes())
        defaults = PROJECT/"bimanual_teleop/sim/configs/default.yaml"
        version = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT, capture_output=True, text=True, check=True).stdout
        code_paths = [*HERE.glob("*.py"), PROJECT/"bimanual_teleop/sim/engine.py", PROJECT/"bimanual_teleop/sim/local_session.py",
                      PROJECT/"bimanual_teleop/sim/task.py",
                      PROJECT/"bimanual_teleop/sim/assets/generated/manifest.json",
                      PROJECT/"bimanual_teleop/sim/configs/wrist_camera_calibrated.yaml"]
        code_hashes = {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_paths}
        (output/"metadata.json").write_text(json.dumps({"git_commit": version, "git_dirty": bool(dirty), "file_sha256": code_hashes,
            "default_yaml_sha256": hashlib.sha256(defaults.read_bytes()).hexdigest(), "test_data": True,
            "runtime": "Isaac Sim 5.1.0", "mode": args.mode, "until": args.until, "openblas_threads": 1,
            "replay": str(args.replay) if args.replay else None}, indent=2))
        mapped = Path("/workspace/bimanual-teleop")/output.relative_to(PROJECT)
        command = [str(PROJECT/"bimanual_teleop/sim/docker/exec-apptainer.sh"), "/isaac-sim/python.sh",
                   "/workspace/bimanual-teleop/workspace/experiments/rail_grasp_insert/run.py",
                   "--inside-container", "--config", str(mapped/"input.yaml"), "--output", str(mapped),
                   "--mode", args.mode, "--until", args.until, "--steps", str(args.steps)]
        command += ["--repeat", str(args.repeat)]
        for flag, enabled in (("--headless", args.headless), ("--record", args.record), ("--snapshots", args.snapshots), ("--preflight", args.preflight)):
            if enabled:
                command.append(flag)
        if args.initial_state:
            command += ["--initial-state", str(mapped/"initial-state.json"), "--start-phase", args.start_phase]
        if args.replay:
            command += ["--replay", str(mapped/"replay-input.json")]
        os.environ["BIMANUAL_SIM_GUI"] = "0" if args.headless or args.preflight else "1"
        print(f"Experiment output: {output}", flush=True)
        child_environment = os.environ.copy()
        # Control uses tiny matrices. Parallel BLAS workers otherwise spend most
        # of the cycle synchronizing; keep this override local to the experiment.
        child_environment["APPTAINERENV_OPENBLAS_NUM_THREADS"] = "1"
        completed = subprocess.run(command, env=child_environment)
        result = read_runtime_result(output, completed.returncode)
        summary_path = output/"summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {"completed_trials": 0}
        print(f"Experiment result: {result['status']}; {result.get('reason') or result.get('phase')}", flush=True)
        accepted = result["status"] in ("success", "milestone_passed", "diagnostic_passed", "preflight_passed")
        if not accepted or summary.get("passed_trials", 0) < args.repeat:
            raise SystemExit(completed.returncode or 2)
        raise SystemExit(completed.returncode)
    else:
        import yaml
        from controller import RailController, RailReplayController
        if args.preflight:
            from handover import preflight
            result = preflight(config, task, args.output)
            if result["status"] != "preflight_passed":
                raise SystemExit(2)
            return
        if bimanual:
            from handover import HandoverController, HandoverReplayController
            RailController, RailReplayController = HandoverController, HandoverReplayController
        from bimanual_teleop.sim.engine import run
        config["paths"]["recordings"] = str(args.output/"recordings")
        snapshot = copy.deepcopy(config)
        snapshot["network"]["token"] = "[redacted]"
        (args.output/"effective.yaml").write_text(yaml.safe_dump({"simulation": snapshot, "task": task}, sort_keys=False))
        def factory(sim_config, model):
            if args.replay:
                source = RailReplayController(sim_config, model, task, args.output, args.replay, mode=args.mode,
                                              exit_on_complete=args.headless, snapshots=args.snapshots)
            else:
                source = RailController(sim_config, model, task, args.output, mode=args.mode, until=args.until,
                                      exit_on_complete=args.headless, snapshots=args.snapshots, start_phase=args.start_phase,
                                      initial_positions=initial_positions)
            source.repeat = args.repeat
            # SimulationApp installs its own signal handling. Register after it
            # starts; only set flags here, and let the engine close recordings.
            import signal
            def stop(signum, frame):
                source.stop_reason = f"received {signal.Signals(signum).name}"
                source.should_stop = True
                source.snapshot_requested = False
            signal.signal(signal.SIGINT, stop)
            signal.signal(signal.SIGTERM, stop)
            return source
        run(config, "", gui=not args.headless, standalone=True, record=args.record, steps=args.steps,
            action_source_factory=factory)
        result = json.loads((args.output/"result.json").read_text())
        if result["status"] not in ("milestone_passed", "success", "diagnostic_passed"):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
