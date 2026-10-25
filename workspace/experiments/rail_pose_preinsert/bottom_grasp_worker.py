"""Persistent host service for an existing Isaac grasp scene; exits with that scene."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from bottom_grasp import BottomGraspFit
from host_rpc import RequestServer, WorkerLease, process_identity
from locked_grasp_host import handle


def bootstrap(output):
    from bridge import Bridge
    from reload_settings import ReloadSettings
    # Reset writes a complete trial snapshot, including live task overrides.
    result = json.loads((output / "result.json").read_text())
    snapshot = output / "trials" / f"trial_{result['trial']:03d}" / "settings.json"
    if snapshot.exists():
        saved = json.loads(snapshot.read_text())
    else:
        saved = json.loads((output / "live_settings.json").read_text())
        saved["perception"] = result.get("perception", "oracle")
    simulation, task = saved["simulation"], saved["task"]
    perception = saved.get("perception", "oracle")
    bridge = Bridge(copy.deepcopy(simulation), copy.deepcopy(task), output, perception)
    config = saved["runtime"].get("config_path") or HERE / "config.yaml"
    reloads = ReloadSettings(config, bridge)
    # Reconnecting a host must retain revisions already installed in Isaac.
    for path in sorted((output / "configurations").glob("*/prepared.json")):
        configuration = json.loads(path.read_text())
        revision = configuration["runtime"]["revision"]
        candidate = Bridge(configuration["simulation"], configuration["task"], output, perception)
        raw = yaml.safe_load((path.parent / "input.yaml").read_text())
        assets = json.loads((path.parent / "assets.json").read_text())
        reloads.versions[revision] = candidate
        reloads.inputs[revision] = (raw["simulation"], raw["task"], assets)
    active_revision = saved["runtime"].get("revision", "startup")
    if active_revision not in reloads.versions:
        raise ValueError(f"Cannot restore active configuration revision {active_revision}")
    return reloads


def handle_request(output, reloads, request):
    if request["stage"] == "reload":
        return reloads.prepare(request["id"], current_revision=request.get("revision", "startup"),
                               rebuild=request.get("rebuild", False))
    active = reloads.bridge_for(request)
    if request["stage"] == "bottom_release_check":
        fit = BottomGraspFit(active.geometry.robot, active.config["urdf"], active.config["manifest"], request["state"])
        start = np.asarray(request["state"]["q"])
        opened = start.copy()
        settings = json.loads((output / "live_settings.json").read_text())
        opened[14:34] = settings["runtime"]["open"]
        recovery = settings["task"].get("bottom_recovery_open")
        if recovery:
            opened[14:18] = request["command"][14:18]
        gaps = [float(fit.distances((1-u)*start+u*opened, fit.rail_pose[:3, 3]).min())
                for u in np.linspace(0, 1, 201)]
        minimum = min(gaps)
        safe_retreat = recovery and gaps[0] >= -.0002 and min(gaps) >= gaps[0]-1e-7 and min(gaps[4:]) >= 0
        if minimum < 0 and not safe_retreat:
            raise RuntimeError(f"Opening path intersects hand/rail envelope: {minimum}")
        return {"ok": True, "sampled_release_gap_m": minimum}
    if request["stage"].startswith("locked_grasp_"):
        return handle(active, request)
    if request["stage"] == "four_fingers_check":
        from four_finger_geometry import FourFingerGeometry
        checker = FourFingerGeometry(active.geometry.robot, active.config["urdf"], active.config["manifest"])
        return {"ok": True, "geometry": checker.check(request["state"])}
    if active.perception != "oracle":
        raise ValueError("Reconnect service supports oracle grasp experiments only; restart the full host for FoundationPose")
    return active.handle(request)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scene-pid", type=int, required=True, help="Host PID of the existing Isaac Python process")
    args = parser.parse_args()
    output = args.output.resolve()
    identity = process_identity(args.scene_pid)
    if identity is None:
        parser.error("The specified Isaac process is not alive")
    with WorkerLease(output, args.scene_pid) as lease:
        reloads = bootstrap(output)
        server = RequestServer(output, lambda request: handle_request(output, reloads, request), lease)
        # Terminal trial results do not end this service. Restart/reload creates
        # another request in the same application and must still be answered.
        while process_identity(args.scene_pid) == identity:
            server.poll()
            time.sleep(.05)


if __name__ == "__main__":
    main()
