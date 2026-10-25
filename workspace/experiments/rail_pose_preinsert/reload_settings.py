"""Host-side preparation of complete, versioned task configurations."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import yaml
from common import pose, seated_rail, write_json
from scipy.spatial.transform import Rotation
from task_settings import ready_joints, rebuild_changes, validate_task

from bimanual_teleop.sim.config import load_config, sim_path


def load_settings(path):
    document = yaml.safe_load(Path(path).read_text())
    config = load_config(overrides=document["simulation"])
    task = document["task"]
    validate_task(task)
    for section, keys in (("robot", ("arm_stiffness", "arm_damping")),
                          ("runtime", ("joint_speed_rad_s", "hand_speed_rad_s"))):
        if any(not np.isfinite(config[section][key]) or config[section][key] <= 0 for key in keys):
            raise ValueError(f"Invalid positive settings in {section}")
    return config, task


def runtime_bundle(bridge, config_path=None, revision="startup"):
    manifest = json.loads(Path(bridge.config["manifest"]).read_text())
    goal = seated_rail(bridge.simulation, manifest)
    setting = bridge.task.get("initial_rail_pose_world")
    initial = pose(setting["xyz_m"], setting["rpy_deg"]) if setting is not None else goal
    bridge.simulation["rail"]["initial_center"] = initial[:3, 3].tolist()
    bridge.simulation["rail"]["initial_rpy_deg"] = Rotation.from_matrix(initial[:3, :3]).as_euler("xyz", degrees=True).tolist()
    return {"open": bridge.references["open"].tolist(), "grasp": bridge.references["grasp"].tolist(),
            "T_B_W": bridge.T_B_W.tolist(), "T_S_CAD": bridge.S_O.tolist(), "goal_rail": goal.tolist(),
            "config_path": str(config_path) if config_path else None, "revision": revision}


class ReloadSettings:
    def __init__(self, path, bridge, until_override=None):
        self.path, self.initial, self.until_override = Path(path).resolve(), bridge, until_override
        self.input_simulation, self.input_task = load_settings(self.path)
        self.assets = self.asset_snapshot(bridge)
        self.versions = {"startup": bridge}
        self.inputs = {"startup": (copy.deepcopy(self.input_simulation), copy.deepcopy(self.input_task), self.assets)}

    @staticmethod
    def asset_snapshot(bridge):
        from rail_preinsert.core import digest
        asset_paths = {Path(bridge.config[key]) for key in ("urdf", "manifest", "mesh")}
        generated = Path(bridge.config["manifest"]).parent
        asset_paths.update(path for path in generated.rglob("*") if path.is_file())
        calibration = bridge.simulation["cameras"].get("wrist_calibration_file")
        if calibration:
            asset_paths.add(sim_path(calibration))
        # Also check meshes referenced by an alternate, compatible URDF.
        import xml.etree.ElementTree as ET
        urdf = Path(bridge.config["urdf"])
        for mesh in ET.parse(urdf).getroot().iter("mesh"):
            asset_paths.add((urdf.parent / mesh.attrib["filename"]).resolve())
        return {str(path): digest(path) for path in asset_paths}

    def prepare(self, revision, *, current_revision="startup", rebuild=False):
        from bridge import Bridge
        if current_revision not in self.versions:
            raise ValueError(f"Unknown configuration revision: {current_revision}")
        simulation, task = load_settings(self.path)
        raw_input = copy.deepcopy((simulation, task))
        before, old_task, assets = self.inputs[current_revision]
        active = self.versions[current_revision]
        changes = rebuild_changes(before, simulation, old_task, task)
        if self.asset_snapshot(active) != assets:
            changes.add("assets")
        rebuild = rebuild or bool(changes)
        if self.until_override:
            task["until"] = self.until_override
        candidate = (Bridge(simulation, task, active.output, active.perception) if rebuild
                     else active.reconfigured(simulation, task))
        runtime = runtime_bundle(candidate, self.path, revision)
        ready_joints(candidate.simulation, runtime, candidate.geometry.robot)
        candidate_assets = self.asset_snapshot(candidate)
        configuration = {"simulation": candidate.simulation, "task": candidate.task, "runtime": runtime}
        directory = candidate.output / "configurations" / revision
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "input.yaml").write_text(yaml.safe_dump({"simulation": raw_input[0], "task": raw_input[1]}, allow_unicode=True))
        write_json(directory / "prepared.json", configuration)
        write_json(directory / "assets.json", candidate_assets)
        self.versions[revision] = candidate
        self.inputs[revision] = (*raw_input, candidate_assets)
        return {"ok": True, "configuration": copy.deepcopy(configuration),
                "operation": "rebuild" if rebuild else "reset", "changed_structure": sorted(changes)}

    def bridge_for(self, request):
        revision = request.get("revision", "startup")
        if revision not in self.versions:
            raise ValueError(f"Unknown configuration revision: {revision}")
        return self.versions[revision]
