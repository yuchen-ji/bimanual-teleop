"""Host-side adapter to the existing preinsert pipeline and FoundationPose."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
from common import cad_to_sim, pose, write_json

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
WYS = PROJECT.parent
LIVE = WYS / "diffusion_policy/workspace/rail_pose_insert"
sys.path[:0] = [str(LIVE), str(WYS), str(WYS / "diffusion_policy")]

from diffusion_policy.third_party.bimanual_teleop.devices.tianji.model import (
    TianjiKinematics,
)
from rail_preinsert.core import digest, load_config, mean_pose
from rail_preinsert.geometry import RobotGeometry
from rail_preinsert.workflow import hand_references, prepare_plan, verify_preinsert


class Bridge:
    def __init__(self, simulation, task, output, perception="oracle"):
        self.simulation, self.task, self.output, self.perception = simulation, task, Path(output), perception
        self.config = load_config(LIVE / "config.yaml")
        # Planning and reconstruction must read the same generated assets.
        from bimanual_teleop.sim import ROOT
        from bimanual_teleop.sim.config import sim_path
        from bimanual_teleop.sim.kinematics import RobotModel
        self.config["urdf"] = str(sim_path(simulation["robot"]["urdf"]))
        self.config["manifest"] = str(ROOT / "assets/generated/manifest.json")
        self.kinematics = TianjiKinematics()  # Official offline math only; no driver.
        self.geometry = RobotGeometry(self.config, self.kinematics)
        simulated = RobotModel(sim_path(simulation["robot"]["urdf"]), simulation["robot"]["world_yaw_deg"])
        if (simulated.names != self.geometry.robot.names
                or not np.allclose(simulated.world, self.geometry.robot.world)):
            raise ValueError("Simulation robot frames/joints do not match the existing planner; regenerate compatible assets")
        self.T_B_W = self.geometry.T_B_W
        self.S_O = cad_to_sim(__import__("json").loads(Path(self.config["manifest"]).read_text()))
        self.references = hand_references(self.config)
        self.base_references = copy.deepcopy(self.references)
        for name in ("calibration", "target", "scene"):
            self.config[name] = str(self.output / "inputs" / f"{name}.json")
        self.config["joint_speed_rad_s"] = task["joint_speed_rad_s"]
        self.config["joint_acceleration_rad_s2"] = task["joint_acceleration_rad_s2"]
        self.plan = None
        self.engine = None
        self.accuracy = {}
        if task.get("grasp_template") == "demonstrated_power_wrap":
            self.configure_demonstrated_grasp()
        from task_settings import apply_overrides
        self.references = apply_overrides(self.task, self.references, self.geometry.robot)
        self.configure_initial_direction()

    def reconfigured(self, simulation, task):
        """Prepare a candidate without changing the active planner or references."""
        result = copy.copy(self)
        result.simulation, result.task = copy.deepcopy(simulation), copy.deepcopy(task)
        result.config = copy.deepcopy(self.config)
        result.references = copy.deepcopy(self.base_references)
        result.plan, result.accuracy = None, {}
        if task.get("grasp_template") == "demonstrated_power_wrap":
            result.configure_demonstrated_grasp()
        from task_settings import apply_overrides
        result.references = apply_overrides(result.task, result.references, result.geometry.robot)
        result.config.update(joint_speed_rad_s=task["joint_speed_rad_s"],
                             joint_acceleration_rad_s2=task["joint_acceleration_rad_s2"])
        result.configure_initial_direction()
        return result

    def configure_initial_direction(self):
        direction = self.task.get("initial_flange_z_axis_world")
        if direction is None:
            return
        from common import align_flange_z, pose_error
        from rail_preinsert.core import pose as sdk_pose
        from task_settings import ready_joints
        model = self.geometry.robot
        runtime = {key: self.references[key] for key in ("open", "grasp")}
        joints = ready_joints(self.simulation, runtime, model)
        initial = model.flange("left", joints, world=True)
        target = align_flange_z(initial, direction)
        joints[:7] = self.kinematics.ik("left", sdk_pose(self.T_B_W @ target), tuple(joints[:7]))
        actual = model.flange("left", joints, world=True)
        error = pose_error(actual, target)
        if error[0] > .001 or error[1] > .002:
            raise ValueError("Initial flange direction IK did not meet pose tolerance")
        self.simulation["robot"]["ready_deg"]["left"] = np.degrees(joints[:7]).tolist()
        write_json(self.output / "initial_flange_alignment.json", {
            "direction_world": (np.asarray(direction) / np.linalg.norm(direction)).tolist(),
            "T_W_F_before": initial.tolist(), "T_W_F_target": target.tolist(), "T_W_F_actual": actual.tolist(),
            "position_residual_m": error[0], "rotation_residual_rad": error[1],
            "joint_target_deg": np.degrees(joints[:7]).tolist(), "test_data": True})

    def configure_demonstrated_grasp(self):
        from common import seated_rail
        from grasp_template import fit_demonstrated_wrap, load_reference
        from rail_preinsert.core import pose as sdk_pose
        from scipy.spatial.transform import Rotation

        reference = load_reference()
        template = fit_demonstrated_wrap(self.geometry.robot, reference, self.task)
        write_json(self.output / "grasp_template.json", template)
        if max(template["tip_fit_error_m"].values()) > .002:
            raise ValueError("The five-finger wrap cannot fit the rail while preserving the demonstration")
        self.references = {"open": np.asarray(template["open"]), "grasp": np.asarray(template["grasp"])}
        F_S = np.asarray(template["T_F_S"])
        self.task["rail_in_flange_xyz_m"] = F_S[:3, 3].tolist()
        self.task["rail_in_flange_rpy_deg"] = Rotation.from_matrix(F_S[:3, :3]).as_euler("xyz", degrees=True).tolist()
        self.config["hand_reference"] = str(HERE / "hand_profiles/human_left_wrap.json")
        manifest = __import__("json").loads(Path(self.config["manifest"]).read_text())
        initial = seated_rail(self.simulation, manifest)
        initial[:3, 3] += np.asarray(self.task["initial_rail_offset_world_m"])
        flange = self.T_B_W @ initial @ np.asarray(template["T_S_F"])
        seed = np.asarray(reference["measured_arm_rad"])
        choices = []
        for start in [seed, np.radians([30, -60, -34, -52, 30, 12, 4]),
                      np.radians([50, -40, -30, -80, 10, 25, -10])]:
            try:
                q = np.asarray(self.kinematics.ik("left", sdk_pose(flange), tuple(start)))
                choices.append((float(np.linalg.norm(q-seed)), q))
            except RuntimeError:
                continue
        if not choices:
            raise ValueError("No initial IK preserves the demonstrated wrist orientation")
        self.simulation["robot"]["ready_deg"]["left"] = np.degrees(min(choices, key=lambda x: x[0])[1]).tolist()

    def scene(self):
        scene, obstacles = self.simulation["scene"], []
        def box(name, centre, size):
            obstacles.append({"name": name, "type": "box", "size_m": list(size),
                              "T_B_obstacle": (self.T_B_W @ pose(centre)).tolist()})
        # Conservative table envelope, including the volume below its top.
        size = np.asarray(scene["table_size"])
        box("table", np.asarray(scene["table_center"]) + [0, 0, size[2]/2], size)
        from bimanual_teleop.sim.bin_geometry import bin_geometry
        crate = bin_geometry(scene)
        for i, centre in enumerate(scene["bin_centers"]):
            # Use a conservative solid envelope of each full stack. We never
            # plan through the bin cavities in this hand-delivery experiment.
            lower = np.asarray(centre).copy()
            lower[2] -= (scene["bin_stack_count"]-1)*scene["bin_stack_pitch_m"]
            high = np.asarray(centre).copy(); high[2] += crate["rim_height_m"]
            xy = np.asarray(scene["bin_inner_size"][:2]) + .06
            box(f"bin_stack_{i}", [(lower[0]+high[0])/2, (lower[1]+high[1])/2, (lower[2]+high[2])/2],
                [*xy, high[2]-lower[2]])
        write_json(self.config["scene"], {"parent_frame": "tianji_left_base", "unit": "m",
                   "test_data": True, "source": "simulation scene; conservative table/bin envelopes",
                   "obstacles": obstacles})

    def observation(self, request):
        records = []
        if self.perception == "foundationpose" and self.engine is None:
            # This Bridge runs in the control environment. Actual inference is
            # delegated to the dedicated GPU worker by the host entry point.
            raise RuntimeError("FoundationPose observations must be provided by the perception worker")
        for frame in request["frames"]:
            W_C = np.asarray(frame["T_W_C"])
            W_O = np.asarray(frame["T_W_S"]) @ self.S_O
            records.append({"valid": True, "timestamp_ns": frame["state"]["time_ns"],
                            "timestamp_source": "simulation_clock", "T_C_CAD": (np.linalg.inv(W_C) @ W_O).tolist()})
        return {"unit": "m", "frame": "T_C_CAD", "camera_serial": "simulation-head",
                "mesh_sha256": digest(self.config["mesh"]), "test_data": True,
                "source": "simulation ground-truth pose; not vision", "records": records}

    def handle(self, request, observation=None):
        from task_settings import positive_settings
        self.config.update(positive_settings(request.get("task_settings", {})))
        observation = observation if observation is not None else self.observation(request)
        stage = request["stage"]
        # Independent diagnostics only: never replace, align, flip or seed the
        # perception output with simulator object poses.
        import trimesh
        from common import perception_errors
        centre = np.r_[trimesh.load(self.config["mesh"], force="mesh").bounds.mean(axis=0), 1.]
        accuracy = perception_errors(request["frames"], observation["records"], self.S_O, centre)
        self.accuracy[stage] = {"source": observation.get("source"), "frames": accuracy,
                               "used_for_pose_initialization_or_correction": False}
        write_json(self.output / "perception_accuracy.json", self.accuracy)
        write_json(self.output / f"{stage}_observation.json", observation)
        if stage in ("measure", "grasp_review"):
            from geometry_audit import audit_grasp
            from rail_preinsert.core import matrix

            from bimanual_teleop.sim.kinematics import pose_matrix
            frame = request["frames"][-1]
            state = frame["state"]
            q = np.asarray(state["q"])
            if "eef" in state:
                W_F = self.geometry.robot.base_pose("left") @ pose_matrix(state["eef"][0])
            else:
                W_F = np.linalg.inv(self.T_B_W) @ matrix(self.kinematics.fk("left", tuple(q[:7])))
            report = audit_grasp(self.geometry.robot, self.config["urdf"], self.config["manifest"], q,
                                 np.linalg.inv(frame["T_W_S"]) @ W_F)
            write_json(self.output / "grasp_geometry.json", report)
            if report["maximum_sampled_intrusion_m"] > .0002:
                raise ValueError("Actual hand/rail meshes intersect by over 0.2 mm in the sampled solid audit; grasp is invalid")
        if stage == "teach":
            frame = request["frames"][0]
            write_json(self.config["calibration"], {"parent_frame": "tianji_left_base",
                       "child_frame": "head_color_optical", "unit": "m", "camera_serial": "simulation-head",
                       "test_data": True, "T_B_C": (self.T_B_W @ np.asarray(frame["T_W_C"])).tolist()})
            write_json(self.config["target"], {"unit": "m", "camera_serial": "simulation-head", "test_data": True,
                       "mesh_sha256": digest(self.config["mesh"]),
                       "T_C_O_goal": mean_pose([r["T_C_CAD"] for r in observation["records"]]).tolist()})
            self.scene()
            return {"ok": True}
        if stage == "measure":
            state = {"full_joints": request["frames"][-1]["state"]["q"],
                     "left_joint_history": request["left_joint_history"], "test_data": True,
                     "model_sha256": self.kinematics.model.digest}
            write_json(self.output / "initial_state.json", state)
            _, self.plan, _ = prepare_plan(self.config, self.kinematics, state, observation, allow_test_data=True)
            self.plan["simulation_only"] = True
            write_json(self.output / "plan.json", self.plan)
            return {"ok": True, "plan": self.plan}
        if stage == "grasp_review":
            return {"ok": True}
        if stage == "verify":
            return {"ok": True, "verification": verify_preinsert(self.config, self.plan, observation, allow_test_data=True)}
        raise ValueError(f"Unknown stage: {stage}")
