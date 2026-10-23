"""Isaac-only preinsert state machine with an explicit simulated hand delivery."""
from __future__ import annotations

import json
import time
import uuid
from collections import deque
from pathlib import Path

import numpy as np
from common import (
    finger_forces,
    opposed_grasp,
    pose,
    pose_error,
    sample_path,
    write_json,
)
from scipy.spatial.transform import Rotation

from bimanual_teleop.sim.kinematics import pose_matrix


class PreinsertController:
    name = "rail_pose_preinsert_simulation"
    capture_snapshots = True

    def __init__(self, config, model, task, output, runtime, *, mode, headless, perception, grasp):
        self.config, self.model, self.task, self.output, self.runtime = config, model, task, Path(output), runtime
        self.mode, self.headless, self.perception, self.grasp_mode = mode, headless, perception, grasp
        self.ready = np.r_[np.radians(config["robot"]["ready_deg"]["left"]),
                          np.radians(config["robot"]["ready_deg"]["right"]), runtime["open"], np.zeros(20)]
        self.joint_gain_overrides = {name: (float(kp), float(kd)) for name, kp, kd in
            zip(model.hand_names["left"], np.tile([5, 5, 3, 2], 5), np.tile([.1, .1, .06, .06], 5))}
        self.trace = (self.output / "trace.jsonl").open("w")
        self.events = (self.output / "events.jsonl").open("w")
        self.label = None
        self.bound = False
        self.reset()

    def reset(self):
        self.reset_index = getattr(self, "reset_index", -1) + 1
        self.phase, self.status = "settle_teach", "running"
        self.command = self.ready.copy()
        self.elapsed = self.stable = 0.
        self.should_stop = self.snapshot_requested = self.finished = False
        self.pending_placement = self.pending_attachment = False
        self.frames, self.history = [], deque(maxlen=3000)
        self.request_id = self.capture_stage = self.last_state = self.plan = self.grip = None
        self.waiting = self.paused = False
        self.metrics = {}
        self.support_active = True
        self.failure_snapshot_pending = False
        self.contact_lost_since = self.stationary_reference = None
        if self.bound:
            self.joint.CreateJointEnabledAttr(False)
            self.body.CreateKinematicEnabledAttr(False)
            self.ccd.CreateEnableCCDAttr(False)
        self.report()

    def scene_ready(self, drives):
        write_json(self.output / "actual_drives.json", drives)

    def event(self, kind, **values):
        value = {"kind": kind, "phase": self.phase, "time_ns": (self.last_state or {}).get("time_ns"), **values}
        self.events.write(json.dumps(value, allow_nan=False) + "\n"); self.events.flush()
        print("Preinsert: " + json.dumps(value), flush=True)

    def report(self, reason=None):
        result = {"status": self.status, "phase": self.phase, "reason": reason, "test_data": True,
                  "simulation_only": True, "perception": self.perception, "grasp_model": self.grasp_mode,
                  "physical_grasp_validated": self.grasp_mode == "physics" and self.status in ("simulation_preinsert_passed", "simulation_grasp_passed"),
                  "grasp_template": self.task.get("grasp_template", "legacy_geometry"),
                  "real_grasp_validated": False, "end_stage": "preinsert", "insertion_executed": False,
                  "requested_stop": self.task.get("until", "preinsert"),
                  "waiting": self.waiting, "metrics": self.metrics}
        write_json(self.output / "result.json", result)
        if self.label is not None:
            self.label.text = f"{self.phase}: {self.status}" + (" | Next stage" if self.waiting else "")

    def finish(self, status, reason=None):
        if self.finished:
            return
        self.finished, self.status = True, status
        self.should_stop = self.headless
        self.failure_snapshot_pending = status == "failed" and self.last_state is not None
        self.snapshot_requested = self.failure_snapshot_pending
        self.event("finished", status=status, reason=reason)
        self.report(reason)

    def bind_simulation(self, *, world, robot, rail, robot_path, camera_paths):
        import omni.usd
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade
        self.world, self.robot, self.rail = world, robot, rail
        self.stage = omni.usd.get_context().get_stage()
        self.camera_path = camera_paths[0]
        self.Gf, self.UsdGeom = Gf, UsdGeom
        self.body = UsdPhysics.RigidBodyAPI(self.stage.GetPrimAtPath("/World/Rail"))
        # Enter supported mode only after the engine's initial resets. Isaac's
        # reset path writes velocities, which PhysX rejects for kinematic bodies.
        self.body.CreateKinematicEnabledAttr(False)
        self.ccd = PhysxSchema.PhysxRigidBodyAPI(self.stage.GetPrimAtPath("/World/Rail"))
        self.ccd.CreateEnableCCDAttr(False)
        self.support_active = True
        self.joint = UsdPhysics.FixedJoint.Define(self.stage, "/World/ExperimentGrasp")
        self.joint.CreateBody0Rel().SetTargets([robot_path + "/TCP_Link_L"])
        self.joint.CreateBody1Rel().SetTargets(["/World/Rail"])
        self.joint.CreateExcludeFromArticulationAttr(True)
        self.joint.CreateJointEnabledAttr(False)
        self.joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        self.joint.CreateLocalRot1Attr(Gf.Quatf(1, Gf.Vec3f(0, 0, 0)))
        if self.grasp_mode == "fixed":
            # Same rigid compound assumption as the planner. This mode cannot
            # validate finger contact, friction or grip force.
            relation = UsdPhysics.FilteredPairsAPI.Apply(self.stage.GetPrimAtPath("/World/Rail")).CreateFilteredPairsRel()
            for name in self.model.collidable_links:
                if name == "l_wrist" or name.startswith(("l_thumb_", "l_index_finger_", "l_middle_finger_", "l_ring_finger_", "l_pinky_")):
                    relation.AddTarget(robot_path + "/" + name)
        # Imported collision shapes may be instance proxies. Bind at editable
        # rigid links so the configured finger material actually reaches them.
        material = UsdShade.Material.Get(self.stage, "/World/Materials/Fingers")
        for name in self.model.collidable_links:
            if name.startswith(("l_", "r_")):
                link = self.stage.GetPrimAtPath(robot_path + "/" + name)
                # PhysX collision offsets are shape attributes, not inherited
                # materials. Make this experiment's hand collision instances
                # editable before applying the millimetre-scale settings.
                for prim in list(Usd.PrimRange(link)):
                    if prim.IsInstance():
                        prim.SetInstanceable(False)
                UsdShade.MaterialBindingAPI.Apply(link).Bind(
                    material, bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics")
                for prim in Usd.PrimRange(link):
                    if prim.HasAPI(UsdPhysics.CollisionAPI):
                        collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                        collision.CreateContactOffsetAttr(self.config["rail"]["contact_offset_m"])
                        collision.CreateRestOffsetAttr(self.config["rail"]["rest_offset_m"])
        colliders = []
        for root in (robot_path, "/World/Rail"):
            for prim in Usd.PrimRange(self.stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
                physics = UsdPhysics.MaterialAPI(bound.GetPrim()) if bound else None
                record = {"path": str(prim.GetPath()), "type": prim.GetTypeName(),
                          "instance_proxy": prim.IsInstanceProxy(),
                          "collision_enabled": UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(),
                          "approximation": UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get(),
                          "contact_offset_m": PhysxSchema.PhysxCollisionAPI(prim).GetContactOffsetAttr().Get(),
                          "rest_offset_m": PhysxSchema.PhysxCollisionAPI(prim).GetRestOffsetAttr().Get(),
                          "world_scale": list(Gf.Transform(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).GetScale()),
                          "material": str(bound.GetPath()) if bound else None,
                          "static_friction": physics.GetStaticFrictionAttr().Get() if physics else None,
                          "dynamic_friction": physics.GetDynamicFrictionAttr().Get() if physics else None}
                name = str(prim.GetPath()).removeprefix(robot_path + "/").split("/")[0]
                if name.startswith(("l_", "r_")) and (physics is None or any(
                        value is None or abs(value-self.config["rail"]["finger_friction"]) > 1e-6
                        for value in (record["static_friction"], record["dynamic_friction"]))):
                    raise RuntimeError("Finger physics material did not reach collider: " + str(prim.GetPath()))
                colliders.append(record)
        write_json(self.output / "collision_approximations.json", {
            "colliders": colliders, "hand_rail_collision_filtered": self.grasp_mode == "fixed",
            "finger_material_inheritance_verified": True,
            "limitation": "USD collision settings; convex cooking can change the source mesh"})
        if self.perception == "foundationpose":
            import omni.replicator.core as rep
            from isaacsim.core.utils.semantics import add_update_semantics
            add_update_semantics(self.stage.GetPrimAtPath("/World/Rail"), "rail")
            size = (self.config["cameras"]["width"], self.config["cameras"]["height"])
            self.product = rep.create.render_product(self.camera_path, size)
            self.depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane", device="cpu")
            self.segmentation = rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": False}, device="cpu")
            self.depth.attach(self.product); self.segmentation.attach(self.product)
        self.bound = True

    def flange(self, state):
        # eef is an actual physics link transform in the URDF joint-origin base.
        return self.model.base_pose("left") @ pose_matrix(state["eef"][0])

    def before_physics_step(self):
        if self.support_active and not self.body.GetKinematicEnabledAttr().Get():
            self.body.CreateKinematicEnabledAttr(True)
        if self.pending_placement:
            W_S = self.flange(self.last_state) @ pose(self.task["rail_in_flange_xyz_m"], self.task["rail_in_flange_rpy_deg"])
            q = Rotation.from_matrix(W_S[:3, :3]).as_quat()
            self.rail.set_world_pose(W_S[:3, 3], q[[3, 0, 1, 2]])
            self.pending_placement = False
            self.event("human_delivery_simulated", T_W_S=W_S.tolist(), note="One-time placement; not a grasp policy")
        if self.pending_attachment:
            self.grip = np.linalg.inv(self.flange(self.last_state)) @ pose_matrix(self.last_state["rail"])
            if self.grasp_mode == "fixed":
                q = Rotation.from_matrix(self.grip[:3, :3]).as_quat()
                self.joint.CreateLocalPos0Attr(self.Gf.Vec3f(*self.grip[:3, 3]))
                self.joint.CreateLocalRot0Attr(self.Gf.Quatf(float(q[3]), self.Gf.Vec3f(*q[:3])))
                self.joint.CreateJointEnabledAttr(True)
            self.body.CreateKinematicEnabledAttr(False)
            self.ccd.CreateEnableCCDAttr(True)
            self.support_active = False
            self.pending_attachment = False
            self.event("delivery_support_removed", fixed_constraint=self.grasp_mode == "fixed")

    def transition(self, phase, *, wait=False):
        self.phase, self.elapsed, self.stable = phase, 0., 0.
        self.waiting = wait and self.mode == "step"
        self.event("phase_started", waiting=self.waiting)
        self.report()

    def capture(self, stage):
        self.capture_stage, self.frames = stage, []
        self.snapshot_requested = True
        self.transition(stage + "_capture")

    def send_request(self):
        self.request_id = uuid.uuid4().hex
        write_json(self.output / "rpc/request.json", {"id": self.request_id, "stage": self.capture_stage,
                   "frames": self.frames, "left_joint_history": list(self.history)})
        self.snapshot_requested = False
        self.transition(self.capture_stage + "_wait")
        self.request_started = time.monotonic()

    def response(self):
        path = self.output / "rpc/response.json"
        value = json.loads(path.read_text()) if path.is_file() else {}
        if value.get("id") != self.request_id:
            if time.monotonic() - self.request_started > 240:
                self.finish("failed", "Host perception/planning request timed out")
            return None
        if not value.get("ok"):
            self.finish("failed", value.get("error", "Host request failed")); return None
        return value

    def step(self, state, dt):
        if self.finished or self.paused or self.waiting:
            return self.command.copy()
        self.elapsed += dt
        desired = self.command.copy()
        if self.phase == "settle_teach" and self.elapsed >= self.task["settle_s"]:
            self.capture("teach")
        elif self.phase == "teach_wait" and self.response() is not None:
            self.pending_placement = True
            self.transition("close", wait=True)
        elif self.phase == "close":
            u = min(self.elapsed / self.task["hand_transition_s"], 1.)
            u = 10*u**3 - 15*u**4 + 6*u**5
            desired[14:34] = (1-u)*np.asarray(self.runtime["open"]) + u*np.asarray(self.runtime["grasp"])
            if self.elapsed >= self.task["hand_transition_s"] + self.task["settle_s"]:
                self.pending_attachment = True
                self.transition("settle_grasp")
        elif self.phase == "settle_grasp" and self.elapsed >= self.task["settle_s"]:
            gripping = opposed_grasp(state, minimum_support_fingers=self.task.get("minimum_support_fingers", 4))
            self.stable = self.stable + dt if gripping else 0.
            if self.grasp_mode == "fixed" or self.stable >= .5:
                self.stationary_reference = np.asarray(state["q"]).copy()
                self.grip = np.linalg.inv(self.flange(state)) @ pose_matrix(state["rail"])
                self.event("grasp_reference_established", finger_force_n=finger_forces(state),
                           fixed_constraint=self.grasp_mode == "fixed")
                self.capture("grasp_review" if self.task.get("until") == "grasp" else "measure")
            elif self.elapsed >= self.task["settle_s"] + 2:
                self.finish("failed", "A thumb plus four supporting fingers are required; multi-finger contact did not stabilize")
        elif self.phase == "grasp_review_wait" and self.response() is not None:
            self.finish("grasp_preview_complete" if self.grasp_mode == "fixed" else "simulation_grasp_passed")
        elif self.phase == "measure_wait":
            response = self.response()
            if response is not None:
                self.plan = response["plan"]
                self.transition("execute", wait=True)
        elif self.phase == "execute":
            desired[:7] = sample_path(self.plan, self.elapsed)
            if self.elapsed >= self.plan["duration_s"]:
                self.transition("settle_endpoint")
        elif self.phase == "settle_endpoint":
            desired[:7] = self.plan["joint_waypoints_rad"][-1]
            B_F = np.asarray(self.runtime["T_B_W"]) @ self.flange(state)
            error = pose_error(B_F, np.asarray(self.plan["targets"]["T_B_F_preinsert"]))
            self.metrics["flange_position_error_m"], self.metrics["flange_rotation_error_rad"] = error
            passed = error[0] <= self.task["position_tolerance_m"] and error[1] <= self.task["rotation_tolerance_rad"]
            self.stable = self.stable + dt if passed else 0.
            if self.stable >= .5:
                self.capture("verify")
            elif self.elapsed > self.task["endpoint_timeout_s"]:
                self.finish("failed", "Measured flange did not reach the preinsert tolerance")
        elif self.phase == "verify_wait":
            response = self.response()
            if response is not None:
                verification = response["verification"]
                write_json(self.output / "verification.json", verification)
                self.metrics["visual_verification"] = verification
                if verification["status"] != "preinsert_reached":
                    self.finish("failed", "End-point rail pose did not pass verification")
                else:
                    self.transition("hold")
        elif self.phase == "hold" and self.elapsed >= self.task["hold_s"]:
            expected = np.asarray(self.runtime["goal_rail"]).copy()
            expected[:3, 3] += self.task["preinsert_offset_m"] * expected[:3, 2]
            error = pose_error(pose_matrix(state["rail"]), expected)
            self.metrics["ground_truth_rail_position_error_m"], self.metrics["ground_truth_rail_rotation_error_rad"] = error
            if error[0] > self.task["position_tolerance_m"] or error[1] > self.task["rotation_tolerance_rad"]:
                self.finish("failed", "Actual rail pose differs from the scene's true preinsert pose")
            else:
                self.finish("simulation_preinsert_passed")
        limits = np.r_[np.full(7, self.task["joint_speed_rad_s"]), np.full(7, .12), np.full(40, .8)] * dt
        self.command += np.clip(desired-self.command, -limits, limits)
        self.command = np.clip(self.command, self.model.lower, self.model.upper)
        return self.command.copy()

    def observe(self, state, images=None):
        self.last_state = state
        self.history.append([state["time_ns"], state["q"][:7]])
        self.metrics["finger_contact_force_n"] = finger_forces(state)
        if self.failure_snapshot_pending and images is not None:
            from PIL import Image
            directory = self.output / "captures" / f"trial_{self.reset_index:03d}" / "failure"
            directory.mkdir(parents=True, exist_ok=True)
            for i, rgb in enumerate(images):
                Image.fromarray(rgb).save(directory / f"camera_{i}.png")
            write_json(directory / "state.json", state)
            self.failure_snapshot_pending = self.snapshot_requested = False
        if self.grip is not None and not self.finished:
            actual = np.linalg.inv(self.flange(state)) @ pose_matrix(state["rail"])
            error = pose_error(actual, self.grip)
            self.metrics["grasp_translation_error_m"], self.metrics["grasp_rotation_error_rad"] = error
            position_limit = self.task.get("loading_translation_limit_m", .02) if self.phase == "settle_grasp" else self.task["max_grasp_translation_m"]
            rotation_limit = self.task.get("loading_rotation_limit_rad", .25) if self.phase == "settle_grasp" else self.task["max_grasp_rotation_rad"]
            if error[0] > position_limit or error[1] > rotation_limit:
                self.finish("failed", "Rail slipped relative to flange; grasp assumption failed")
        if self.phase in ("execute", "settle_endpoint", "verify_capture", "verify_wait", "hold") and not self.finished:
            if self.stationary_reference is not None:
                q = np.asarray(state["q"])
                if np.max(np.abs(q[7:14]-self.stationary_reference[7:14])) > .015:
                    self.finish("failed", "Idle right arm moved after planning")
                if np.max(np.abs(q[14:34]-self.stationary_reference[14:34])) > .10:
                    self.finish("failed", "Left hand shape changed after planning")
            if self.grasp_mode == "physics":
                if opposed_grasp(state, minimum_support_fingers=self.task.get("minimum_support_fingers", 4)):
                    self.contact_lost_since = None
                elif self.contact_lost_since is None:
                    self.contact_lost_since = state["time_ns"]
                elif state["time_ns"]-self.contact_lost_since > 100_000_000:
                    self.finish("failed", "Opposed grasp contacts were lost during transport")
            tracking = float(np.max(np.abs(np.asarray(state["q"])[:7] - np.asarray(state["q_target"])[:7])))
            self.metrics["joint_tracking_error_rad"] = tracking
            if tracking > self.task["max_tracking_error_rad"]:
                self.finish("failed", "Arm joint tracking error exceeded its bound")
            if max(state.get("environment_contacts", {}).values(), default=0) > self.task["environment_force_limit_n"]:
                self.finish("failed", "Robot contacted a table, fixture or bin")
            if max(state.get("rail_support_contacts", {}).values(), default=0) > self.task["environment_force_limit_n"]:
                self.finish("failed", "Held rail contacted the environment")
        if self.snapshot_requested and images is not None and not self.finished:
            from PIL import Image
            index = len(self.frames)
            directory = self.output / "captures" / f"trial_{self.reset_index:03d}" / f"{self.capture_stage}_{index:02d}"
            directory.mkdir(parents=True, exist_ok=False)
            for i, rgb in enumerate(images):
                Image.fromarray(rgb).save(directory / f"camera_{i}.png")
            W_C = np.array(self.UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.camera_path))).T
            W_C[:3, :3] = W_C[:3, :3] @ np.diag([1., -1., -1.])
            frame = {"T_W_C": W_C.tolist(), "T_W_S": pose_matrix(state["rail"]).tolist(), "state": state,
                     "directory": str(directory.relative_to(self.output))}
            write_json(directory / "state.json", frame)
            if self.perception == "foundationpose":
                depth = np.asarray(self.depth.get_data(), dtype=np.float32)
                segmentation = self.segmentation.get_data()
                labels = segmentation["info"]["idToLabels"]
                ids = [int(key) for key, value in labels.items() if (value.get("class") if isinstance(value, dict) else value) == "rail"]
                mask = np.isin(np.asarray(segmentation["data"]), ids).astype(np.uint8) * 255
                if mask.sum() == 0:
                    self.finish("failed", "No visible rail pixels in the head camera"); return
                np.save(directory / "depth_m.npy", depth)
                Image.fromarray(mask).save(directory / "mask.png")
            self.frames.append(frame)
            if len(self.frames) >= 3:
                self.send_request()
        if state["tick"] % 12 == 0:
            self.trace.write(json.dumps({"phase": self.phase, "state": state, "metrics": self.metrics}, allow_nan=False) + "\n")
            self.trace.flush()

    def build_ui(self, ui):
        self.label = ui.Label("Preinsert: teach -> simulated delivery -> close -> measure -> plan -> preinsert")
        ui.Button("Next stage", clicked_fn=lambda: setattr(self, "waiting", False))
        ui.Button("Run automatically", clicked_fn=self.automatic)
        ui.Button("Pause task / Resume", clicked_fn=lambda: setattr(self, "paused", not self.paused))

    def automatic(self):
        self.mode, self.waiting = "auto", False

    def close(self, error=None):
        if not self.finished:
            self.finish("failed" if error else "interrupted", error or "Simulation closed before acceptance")
        self.trace.close(); self.events.close()
