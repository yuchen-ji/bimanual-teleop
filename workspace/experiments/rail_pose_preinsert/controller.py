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

    def __init__(self, config, model, task, output, runtime, *, mode, headless, perception, grasp, trial_start=0):
        self.config, self.model, self.task, self.output, self.runtime = config, model, task, Path(output), runtime
        self.mode, self.headless, self.perception, self.grasp_mode = mode, headless, perception, grasp
        self.default_grasp_mode = grasp
        from task_settings import ready_joints
        self.ready = ready_joints(config, runtime, model)
        self.configure_gains()
        self.reload_requested = False
        self.reload_id = None
        self.pending_configuration = None
        self.scene_rebuild_requested = False
        self.force_rebuild = False
        self.config_message = ""
        self.reset_index = trial_start - 1
        self.trace = (self.output / "trace.jsonl").open("a" if trial_start else "w")
        self.events = (self.output / "events.jsonl").open("a" if trial_start else "w")
        self.label = None
        self.desktop = None
        self.bound = False
        self.reset()

    def configure_gains(self):
        self.joint_gain_overrides = {name: (float(kp), float(kd)) for name, kp, kd in
            zip(self.model.hand_names["left"], self.task.get("left_hand_stiffness", np.tile([5, 5, 3, 2], 5)),
                self.task.get("left_hand_damping", np.tile([.1, .1, .06, .06], 5)))}

    def reset(self):
        start_on_reset = getattr(self, "start_on_reset", False)
        self.start_on_reset = False
        self.reset_index = getattr(self, "reset_index", -1) + 1
        self.phase, self.status = "settle_teach", "running"
        if self.task.get("until") == "four_fingers":
            self.phase = "settle_four_fingers"
        if self.task.get("orientation_locked_grasp") and self.task.get("until") == "grasp":
            self.phase = "settle_locked_grasp"
        self.grasp_mode = getattr(self, "default_grasp_mode", self.grasp_mode)
        self.locked_grasp = None
        self.four_fingers = None
        self.four_geometry_id = None
        self.reason = None
        self.config_message = ""
        self.command = self.ready.copy()
        self.elapsed = self.stable = 0.
        self.should_stop = self.snapshot_requested = self.finished = False
        self.pause_physics_requested = self.reset_requested = False
        self.physics_action_requested = None
        self.pending_placement = self.pending_attachment = False
        self.frames, self.history = [], deque(maxlen=3000)
        self.request_id = self.capture_stage = self.last_state = self.plan = self.grip = None
        self.waiting = not self.headless and self.mode == "step" and not start_on_reset
        self.paused = False
        self.metrics = {}
        self.support_active = True
        self.failure_snapshot_pending = False
        self.contact_lost_since = self.stationary_reference = None
        if self.bound:
            self.joint.CreateJointEnabledAttr(False)
            self.body.CreateKinematicEnabledAttr(False)
            self.ccd.CreateEnableCCDAttr(False)
        write_json(self.trial_directory / "settings.json", {
            "task": self.task, "simulation": self.config, "runtime": self.runtime,
            "grasp_rad": self.runtime["grasp"],
            "perception": self.perception, "grasp_mode": self.grasp_mode, "test_data": True})
        self.report()

    @property
    def trial_directory(self):
        return self.output / "trials" / f"trial_{self.reset_index:03d}"

    def scene_ready(self, drives):
        write_json(self.output / "actual_drives.json", drives)
        import os
        self.event("scene_ready", revision=self.runtime.get("revision", "startup"), pid=os.getpid())

    def setup_gui(self):
        from desktop_layout import DesktopLayout, layout_settings
        settings = layout_settings(self.task.get("desktop_layout"))
        if settings["enabled"]:
            self.desktop = DesktopLayout(settings, self.camera_paths, self.output)
            self.desktop.restore()

    def prepare_gui(self):
        from desktop_layout import layout_settings, prepare_extensions
        if layout_settings(self.task.get("desktop_layout"))["enabled"]:
            prepare_extensions()

    def restore_desktop(self):
        if self.desktop is not None:
            self.desktop.restore()

    def event(self, kind, **values):
        value = {"kind": kind, "phase": self.phase, "time_ns": (self.last_state or {}).get("time_ns"), **values}
        self.events.write(json.dumps(value, allow_nan=False) + "\n"); self.events.flush()
        print("Preinsert: " + json.dumps(value), flush=True)

    def report(self, reason=None):
        if reason is not None:
            self.reason = reason
        result = {"status": self.status, "phase": self.phase, "reason": self.reason, "test_data": True,
                  "trial": self.reset_index, "revision": self.runtime.get("revision", "startup"),
                  "simulation_only": True, "perception": self.perception, "grasp_model": self.grasp_mode,
                  "physical_grasp_validated": self.grasp_mode == "physics" and self.status in ("simulation_preinsert_passed", "simulation_grasp_passed"),
                  "grasp_template": self.task.get("grasp_template", "legacy_geometry"),
                  "real_grasp_validated": False, "end_stage": self.task.get("until", "preinsert"), "insertion_executed": False,
                  "delivery_support_active": self.support_active,
                  "requested_stop": self.task.get("until", "preinsert"),
                  "waiting": self.waiting, "metrics": self.metrics}
        write_json(self.output / "result.json", result)
        write_json(self.trial_directory / "result.json", result)
        if self.label is not None:
            self.label.text = (f"{self.phase}: {self.status}" + (" | Next stage" if self.waiting else "")
                               + (f"\n{self.reason}" if self.reason else ""))
            if self.config_message:
                self.label.text += "\n" + self.config_message

    def finish(self, status, reason=None):
        if self.finished:
            return
        self.finished, self.status = True, status
        self.should_stop = self.headless
        self.pause_physics_requested = status == "failed" and not self.headless
        self.failure_snapshot_pending = status == "failed" and self.last_state is not None
        self.snapshot_requested = self.failure_snapshot_pending
        if self.failure_snapshot_pending:
            # Rendering can happen several physics steps later. Preserve the
            # trigger state separately from the eventual diagnostic image.
            failure = {
                "phase": self.phase, "reason": reason, "test_data": True,
                "trial": self.reset_index, "state": self.last_state, "metrics": self.metrics,
            }
            write_json(self.output / "failure_state.json", failure)
            write_json(self.trial_directory / "failure_state.json", failure)
        self.event("finished", status=status, reason=reason)
        self.report(reason)

    def bind_simulation(self, *, world, robot, rail, robot_path, camera_paths):
        import omni.usd
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade
        self.world, self.robot, self.rail = world, robot, rail
        self.robot_path = robot_path
        self.stage = omni.usd.get_context().get_stage()
        self.camera_path = camera_paths[0]
        self.camera_paths = camera_paths
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
            resources = getattr(self, "render_resources", None)
            if resources is not None:
                resources.product(self.product)
            self.depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane", device="cpu")
            self.segmentation = rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": False}, device="cpu")
            if resources is not None:
                resources.annotator(self.depth)
                resources.annotator(self.segmentation)
            self.depth.attach(self.product); self.segmentation.attach(self.product)
        self.bound = True

    def flange(self, state):
        # eef is an actual physics link transform in the URDF joint-origin base.
        return self.model.base_pose("left") @ pose_matrix(state["eef"][0])

    def before_physics_step(self):
        if self.support_active and not self.body.GetKinematicEnabledAttr().Get():
            self.body.CreateKinematicEnabledAttr(True)
        if self.support_active and self.locked_grasp is not None:
            grasp = self.locked_grasp
            self.rail.set_world_pose(grasp.rail_xyz, grasp.quaternion[[3, 0, 1, 2]])
        if self.pending_placement:
            W_S = self.flange(self.last_state) @ pose(self.task["rail_in_flange_xyz_m"], self.task["rail_in_flange_rpy_deg"])
            q = Rotation.from_matrix(W_S[:3, :3]).as_quat()
            self.rail.set_world_pose(W_S[:3, 3], q[[3, 0, 1, 2]])
            self.pending_placement = False
            self.event("human_delivery_simulated", T_W_S=W_S.tolist(), note="One-time placement; not a grasp policy")
        if self.pending_attachment:
            self.grip = np.linalg.inv(self.flange(self.last_state)) @ pose_matrix(self.last_state["rail"])
            if self.grasp_mode in ("fixed", "contact_fixed"):
                q = Rotation.from_matrix(self.grip[:3, :3]).as_quat()
                self.joint.CreateLocalPos0Attr(self.Gf.Vec3f(*self.grip[:3, 3]))
                self.joint.CreateLocalRot0Attr(self.Gf.Quatf(float(q[3]), self.Gf.Vec3f(*q[:3])))
                self.joint.CreateJointEnabledAttr(True)
            self.body.CreateKinematicEnabledAttr(False)
            self.ccd.CreateEnableCCDAttr(True)
            self.support_active = False
            self.pending_attachment = False
            self.event("delivery_support_removed", fixed_constraint=self.grasp_mode in ("fixed", "contact_fixed"))

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
                   "revision": self.runtime.get("revision", "startup"),
                   "frames": self.frames, "left_joint_history": list(self.history),
                   "trial": self.reset_index, "task_settings": {
                       key: self.task[key] for key in ("joint_speed_rad_s", "joint_acceleration_rad_s2")}})
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
        if self.phase == "settle_locked_grasp" and self.elapsed >= self.task["settle_s"]:
            from locked_grasp import LockedGrasp
            self.locked_grasp = LockedGrasp(self, state)
            self.transition("locked_grasp_close")
        elif self.phase == "locked_grasp_close":
            desired = self.locked_grasp.step(state, dt)
        elif self.phase == "settle_four_fingers" and self.elapsed >= self.task["settle_s"]:
            from four_fingers import FourFingerClosure
            grasp = np.array(self.runtime["grasp"]).copy()
            if "four_finger_flexion_limit_rad" in self.task:
                for i in range(4):
                    grasp[np.array([4+4*i, 6+4*i, 7+4*i])] = self.task["four_finger_flexion_limit_rad"]
            self.four_fingers = FourFingerClosure(self.command, grasp, self.task["hand_transition_s"], state)
            write_json(self.trial_directory / "four_fingers_start.json", state)
            self.transition("four_finger_close")
        elif self.phase == "four_finger_close":
            desired = self.four_fingers.step(state, dt)
            if self.four_fingers.phase == "check":
                if self.four_geometry_id is None:
                    self.four_geometry_id = uuid.uuid4().hex
                    self.four_geometry_started = time.monotonic()
                    write_json(self.output / "rpc/request.json", {"id": self.four_geometry_id,
                               "stage": "four_fingers_check", "state": state, "trial": self.reset_index,
                               "revision": self.runtime.get("revision", "startup")})
                else:
                    path = self.output / "rpc/response.json"
                    response = json.loads(path.read_text()) if path.is_file() else {}
                    if response.get("id") == self.four_geometry_id:
                        self.four_geometry_id = None
                        if response.get("ok"):
                            self.four_fingers.review(response["geometry"])
                        else:
                            self.finish("failed", response.get("error", "Four-finger geometry check failed"))
                    elif time.monotonic()-self.four_geometry_started > 60:
                        self.finish("failed", "Four-finger geometry check timed out")
            else:
                self.four_geometry_id = None
            for event in self.four_fingers.events:
                self.event(**event)
            self.four_fingers.events.clear()
            if self.four_fingers.failure:
                self.finish("failed", self.four_fingers.failure)
            elif self.four_fingers.done:
                self.snapshot_requested = True
                self.transition("four_finger_review")
            elif self.elapsed > 300:
                self.finish("failed", "Incremental four-finger closure exceeded 300 simulated seconds")
        elif self.phase == "settle_teach" and self.elapsed >= self.task["settle_s"]:
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
        if self.phase == "locked_grasp_review" and images is not None and not self.finished:
            from PIL import Image
            directory = self.output / "captures" / f"trial_{self.reset_index:03d}" / "locked_grasp"
            directory.mkdir(parents=True, exist_ok=True)
            for i, rgb in enumerate(images):
                Image.fromarray(rgb).save(directory / f"camera_{i}.png")
            write_json(directory / "state.json", state)
            self.snapshot_requested = False
            self.finish("constrained_grasp_complete", "Five distal contacts verified; fixed flange/rail constraint enabled")
            self.pause_physics_requested = not self.headless
            return
        if self.phase == "four_finger_review" and images is not None and not self.finished:
            from four_fingers import FINGERS
            from PIL import Image
            directory = self.output / "captures" / f"trial_{self.reset_index:03d}" / "four_fingers"
            directory.mkdir(parents=True, exist_ok=True)
            for i, rgb in enumerate(images):
                Image.fromarray(rgb).save(directory / f"camera_{i}.png")
            write_json(directory / "state.json", state)
            missing = [f for f in FINGERS if finger_forces(state)[f] < .2]
            self.metrics["four_fingers_missing_contact"] = missing
            self.metrics["four_fingers_contact_stable_s"] = self.four_fingers.stable
            self.metrics["four_fingers_blocked"] = [FINGERS[i] for i in sorted(self.four_fingers.blocked)]
            self.metrics["four_fingers_geometry"] = self.four_fingers.last_review
            write_json(self.trial_directory / "four_fingers_pose.json", {
                "simulation_only": True, "delivery_support_active": True,
                "joint_names": self.model.hand_names["left"],
                "command_position_rad": self.command[14:34].tolist(),
                "measured_position_rad": state["q"][14:34],
                "finger_contact_force_n": finger_forces(state)})
            self.snapshot_requested = False
            supported = not missing and self.four_fingers.stable >= .5
            self.finish("supported_four_finger_contact" if supported else "four_finger_preview_complete",
                        "Delivery support retained; thumb unchanged; no unsupported grasp validation")
            self.pause_physics_requested = not self.headless
            return
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
            self.metrics["grasp_translation_limit_m"] = position_limit
            self.metrics["grasp_rotation_limit_rad"] = rotation_limit
            if error[0] > position_limit or error[1] > rotation_limit:
                self.finish("failed", "Rail slipped relative to flange; "
                            f"translation {error[0]*1000:.2f} mm (limit {position_limit*1000:.2f} mm), "
                            f"rotation {np.degrees(error[1]):.2f} deg (limit {np.degrees(rotation_limit):.2f} deg)")
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
        self.label = ui.Label("Preinsert: teach -> close -> measure -> preinsert", height=70, word_wrap=True)
        ui.Button("Restart task", clicked_fn=self.request_reset)
        ui.Button("Apply config / Restart task", clicked_fn=self.request_reload)
        ui.Button("Apply config / Rebuild scene", clicked_fn=self.request_rebuild)
        ui.Button("Start / Next stage", clicked_fn=self.next_stage)
        ui.Button("Run automatically", clicked_fn=self.automatic)
        ui.Button("Pause / Resume physics", clicked_fn=self.toggle_physics)
        ui.Button("Restore default layout", clicked_fn=self.restore_desktop)
        self.report()

    def next_stage(self):
        if self.finished:
            self.request_reset()
            return
        self.waiting = False
        if self.bound:
            self.physics_action_requested = "play"
        self.report()

    def toggle_physics(self):
        if self.bound:
            self.physics_action_requested = "pause" if self.finished else "toggle"
            if self.finished:
                self.config_message = "Task finished. Use Start / Next stage or Restart task for a new trial."
                self.report()

    def request_reset(self):
        if not self.bound:
            print("Wait for the scene to finish loading.", flush=True)
            return
        if self.reload_requested or self.reload_id:
            print("Wait for configuration reload to finish.", flush=True)
            return
        if not self.finished:
            self.finish("interrupted", "Task reset by operator")
        self.snapshot_requested = self.failure_snapshot_pending = False
        self.pause_physics_requested = False
        self.start_on_reset = True
        self.reset_requested = True
        self.physics_action_requested = "play"

    def request_reload(self):
        if (not self.bound and not self.force_rebuild) or self.reload_id or self.reload_requested:
            return
        self.reload_requested = True

    def request_rebuild(self):
        if self.reload_id or self.reload_requested or self.scene_rebuild_requested:
            return
        self.force_rebuild = True
        self.request_reload()

    def service_lifecycle(self):
        """Keep completed trials paused, including native toolbar Play events."""
        if self.finished and not self.reload_id:
            if (self.bound and not self.snapshot_requested and not self.reset_requested and not self.reload_requested and
                    not self.reload_id and self.world.is_playing()):
                self.physics_action_requested = "pause"
                message = "Task finished. Use Start / Next stage or Restart task for a new trial."
                if self.config_message != message:
                    self.config_message = message
                    self.event("completed_trial_play_blocked")
                    self.report()
            return
        grasp = self.locked_grasp
        request_id = self.reload_id or getattr(grasp, "request_id", None) or self.request_id or self.four_geometry_id
        if request_id is None:
            return
        path = self.output / "rpc/request.json"
        if not path.is_file() or json.loads(path.read_text()).get("id") != request_id:
            return
        response = self.output / "rpc/response.json"
        if response.is_file() and json.loads(response.read_text()).get("id") == request_id:
            return
        # The worker heartbeat runs on another thread even during long solves.
        heartbeat = self.output / "rpc/worker.json"
        health = json.loads(heartbeat.read_text()) if heartbeat.is_file() else {}
        now = time.time()
        if now-path.stat().st_mtime < 5.:
            return
        if health.get("status") in ("idle", "busy") and now-health.get("updated_at", 0.) < 5.:
            return
        self.reload_id = None
        self.reload_requested = self.force_rebuild = False
        self.finished = False
        self.config_message = "Planner offline. Reconnect the host service, then restart the task."
        self.finish("failed", self.config_message)
        # A paused scene cannot produce a fresh physics-labelled snapshot.
        # finish() already preserves the last measured state on disk.
        self.snapshot_requested = self.failure_snapshot_pending = False
        self.physics_action_requested = "pause"

    def service_reload(self):
        """Called by the engine even while paused or waiting after an error."""
        self.service_lifecycle()
        if self.reload_requested:
            self.reload_requested = False
            if not self.finished:
                self.finish("interrupted", "Applying configuration")
            self.snapshot_requested = self.failure_snapshot_pending = False
            self.physics_action_requested = "pause"
            self.reload_id = uuid.uuid4().hex
            self.reload_started = time.monotonic()
            self.config_message = "Loading " + str(self.runtime.get("config_path", "configuration"))
            write_json(self.output / "rpc/request.json", {"id": self.reload_id, "stage": "reload",
                       "revision": self.runtime.get("revision", "startup"), "rebuild": self.force_rebuild})
            self.report()
        if self.reload_id is None:
            return
        path = self.output / "rpc/response.json"
        response = json.loads(path.read_text()) if path.is_file() else {}
        if response.get("id") != self.reload_id:
            if time.monotonic() - self.reload_started < 240:
                return
            response = {"ok": False, "error": "Configuration reload timed out"}
        self.reload_id = None
        if not response.get("ok"):
            self.force_rebuild = False
            self.config_message = "Config rejected: " + response.get("error", "Unknown error")
            self.event("config_rejected", error=self.config_message)
            self.report()
            return
        self.pending_configuration = response["configuration"]
        if self.force_rebuild or response.get("operation") == "rebuild":
            self.force_rebuild = False
            self.scene_rebuild_requested = True
            self.config_message = "Rebuilding scene in the same Isaac application"
            self.report()
            return
        self.config_message = "Applying validated configuration"
        self.request_reset()

    def take_scene_replacement(self):
        """Called only by the engine, outside UI/render/physics callbacks."""
        import copy
        from functools import partial

        from bimanual_teleop.sim.scene_lifecycle import SceneReplacement
        supplied = copy.deepcopy(self.pending_configuration)
        supplied["simulation"]["paths"]["recordings"] = self.config["paths"]["recordings"]
        self.event("scene_rebuild_requested", revision=supplied["runtime"]["revision"])
        # Capture only data; retaining self here would retain all old native handles.
        factory = partial(type(self), task=supplied["task"], output=self.output, runtime=supplied["runtime"],
                          mode=self.mode, headless=self.headless, perception=self.perception, grasp=self.grasp_mode,
                          trial_start=self.reset_index + 1)
        import yaml
        (self.output / "effective.yaml").write_text(yaml.safe_dump(
            {**supplied["simulation"], "experiment": supplied["task"]}, allow_unicode=True))
        write_json(self.output / "runtime.json", supplied["runtime"])
        return SceneReplacement(supplied["simulation"], factory)

    def install_pending_settings(self):
        """Install a host-validated revision only at the reset boundary."""
        if self.pending_configuration is None:
            return
        from task_settings import ready_joints
        supplied = self.pending_configuration
        ready = ready_joints(supplied["simulation"], supplied["runtime"], self.model)
        recordings = self.config["paths"]["recordings"]
        self.config.clear()
        self.config.update(supplied["simulation"])
        self.config["paths"]["recordings"] = recordings
        self.task, self.runtime, self.ready = supplied["task"], supplied["runtime"], ready
        self.configure_gains()
        if self.bound and not self.headless:
            from desktop_layout import layout_settings
            settings = layout_settings(self.task.get("desktop_layout"))
            if settings["enabled"] and self.desktop is None:
                self.setup_gui()
            elif self.desktop is not None and settings != self.desktop.settings and settings["enabled"]:
                self.desktop.settings = settings
                self.desktop.restore()
        import yaml
        effective = {**self.config, "experiment": self.task}
        (self.output / "effective.yaml").write_text(yaml.safe_dump(effective, allow_unicode=True))
        write_json(self.output / "runtime.json", self.runtime)
        self.event("configuration_installed", revision=self.runtime["revision"])
        self.pending_configuration = None

    def update_scene_physics(self):
        from pxr import PhysxSchema, Usd, UsdPhysics
        rail = self.config["rail"]
        old_mass = float(self.rail.get_mass())
        view = self.rail._rigid_prim_view
        inertias = np.asarray(view.get_inertias()).copy()
        self.rail.set_mass(rail["mass_kg"])
        view.set_inertias(inertias * (rail["mass_kg"] / old_mass))
        UsdPhysics.MassAPI(self.stage.GetPrimAtPath("/World/Rail")).GetMassAttr().Set(rail["mass_kg"])
        for name in ("Rail", "Fixture", "PinTops", "CameraHousing", "Fingers"):
            material = UsdPhysics.MaterialAPI(self.stage.GetPrimAtPath("/World/Materials/" + name))
            friction = rail["finger_friction"] if name == "Fingers" else rail["friction"]
            material.GetStaticFrictionAttr().Set(friction)
            material.GetDynamicFrictionAttr().Set(friction)
        roots = ["/World/Rail"] + [self.robot_path + "/" + name for name in self.model.collidable_links
                                  if name.startswith(("l_", "r_"))]
        for root in roots:
            for prim in Usd.PrimRange(self.stage.GetPrimAtPath(root)):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                old_contact = collision.GetContactOffsetAttr().Get()
                if old_contact is not None and rail["rest_offset_m"] < old_contact:
                    collision.CreateRestOffsetAttr(rail["rest_offset_m"])
                collision.CreateContactOffsetAttr(rail["contact_offset_m"])
                collision.CreateRestOffsetAttr(rail["rest_offset_m"])
        self.config_message = "Configuration applied"
        write_json(self.output / "applied_physics.json", {
            "revision": self.runtime.get("revision", "startup"), "rail_mass_kg": float(self.rail.get_mass()),
            "inertia_kg_m2": np.asarray(view.get_inertias()).tolist(),
            "rail_friction": rail["friction"], "finger_friction": rail["finger_friction"],
            "contact_offset_m": rail["contact_offset_m"], "rest_offset_m": rail["rest_offset_m"]})

    def before_reset(self):
        # Run outside UI callbacks: Kit rendering must not be re-entered by a
        # synchronous World.play/reset while it dispatches a button event.
        self.joint.CreateJointEnabledAttr(False)
        self.body.CreateKinematicEnabledAttr(False)
        self.ccd.CreateEnableCCDAttr(False)

    def automatic(self):
        self.mode = "auto"
        self.next_stage()

    def close(self, error=None):
        if self.desktop is not None:
            self.desktop.close()
        if not self.finished:
            self.finish("failed" if error else "interrupted", error or "Simulation closed before acceptance")
        self.trace.close(); self.events.close()
