"""Container-only Isaac engine. No real robot SDK is imported here."""

from __future__ import annotations

import argparse
import time

from bimanual_teleop.sim._dependencies import np

from . import ROOT
from .config import load_config, sim_path
from .kinematics import RobotModel, pose_vector
from .protocol import ImageRing, pack, unpack


def run(config, endpoint, *, gui=False, standalone=False, record=False, trajectory=None, steps=0, codec="h264_nvenc",
        action_source_factory=None):
    if action_source_factory is not None and (not standalone or trajectory is not None):
        raise ValueError("Custom action sources require standalone mode without a trajectory")
    # SimulationApp must exist before importing any Omniverse/pxr modules.
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": not gui, "hide_ui": not gui, "renderer": "RaytracedLighting",
                         "width": 1280 if gui else 640, "height": 800 if gui else 480,
                         "anti_aliasing": 1, "disable_viewport_updates": not gui, "limit_cpu_threads": 8})
    import zmq
    from isaacsim.core.api import World
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.core.prims import SingleRigidPrim
    from isaacsim.core.api.sensors import RigidContactView
    import omni.replicator.core as rep
    import carb
    carb.settings.get_settings().set_int("/persistent/physics/numThreads", 0)
    from .rendering import configure_camera_rendering
    configure_camera_rendering()
    from .scene import build_scene, rgb_frames
    rep.orchestrator.set_capture_on_play(False)

    context = socket = ring = local = source = None
    failure = None
    capture = not standalone or record
    if not standalone:
        context = zmq.Context(); socket = context.socket(zmq.DEALER)
        socket.setsockopt(zmq.IDENTITY, b"isaac-engine"); socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDHWM, 1024); socket.setsockopt(zmq.SNDTIMEO, 1000); socket.connect(endpoint)
    try:
        model = RobotModel(sim_path(config["robot"]["urdf"]), config["robot"]["world_yaw_deg"])
        source = action_source_factory(config, model) if action_source_factory is not None else None
        capture = capture or bool(source is not None and source.capture_snapshots)
        world = World(physics_dt=1/config["runtime"]["physics_hz"], rendering_dt=1/config["runtime"]["camera_hz"], stage_units_in_meters=1.)
        world.get_physics_context().enable_gpu_dynamics(False)
        world.get_physics_context().set_broadphase_type("MBP")
        world.get_physics_context().enable_ccd(True)
        robot, annotators, geometry, robot_path, camera_paths = build_scene(world, config, model, capture=capture)
        if standalone:
            from .local_session import LocalSession
            local = LocalSession(config, model, record, codec, trajectory, gui,
                                 status_path=source.output/"session.json" if source is not None else None,
                                 action_source=source)
        if gui:
            from isaacsim.core.utils.viewports import set_camera_view
            from omni.kit.viewport.utility import get_active_viewport

            def frame_scene():
                viewport = get_active_viewport()
                if viewport is not None:
                    viewport.camera_path = "/OmniverseKit_Persp"
                    viewport.fill_frame = True
                set_camera_view(eye=np.array([2.5, -2.5, 2.0]), target=np.array([.45, 0., .72]))

            def view_head_camera():
                viewport = get_active_viewport()
                if viewport is not None:
                    viewport.camera_path = camera_paths[0]
                    viewport.fill_frame = False
                    scale = config["cameras"]["preview_scale"]
                    viewport.resolution = (config["cameras"]["width"]*scale, config["cameras"]["height"]*scale)

            frame_scene()
            if local:
                local.build_ui(frame_scene, view_head_camera)
        rail = world.scene.add(SingleRigidPrim("/World/Rail", name="rail"))
        if source is not None and hasattr(source, "bind_simulation"):
            source.bind_simulation(world=world, robot=robot, rail=rail, robot_path=robot_path,
                                   camera_paths=camera_paths)
        contact_views, contact_names = [], []
        for prefix in ("l", "r"):
            paths = [robot_path+"/"+link for link in model.collidable_links if link.startswith(prefix+"_")]
            contact_views.append(RigidContactView(paths, [["/World/Rail"] for _ in paths],
                                                  name=f"{prefix}_rail_contacts", max_contact_count=4096))
            contact_names.append([p.rsplit("/", 1)[-1] for p in paths])
        environment_view = None
        if source is not None:
            environment_links = list(model.collidable_links)
            paths = [robot_path+"/"+link.replace(" ", "_") for link in environment_links]
            filters = ["/World/Fixture/Collision", "/World/Table/Mesh",
                       f"/World/Bin_0/Layer_{config['scene']['bin_stack_count']}/Mesh",
                       f"/World/Bin_1/Layer_{config['scene']['bin_stack_count']}/Mesh", "/World/Rail", robot_path+"/base_link"]
            paths.append("/World/Rail")
            environment_view = RigidContactView(paths, [filters for _ in paths], name="experiment_environment_contacts",
                                                max_contact_count=8192)
        world.reset()
        robot.set_solver_position_iteration_count(16)
        robot.set_solver_velocity_iteration_count(4)
        indices = np.array([robot.get_dof_index(name) for name in model.names], dtype=np.int32)
        ready = np.r_[np.deg2rad(config["robot"]["ready_deg"]["left"]), np.deg2rad(config["robot"]["ready_deg"]["right"]), np.zeros(40)]
        ready = np.clip(ready, model.lower, model.upper)
        if source is not None:
            ready = source.ready.copy()
        robot.set_joint_positions(ready, joint_indices=indices)
        gains = np.r_[np.full(14, config["robot"]["arm_stiffness"]), np.tile([2, 2, 1, .8], 10)]
        damping = np.r_[np.full(14, config["robot"]["arm_damping"]), np.tile([.05, .05, .03, .03], 10)]
        if source is not None:
            for name, supplied in getattr(source, "joint_gain_overrides", {}).items():
                values = np.asarray(supplied, dtype=float)
                if name not in model.names or values.shape != (2,) or not np.isfinite(values).all() or np.any(values <= 0):
                    raise ValueError("Invalid simulation joint gain override")
                index = model.names.index(name)
                gains[index], damping[index] = values
        controller = robot.get_articulation_controller()
        # SingleArticulationController.set_gains accepts the full DOF arrays.
        full_kp, full_kd = controller.get_gains()
        full_kp[indices], full_kd[indices] = gains, damping
        controller.set_gains(kps=full_kp, kds=full_kd)
        controller.set_max_efforts(model.effort, joint_indices=indices)
        target = ready.copy(); eef_target = [pose_vector(model.flange(side, ready)).tolist() for side in ("left", "right")]
        command_tick = 0; command_deadline = 0; command_id = "initial_hold"
        camera_stride = config["runtime"]["physics_hz"] // config["runtime"]["camera_hz"]
        if socket is not None:
            socket.send(pack({"kind": "ready", "camera_paths": camera_paths}))
            deadline = time.monotonic()+30
            settings = None
            while time.monotonic() < deadline:
                if not socket.poll(1000):
                    continue
                message = unpack(socket.recv())
                if message["kind"] == "stop":
                    return
                if message["kind"] == "initialize":
                    settings = message; break
            if settings is None:
                raise RuntimeError("Simulation host did not provide an image ring")
            ring = ImageRing(**settings["ring"])
        start = time.monotonic(); tick = 0
        for _ in range(16):
            world.step(render=True)
        if capture:
            for _ in range(4):
                rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)
        world.reset(); robot.set_joint_positions(ready, joint_indices=indices)
        for view in contact_views:
            view.initialize()
        if environment_view is not None:
            environment_view.initialize()
        # Fixed-joint incoming forces are indexed by the child rigid link.
        tensor_view = robot._articulation_view._physics_view
        link_names = list(tensor_view.shared_metatype.link_names)
        if link_names and isinstance(link_names[0], (list, tuple)):
            link_names = list(link_names[0])
        force_links = (("l_force_sensor", "r_force_sensor") if config["scene"]["asset_profile"] == "rail_task_cad_v1"
                       else ("l_mount", "r_mount"))
        force_indices = [link_names.index(name) for name in force_links]
        flange_indices = [link_names.index(name) for name in ("TCP_Link_L", "TCP_Link_R")]
        applied_target = ready.copy(); applied_eef_target = eef_target
        applied_tick = 0; applied_command_id = command_id
        def report_source_drives():
            if source is not None and hasattr(source, "scene_ready"):
                kp, kd = controller.get_gains()
                source.scene_ready({"joint_names": list(model.names),
                    "stiffness": np.asarray(kp)[indices].tolist(), "damping": np.asarray(kd)[indices].tolist(),
                    "max_efforts": np.asarray(controller.get_max_efforts())[indices].tolist(),
                    "joint_limits": np.asarray(tensor_view.get_dof_limits())[0, indices].tolist()})
        report_source_drives()
        print(f"Isaac scene ready; 54 joints; {'three RGB cameras' if capture else 'view only'}; {'native GUI' if gui else 'headless'}", flush=True)
        start = time.monotonic()
        last_host_ns = time.monotonic_ns()
        timeline_stopped = False
        timeline_paused = False
        while app.is_running():
            if steps and tick >= steps:
                break
            if source is not None and source.should_stop and not source.snapshot_requested:
                break
            if gui and not world.is_playing():
                if local and not timeline_paused and local.capturing and local.recorder:
                    local.recorder.pause(tick, "timeline_paused"); local.pending.clear()
                timeline_paused = True
                timeline_stopped = timeline_stopped or world.is_stopped()
                app.update(); time.sleep(.01)
                # Do not try to catch up all wall time spent in the GUI pause.
                start = time.monotonic()-tick/config["runtime"]["physics_hz"]
                continue
            if local and timeline_paused and local.capturing and local.recorder:
                local.recorder.resume(tick)
            timeline_paused = False
            now_ns = time.monotonic_ns()
            messages = []
            repeat_reset = source is not None and getattr(source, "reset_requested", False) and not source.snapshot_requested
            if local and (local.reset_requested or timeline_stopped or repeat_reset):
                local.reset_requested = False; timeline_stopped = False
                local.save(); local.task.reset()
                messages.append({"kind": "reset"})
            while socket is not None and socket.poll(0):
                messages.append(unpack(socket.recv()))
                last_host_ns = time.monotonic_ns()
            for message in messages:
                if message["kind"] == "stop":
                    return
                if message["kind"] == "reset":
                    world.reset(); robot.set_joint_positions(ready, joint_indices=indices)
                    for view in contact_views:
                        view.initialize()
                    if environment_view is not None:
                        environment_view.initialize()
                    tensor_view = robot._articulation_view._physics_view
                    target = ready.copy(); command_deadline = 0
                    eef_target = [pose_vector(model.flange(side, ready)).tolist() for side in ("left", "right")]
                    command_tick = tick; command_id = "reset_hold"
                    controller.apply_action(ArticulationAction(joint_positions=ready, joint_indices=indices))
                    applied_target, applied_eef_target = ready.copy(), eef_target
                    applied_tick, applied_command_id = tick, command_id
                    from ._dependencies import Rotation
                    quaternion = Rotation.from_euler("xyz", config["rail"]["initial_rpy_deg"], degrees=True).as_quat()
                    rail.set_world_pose(np.array(config["rail"]["initial_center"]), quaternion[[3, 0, 1, 2]])
                    rail.set_linear_velocity(np.zeros(3)); rail.set_angular_velocity(np.zeros(3))
                    if local:
                        local.on_reset(tick)
                    report_source_drives()
                elif message["kind"] == "target":
                    supplied = np.asarray(message["q"], dtype=float)
                    if supplied.shape != (54,) or not np.isfinite(supplied).all():
                        raise ValueError("Invalid simulation joint target")
                    target = np.clip(supplied, model.lower, model.upper)
                    eef_target = message["eef_target"]; command_tick = tick
                    command_deadline = message["expires_ns"]; command_id = message["command_id"]
            if socket is not None and now_ns-last_host_ns > 2_000_000_000:
                raise RuntimeError("Simulation host heartbeat expired")
            if command_deadline and now_ns >= command_deadline:
                target = robot.get_joint_positions(joint_indices=indices).copy()
                eef_target = [pose_vector(model.flange(side, target)).tolist() for side in ("left", "right")]
                command_deadline = 0; command_id = "expired_hold"; command_tick = tick
            if tick % (config["runtime"]["physics_hz"]//config["runtime"]["control_hz"]) == 0:
                if local:
                    target = local.targets(tick)
                    eef_target = [pose_vector(model.flange(side, target)).tolist() for side in ("left", "right")]
                    command_id = source.name if source is not None else ("joint_trajectory" if local.program else "local_hold")
                    command_tick = tick
                controller.apply_action(ArticulationAction(joint_positions=target, joint_indices=indices))
                applied_target = target.copy(); applied_eef_target = eef_target
                applied_tick = tick; applied_command_id = command_id
            before_physics = world.current_time if source is not None else None
            if source is not None and hasattr(source, "before_physics_step"):
                source.before_physics_step()
            world.step(render=False); tick += 1
            if source is not None and abs(world.current_time-before_physics-1/config["runtime"]["physics_hz"]) > 1e-6:
                raise RuntimeError("Physics step did not advance the simulation clock by the configured timestep")
            q = robot.get_joint_positions(joint_indices=indices)
            eef = []
            link_poses = tensor_view.get_link_transforms()[0]
            for side, flange_index in zip(("left", "right"), flange_indices):
                from .kinematics import pose_matrix
                actual_world = pose_matrix(link_poses[flange_index])
                actual_base = np.linalg.inv(model.base_pose(side)) @ actual_world
                eef.append(pose_vector(actual_base).tolist())
            forces = robot.get_measured_joint_forces(joint_indices=np.array(force_indices))
            rail_position, rail_quaternion = rail.get_world_pose()
            state = {"kind": "state", "tick": tick, "time_ns": round(tick*1e9/config["runtime"]["physics_hz"]),
                     "q": q.tolist(), "eef": eef, "wrench": forces.reshape(2, 6).tolist(),
                     "q_target": applied_target.tolist(), "eef_target": applied_eef_target,
                     "command_tick": applied_tick, "command_id": applied_command_id,
                     "rail": np.r_[rail_position, rail_quaternion[[1, 2, 3, 0]]].tolist()}
            matrices = [np.asarray(view.get_contact_force_matrix(dt=1/config["runtime"]["physics_hz"])) for view in contact_views]
            state["hand_contact_force"] = [float(np.linalg.norm(matrix, axis=-1).sum()) for matrix in matrices]
            if source is not None:
                state["physics_time_s"] = float(world.current_time)
                state["rail_contacts"] = {name: float(force) for names, matrix in zip(contact_names, matrices)
                                          for name, force in zip(names, np.linalg.norm(matrix, axis=-1).sum(axis=1)) if force > 1e-6}
                state["rail_contact_vectors"] = {name: vector.tolist() for names, matrix in zip(contact_names, matrices)
                                                  for name, vector in zip(names, matrix.sum(axis=1)) if np.linalg.norm(vector) > 1e-6}
                forces = np.linalg.norm(np.asarray(environment_view.get_contact_force_matrix(dt=1/config["runtime"]["physics_hz"])), axis=-1)
                state["environment_contacts"] = {name: float(force) for name, force in zip(environment_links, forces[:-1, :4].sum(axis=1)) if force > 1e-6}
                state["rail_support_contacts"] = {name: float(force) for name, force in zip(("fixture", "table", "input_bin", "output_bin"), forces[-1, :4]) if force > 1e-6}
                state["robot_base_contacts"] = {name: float(force) for name, force in zip(environment_links, forces[:-1, 5]) if force > 1e-6}
                state["rail_robot_base_force_n"] = float(forces[-1, 5])
                state["arm_joint_effort"] = robot.get_measured_joint_efforts(joint_indices=indices[:14]).tolist()
                state["arm_drive_target"] = np.asarray(tensor_view.get_dof_position_targets())[0, indices[:14]].tolist()
                state["rail_contacts"].update({name: float(force) for name, force in zip(environment_links, forces[:-1, 4])
                                               if force > 1e-6 and not name.startswith(("l_", "r_"))})
            images = None
            if capture and tick % camera_stride == 0 and (not standalone or local.capturing or
                                                          (source is not None and source.snapshot_requested)):
                before = world.current_time
                rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)
                if abs(world.current_time-before) > 1e-8:
                    raise RuntimeError("Synchronous camera rendering advanced physics; cannot label these frames with the frozen state")
                images = rgb_frames(annotators)
                if any(image.shape != (config["cameras"]["height"], config["cameras"]["width"], 3) for image in images):
                    raise RuntimeError("Camera render did not produce all three RGB frames")
                if ring is not None:
                    state["image_slot"] = ring.put(np.stack(images))
            elif gui and tick % camera_stride == 0:
                world.render()
            if local:
                local.accept(state, np.stack(images) if images is not None else None)
            if socket is not None:
                socket.send(pack(state))
            delay = start+tick/config["runtime"]["physics_hz"]-time.monotonic()
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        print("Simulation stopped by operator; saving recording", flush=True)
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        print(f"Isaac engine error: {type(error).__name__}: {error}", flush=True)
        try:
            if socket is not None:
                socket.send(pack({"kind": "error", "error": str(error)}))
        except Exception:
            pass
        raise
    finally:
        try:
            if local is not None:
                local.close(failure)
            elif source is not None:
                source.close(failure or "Scene initialization did not complete")
            if ring is not None:
                ring.close()
            if socket is not None:
                socket.close(); context.term()
        finally:
            app.close()


def main():
    parser = argparse.ArgumentParser(description="Run the Isaac engine inside the official container")
    parser.add_argument("--config")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5558")
    parser.add_argument("--standalone", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--trajectory")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--codec", choices=("h264_nvenc", "libx264"), default="h264_nvenc")
    args = parser.parse_args()
    if (args.record or args.trajectory or args.steps) and not args.standalone:
        parser.error("Local recording, trajectories and step limits require --standalone")
    run(load_config(args.config), args.endpoint, gui=args.gui, standalone=args.standalone,
        record=args.record, trajectory=args.trajectory, steps=args.steps, codec=args.codec)


if __name__ == "__main__":
    main()
