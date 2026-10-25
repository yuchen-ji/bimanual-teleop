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
        action_source_factory=None, keep_gui_on_error=False):
    if action_source_factory is not None and (not standalone or trajectory is not None):
        raise ValueError("Custom action sources require standalone mode without a trajectory")
    # SimulationApp must exist before importing any Omniverse/pxr modules.
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": not gui, "hide_ui": not gui, "renderer": "RaytracedLighting",
                         "width": 1280 if gui else 640, "height": 800 if gui else 480,
                         "anti_aliasing": 1, "disable_viewport_updates": not gui, "limit_cpu_threads": 8})
    from .scene_lifecycle import run_scenes

    def execute(app, settings, factory):
        return _run_scene(app, settings, endpoint, gui=gui, standalone=standalone, record=record,
                          trajectory=trajectory, steps=steps, codec=codec,
                          action_source_factory=factory, keep_gui_on_error=keep_gui_on_error)

    def clear_stage():
        import gc

        import omni.usd
        from isaacsim.core.api import World
        # All scene-local callbacks, recorders and render products are released
        # before returning from _run_scene. No stale tensor views cross here.
        World.clear_instance()
        omni.usd.get_context().close_stage()
        gc.collect()
        omni.usd.get_context().new_stage()
        app.update()

    try:
        run_scenes(app, config, action_source_factory, execute, clear_stage)
    finally:
        app.close()


def _run_scene(app, config, endpoint, *, gui, standalone, record, trajectory, steps, codec,
               action_source_factory, keep_gui_on_error):
    import carb
    import omni.replicator.core as rep
    import zmq
    from isaacsim.core.api import World
    from isaacsim.core.api.sensors import RigidContactView
    from isaacsim.core.prims import SingleRigidPrim
    from isaacsim.core.utils.types import ArticulationAction
    carb.settings.get_settings().set_int("/persistent/physics/numThreads", 0)
    from .rendering import configure_camera_rendering
    configure_camera_rendering()
    from .scene import build_scene, rgb_frames
    rep.orchestrator.set_capture_on_play(False)

    context = socket = ring = local = source = world = None
    failure = None
    from .scene_lifecycle import RenderResources
    resources = RenderResources()
    capture = not standalone or record
    if not standalone:
        context = zmq.Context(); socket = context.socket(zmq.DEALER)
        socket.setsockopt(zmq.IDENTITY, b"isaac-engine"); socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDHWM, 1024); socket.setsockopt(zmq.SNDTIMEO, 1000); socket.connect(endpoint)
    try:
        model = RobotModel(sim_path(config["robot"]["urdf"]), config["robot"]["world_yaw_deg"])
        source = action_source_factory(config, model) if action_source_factory is not None else None
        if gui and source is not None and hasattr(source, "prepare_gui"):
            source.prepare_gui()
        capture = capture or bool(source is not None and source.capture_snapshots)
        world = World(physics_dt=1/config["runtime"]["physics_hz"], rendering_dt=1/config["runtime"]["camera_hz"], stage_units_in_meters=1.)
        world.get_physics_context().enable_gpu_dynamics(False)
        world.get_physics_context().set_broadphase_type("MBP")
        world.get_physics_context().enable_ccd(True)
        robot, annotators, _geometry, robot_path, camera_paths = build_scene(world, config, model, capture=capture, resources=resources)
        if standalone:
            from .local_session import LocalSession
            local = LocalSession(config, model, record, codec, trajectory, gui,
                                 status_path=source.output/"session.json" if source is not None else None,
                                 action_source=source)
        if gui:
            from isaacsim.core.utils.viewports import set_camera_view
            from omni.kit.viewport.utility import get_active_viewport

            def frame_scene():
                if getattr(source, "desktop", None) is not None:
                    source.desktop.frame_scene()
                    return
                viewport = get_active_viewport()
                if viewport is not None:
                    viewport.camera_path = "/OmniverseKit_Persp"
                    viewport.fill_frame = True
                set_camera_view(eye=np.array([2.5, -2.5, 2.0]), target=np.array([.45, 0., .72]))

            def view_head_camera():
                if getattr(source, "desktop", None) is not None:
                    source.desktop.view_head_camera()
                    return
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
            source.render_resources = resources
            source.bind_simulation(world=world, robot=robot, rail=rail, robot_path=robot_path,
                                   camera_paths=camera_paths)
        import omni.usd
        from pxr import UsdPhysics

        from .collisions import inspect_robot_collisions
        from .recording import atomic_json
        stage = omni.usd.get_context().get_stage()
        # Check after task callbacks too, so the report records effective physics.
        extra = getattr(source, "task", {}).get("convex_decomposition_links", ())
        physics_report = inspect_robot_collisions(stage, robot_path, model, config, extra)
        physics_report["rail_mass_kg"] = UsdPhysics.MassAPI(stage.GetPrimAtPath("/World/Rail")).GetMassAttr().Get()
        if abs(physics_report["rail_mass_kg"]-config["rail"]["mass_kg"]) > 1e-6:
            raise RuntimeError("Configured rail mass did not reach the simulated body")
        report_path = source.output/"scene_physics.json" if source is not None else ROOT/"data/local/scene_physics.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(report_path, physics_report)
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
        if gui:
            from omni.kit.viewport.window import get_viewport_window_instances
            for window in get_viewport_window_instances():
                viewport = window.viewport_api
                if not stage.GetPrimAtPath(viewport.camera_path):
                    viewport.camera_path = "/OmniverseKit_Persp"
                viewport.updates_enabled = True
        if gui and source is not None and hasattr(source, "setup_gui"):
            source.setup_gui()
        print(f"Isaac scene ready; 54 joints; {'three RGB cameras' if capture else 'view only'}; {'native GUI' if gui else 'headless'}", flush=True)
        start = time.monotonic()
        last_host_ns = time.monotonic_ns()
        timeline_stopped = False
        timeline_paused = False
        def reset_runtime(hard=False):
            nonlocal controller, tensor_view, indices, force_indices, flange_indices, target, command_deadline
            nonlocal eef_target, command_tick, command_id, applied_target, applied_eef_target, applied_tick, applied_command_id
            nonlocal timeline_stopped, timeline_paused, start, failure, ready, gains, damping
            if local:
                local.reset_requested = False
                local.save(); local.task.reset()
            if source is not None and hasattr(source, "before_reset"):
                source.before_reset()
            if source is not None and hasattr(source, "install_pending_settings"):
                source.install_pending_settings()
                ready = source.ready.copy()
                local.ready = ready.copy()
                gains = np.r_[np.full(14, config["robot"]["arm_stiffness"]), np.tile([2, 2, 1, .8], 10)]
                damping = np.r_[np.full(14, config["robot"]["arm_damping"]), np.tile([.05, .05, .03, .03], 10)]
                for name, (kp, kd) in source.joint_gain_overrides.items():
                    index = model.names.index(name)
                    gains[index], damping[index] = kp, kd
            soft = bool(gui and keep_gui_on_error and not hard)
            world.reset(soft=soft)
            if gui and keep_gui_on_error:
                # A stopped timeline invalidates tensor handles. Rebind them
                # before using a controller, including the native error path.
                robot.initialize(physics_sim_view=world.physics_sim_view)
                rail.initialize(physics_sim_view=world.physics_sim_view)
                controller = robot.get_articulation_controller()
                indices = np.array([robot.get_dof_index(name) for name in model.names], dtype=np.int32)
                full_kp, full_kd = controller.get_gains()
                full_kp[indices], full_kd[indices] = gains, damping
                controller.set_gains(kps=full_kp, kds=full_kd)
                controller.set_max_efforts(model.effort, joint_indices=indices)
            if source is not None and hasattr(source, "update_scene_physics"):
                source.update_scene_physics()
            robot.set_joint_positions(ready, joint_indices=indices)
            for view in contact_views:
                view.initialize(physics_sim_view=world.physics_sim_view)
            if environment_view is not None:
                environment_view.initialize(physics_sim_view=world.physics_sim_view)
            tensor_view = robot._articulation_view._physics_view
            names = list(tensor_view.shared_metatype.link_names)
            if names and isinstance(names[0], (list, tuple)):
                names = list(names[0])
            force_indices = [names.index(name) for name in force_links]
            flange_indices = [names.index(name) for name in ("TCP_Link_L", "TCP_Link_R")]
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
            if not world.is_playing():
                world.play()
            timeline_stopped = timeline_paused = False
            start = time.monotonic()-tick/config["runtime"]["physics_hz"]
            failure = None
            if gui and keep_gui_on_error and source is not None and hasattr(source, "event"):
                source.event("task_restarted", physics_reset="soft" if soft else "hard")

        runtime_error = None
        hard_reset_requested = False
        while app.is_running():
            if source is not None and hasattr(source, "service_reload"):
                source.service_reload()
                if getattr(source, "scene_rebuild_requested", False):
                    return source.take_scene_replacement()
            if runtime_error is not None:
                requested = local and (local.reset_requested or getattr(source, "reset_requested", False))
                if not requested:
                    app.update(); time.sleep(.01)
                    continue
                runtime_error = None
                hard_reset_requested = True
            try:
                if steps and tick >= steps:
                    break
                if source is not None and source.should_stop and not source.snapshot_requested:
                    break
                if local and (local.reset_requested or (source is not None and getattr(source, "reset_requested", False))):
                    reset_runtime(hard=hard_reset_requested or world.is_stopped())
                    hard_reset_requested = False
                if gui and source is not None:
                    physics_action = getattr(source, "physics_action_requested", None)
                    if physics_action is not None:
                        source.physics_action_requested = None
                        if physics_action in ("play", "toggle") and not world.is_playing():
                            world.play()
                        elif physics_action in ("pause", "toggle"):
                            world.pause()
                if (gui and source is not None and getattr(source, "pause_physics_requested", False)
                        and not source.snapshot_requested):
                    source.pause_physics_requested = False
                    world.pause()
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
                if local and timeline_stopped:
                    messages.append({"kind": "reset", "hard": True})
                while socket is not None and socket.poll(0):
                    messages.append(unpack(socket.recv()))
                    last_host_ns = time.monotonic_ns()
                for message in messages:
                    if message["kind"] == "stop":
                        return
                    if message["kind"] == "reset":
                        reset_runtime(hard=message.get("hard", False))
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
            except Exception as error:
                if not (gui and keep_gui_on_error and source is not None and local is not None):
                    raise
                import traceback

                from .recording import atomic_json
                runtime_error = failure = f"{type(error).__name__}: {error}"
                traceback.print_exc()
                atomic_json(source.output / "engine_error.json", {
                    "error": failure, "traceback": traceback.format_exc(), "retry_available": True})
                try:
                    world.pause()
                except Exception as pause_error:  # noqa: BLE001 -- Keep the original failure visible.
                    print(f"Could not pause the failed scene: {pause_error}", flush=True)
                source.finished = False
                source.finish("failed", failure)
                source.reset_requested = False
                source.physics_action_requested = None
                source.pause_physics_requested = False
                source.snapshot_requested = False
                if hasattr(source, "failure_snapshot_pending"):
                    source.failure_snapshot_pending = False
                print("Task paused after an engine error. Click Restart task to retry.", flush=True)
    except KeyboardInterrupt:
        print("Simulation stopped by operator; saving recording", flush=True)
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        print(f"Isaac engine error: {type(error).__name__}: {error}", flush=True)
        try:
            if socket is not None:
                socket.send(pack({"kind": "error", "error": str(error)}))
        except Exception as send_error:  # noqa: BLE001 -- Preserve the original engine error.
            print(f"Could not report engine error to host: {send_error}", flush=True)
        if gui and keep_gui_on_error and isinstance(error, Exception):
            # Opt-in for interactive task debugging. Keep Kit responsive after
            # Python failures; this does not recover a native process/GPU crash.
            import traceback

            from omni import ui
            traceback.print_exc()
            if world is not None:
                try:
                    world.pause()
                except Exception as pause_error:  # noqa: BLE001 -- Preserve the original engine failure.
                    print(f"Could not pause the failed scene: {pause_error}", flush=True)
            if source is not None:
                from .recording import atomic_json
                atomic_json(source.output / "engine_error.json", {"error": failure, "traceback": traceback.format_exc()})
                try:
                    source.finish("failed", failure)
                except Exception:  # noqa: BLE001 -- Error reporting must not hide the original exception.
                    traceback.print_exc()
            if local is not None and getattr(local, "window", None) is not None:
                local.window.frame.enabled = False
            error_window = ui.Window("Experiment error - scene retained", width=600, height=220)
            with error_window.frame, ui.VStack(spacing=8):
                ui.Label(failure, word_wrap=True)
                ui.Label("Scene construction failed. Edit the configuration and rebuild without closing Isaac.", word_wrap=True)
                if source is not None and hasattr(source, "request_rebuild"):
                    ui.Button("Apply config / Rebuild scene", clicked_fn=source.request_rebuild)
                rebuild_status = ui.Label("", word_wrap=True)
            while app.is_running():
                if source is not None and hasattr(source, "service_reload"):
                    source.service_reload()
                    if getattr(source, "scene_rebuild_requested", False):
                        error_window.destroy()
                        return source.take_scene_replacement()
                    if source.config_message:
                        rebuild_status.text = source.config_message
                app.update()
                time.sleep(.01)
            error_window.destroy()
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
            try:
                if gui:
                    from omni.kit.viewport.window import get_viewport_window_instances
                    # Existing viewport widgets outlive the USD stage. Stop
                    # queued image callbacks from dereferencing deleted cameras.
                    for window in get_viewport_window_instances():
                        window.viewport_api.updates_enabled = False
                if world is not None:
                    world.stop()
                resources.close()
            finally:
                if world is not None:
                    world.clear()
                    World.clear_instance()


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
