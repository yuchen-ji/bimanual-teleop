"""Physical scene checks in the official runtime; never creates learning data."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import ROOT
from ._dependencies import np, Rotation
from .config import load_config, sim_path
from .kinematics import RobotModel, pose_matrix


def run(config, output):
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "anti_aliasing": 2,
                         "disable_viewport_updates": True, "limit_cpu_threads": 8})
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleRigidPrim, RigidPrim
    from isaacsim.core.utils.types import ArticulationAction
    import omni.replicator.core as rep
    import carb
    carb.settings.get_settings().set_int("/persistent/physics/numThreads", 0)
    rep.orchestrator.set_capture_on_play(False)
    from PIL import Image
    from .scene import build_scene, rgb_frames

    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    report = {"runtime": "official Isaac Sim 5.1.0", "passed": False, "checks": {}, "learning_data": False}
    try:
        model = RobotModel(sim_path(config["robot"]["urdf"]), config["robot"]["world_yaw_deg"])
        world = World(physics_dt=1/config["runtime"]["physics_hz"], rendering_dt=1/30, stage_units_in_meters=1.)
        world.get_physics_context().enable_gpu_dynamics(False)
        world.get_physics_context().set_broadphase_type("MBP")
        world.get_physics_context().enable_ccd(True)
        robot, annotators, geometry, robot_path, cameras = build_scene(world, config, model)
        rail = world.scene.add(SingleRigidPrim("/World/Rail", name="rail"))
        cad_scene = config["scene"]["asset_profile"] == "rail_task_cad_v1"
        rail_contacts = world.scene.add(RigidPrim("/World/Rail", name="rail_contacts", reset_xform_properties=False,
                                                track_contact_forces=True, max_contact_count=4096,
                                                contact_filter_prim_paths_expr=(["/World/Fixture/Collision", "/World/Table/Mesh"] if cad_scene
                                                                               else ["/World/Fixture/Mesh", "/World/Table"])))
        flanges = [world.scene.add(SingleRigidPrim(robot_path+f"/TCP_Link_{suffix}", name=f"flange_{suffix}",
                                                 reset_xform_properties=False)) for suffix in ("L", "R")]
        world.reset()
        indices = np.array([robot.get_dof_index(name) for name in model.names], dtype=np.int32)
        ready = np.clip(np.r_[np.deg2rad(config["robot"]["ready_deg"]["left"]),
                              np.deg2rad(config["robot"]["ready_deg"]["right"]), np.zeros(40)], model.lower, model.upper)
        robot.set_joint_positions(ready, joint_indices=indices)
        controller = robot.get_articulation_controller()
        kp, kd = controller.get_gains()
        kp[indices] = np.r_[np.full(14, config["robot"]["arm_stiffness"]), np.tile([2, 2, 1, .8], 10)]
        kd[indices] = np.r_[np.full(14, config["robot"]["arm_damping"]), np.tile([.05, .05, .03, .03], 10)]
        controller.set_gains(kps=kp, kds=kd)
        controller.set_max_efforts(model.effort, joint_indices=indices)
        robot.set_solver_position_iteration_count(16); robot.set_solver_velocity_iteration_count(4)

        def step(count):
            for index in range(count):
                if index % 2 == 0:
                    controller.apply_action(ArticulationAction(joint_positions=ready, joint_indices=indices))
                world.step(render=False)

        step(1)
        for _ in range(4):
            rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)
        for index, image in enumerate(rgb_frames(annotators)):
            Image.fromarray(image).save(output / f"ready_camera_{index}.png")
        step(600)
        q = robot.get_joint_positions(joint_indices=indices)
        report["checks"]["joint_count"] = len(q)
        report["checks"]["joint_positions"] = q.tolist()
        assert len(q) == 54 and np.isfinite(q).all(), q
        bad = np.flatnonzero((q < model.lower-.02) | (q > model.upper+.02))
        assert not len(bad), [(model.names[i], float(q[i]), float(model.lower[i]), float(model.upper[i])) for i in bad]
        report["checks"]["ready_arm_error_rad"] = float(np.max(np.abs(q[:14]-ready[:14])))
        errors = []
        for side, flange in zip(("left", "right"), flanges):
            position, wxyz = flange.get_world_pose()
            actual = pose_matrix(np.r_[position, wxyz[[1, 2, 3, 0]]])
            expected = model.flange(side, q, world=True)
            translation = float(np.linalg.norm(actual[:3, 3]-expected[:3, 3]))
            angular = float((Rotation.from_matrix(actual[:3, :3])*Rotation.from_matrix(expected[:3, :3]).inv()).magnitude())
            errors.append({"side": side, "translation_m": translation, "rotation_rad": angular,
                           "actual_position": position.tolist(), "expected_position": expected[:3, 3].tolist()})
            report["checks"]["physical_fk"] = errors
            assert translation < .003 and angular < .01, errors
        report["checks"]["physical_fk"] = errors
        position, orientation = rail.get_world_pose()
        report["checks"]["default_rail_quaternion_wxyz"] = orientation.tolist()
        report["checks"]["default_rail_velocity"] = rail.get_linear_velocity().tolist()
        report["checks"]["default_rail_contact_forces"] = rail_contacts.get_contact_force_matrix(dt=1/240).tolist()
        rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)
        for index, image in enumerate(rgb_frames(annotators)):
            Image.fromarray(image).save(output / f"settled_camera_{index}.png")
        if cad_scene:
            expected_z = config["rail"]["initial_center"][2]-.0004
            report["checks"]["default_rail_position_m"] = position.tolist()
            report["checks"]["default_fixture_height_error_m"] = float(abs(position[2]-expected_z))
            from pxr import UsdGeom, Usd
            box_paths = [str(prim.GetPath()) for prim in Usd.PrimRange(world.stage.GetPrimAtPath("/World"))
                         if prim.IsA(UsdGeom.Mesh) and "/Layer_" in str(prim.GetPath())]
            report["checks"]["transfer_box_meshes"] = box_paths
            assert len(box_paths) == 10, box_paths
            names = list(robot._articulation_view._physics_view.shared_metatype.link_names)
            if names and isinstance(names[0], (list, tuple)):
                names = names[0]
            sensors = np.array([names.index(f"{p}_force_sensor") for p in ("l", "r")])
            forces = robot.get_measured_joint_forces(joint_indices=sensors)
            report["checks"]["sensor_input_wrenches"] = forces.tolist()
            assert forces.size == 12 and np.isfinite(forces).all() and np.linalg.norm(forces) > .1
        else:
            floor = config["scene"]["bin_centers"][0][2]+config["scene"]["bin_wall"]
            expected_z = floor+config["rail"]["size_m"][2]/2
            report["checks"]["input_bin_height_error_m"] = float(abs(position[2]-expected_z))
        assert abs(position[2]-expected_z) < .002, position
        print("Physics validation: default fixture placement and tool wrenches passed", flush=True)

        # Render frozen states after warmup, then verify consecutive physics and
        # camera samples without recording transport-test demonstrations.
        for _ in range(4):
            rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)
        started = time.monotonic(); physics_start = world.current_time
        renders = 0
        stepping_seconds = 0.; rendering_seconds = 0.
        for _ in range(15):
            phase = time.monotonic()
            step(8)
            stepping_seconds += time.monotonic()-phase
            before = world.current_time
            phase = time.monotonic()
            rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)
            rendering_seconds += time.monotonic()-phase
            assert abs(world.current_time-before) < 1e-8
            for index, image in enumerate(rgb_frames(annotators)):
                assert image.shape == (config["cameras"]["height"], config["cameras"]["width"], 3)
                assert image.dtype == np.uint8 and image.std() > 2
                if renders == 0:
                    Image.fromarray(image).save(output / f"camera_{index}.png")
            renders += 1
        duration = time.monotonic()-started
        report["checks"]["three_rgb_cameras"] = cameras
        report["checks"]["frozen_camera_physics"] = True
        report["checks"]["wall_camera_hz"] = renders/duration
        report["checks"]["real_time_factor"] = (world.current_time-physics_start)/duration
        report["checks"]["physics_seconds_per_camera_frame"] = stepping_seconds/renders
        report["checks"]["render_seconds_per_camera_frame"] = rendering_seconds/renders
        print("Physics validation: three synchronous RGB streams passed", flush=True)

        expected = (np.array(config["rail"]["initial_center"])-[0, 0, .0004] if cad_scene else
                    np.array(config["scene"]["fixture_origin"])+[0, 0, geometry["support_height_m"]+config["rail"]["size_m"][2]/2])
        yaw = np.deg2rad(config["rail"]["initial_rpy_deg"][2])

        def release(offset, yaw_error=0):
            angle = yaw+yaw_error
            rotation = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
            # The supplied pin pair has only 0.66 mm minimum radial clearance.
            # Validate release after guided insertion; a 20 mm free drop is not
            # a substitute for the arms' alignment and controlled lowering.
            clearance = .001 if cad_scene and not yaw_error else .02
            rail.set_world_pose(expected+offset+[0, 0, clearance], rotation)
            rail.set_linear_velocity(np.zeros(3)); rail.set_angular_velocity(np.zeros(3))
            report["checks"]["release_initial"] = {"pose": [v.tolist() for v in rail.get_world_pose()], "playing": world.is_playing()}
            step(480)
            report["checks"]["release_final"] = {"pose": [v.tolist() for v in rail.get_world_pose()],
                                                "velocity": rail.get_linear_velocity().tolist(), "playing": world.is_playing(),
                                                "contacts": rail_contacts.get_contact_force_matrix(dt=1/240).tolist(),
                                                "total_force": rail_contacts.get_net_contact_forces(dt=1/240).tolist()}
            rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)
            name = "fixture_probe_misaligned.png" if yaw_error else "fixture_probe_aligned.png"
            Image.fromarray(rgb_frames(annotators)[0]).save(output / name)
            return rail.get_world_pose()[0]

        aligned = release(np.zeros(3))
        report["checks"]["aligned_fixture_error_m"] = float(abs(aligned[2]-expected[2]))
        assert abs(aligned[2]-expected[2]) < .002, aligned
        report["checks"]["guided_release_clearance_m"] = .001 if cad_scene else .02
        if not cad_scene:
            # Legacy rectangular rail allows a free-drop blocking check. The
            # shaped CAD rail requires controlled insertion, checked above.
            blocked = release(np.zeros(3), np.deg2rad(8))
            report["checks"]["misaligned_pin_height_m"] = float(blocked[2]-expected[2])
            assert blocked[2]-expected[2] > .004, blocked
        print("Physics validation: guided release onto fixture passed", flush=True)

        # A static image with the correct shape does not prove fresh captures.
        # Move visible camera-local probes across all three views, without
        # stepping physics, and require their measured pixels to move too.
        from pxr import Gf, Sdf, Usd, UsdGeom
        from .scene import _material, _bind
        stage = world.stage
        material = _material(stage, "/World/Materials/CameraProbe", (1, 0, 1))
        shader = stage.GetPrimAtPath("/World/Materials/CameraProbe/Shader")
        shader.CreateAttribute("inputs:emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(5, 0, 5))
        probes = []
        for path in cameras:
            cube = UsdGeom.Cube.Define(stage, path+"/ValidationProbe")
            cube.CreateSizeAttr(.008)
            probes.append(cube.AddTranslateOp())
            _bind(cube, material)
        pixels = []
        for x in (-.025, .025):
            for operation in probes:
                operation.Set(Gf.Vec3d(x, 0, -.07))
            before = world.current_time
            rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)
            assert abs(world.current_time-before) < 1e-8
            locations = []
            for image in rgb_frames(annotators):
                mask = (image[:, :, 0] > 150) & (image[:, :, 2] > 100) & (image[:, :, 1] < 100)
                y, u = np.nonzero(mask)
                assert len(u) > 25, "Fresh camera probe not visible"
                locations.append(float(np.median(u)))
            pixels.append(locations)
        report["checks"]["fresh_camera_probe_pixels"] = pixels
        assert all(right-left > 200 for left, right in zip(*pixels)), pixels
        for path in cameras:
            stage.RemovePrim(path+"/ValidationProbe")

        # Compare the renderer's rigid camera transforms to independently
        # evaluated URDF kinematics, before and after physical wrist motion.
        camera_motion = []
        previous_camera_poses = None
        parent_links = (("Head depth_Link", "l_camera_bracket", "r_camera_bracket") if cad_scene
                        else ("Head depth_Link", "l_mount", "r_mount"))
        for moved in (False, True):
            if moved:
                ready[6] += .2; ready[13] -= .2
                step(240)
                rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)
            q = robot.get_joint_positions(joint_indices=indices)
            rendered_poses = []
            for path, parent in zip(cameras, parent_links):
                camera = UsdGeom.Xformable(stage.GetPrimAtPath(path))
                rendered = np.array(camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
                expected_camera = model.link_pose(parent, q) @ np.array(camera.GetLocalTransformation()).T
                distance = float(np.linalg.norm(rendered[:3, 3]-expected_camera[:3, 3]))
                angle = float((Rotation.from_matrix(rendered[:3, :3])*Rotation.from_matrix(expected_camera[:3, :3]).inv()).magnitude())
                camera_motion.append({"camera": path, "moved": moved, "translation_error_m": distance,
                                      "rotation_error_rad": angle})
                assert distance < .003 and angle < .01, camera_motion[-1]
                rendered_poses.append(rendered)
            if previous_camera_poses:
                for before, after in zip(previous_camera_poses[1:], rendered_poses[1:]):
                    assert (Rotation.from_matrix(before[:3, :3])*Rotation.from_matrix(after[:3, :3]).inv()).magnitude() > .05
            previous_camera_poses = rendered_poses
        report["checks"]["rigid_camera_motion"] = camera_motion
        report["passed"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        print(report["error"], flush=True)
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        app.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Validate actual Isaac physics and three cameras")
    parser.add_argument("--config")
    parser.add_argument("--output", default=str(ROOT / "data/isaac-validation"))
    args = parser.parse_args()
    run(load_config(args.config), args.output)


if __name__ == "__main__":
    main()
