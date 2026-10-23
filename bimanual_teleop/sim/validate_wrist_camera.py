"""Render measured wrist optics in Isaac, including an independent calibration-board pose."""

import argparse
import faulthandler
import json
from pathlib import Path

from . import ROOT
from ._dependencies import np, Rotation
from .config import load_config, sim_path


def run(result_file, output):
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "anti_aliasing": 1,
                         "disable_viewport_updates": True, "limit_cpu_threads": 8})
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    result_file = Path(result_file)
    calibration = json.loads(result_file.read_text())
    report = {"passed": False, "learning_data": False, "wrists": {}}
    faulthandler.dump_traceback_later(90, repeat=True)
    try:
        import carb
        import omni.replicator.core as rep
        from isaacsim.core.api import World
        from isaacsim.core.utils.types import ArticulationAction
        from pxr import Gf, Sdf, Usd, UsdGeom
        from PIL import Image
        from .camera_geometry import wrist_camera_local_pose
        from .kinematics import RobotModel
        from .rendering import configure_camera_rendering
        from .scene import build_scene, rgb_frames, _box, _material, _bind

        carb.settings.get_settings().set_int("/persistent/physics/numThreads", 0)
        configure_camera_rendering()
        config = load_config()
        model = RobotModel(sim_path(config["robot"]["urdf"]), config["robot"]["world_yaw_deg"])
        world = World(physics_dt=1/240, rendering_dt=1/30, stage_units_in_meters=1.)
        world.get_physics_context().enable_gpu_dynamics(False)
        world.get_physics_context().set_broadphase_type("MBP")
        rep.orchestrator.set_capture_on_play(False)
        print("Building measured wrist-camera validation scene", flush=True)
        robot, annotators, _, _, cameras = build_scene(world, config, model)
        world.reset()
        indices = np.array([robot.get_dof_index(name) for name in model.names], dtype=np.int32)
        ready = np.r_[np.deg2rad(config["robot"]["ready_deg"]["left"]),
                      np.deg2rad(config["robot"]["ready_deg"]["right"]), np.zeros(40)]
        manifest = json.loads((ROOT/"assets/generated/manifest.json").read_text())
        controller = robot.get_articulation_controller()

        def capture(positions):
            controller.apply_action(ArticulationAction(joint_positions=positions, joint_indices=indices))
            robot.set_joint_positions(positions, joint_indices=indices)
            robot.set_joint_velocities(np.zeros(54), joint_indices=indices)
            # PhysX must publish the link transforms before RTX renders attached cameras.
            # A zero-time replicator step alone leaves them at the previous articulation pose.
            world.step(render=False)
            for _ in range(8):
                rep.orchestrator.step(rt_subframes=4, delta_time=0., pause_timeline=False)
            return rgb_frames(annotators)

        def set_optics(side, measured):
            position, quaternion = wrist_camera_local_pose(manifest["cad"]["tool_stack"], measured)
            index = 1 if side == "left" else 2
            prim = world.stage.GetPrimAtPath(cameras[index])
            prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*position))
            prim.GetAttribute("xformOp:orient").Set(Gf.Quatf(float(quaternion[3]), Gf.Vec3f(*quaternion[:3])))

        # Same scene and ready joints for the before/after comparison.
        for side in ("left", "right"):
            set_optics(side, None)
        frames = capture(ready)
        for side, index in (("left", 1), ("right", 2)):
            Image.fromarray(frames[index]).save(output/f"{side}_before.png")
            set_optics(side, config["cameras"]["wrist"][side]["pose_in_bracket"])
        frames = capture(ready)
        Image.fromarray(frames[0]).save(output/"head_ready.png")
        for side, index in (("left", 1), ("right", 2)):
            Image.fromarray(frames[index]).save(output/f"{side}_ready.png")

        black = _material(world.stage, "/World/Materials/CalibrationBlack", (.008, .008, .008), roughness=1.)
        white = _material(world.stage, "/World/Materials/CalibrationWhite", (.8, .8, .8), roughness=1.)
        shader = world.stage.GetPrimAtPath("/World/Materials/CalibrationWhite/Shader")
        shader.CreateAttribute("inputs:emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.8, .8, .8))
        # Board dimensions are visual only; all calibration uses the confirmed center spacing.
        points = np.array([[(2*j+i % 2)*.05, i*.05, 0.] for i in range(5) for j in range(4)])
        for side, item in calibration["calibrations"].items():
            assert item["passed"] and item["reference"]["index"] in item["quality"]["held_out_indices"]
            path = "/World/WristValidationBoard"
            board = UsdGeom.Xform.Define(world.stage, path)
            pose = np.asarray(item["T_world_board"])
            board.AddTranslateOp().Set(Gf.Vec3d(*pose[:3, 3]))
            q = Rotation.from_matrix(pose[:3, :3]).as_quat()
            board.AddOrientOp().Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3])))
            _box(world.stage, path+"/Backing", (.175, .1, .002), (.43, .28, .002), black, collision=False)
            for i, point in enumerate(points):
                mesh = UsdGeom.Mesh.Define(world.stage, path+f"/Circle_{i}")
                theta = np.arange(80)*2*np.pi/80
                vertices = [point.tolist(), *(point+np.c_[.025*np.cos(theta), .025*np.sin(theta), np.zeros(80)]).tolist()]
                mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in vertices])
                mesh.CreateFaceVertexCountsAttr([3]*80)
                mesh.CreateFaceVertexIndicesAttr([v for k in range(80) for v in (0, k+1, (k+1) % 80+1)])
                mesh.CreateDoubleSidedAttr(True); _bind(mesh, white)
            positions = ready.copy(); offset = 0 if side == "left" else 7
            positions[offset:offset+7] = item["reference"]["joints_rad"]
            frames = capture(positions); index = 1 if side == "left" else 2
            Image.fromarray(frames[index]).save(output/f"{side}_reference_sim.png")
            real = np.asarray(Image.open(result_file.parent/item["reference"]["image"]).convert("RGB"))
            if item["camera"]["output_rotation_deg"] == 180:
                real = real[::-1, ::-1]
            Image.fromarray(real).save(output/f"{side}_reference_real.png")
            camera = UsdGeom.Camera.Get(world.stage, cameras[index])
            world_usd = np.array(camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
            world_cv = world_usd.copy(); world_cv[:3, :3] = world_usd[:3, :3] @ np.diag([1., -1., -1.])
            actual_positions = np.asarray(robot.get_joint_positions(joint_indices=indices))
            expected = model.link_pose(("l" if side == "left" else "r")+"_camera_bracket", actual_positions) @ np.asarray(item["T_bracket_output_opencv"])
            report["wrists"][side] = {"camera_path": cameras[index], "reference": item["reference"],
                "T_world_camera_opencv": world_cv.tolist(), "T_world_camera_expected": expected.tolist(),
                "max_transform_error": float(np.abs(world_cv-expected).max()),
                "max_arm_joint_pose_error_rad": float(np.abs(actual_positions[:14]-positions[:14]).max()),
                "T_world_board": pose.tolist(), "intrinsics": item["settings"]["intrinsics"],
                "distortion_coefficients": item["settings"]["distortion_coefficients"]}
            world.stage.RemovePrim(path)
            print(f"Rendered {side} wrist: ready and held-out reference", flush=True)
        report["passed"] = all(item["max_transform_error"] < 1e-5 for item in report["wrists"].values())
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        (output/"render_report.json").write_text(json.dumps(report, indent=2)+"\n")
        app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.result, args.output)


if __name__ == "__main__":
    main()
