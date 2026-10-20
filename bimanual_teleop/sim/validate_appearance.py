"""Render the real-scene palette and inspect visual material bindings in Isaac."""

import argparse
import faulthandler
import json
from pathlib import Path

from . import ROOT
from ._dependencies import np, Rotation
from .config import load_config, sim_path


def run(output):
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "anti_aliasing": 1,
                         "disable_viewport_updates": True, "limit_cpu_threads": 8})
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    report = {"passed": False, "learning_data": False, "visual_materials": {}}
    faulthandler.dump_traceback_later(90, repeat=True)
    try:
        import carb
        import omni.replicator.core as rep
        from isaacsim.core.api import World
        from isaacsim.core.utils.types import ArticulationAction
        from PIL import Image
        from pxr import Gf, Usd, UsdGeom, UsdShade
        from .camera_calibration import wrist_camera_local_pose
        from .kinematics import RobotModel
        from .rendering import configure_camera_rendering
        from .scene import build_scene, look_at, rgb_frames

        carb.settings.get_settings().set_int("/persistent/physics/numThreads", 0)
        configure_camera_rendering(); config = load_config()
        model = RobotModel(sim_path(config["robot"]["urdf"]), config["robot"]["world_yaw_deg"])
        world = World(physics_dt=1/240, rendering_dt=1/30, stage_units_in_meters=1.)
        world.get_physics_context().enable_gpu_dynamics(False)
        world.get_physics_context().set_broadphase_type("MBP")
        rep.orchestrator.set_capture_on_play(False)
        robot, annotators, _, robot_path, cameras = build_scene(world, config, model)
        world.reset()
        indices = np.array([robot.get_dof_index(name) for name in model.names], dtype=np.int32)
        positions = np.r_[np.deg2rad(config["robot"]["ready_deg"]["left"]),
                          np.deg2rad(config["robot"]["ready_deg"]["right"]), np.zeros(40)]
        robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=positions, joint_indices=indices))
        robot.set_joint_positions(positions, joint_indices=indices)
        robot.set_joint_velocities(np.zeros(54), joint_indices=indices); world.step(render=False)

        for side in ("L", "R"):
            for number in range(1, 8):
                name = f"Link{number}_{side}"
                visuals = world.stage.GetPrimAtPath(robot_path+f"/{name}/visuals")
                meshes = [p for p in Usd.PrimRange(visuals, Usd.TraverseInstanceProxies()) if p.IsA(UsdGeom.Mesh)]
                assert meshes, name
                for mesh in meshes:
                    material, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
                    assert material.GetPrim().GetName() == "RobotShellWhite", str(mesh.GetPath())
                report["visual_materials"][name] = "RobotShellWhite"
        # Inspect actual finger visual bindings, including the URDF's mesh instances.
        for name in ("l_wrist", "r_wrist", "l_index_finger_proximal", "r_index_finger_proximal"):
            visuals = world.stage.GetPrimAtPath(robot_path+f"/{name}/visuals")
            for mesh in Usd.PrimRange(visuals, Usd.TraverseInstanceProxies()):
                if mesh.IsA(UsdGeom.Mesh):
                    material, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
                    assert material.GetPrim().GetName() == "RobotHandDark", str(mesh.GetPath())
                    report["visual_materials"][name] = "RobotHandDark"

        base_visuals = world.stage.GetPrimAtPath(robot_path+"/base_link/visuals")
        report["base_materials"] = []
        for mesh in Usd.PrimRange(base_visuals, Usd.TraverseInstanceProxies()):
            if not mesh.IsA(UsdGeom.Mesh):
                continue
            binding = UsdShade.MaterialBindingAPI(mesh)
            material, relationship = binding.ComputeBoundMaterial()
            item = {"path": str(mesh.GetPath()), "material": str(material.GetPath()),
                    "binding": str(relationship.GetPath()), "subsets": {}}
            for subset in binding.GetMaterialBindSubsets():
                submaterial, _ = UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()
                item["subsets"][subset.GetPrim().GetName()] = str(submaterial.GetPath())
            report["base_materials"].append(item)
            print(json.dumps(item), flush=True)
            assert material.GetPrim().GetName() == "RobotTrim", item
            assert item["subsets"].get("WhiteTorso", "").endswith("/RobotShellWhite"), item
        assert report["base_materials"], "No base visual meshes inspected"

        observers = []
        for name, eye, target in (("scene_overview", [2.2, -1.9, 1.85], [.40, 0., .72]),
                                  ("robot_closeup", [1.1, -1.3, 1.50], [.05, 0., .95])):
            camera = UsdGeom.Camera.Define(world.stage, "/World/Validation/"+name)
            camera.AddTranslateOp().Set(Gf.Vec3d(*eye))
            q = look_at(eye, target)
            camera.AddOrientOp().Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3])))
            camera.CreateFocalLengthAttr(24.)
            camera.CreateHorizontalApertureAttr(32.); camera.CreateVerticalApertureAttr(24.)
            camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 10.)); camera.CreateFStopAttr(0.)
            product = rep.create.render_product(str(camera.GetPath()), (1280, 960))
            annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cuda")
            annotator.attach(product); observers.append(annotator)
        for _ in range(8):
            rep.orchestrator.step(rt_subframes=4, delta_time=0., pause_timeline=False)
        for name, frame in zip(("head_camera", "left_wrist", "right_wrist", "scene_overview", "robot_closeup"), rgb_frames([*annotators, *observers])):
            assert frame.std() > 2, name
            Image.fromarray(frame).save(output/(name+".png"))
        # Appearance overrides must leave the measured camera transforms intact.
        manifest = json.loads((ROOT/"assets/generated/manifest.json").read_text())
        actual = np.asarray(robot.get_joint_positions(joint_indices=indices))
        for side, index in (("left", 1), ("right", 2)):
            camera = UsdGeom.Camera.Get(world.stage, cameras[index])
            position, quaternion = wrist_camera_local_pose(manifest["cad"]["tool_stack"], config["cameras"]["wrist"][side]["pose_in_bracket"])
            local = np.eye(4); local[:3, :3] = Rotation.from_quat(quaternion).as_matrix(); local[:3, 3] = position
            expected = model.link_pose(("l" if side == "left" else "r")+"_camera_bracket", actual) @ local
            native = np.array(camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
            error = float(np.abs(native-expected).max()); assert error < 1e-5, (side, error)
            report[side+"_camera_transform_error"] = error
        report["robot_appearance"] = config["robot"]["appearance"]; report["passed"] = True
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
        app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
