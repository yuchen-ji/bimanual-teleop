"""Render the calibrated camera and verify its RTX pixel projection without recording data."""

from __future__ import annotations

import argparse
import faulthandler
import json
from pathlib import Path

from . import ROOT
from ._dependencies import np, Rotation
from .config import load_config, sim_path


def run(config, output):
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "anti_aliasing": 1,
                         "disable_viewport_updates": True, "limit_cpu_threads": 8})
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    report = {"passed": False, "learning_data": False}
    faulthandler.dump_traceback_later(90, repeat=True)
    try:
        import omni.replicator.core as rep
        import carb
        from isaacsim.core.api import World
        from pxr import Gf, Sdf, Usd, UsdGeom
        from PIL import Image
        from .kinematics import RobotModel
        from .scene import build_scene, rgb_frames, _bind, _material
        from .rendering import configure_camera_rendering

        carb.settings.get_settings().set_int("/persistent/physics/numThreads", 0)
        report["render_settings"] = configure_camera_rendering()
        model = RobotModel(sim_path(config["robot"]["urdf"]), config["robot"]["world_yaw_deg"])
        world = World(physics_dt=1/240, rendering_dt=1/30, stage_units_in_meters=1.)
        world.get_physics_context().enable_gpu_dynamics(False)
        world.get_physics_context().set_broadphase_type("MBP")
        rep.orchestrator.set_capture_on_play(False)
        print("Building head-camera validation scene", flush=True)
        robot, annotators, geometry, robot_path, cameras = build_scene(world, config, model)
        print("Initializing scene physics", flush=True)
        world.reset()
        print("Rendering head-camera validation images", flush=True)
        indices = np.array([robot.get_dof_index(name) for name in model.names], dtype=np.int32)
        ready = np.r_[np.deg2rad(config["robot"]["ready_deg"]["left"]),
                      np.deg2rad(config["robot"]["ready_deg"]["right"]), np.zeros(40)]
        robot.set_joint_positions(np.clip(ready, model.lower, model.upper), joint_indices=indices)

        def capture():
            rep.orchestrator.step(rt_subframes=4, delta_time=0., pause_timeline=False)
            return rgb_frames(annotators)[0]

        for index in range(3):
            capture()
            print(f"Camera warmup {index+1}/3 complete", flush=True)
        image = capture()
        Image.fromarray(image).save(output / "head_camera.png")
        assert image.std() > 2
        for index, frame in enumerate(rgb_frames(annotators)[1:], 1):
            Image.fromarray(frame).save(output / f"wrist_camera_{index}_ready.png")
        from .camera_calibration import wrist_camera_local_pose
        import trimesh
        manifest = json.loads((ROOT / "assets/generated/manifest.json").read_text())
        report["wrist_optics"] = {}
        for side in ("left", "right"):
            measured = config["cameras"]["wrist"][side]["pose_in_bracket"]
            optical_position, optical_quaternion = wrist_camera_local_pose(manifest["cad"]["tool_stack"], measured)
            optical_forward = Rotation.from_quat(optical_quaternion).apply([0, 0, -1])
            margins = []
            for part in (1, 2):
                mesh = trimesh.load(ROOT / f"assets/generated/task/wrist_part_{part}.obj", force="mesh", process=False)
                margins.append(float(((mesh.vertices-optical_position) @ optical_forward).max()))
            if measured is None:
                assert max(margins) < -.001, margins
            report["wrist_optics"][side] = {"source": "measured" if measured is not None else "provisional_cad",
                "position_in_bracket_m": optical_position.tolist(), "forward_in_bracket": optical_forward.tolist(),
                "max_housing_forward_m": margins}
        robot.set_joint_positions(np.clip(np.zeros(54), model.lower, model.upper), joint_indices=indices)
        for _ in range(3):
            capture()
        for index, frame in enumerate(rgb_frames(annotators)[1:], 1):
            Image.fromarray(frame).save(output / f"wrist_camera_{index}_zero.png")
        robot.set_joint_positions(np.clip(ready, model.lower, model.upper), joint_indices=indices)
        for _ in range(3):
            capture()
        actual_settings = carb.settings.get_settings()
        report["render_settings"] = {path: actual_settings.get(path) for path in report["render_settings"]}
        report["depth_of_field"] = actual_settings.get("/rtx/post/dof/enabled")
        report["motion_blur"] = actual_settings.get("/rtx/post/motionblur/enabled")
        # Compare smooth floor noise at the same pixels as the earlier quality check.
        gray = image.astype(float).mean(axis=2)
        residual = gray[21:79, 101:539] - .25 * (gray[20:78, 101:539] + gray[22:80, 101:539]
                                               + gray[21:79, 100:538] + gray[21:79, 102:540])
        report["floor_highpass_std"] = float(residual.std())
        scale = config["cameras"]["preview_scale"]
        preview_size = (config["cameras"]["width"]*scale, config["cameras"]["height"]*scale)
        preview_product = rep.create.render_product(cameras[0], preview_size)
        preview_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cuda")
        preview_annotator.attach(preview_product)
        for _ in range(3):
            capture()
        Image.fromarray(rgb_frames([preview_annotator])[0]).save(output / "head_camera_preview.png")
        report["preview_resolution"] = list(preview_size)
        head = config["cameras"]["head"]
        camera = UsdGeom.Camera.Get(world.stage, cameras[0])
        world_usd = np.array(camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        world_cv = world_usd.copy(); world_cv[:3, :3] = world_usd[:3, :3] @ np.diag([1., -1., -1.])
        report["T_world_camera_opencv"] = world_cv.tolist()
        assert np.linalg.norm(world_cv[:3, 3] - head["position_world_m"]) < 1e-6
        assert np.allclose(world_cv[:3, :3], Rotation.from_quat(head["quaternion_xyzw"]).as_matrix(), atol=1e-6)

        # Verify the actual RTX result at three positions, including both image corners.
        # These small emissive spheres are visual probes only, never physics objects.
        material = _material(world.stage, "/World/Materials/ProjectionProbe", (1, 0, 1))
        shader = world.stage.GetPrimAtPath("/World/Materials/ProjectionProbe/Shader")
        shader.CreateAttribute("inputs:emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(5, 0, 5))
        shader.GetAttribute("inputs:diffuseColor").Set(Gf.Vec3f(0, 0, 0))
        shader.CreateAttribute("inputs:ior", Sdf.ValueTypeNames.Float).Set(1.)
        intr = head["intrinsics"]; pixels = []; preview_pixels = []
        report["projection_probes"] = pixels
        coeff = [*head["distortion_coefficients"], *([0.] * (12 - len(head["distortion_coefficients"])))]
        k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4 = coeff
        for u, v in ((80, 60), (320, 240), (560, 420)):
            x, y = (u-intr["cx"])/intr["fx"], (v-intr["cy"])/intr["fy"]
            sphere = UsdGeom.Sphere.Define(world.stage, cameras[0]+"/ProjectionProbe")
            sphere.CreateRadiusAttr(.0016)
            sphere.AddTranslateOp().Set(Gf.Vec3d(x*.15, -y*.15, -.15)); _bind(sphere, material)
            # TAA applies subpixel jitter. Average eight static observations so a
            # different warmup history does not masquerade as a calibration change.
            observations, preview_observations = [], []
            for _ in range(8):
                measured = capture()
                preview = rgb_frames([preview_annotator])[0]
                for values, observations_at_size in ((measured, observations), (preview, preview_observations)):
                    mask = (values[:, :, 0] > 180) & (values[:, :, 2] > 180) & (values[:, :, 1] < 100)
                    yy, xx = np.nonzero(mask)
                    assert len(xx) >= 3, f"Projection probe missing at {(u, v)}"
                    # Raster indices address cells; compare their centers to continuous coordinates.
                    observations_at_size.append([xx.mean()+.5, yy.mean()+.5])
            Image.fromarray(measured).save(output / f"projection_probe_{len(pixels)}.png")
            observed = np.mean(observations, axis=0)
            r2 = x*x+y*y
            radial = (1+k1*r2+k2*r2**2+k3*r2**3)/(1+k4*r2+k5*r2**2+k6*r2**3)
            xd = x*radial+2*p1*x*y+p2*(r2+2*x*x)+s1*r2+s2*r2**2
            yd = y*radial+p1*(r2+2*y*y)+2*p2*x*y+s3*r2+s4*r2**2
            expected = np.array([intr["fx"]*xd+intr["cx"], intr["fy"]*yd+intr["cy"]])
            error = float(np.linalg.norm(observed-expected))
            pixels.append({"expected": expected.tolist(), "rendered": observed.tolist(), "error_px": error,
                           "sample_std_px": np.std(observations, axis=0).tolist()})
            observed_preview = np.mean(preview_observations, axis=0)
            preview_pixels.append({"expected": (expected*scale).tolist(), "rendered": observed_preview.tolist(),
                                   "error_base_px": float(np.linalg.norm(observed_preview/scale - expected)),
                                   "scale_consistency_error_px": float(np.linalg.norm(observed_preview/scale - observed))})
            world.stage.RemovePrim(str(sphere.GetPath()))
        report["projection_probes"] = pixels
        report["projection_samples_per_probe"] = 8
        report["max_projection_error_px"] = max(p["error_px"] for p in pixels)
        assert report["max_projection_error_px"] < 1.5, pixels
        report["preview_projection_probes"] = preview_pixels
        report["max_preview_projection_error_base_px"] = max(p["error_base_px"] for p in preview_pixels)
        # Verify preview scaling against the actual base-resolution projection.
        # This isolates resolution scaling from the RTX lens model's absolute approximation error.
        report["max_preview_scale_consistency_error_px"] = max(p["scale_consistency_error_px"] for p in preview_pixels)
        assert report["max_preview_scale_consistency_error_px"] < .5, preview_pixels
        report["camera_path"] = cameras[0]
        report["resolution"] = [config["cameras"]["width"], config["cameras"]["height"]]
        report["intrinsics"] = intr
        report["distortion_coefficients"] = head["distortion_coefficients"]
        report["passed"] = True
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        (output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
        app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--output", default=str(ROOT / "data/head-camera-alignment"))
    args = parser.parse_args()
    run(load_config(args.config), args.output)


if __name__ == "__main__":
    main()
