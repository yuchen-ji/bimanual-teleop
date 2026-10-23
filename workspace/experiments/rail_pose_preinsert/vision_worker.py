"""Control-environment image adapter; GPU inference uses the existing external worker."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def infer_rendered(request, output, live_config, cancelled=lambda: False):
    import cv2
    import yaml
    root = Path(live_config["foundationpose_root"])
    sys.path[:0] = [str(root / "workspace/src")]
    from rail_pose.data import (  # Pure file types, no GPU or device imports.
        Frame,
        SceneWriter,
    )
    output = Path(output)
    simulation = yaml.safe_load((output / "effective.yaml").read_text())
    lens = simulation["cameras"]["head"]
    intr = lens["intrinsics"]
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.]])
    size = (simulation["cameras"]["width"], simulation["cameras"]["height"])
    maps = cv2.initUndistortRectifyMap(K, np.array(lens["distortion_coefficients"]), None, K, size, cv2.CV_32FC1)
    name = request["stage"] + "_" + request["id"]
    scene = output / "vision_scenes" / name
    scene.parent.mkdir(parents=True, exist_ok=True)
    writer = SceneWriter(scene, {"serial": "simulation-head", "K": K.tolist()})
    for i, item in enumerate(request["frames"]):
        directory = output / item["directory"]
        rgb = cv2.remap(np.asarray(Image.open(directory / "camera_0.png")), *maps, cv2.INTER_LINEAR)
        depth = cv2.remap(np.load(directory / "depth_m.npy"), *maps, cv2.INTER_NEAREST)
        mask = cv2.remap(np.asarray(Image.open(directory / "mask.png")), *maps, cv2.INTER_NEAREST) > 0
        depth[~np.isfinite(depth) | (depth <= 0) | (depth > 3)] = 0
        writer.write(Frame(str(i), rgb, depth, K, item["state"]["time_ns"], mask=mask,
                           metadata={"timestamp_source": "simulation_clock", "test_data": True}))
    destination = output / "inference" / name
    worker = Path(live_config["config_path"]).parent / "perception_worker.py"
    command = [live_config["foundationpose_python"], str(worker), "--foundationpose-root", str(root),
               "--mesh", live_config["mesh"], "--scene", str(scene), "--samples", str(len(request["frames"])),
               "--output", str(destination)]
    # Only the external project's established GPU entry uses its own env.
    # Python belonging to bimanual-teleop stays in bimanual-teleop.
    with (output / "foundationpose.log").open("a") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"))
        try:
            deadline = time.monotonic() + 180
            while process.poll() is None:
                if cancelled() or time.monotonic() > deadline:
                    raise RuntimeError("FoundationPose cancelled or timed out; inspect foundationpose.log")
                time.sleep(.05)
            if process.returncode:
                raise RuntimeError("FoundationPose failed; inspect foundationpose.log")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=5)
    observation = json.loads((destination / "observation.json").read_text())
    observation.update(test_data=True, source="FoundationPose on rendered RGB-D; ground-truth visible mask only")
    for i, item in enumerate(request["frames"]):
        shutil.copyfile(destination / "measurement" / f"pose_{i:02d}.png", output / item["directory"] / "foundationpose.png")
    return observation
