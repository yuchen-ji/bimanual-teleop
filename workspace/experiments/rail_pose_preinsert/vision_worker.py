"""Control-environment image adapter; GPU inference uses the existing external worker."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image


class PersistentVision:
    def __init__(self, output, live_config, *, timeout_s=180):
        self.output, self.config = Path(output), live_config
        self.timeout_s = timeout_s
        self.process = self.log = self.directory = None

    def wait_json(self, name, cancelled, request_id=None):
        deadline = time.monotonic() + self.timeout_s
        while True:
            if cancelled() or time.monotonic() > deadline:
                self.close()
                raise RuntimeError("FoundationPose cancelled or timed out; inspect foundationpose.log")
            path = self.directory / name
            if path.is_file():
                value = json.loads(path.read_text())
                if request_id is None or value.get("id") == request_id:
                    return value
            if self.process.poll() is not None:
                raise RuntimeError("FoundationPose worker exited; inspect foundationpose.log")
            time.sleep(.03)

    def start(self, cancelled):
        if self.process is not None and self.process.poll() is None:
            return
        self.close()
        self.directory = self.output / "vision_worker" / uuid.uuid4().hex
        self.directory.mkdir(parents=True)
        self.log = (self.output / "foundationpose.log").open("a")
        from rail_preinsert.perception import python_command
        command = python_command(self.config) + [str(Path(__file__).with_name("perception_service.py")),
                   "--foundationpose-root", self.config["foundationpose_root"], "--mesh", self.config["mesh"],
                   "--serve", str(self.directory), "--parent-pid", str(os.getpid())]
        self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT,
                                        env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"))
        ready = self.wait_json("ready.json", cancelled)
        if not ready.get("ready"):
            self.close()
            raise RuntimeError("FoundationPose initialization failed: " + ready.get("error", "unknown error"))

    def measure(self, scene, destination, samples, cancelled=lambda: False):
        from common import write_json
        self.start(cancelled)
        request_id = uuid.uuid4().hex
        write_json(self.directory / "request.json", {"id": request_id, "command": "measure",
                   "scene": str(scene), "output": str(destination), "samples": samples})
        response = self.wait_json("response.json", cancelled, request_id)
        if not response.get("ok"):
            raise RuntimeError("FoundationPose measurement failed: " + response.get("error", "unknown error"))
        return response["observation"]

    def close(self):
        from common import write_json
        if self.process is not None and self.process.poll() is None:
            write_json(self.directory / "request.json", {"id": uuid.uuid4().hex, "command": "shutdown"})
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None


def infer_rendered(request, output, live_config, cancelled=lambda: False, *, worker=None, simulation=None):
    import cv2
    import yaml
    root = Path(live_config["foundationpose_root"])
    sys.path[:0] = [str(root / "workspace/src")]
    from rail_pose.data import (  # Pure file types, no GPU or device imports.
        Frame,
        SceneWriter,
    )
    output = Path(output)
    simulation = simulation or yaml.safe_load((output / "effective.yaml").read_text())
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
    own_worker = worker is None
    worker = worker or PersistentVision(output, live_config)
    try:
        observation = worker.measure(scene, destination, len(request["frames"]), cancelled)
    finally:
        if own_worker:
            worker.close()
    observation.update(test_data=True, source="FoundationPose on rendered RGB-D; ground-truth visible mask only")
    for i, item in enumerate(request["frames"]):
        shutil.copyfile(destination / "measurement" / f"pose_{i:02d}.png", output / item["directory"] / "foundationpose.png")
    return observation
