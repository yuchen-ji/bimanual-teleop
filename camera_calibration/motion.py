"""Keyboard position control independent of camera detection and file writes."""

import queue
import threading
import time

import numpy as np

from bimanual_teleop.control.arm.jog import ROTATION_KEYS, TRANSLATION_KEYS, run_jog, select_side_profile
from bimanual_teleop.devices.tianji.config import load_config
from bimanual_teleop.devices.tianji.driver import TianjiDriver
from bimanual_teleop.types import ControlProfile


class KeyboardInput:
    """Dispatch jog and capture keys independently of detection and file writes."""

    def __init__(self, terminal, robot, on_capture):
        self.terminal, self.robot, self.on_capture = terminal, robot, on_capture
        self.stopped = threading.Event()
        self.error = None

    def stop(self):
        self.robot.send_key(" ")
        self.stopped.set()

    def key(self, key):
        key = (key or "").lower()
        if not key or self.stopped.is_set():
            return
        if key in ("q", "\x03", "\x04", "ctrl+c", "ctrl+d"):
            self.stop()
        elif key == "c":
            self.on_capture(time.monotonic_ns())
        elif key in ("enter", "return"):
            self.robot.send_key("\r")
        elif key == "space":
            self.robot.send_key(" ")
        elif key in TRANSLATION_KEYS or key in ROTATION_KEYS or key in (" ", "\r", "\n"):
            self.robot.send_key(key)

    def _run(self):
        try:
            while not self.stopped.is_set():
                keys = self.terminal.read(.01)
                if keys == "" or any(k.lower() in ("q", "\x03", "\x04") for k in keys or ""):
                    self.stop()
                    return
                for key in keys or "":
                    self.key(key)
                    if self.stopped.is_set():
                        return
        except Exception as error:
            self.error = str(error)
            self.stop()

    def check(self):
        if self.error:
            raise RuntimeError(f"键盘读取失败：{self.error}")

    def __enter__(self):
        self.thread = threading.Thread(target=self._run, name="calibration-input", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        self.thread.join(1)
        if self.thread.is_alive():
            raise RuntimeError("标定键盘读取线程未能停止")


class JogKeys:
    def __init__(self):
        self.queue = queue.Queue(maxsize=16)
        self.ready = threading.Event()
        self.stopped = threading.Event()

    def put(self, key):
        if not self.ready.is_set() or self.stopped.is_set():
            return
        if key == " ":
            while True:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
        try:
            self.queue.put_nowait((time.monotonic_ns(), key))
        except queue.Full:
            pass

    def __enter__(self):
        self.ready.set()
        return self

    def read(self, timeout):
        if self.stopped.is_set():
            return ""
        while True:
            try:
                timestamp, key = self.queue.get_nowait()
            except queue.Empty:
                return None
            if key == " " or time.monotonic_ns() - timestamp <= 100_000_000:
                return key

    def __exit__(self, *exc):
        self.ready.clear()


class KeyboardRobot:
    def __init__(self, args, kinematics):
        settings = load_config(args.tianji_config, args.robot_ip)
        self.ip = settings["controller_ip"]
        self.profile = select_side_profile(ControlProfile(**settings["profile"]), args.side,
                                           model_path=args.model)
        self.args, self.kinematics = args, kinematics
        self.keys = JogKeys()
        self.driver = None
        self.thread = None
        self.error = None

    def __enter__(self):
        self.driver = TianjiDriver(self.ip, self.args.sdk_root, model_path=self.args.model)
        try:
            self.driver.start(record_reported_config=False)

            def control():
                try:
                    run_jog(self.driver, self.kinematics, self.profile, side=self.args.side,
                            translation_m=self.args.step_mm / 1000,
                            rotation_rad=np.deg2rad(self.args.step_deg), transition_s=self.args.transition_s,
                            terminal=self.keys, configure_on_engage=True, control_mode="position",
                            emit=lambda text: print(text, flush=True))
                except Exception as error:
                    self.error = str(error)

            self.thread = threading.Thread(target=control, name="calibration-jog", daemon=True)
            self.thread.start()
            deadline = time.monotonic() + 3
            while not self.keys.ready.wait(.01):
                if self.error or not self.thread.is_alive() or time.monotonic() > deadline:
                    raise RuntimeError(self.error or "键盘控制启动超时")
            return self
        except BaseException:
            self.close()
            raise

    def snapshot(self):
        """Copy current actual feedback. Stopping is confirmed by the operator."""
        sample = self.driver.get_latest()
        if sample is None:
            raise ValueError("尚未收到机器人实际关节反馈")
        health = self.driver.health(sides=(self.args.side,))
        if not health.ready:
            raise ValueError(f"机器人反馈不可用：{health.detail}")
        arm = sample.payload.arms[self.args.side]
        q = np.asarray(arm.joints.position_rad, dtype=float)
        if arm.error or q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError(f"实际关节反馈无效，控制器错误码 {arm.error}")
        return {"joint_time_ns": sample.header.received_monotonic_ns,
                "joint_sequence": int(arm.source_sequence), "joints_rad": q.tolist()}

    def check(self):
        if self.error or self.thread is None or not self.thread.is_alive():
            raise RuntimeError(self.error or "键盘控制线程意外退出")

    def send_key(self, key):
        self.keys.put(key)

    def close(self):
        self.keys.stopped.set()
        try:
            if self.thread is not None:
                self.thread.join(2)
        finally:
            if self.driver is not None:
                self.driver.close()
                self.driver = None
        if self.thread is not None:
            self.thread.join(2)
            if self.thread.is_alive():
                raise RuntimeError("相机标定控制线程未能停止")

    def __exit__(self, *exc):
        self.close()
