"""Independent, one-arm keyboard Cartesian jog in the arm base frame."""

from __future__ import annotations

import argparse

from bimanual_teleop.devices.tianji.sdk import add_sdk_argument
from bimanual_teleop.common.config import positive_number
from dataclasses import replace
import math
from pathlib import Path
import time

from bimanual_teleop.common.terminal import NonblockingTerminal, confirm_motion
from bimanual_teleop.common.console import LiveProgress, StatusConsole, configure_runtime_logging, print_message
from bimanual_teleop.devices.tianji.driver import TianjiDriver
from bimanual_teleop.devices.tianji.config import DEFAULT_CONFIG, load_config, select_profile_side
from bimanual_teleop.devices.tianji.model import MotionProfile, TianjiKinematics
from bimanual_teleop.control.arm.cartesian import PERIOD_NS, TianjiCartesianExecutor
from bimanual_teleop.control.arm.preparation import prepare_initial_pose
from bimanual_teleop.types import ControlProfile, Pose, RobotTarget, Side


TRANSLATION_KEYS = {"w": (0, 1), "s": (0, -1), "a": (1, 1),
                    "d": (1, -1), "r": (2, 1), "f": (2, -1)}
ROTATION_KEYS = {"i": (0, 1), "k": (0, -1), "j": (1, 1),
                 "l": (1, -1), "u": (2, 1), "o": (2, -1)}
HELP = ("Enter 接合/恢复 · Space 暂停 · Q 退出\n"
        "W/S X+/− · A/D Y+/− · R/F Z+/−（原生基座）\n"
        "I/K 绕 X · J/L 绕 Y · U/O 绕 Z（基座坐标系）")


def select_side_profile(profile: ControlProfile, side: Side, *, model_path=None) -> ControlProfile:
    """Keep only the selected arm, including the native driver's health scope."""
    parsed = MotionProfile.from_control_profile(profile, model_path=model_path)
    if side not in parsed.active_arms:
        raise ValueError(f"{side} is absent from the motion profile")
    selected = select_profile_side(profile, side)
    MotionProfile.from_control_profile(selected, model_path=model_path)
    return selected


def _multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
            aw*bw - ax*bx - ay*by - az*bz)


def _normalized(quaternion):
    norm = math.sqrt(sum(value*value for value in quaternion))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("pose orientation is invalid")
    return tuple(value / norm for value in quaternion)


def jog_pose(pose: Pose, key: str, translation_m: float = .005,
             rotation_rad: float = math.radians(2)) -> Pose:
    """A single base-frame step; world-axis rotation is left-multiplied."""
    positive_number(translation_m, "translation_m")
    positive_number(rotation_rad, "rotation_rad")
    key = key.lower()
    if key in TRANSLATION_KEYS:
        axis, sign = TRANSLATION_KEYS[key]
        xyz = list(pose.position_m)
        xyz[axis] += sign * translation_m
        return replace(pose, position_m=tuple(xyz))
    if key in ROTATION_KEYS:
        axis, sign = ROTATION_KEYS[key]
        angle = sign * rotation_rad / 2
        delta = [0., 0., 0., math.cos(angle)]
        delta[axis] = math.sin(angle)
        return replace(pose, orientation_xyzw=_normalized(
            _multiply(tuple(delta), pose.orientation_xyzw)))
    raise ValueError(f"unsupported jog key: {key}")


def interpolate_pose(start: Pose, goal: Pose, fraction: float) -> Pose:
    t = max(0., min(1., fraction))
    t = t*t*(3 - 2*t)
    position = tuple(a + (b-a)*t for a, b in zip(start.position_m, goal.position_m))
    first, last = start.orientation_xyzw, goal.orientation_xyzw
    if sum(a*b for a, b in zip(first, last)) < 0:
        last = tuple(-value for value in last)
    orientation = _normalized(tuple(a + (b-a)*t for a, b in zip(first, last)))
    return replace(start, position_m=position, orientation_xyzw=orientation)


class JogPlanner:
    def __init__(self, initial: Pose, *, transition_s: float = .25):
        self.transition_ns = round(positive_number(transition_s, "transition_s") * 1e9)
        self.start = self.goal = initial
        self.started_ns = 0

    def pose_at(self, now_ns: int) -> Pose:
        return interpolate_pose(self.start, self.goal,
                                (now_ns - self.started_ns) / self.transition_ns)

    def step(self, key: str, now_ns: int, *, side: Side, kinematics,
             reference_rad: tuple[float, ...], translation_m: float,
             rotation_rad: float) -> Pose:
        candidate = jog_pose(self.goal, key, translation_m, rotation_rad)
        kinematics.ik(side, candidate, reference_rad)
        self.start = self.pose_at(now_ns)
        self.goal = candidate
        self.started_ns = now_ns
        return candidate


class SideFeedbackTracker:
    """Track freshness and valid joint feedback for the selected arm only."""

    def __init__(self, side: Side, timeout_ns: int):
        self.side, self.timeout_ns = side, timeout_ns
        self.sequence = None
        self.advanced_ns = 0

    def observe(self, sample, now_ns: int) -> tuple[float, ...] | None:
        if sample is None:
            return None
        arm = sample.payload.arms[self.side]
        if arm.source_sequence != self.sequence:
            self.sequence = arm.source_sequence
            self.advanced_ns = sample.header.received_monotonic_ns
        if (not self.advanced_ns or now_ns - self.advanced_ns >= self.timeout_ns
                or arm.error):
            return None
        joints = arm.joints.position_rad
        if len(joints) != 7 or any(q is None or not math.isfinite(q) for q in joints):
            return None
        return tuple(joints)


def run_jog(driver: TianjiDriver, kinematics: TianjiKinematics, profile: ControlProfile,
            *, side: Side, translation_m: float, rotation_rad: float,
            transition_s: float, terminal=None, emit=None,
            configure_on_engage: bool = False,
            control_mode: str = "cartesian") -> None:
    """Process keys and refresh the selected target at 200 Hz while engaged."""
    if control_mode not in ("cartesian", "position"):
        raise ValueError("control_mode must be cartesian or position")
    progress = LiveProgress()
    status = StatusConsole()

    def say(message, level="info"):
        progress.clear()
        if emit is None:
            status.state(message, level)
        else:
            emit(message)

    def show_target(message):
        if emit is None:
            progress.update(message)
        else:
            emit(message)

    executor = TianjiCartesianExecutor(driver, kinematics)
    configured = False
    if not configure_on_engage:
        executor.configure(profile)
        configured = True
    tracker = SideFeedbackTracker(side, driver.watchdog_ns)
    planner = None
    engaged = False
    count = 0
    say(HELP)
    say("按 Enter 接合所选机械臂。")
    terminal = terminal or NonblockingTerminal()
    try:
        with terminal as keys:
            deadline = time.monotonic_ns()
            while True:
                now = time.monotonic_ns()
                if now < deadline:
                    time.sleep((deadline - now) / 1e9)
                    now = time.monotonic_ns()
                deadline += PERIOD_NS
                if deadline <= now:
                    deadline = now + PERIOD_NS
                reference = tracker.observe(driver.get_latest(), now)
                key_batch = keys.read(0)
                if key_batch == "":
                    break
                for key in key_batch or "":
                    key = key.lower()
                    if key == "q":
                        return
                    if key == " ":
                        if engaged:
                            executor.request_hold("keyboard pause")
                            engaged = False
                            planner = None
                        say("已暂停。", "warning")
                        continue
                    if key in ("\r", "\n"):
                        if not engaged:
                            try:
                                if not configured:
                                    executor.configure(profile)
                                    configured = True
                                if control_mode == "position":
                                    executor.engage(control_mode="position")
                                else:
                                    executor.engage()
                                planner = JogPlanner(executor.engagement_poses[side], transition_s=transition_s)
                                engaged = True
                                say("已接合所选机械臂。", "ready")
                            except (RuntimeError, ValueError) as error:
                                say(f"接合失败：{error}", "error")
                        continue
                    if key not in TRANSLATION_KEYS and key not in ROTATION_KEYS:
                        continue
                    if planner is None:
                        say("请先按 Enter 接合。", "warning")
                        continue
                    if reference is None:
                        if engaged:
                            executor.request_hold("selected arm feedback lost")
                            engaged = False
                            planner = None
                        health = driver.health(sides=(side,))
                        say(f"所选机械臂反馈无效或超时，已暂停：{health.detail or '关节反馈无效或源序号未更新'}", "warning")
                        continue
                    try:
                        goal = planner.step(key, now, side=side, kinematics=kinematics,
                                            reference_rad=reference, translation_m=translation_m,
                                            rotation_rad=rotation_rad)
                        show_target(f"{side} {key.upper()} → 基座位姿 {tuple(round(x, 4) for x in goal.position_m)} m")
                    except (RuntimeError, ValueError) as error:
                        if engaged:
                            executor.request_hold(f"keyboard jog IK rejected: {error}")
                            engaged = False
                            planner = None
                        say(f"目标不可用，已暂停：{error}", "error")
                if engaged:
                    if planner is None:
                        raise RuntimeError("engaged jog lost its target planner")
                    count += 1
                    submit_ns = time.monotonic_ns()
                    target = RobotTarget(
                        f"keyboard-jog-{count}", {side: planner.pose_at(submit_ns)}, (executor.engagement_ref,), submit_ns, submit_ns + driver.watchdog_ns,
                        profile.profile_id)
                    result = executor.submit(target)
                    if not result.accepted:
                        engaged = False
                        planner = None
                        say(f"已暂停：{result.reason}", "warning")
    finally:
        progress.clear()
        if engaged:
            executor.request_hold("keyboard jog exited")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--tianji-config", "--config", dest="config", type=Path, default=DEFAULT_CONFIG,
                        help="天机统一配置 YAML；默认 configs/tianji_teleop.yaml")
    parser.add_argument("--ip", help="临时覆盖设备配置中的天机控制器 IP")
    parser.add_argument("--model", type=Path)
    add_sdk_argument(parser)
    parser.add_argument("--step-mm", type=float, default=5., help="单次平移，默认 5 mm")
    parser.add_argument("--step-deg", type=float, default=2., help="单次旋转，默认 2°")
    parser.add_argument("--transition-s", type=float, default=.25, help="点动平滑时间，默认 0.25 s")
    args = parser.parse_args(argv)
    driver = None
    error = None
    try:
        configure_runtime_logging()
        settings = load_config(args.config, args.ip)
        controller_ip = settings["controller_ip"]
        translation_m = positive_number(args.step_mm, "--step-mm") / 1000
        rotation_rad = math.radians(positive_number(args.step_deg, "--step-deg"))
        transition_s = positive_number(args.transition_s, "--transition-s")
        profile = select_side_profile(ControlProfile(**settings["profile"]), args.side, model_path=args.model)
        with NonblockingTerminal() as terminal:
            if not confirm_motion(terminal, "开始初始回位，完成后等待点动接合"):
                return 0
            prepare_initial_pose(config=args.config, ip=controller_ip, side=args.side,
                                 sdk_root=args.sdk_root, model=args.model, terminal=terminal)
        kinematics = TianjiKinematics(args.sdk_root, args.model)
        driver = TianjiDriver(controller_ip, args.sdk_root, model_path=args.model)
        driver.start()
        run_jog(driver, kinematics, profile, side=args.side,
                translation_m=translation_m,
                rotation_rad=rotation_rad, transition_s=transition_s)
    except KeyboardInterrupt:
        pass
    except Exception as problem:
        error = str(problem)
    finally:
        if driver is not None:
            try:
                driver.close()
            except Exception as problem:
                error = f"{error + '; ' if error else ''}{problem}"
    if error:
        print_message(f"天机点动：{error}", "error")
    else:
        print_message("已退出。", "done")
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
