"""Quest 与 Wuji 双臂双手遥操作；--arms-only 仅控制机械臂。"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from bimanual_teleop.devices.tianji.sdk import add_sdk_argument
from bimanual_teleop.types import ControlProfile
from bimanual_teleop.common.console import configure_runtime_logging, print_message, runtime_message
from bimanual_teleop.common.runlog import default_run_log_path, process_snapshot
from bimanual_teleop.common.terminal import NonblockingTerminal, confirm_motion
from bimanual_teleop.devices.tianji.config import DEFAULT_CONFIG, load_config, select_profile_side
from bimanual_teleop.devices.wuji.config import DEFAULT_CONFIG as DEFAULT_WUJI_CONFIG
from bimanual_teleop.control.arm import preparation
from bimanual_teleop.cli.runtime import TeleopUI, run_loop

MAPPING_HELP = "左手柄控制右臂，右手柄控制左臂；坐标系由配置 quest.coordinate_frame 选择，默认 headset。"


def prepare_initial_pose(args, terminal):
    """Finish the existing position runner before creating any teleop devices."""
    result = preparation.prepare_initial_pose(
        config=args.config, ip=args.robot_ip, terminal=terminal,
        side=getattr(args, "side", "both"), sdk_root=args.sdk_root,
        model=getattr(args, "model", None), verbose=getattr(args, "verbose", False))
    print_message("正在连接 Quest。")
    return result


def create_runtime(args, profile, sink):
    from bimanual_teleop.devices.quest.adapter import QuestSource
    from bimanual_teleop.devices.tianji.driver import TianjiDriver
    from bimanual_teleop.devices.tianji.model import TianjiKinematics
    from bimanual_teleop.control.arm.quest import QuestTianjiTeleop
    from bimanual_teleop.control.arm.cartesian import TianjiCartesianExecutor

    quest = QuestSource(serial=args.serial)
    driver_options = {"record_force": True} if getattr(args, "record", False) else {}
    driver = TianjiDriver(args.robot_ip, args.sdk_root, **driver_options)
    kinematics = TianjiKinematics(args.sdk_root)
    executor = TianjiCartesianExecutor(driver, kinematics)
    arms = QuestTianjiTeleop(quest, driver, executor, profile=profile, sink=sink,
                            side=getattr(args, "side", "both"),
                            coordinate_frame=args.coordinate_frame,
                            ready_pose=getattr(args, "ready_pose", None))
    if args.arms_only:
        return arms
    from bimanual_teleop.control.hand.process import WujiProcess
    from bimanual_teleop.control.combined import QuestTianjiWujiTeleop
    hand_options = {"record_sink": sink} if getattr(args, "record", False) else {}
    hands = WujiProcess(args.wuji_settings, verbose=getattr(args, "verbose", False), **hand_options)
    return QuestTianjiWujiTeleop(arms, hands)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=MAPPING_HELP)
    parser.add_argument("--tianji-config", "--config", dest="config", type=Path, default=DEFAULT_CONFIG,
                        help="天机统一配置 YAML；默认 configs/tianji_teleop.yaml")
    parser.add_argument("--side", choices=("left", "right", "both"), default="both",
                        help="select the robot arm; left uses the right controller and vice versa")
    parser.add_argument("--serial", help="Quest USB adb serial")
    parser.add_argument("--robot-ip", help="临时覆盖设备配置中的天机控制器 IP")
    add_sdk_argument(parser)
    parser.add_argument("--wuji-config", type=Path, default=DEFAULT_WUJI_CONFIG,
                        help="Wuji 配置；默认 configs/wuji_teleop.yaml")
    parser.add_argument("--user-name", help="联合控制使用的已有 Wuji SDK 用户名；默认从配置文件读取")
    parser.add_argument("--arms-only", action="store_true", help="只控制机械臂，不连接 Wuji 手套和灵巧手")
    parser.add_argument("--viewer", action="store_true", help="显示已连接 RealSense 的彩色图像；默认关闭")
    parser.add_argument("--record", action="store_true", help="启用原始采集；C 开始、S 保存、X 作废")
    parser.add_argument("--recording-config", type=Path,
                        help="采集配置 YAML；默认 configs/recording.yaml")
    parser.add_argument("--log-file", type=Path,
                        help="详细运行日志路径；默认写入 logs/teleop_quest_tianji_<时间>_<进程>.jsonl")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="输出跟随受限提示、SDK 告警及完整诊断；默认只显示关键状态和故障")
    args = parser.parse_args(argv)
    if not args.arms_only and args.side != "both":
        parser.error("联合控制需要 --side both；单臂控制请加 --arms-only")
    if args.record and (args.arms_only or args.side != "both"):
        parser.error("数据采集需要双臂双手联合遥操作")
    combined = not args.arms_only
    runtime = ui = None
    preview = None
    recorder = None
    error = None
    timing = None
    runtime_log = None
    try:
        log_path = args.log_file or default_run_log_path()
        runtime_log = configure_runtime_logging(
            wuji=combined, verbose=args.verbose, log_file=log_path)
        runtime_log.start(command="teleop_quest_tianji", arguments=vars(args))
        print_message(f"详细运行日志：{runtime_log.path}")
        settings = load_config(args.config, args.robot_ip)
        args.robot_ip = settings["controller_ip"]
        args.coordinate_frame = settings["quest"]["coordinate_frame"]
        args.ready_pose = settings.get("ready_pose")
        profile = select_profile_side(
            ControlProfile(**settings["profile"]), args.side)
        if combined:
            from bimanual_teleop.control.hand.follow import preflight
            from bimanual_teleop.devices.wuji.config import load_config as load_wuji_config
            args.wuji_settings = load_wuji_config(args.wuji_config)
            if args.user_name is not None:
                from bimanual_teleop.devices.wuji.config import sdk_user_name
                args.wuji_settings["sdk_user_name"] = sdk_user_name({}, user_name=args.user_name)
            preflight()
        runtime_log.event("configuration_loaded", tianji=settings,
                          wuji=args.wuji_settings if combined else None,
                          record=args.record, viewer=args.viewer)
        if args.record:
            from bimanual_teleop.recording.config import load_config as load_recording_config, DEFAULT_CONFIG as RECORDING_CONFIG
            from bimanual_teleop.recording.recorder import Recorder
            recorder = Recorder(load_recording_config(args.recording_config or RECORDING_CONFIG),
                sdk_root=args.sdk_root, viewer=args.viewer,
                metadata={"tianji_config": settings, "wuji_config": args.wuji_settings})
            runtime_log.event("recorder_starting", recorder=recorder.status())
            recorder.start()
            runtime_log.event("recorder_started", recorder=recorder.status())
        with NonblockingTerminal() as terminal:
            if not confirm_motion(terminal, "开始初始回位，完成后等待遥操作接合"):
                runtime_log.event("motion_confirmation_cancelled")
                return 0
            runtime_log.event("motion_confirmation_accepted")
            if args.viewer and not args.record:
                from bimanual_teleop.visualization.realsense import RealSensePreview
                preview = RealSensePreview()
                preview.start()
            runtime_log.event("initial_pose_starting")
            prepare_initial_pose(args, terminal)
            runtime_log.event("initial_pose_completed")
            runtime = create_runtime(args, profile, recorder.sink if recorder else None)
            runtime_log.event("runtime_created", combined=combined)
            ui_type = TeleopUI
            ui_options = {}
            if recorder is not None:
                from bimanual_teleop.recording.ui import RecordingUI, HELP as RECORDING_HELP
                ui_type, ui_options = RecordingUI, {"recorder": recorder}
                print_message(RECORDING_HELP)
            runtime.start()
            runtime_log.event("runtime_started", status=(
                runtime.status(include_target=False) if hasattr(runtime, "status") else {}))
            gesture = None
            if combined and settings["controls"]["gesture_engagement_enabled"]:
                from bimanual_teleop.control.hand.gesture import GestureCommands
                gesture = GestureCommands({side: lambda side=side: runtime.hands.glove_samples()[side]
                    for side in ("left", "right")},
                    timeout_s=runtime.hands.glove_timeout_ns / 1e9)
            ui = ui_type(runtime, profile, gesture=gesture, **ui_options, **settings["controls"],
                            home_enabled=True,
                            verbose=args.verbose,
                            runtime_log=runtime_log,
                            background_engage=combined,
                            following_message="已接合，双臂与双手正在跟随。" if combined else
                                              f"已接合，{'双臂' if args.side == 'both' else '左臂' if args.side == 'left' else '右臂'}正在跟随手柄。")
            print_message("实机遥操作\n" + ui.help_text)
            runtime_log.event("control_loop_starting")
            timing = run_loop(runtime, ui, terminal)
            runtime_log.event("control_loop_completed", timing=timing)
    except KeyboardInterrupt:
        if runtime_log is not None:
            runtime_log.event("keyboard_interrupt")
    except (OSError, RuntimeError, ValueError, TypeError, ImportError, subprocess.SubprocessError) as problem:
        error = str(problem)
        if runtime_log is not None:
            runtime_log.event("unhandled_runtime_error", error=error,
                              process=process_snapshot())
    finally:
        if ui is not None:
            try:
                ui.close()
            except (OSError, RuntimeError, ValueError) as problem:
                error = f"{error + '; ' if error else ''}关闭遥操作失败：{problem}"
                if runtime_log is not None:
                    runtime_log.event("ui_close_failed", error=str(problem))
        if runtime is not None and ui is None:
            try:
                runtime.close()
            except (OSError, RuntimeError, ValueError) as problem:
                error = f"{error + '; ' if error else ''}关闭运行模块失败：{problem}"
                if runtime_log is not None:
                    runtime_log.event("runtime_close_failed", error=str(problem))
        if preview is not None:
            try:
                preview.close()
            except (OSError, RuntimeError, ValueError) as problem:
                error = f"{error + '; ' if error else ''}关闭预览失败：{problem}"
                if runtime_log is not None:
                    runtime_log.event("preview_close_failed", error=str(problem))
        if recorder is not None and ui is None:
            try:
                recorder.close()
            except (OSError, RuntimeError, ValueError) as problem:
                error = f"{error + '; ' if error else ''}关闭采集模块失败：{problem}"
                if runtime_log is not None:
                    runtime_log.event("recorder_close_failed", error=str(problem))
        if runtime_log is not None:
            runtime_log.close(error=error, timing=timing)
    if error:
        print_message(runtime_message(error, verbose=args.verbose), "error")
    else:
        summary = f"已退出，运行 {timing['elapsed_s']:.1f} 秒。" if timing else "已退出。"
        if timing and timing["motion_pauses"]:
            summary += f" 期间暂停 {timing['motion_pauses']} 次。"
        print_message(summary, "done")
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
