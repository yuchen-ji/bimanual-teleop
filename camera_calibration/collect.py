"""Wrist cameras: keyboard position control and explicit C-key captures."""

from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from queue import Empty, SimpleQueue
import socket
import time

import cv2
import numpy as np
from PIL import Image

from bimanual_teleop.common.terminal import NonblockingTerminal, confirm_motion
from .board import CircleBoard, detect_board
from .motion import KeyboardInput, KeyboardRobot
from .preview import detection_overlay
from .robot import base_flange, load_model
from .storage import digest, write_json
from .streams import CameraStreams, require_local_usb


def collect(args):
    require_local_usb()
    board = CircleBoard(same_row_spacing_m=args.same_row_mm / 1000)
    board.points()
    model = load_model(args.model, sdk_root=args.sdk_root)
    sides = (args.side,)
    serials = {side: getattr(args, f"{side}_serial") for side in sides}
    output = Path(args.output).expanduser().resolve()
    counts, latest_frames = dict.fromkeys(sides, 0), {}
    captures = SimpleQueue()
    session = {"schema": "wrist_handeye_capture_v2", "status": "capturing",
               "method": "eye_in_hand", "board_mount": "fixed_in_environment",
               "capture_mode": "manual", "capture_key": "C", "stationary_confirmation": "operator",
               "robot_control": "keyboard", "robot_control_mode": "position",
               "images": "raw RGB sensor orientation", "sides": list(sides), "board": asdict(board),
               "robot_model_sha256": model.model.digest, "robot_model_file": model.model.path.name,
               "host": socket.gethostname(), "created_at": datetime.now().astimezone().isoformat(),
               "counts": counts}
    with NonblockingTerminal() as terminal:
        if not confirm_motion(terminal, "启动腕部标定键盘控制（不会自动回位）"):
            return None
        output.mkdir(parents=True, exist_ok=False)
        try:
            with ExitStack() as cleanup:
                cameras = cleanup.enter_context(CameraStreams(serials, {side: [640, 480] for side in sides}))
                for side, metadata in cameras.metadata.items():
                    metadata["output_rotation_deg"] = 180 if side == "left" else 0
                    (output / side).mkdir()
                session["cameras"] = cameras.metadata
                robot = cleanup.enter_context(KeyboardRobot(args, model))
                session["robot_ip"] = robot.ip
                ledger = cleanup.enter_context((output / "samples.jsonl").open("x"))

                def request_capture(requested_ns):
                    for side in sides:
                        if counts[side] >= args.samples:
                            continue
                        frame = latest_frames.get(side)
                        try:
                            if frame is None or time.monotonic_ns() - frame[0] > 500_000_000:
                                raise ValueError(f"{side} 尚无当前相机图像")
                            captures.put((side, requested_ns, frame, robot.snapshot(), None))
                        except (ValueError, RuntimeError) as error:
                            captures.put((side, requested_ns, None, None, str(error)))

                keys = cleanup.enter_context(KeyboardInput(terminal, robot, request_capture))
                figure, artists = None, {}
                if not args.no_preview:
                    import matplotlib.pyplot as plt
                    if plt.get_backend().lower() == "agg":
                        raise RuntimeError("没有桌面显示，请使用 --no-preview")
                    figure, axes = plt.subplots(1, len(sides), squeeze=False, figsize=(7 * len(sides), 6))
                    cleanup.callback(plt.close, figure)
                    figure.canvas.manager.set_window_title("Wrist calibration - keyboard position control")
                    default = getattr(figure.canvas.manager, "key_press_handler_id", None)
                    if default is not None:
                        figure.canvas.mpl_disconnect(default)
                    figure.canvas.mpl_connect("key_press_event", lambda event: keys.key(event.key))
                    figure.canvas.mpl_connect("close_event", lambda event: keys.key("q"))
                    for side, axis in zip(sides, axes.flat):
                        axis.axis("off")
                        axis.set_title(side)
                        artists[side] = axis.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
                    plt.show(block=False)
                print("腕部标定：标定板固定在环境中，使用位置模式键盘控制所选机械臂。\n"
                      "Enter 接合/恢复，Space 暂停，Q / Ctrl+C 退出；不会自动回位。\n"
                      "W/S X+/−，A/D Y+/−，R/F Z+/−；I/K 绕 X，J/L 绕 Y，U/O 绕 Z。\n"
                      f"默认每键 {args.step_mm:g} mm / {args.step_deg:g}°，均为所选臂原生基座坐标。\n"
                      "人工确认停稳后按 C 保存原图和实际关节反馈；终端和预览窗口均可操作。\n"
                      "不做自动停稳、前后时间窗或重复位姿判断。\n"
                      "预览显示圆点及 0-19 编号，保存原图；检测不完整也可保存。", flush=True)
                write_json(output / "session.json", session)
                while any(counts[side] < args.samples for side in sides):
                    keys.check()
                    robot.check()
                    frames = cameras.poll()
                    latest_frames.update(frames)
                    while True:
                        try:
                            side, requested_ns, frame, actual, error = captures.get_nowait()
                        except Empty:
                            break
                        if error:
                            print(f"本次未记录：{error}", flush=True)
                            continue
                        if counts[side] >= args.samples:
                            continue
                        path = output / side / f"{counts[side]:04d}.png"
                        Image.fromarray(frame[2]).save(path)
                        record = {"side": side, "group": counts[side], **actual,
                                  "capture_request_ns": requested_ns, "image_time_ns": frame[0],
                                  "image_frame_number": frame[1], "image": str(path.relative_to(output)),
                                  "image_sha256": digest(path), "time_error_ns": actual["joint_time_ns"] - frame[0],
                                  "T_base_flange": base_flange(model, side, actual["joints_rad"]).tolist()}
                        ledger.write(json.dumps(record, allow_nan=False) + "\n")
                        ledger.flush()
                        counts[side] += 1
                        write_json(output / "session.json", session)
                        print(f"{side} 已记录 {counts[side]}/{args.samples}", flush=True)
                    if keys.stopped.is_set():
                        break
                    if figure is not None:
                        for side, frame in frames.items():
                            display = frame[2]
                            if cameras.metadata[side]["output_rotation_deg"] == 180:
                                display = display[::-1, ::-1]
                            try:
                                centers = detect_board(display, board)
                            except cv2.error:
                                centers = None
                            artists[side].set_data(detection_overlay(display, centers))
                        for side, axis in zip(sides, axes.flat):
                            axis.set_title(f"{side} | Saved {counts[side]}/{args.samples} | C: capture")
                        figure.canvas.draw_idle()
                        plt.pause(.01)
                    else:
                        time.sleep(.01)
                keys.check()
        except KeyboardInterrupt:
            pass
        except BaseException as error:
            session.update(status="failed", error=str(error))
            raise
        finally:
            if session["status"] != "failed":
                session["status"] = "captured" if any(counts.values()) else "incomplete"
            write_json(output / "session.json", session)
    print(f"腕部采样结束：{counts}；{output}", flush=True)
    return output
