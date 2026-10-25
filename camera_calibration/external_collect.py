"""Manual main-camera capture with live circle positions and index overlay."""

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


def save_sample(output, index, frame, feedback, requested_ns, model, side, ledger):
    """Save one explicit request, without motion, board or duplicate-pose gates."""
    path = output / "external" / f"{index:04d}.png"
    Image.fromarray(frame[2]).save(path)
    record = {"group": index, **feedback, "capture_request_ns": requested_ns,
              "external_time_ns": frame[0], "external_frame_number": frame[1],
              "time_error_ns": feedback["joint_time_ns"] - frame[0],
              "external_image": str(path.relative_to(output)), "external_sha256": digest(path),
              "T_base_flange": base_flange(model, side, feedback["joints_rad"]).tolist()}
    ledger.write(json.dumps(record, allow_nan=False) + "\n")
    ledger.flush()


def collect(args):
    require_local_usb()
    board = CircleBoard(same_row_spacing_m=args.same_row_mm / 1000)
    board.points()
    model = load_model(args.model, sdk_root=args.sdk_root)
    output = Path(args.output).expanduser().resolve()
    with NonblockingTerminal() as terminal:
        if not confirm_motion(terminal, "启动标定键盘控制（不会自动回位）"):
            return None
        output.mkdir(parents=True, exist_ok=False)
        (output / "external").mkdir()
        session = {"schema": "external_eye_to_hand_capture_v3", "status": "capturing",
                   "method": "eye_to_hand", "board_mount": "rigid_on_selected_flange",
                   "stationary_confirmation": "operator", "capture_mode": "manual", "capture_key": "C",
                   "robot_control": "keyboard", "robot_control_mode": "position",
                   "images": "raw RGB sensor orientation", "side": args.side,
                   "reference_frame": f"tianji_{args.side}_base", "robot_model_sha256": model.model.digest,
                   "robot_model_file": model.model.path.name, "board": asdict(board),
                   "host": socket.gethostname(), "created_at": datetime.now().astimezone().isoformat(),
                   "counts": {"groups": 0}}
        count, latest_frame = 0, None
        captures = SimpleQueue()
        try:
            with ExitStack() as cleanup:
                camera = cleanup.enter_context(CameraStreams({"external": args.external_serial},
                                                              {"external": args.external_resolution}))
                session["cameras"] = camera.metadata
                robot = cleanup.enter_context(KeyboardRobot(args, model))
                ledger = cleanup.enter_context((output / "samples.jsonl").open("x"))

                def request_capture(requested_ns):
                    frame = latest_frame
                    try:
                        if frame is None or time.monotonic_ns() - frame[0] > 500_000_000:
                            raise ValueError("尚无当前相机图像")
                        # Freeze the actual values at the key press, not after
                        # detection or disk work. The operator confirms stopping.
                        captures.put((requested_ns, frame, robot.snapshot(), None))
                    except ValueError as error:
                        captures.put((requested_ns, None, None, str(error)))

                keyboard = cleanup.enter_context(KeyboardInput(terminal, robot, request_capture))
                figure = None
                if not args.no_preview:
                    import matplotlib.pyplot as plt
                    if plt.get_backend().lower() == "agg":
                        raise RuntimeError("没有桌面显示，请使用 --no-preview")
                    figure, axis = plt.subplots(figsize=(8, 6))
                    cleanup.callback(plt.close, figure)
                    artist = axis.imshow(np.zeros((args.external_resolution[1], args.external_resolution[0], 3), dtype=np.uint8))
                    axis.axis("off")
                    figure.canvas.manager.set_window_title("Main camera calibration - keyboard focus: terminal")
                    default = getattr(figure.canvas.manager, "key_press_handler_id", None)
                    if default is not None:
                        figure.canvas.mpl_disconnect(default)
                    figure.canvas.mpl_connect("close_event", lambda event: keyboard.stop())
                    plt.show(block=False)
                print("标定使用位置控制。终端按 Enter 接合/恢复，Space 暂停，Q 退出。\n"
                      "W/S、A/D、R/F 平移；I/K、J/L、U/O 旋转。\n"
                      "人工确认机器人停稳后按 C：立即记录当前原始图像和实际关节反馈。\n"
                      "预览显示圆点位置及 0-19 编号；检测不完整也可记录，离线求解时检查。", flush=True)
                write_json(output / "session.json", session)
                while count < args.samples:
                    keyboard.check()
                    robot.check()
                    new_frame = camera.poll().get("external")
                    if new_frame is not None:
                        latest_frame = new_frame
                    while count < args.samples:
                        try:
                            requested_ns, frame, feedback, error = captures.get_nowait()
                        except Empty:
                            break
                        if error:
                            print(f"本次未记录：{error}", flush=True)
                            continue
                        save_sample(output, count, frame, feedback, requested_ns, model, args.side, ledger)
                        count += 1
                        session["counts"]["groups"] = count
                        write_json(output / "session.json", session)
                        print(f"已记录 {count}/{args.samples}", flush=True)
                    if keyboard.stopped.is_set():
                        break
                    if figure is not None:
                        if new_frame is not None:
                            try:
                                centers = detect_board(new_frame[2], board)
                            except cv2.error:
                                centers = None
                            artist.set_data(detection_overlay(new_frame[2], centers))
                        axis.set_title(f"Saved {count}/{args.samples} | C: capture in terminal")
                        figure.canvas.draw_idle()
                        plt.pause(.01)
                    else:
                        time.sleep(.01)
                keyboard.check()
        except KeyboardInterrupt:
            pass
        except BaseException as error:
            session.update(status="failed", error=str(error))
            raise
        finally:
            if session["status"] != "failed":
                session["status"] = "captured" if count else "incomplete"
            session["counts"]["groups"] = count
            write_json(output / "session.json", session)
        print(f"采样结束：{count} 组；{output}", flush=True)
    return output
