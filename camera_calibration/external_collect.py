"""Local paired-camera capture with a stationary board and read-only feedback."""

import json
import socket
import threading
import time
from collections import deque
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .board import CircleBoard, detect_board, estimate_board_pose, opencv_distortion
from .collect import FeedbackReader
from .external import independent_view, view_coverage
from .external_workflow import check_wrist_camera
from .storage import digest, load_wrist_reference, write_json
from .streams import BoardStability, CameraStreams, paired_frame, require_local_usb


def collect(args):
    require_local_usb()
    reference, wrist = load_wrist_reference(args.wrist_result, args.side)
    board = CircleBoard(**reference["board"])
    board.points()
    # The SDK receiver is exclusive; check before opening any camera/robot driver.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.bind(("0.0.0.0", 4730))
        except OSError as error:
            raise RuntimeError(
                "UDP 4730 is occupied; close the other teleop/SDK process"
            ) from error
    from bimanual_teleop.devices.tianji.sdk import ControlSDK

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    serials = {"wrist": wrist["camera"]["serial"], "external": args.external_serial}
    resolutions = {
        "wrist": wrist["camera"]["resolution"],
        "external": args.external_resolution,
    }
    session = {
        "schema": "external_camera_capture_v1",
        "created_at": datetime.now().isoformat(),
        "status": "capturing",
        "side": args.side,
        "stationary_only": True,
        "images": "raw RGB sensor orientation",
        "board": reference["board"],
        "wrist_calibration_sha256": digest(args.wrist_result),
        "robot_ip": args.robot_ip,
    }
    saved, queues = [], {name: deque(maxlen=48) for name in serials}
    stability, stopped = BoardStability(), threading.Event()
    figure, plots = None, {}
    reason, coverage = "waiting for stationary paired images", False
    try:
        with ExitStack() as cleanup:
            cameras = cleanup.enter_context(CameraStreams(serials, resolutions))
            check_wrist_camera(cameras.metadata["wrist"], wrist["camera"])
            for metadata in cameras.metadata.values():
                opencv_distortion(metadata)
            cameras.metadata["wrist"]["output_rotation_deg"] = wrist["camera"][
                "output_rotation_deg"
            ]
            session["cameras"] = cameras.metadata
            reader = FeedbackReader(ControlSDK(), args.robot_ip)
            cleanup.callback(reader.close)
            reader.start()
            ledger = cleanup.enter_context((output / "samples.jsonl").open("x"))
            for name in serials:
                (output / name).mkdir()
            if not args.no_preview:
                import matplotlib.pyplot as plt

                if plt.get_backend().lower() == "agg":
                    raise RuntimeError("No desktop display; use --no-preview")
                figure, axes = plt.subplots(1, 2, figsize=(12, 5))
                cleanup.callback(plt.close, figure)
                figure.canvas.mpl_connect("close_event", lambda event: stopped.set())
                figure.canvas.mpl_connect(
                    "key_press_event", lambda e: stopped.set() if e.key == "q" else None
                )
                for name, axis in zip(serials, axes):
                    plots[name] = axis.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
                    axis.axis("off")
                plt.show(block=False)
            last_notice = 0.0
            while not stopped.is_set() and (len(saved) < args.samples or not coverage):
                if reader.error:
                    raise RuntimeError(reader.error)
                for name, frame in cameras.poll().items():
                    queues[name].append(frame)
                    if plots:
                        display = frame[2]
                        if (
                            name == "wrist"
                            and wrist["camera"]["output_rotation_deg"] == 180
                        ):
                            display = display[::-1, ::-1]
                        plots[name].set_data(display)
                pair = paired_frame(queues, time.monotonic_ns())
                if pair:
                    timestamp = pair[0][0]
                    feedback, reason = reader.stationary(args.side, timestamp)
                    if feedback is not None:
                        centers = [detect_board(frame[2], board) for frame in pair]
                        if any(c is None for c in centers):
                            reason = "show all 20 circles in both views"
                            stability.clear()
                        elif not stability.update(timestamp, centers):
                            reason = "hold the board and wrist still for 1-2 seconds"
                        else:
                            try:
                                poses = []
                                for name, pixels in zip(serials, centers):
                                    if np.any(pixels < 5) or np.any(
                                        pixels > np.asarray(resolutions[name]) - 6
                                    ):
                                        raise ValueError(
                                            "leave a margin around every circle"
                                        )
                                    pose, _ = estimate_board_pose(
                                        pixels, cameras.metadata[name], board
                                    )
                                    poses.append(pose)
                                if not independent_view(*poses, saved):
                                    raise ValueError(
                                        "view already saved; change board position or tilt"
                                    )
                                index = len(saved)
                                record = {
                                    "group": index,
                                    **feedback,
                                    "pair_time_error_ns": pair[1][0] - timestamp,
                                }
                                for name, frame in zip(serials, pair):
                                    path = output / name / f"{index:04d}.png"
                                    Image.fromarray(frame[2]).save(path)
                                    record.update(
                                        {
                                            f"{name}_image": str(
                                                path.relative_to(output)
                                            ),
                                            f"{name}_sha256": digest(path),
                                            f"{name}_time_ns": frame[0],
                                            f"{name}_frame_number": frame[1],
                                        }
                                    )
                                ledger.write(json.dumps(record) + "\n")
                                ledger.flush()
                                saved.append(tuple(poses))
                                stability.clear()
                                try:
                                    view_coverage([p[0] for p in saved])
                                    coverage = True
                                except (ValueError, IndexError):
                                    coverage = False
                                reason = (
                                    "saved"
                                    if coverage
                                    else "saved; add tilt about two different axes"
                                )
                                print(
                                    f"Saved group {index}: {len(saved)}/{args.samples}; {reason}",
                                    flush=True,
                                )
                            except (ValueError, cv2.error) as error:
                                reason = str(error)
                    else:
                        stability.clear()
                if figure is not None:
                    figure.suptitle(
                        f"External calibration {len(saved)}/{args.samples}: {reason}"
                    )
                    figure.canvas.draw_idle()
                    plt.pause(0.01)
                else:
                    time.sleep(0.01)
                if time.monotonic() - last_notice > 5:
                    print(f"{len(saved)}/{args.samples}: {reason}", flush=True)
                    last_notice = time.monotonic()
            session["status"] = (
                "captured" if len(saved) >= 15 and coverage else "incomplete"
            )
    except KeyboardInterrupt:
        session["status"] = (
            "captured" if len(saved) >= 15 and coverage else "incomplete"
        )
    except BaseException as error:
        session.update(status="failed", error=str(error))
        raise
    finally:
        session["counts"] = {"groups": len(saved)}
        write_json(output / "session.json", session)
    return output
