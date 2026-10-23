"""Run on the USB-connected local PC: read images and feedback, issue no motion commands."""

import json
import platform
import socket
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from .board import CircleBoard, detect_board, estimate_board_pose

SERIALS = {"left": "408322072104", "right": "408322070938"}
OUTPUT_ROTATIONS = {"left": 180, "right": 0}


class FeedbackReader:
    """The only robot calls are open, poll_feedback and close on this SDK instance."""

    def __init__(self, sdk, ip):
        self.sdk, self.ip = sdk, ip
        self.buffers = {side: deque(maxlen=1000) for side in SERIALS}
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.error = None
        self.thread = None

    def start(self):
        self.sdk.open(self.ip)
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="handeye-read-feedback"
        )
        self.thread.start()

    def _run(self):
        previous = [None, None]
        try:
            while not self.stopped.is_set():
                sample = self.sdk.poll_feedback()
                if sample is not None:
                    with self.lock:
                        for index, side in enumerate(SERIALS):
                            sequence = sample.sequence[index]
                            if not sequence or sequence == previous[index]:
                                continue
                            previous[index] = sequence
                            q = np.deg2rad(sample.q[index * 7 : (index + 1) * 7])
                            if q.shape != (7,) or not np.isfinite(q).all():
                                raise ValueError(f"Invalid {side} measured joints")
                            self.buffers[side].append((sample.received_ns, sequence, q))
                self.stopped.wait(0.005)
        except Exception as error:  # noqa: BLE001 -- propagate SDK failures across the thread boundary
            self.error = str(error)
            self.stopped.set()

    def stationary(self, side, frame_ns):
        if self.error:
            raise RuntimeError(self.error)
        with self.lock:
            samples = list(self.buffers[side])
        window = [value for value in samples if abs(value[0] - frame_ns) <= 200_000_000]
        if (
            len(window) < 25
            or window[0][0] > frame_ns - 150_000_000
            or window[-1][0] < frame_ns + 150_000_000
        ):
            return None, "waiting for synchronized feedback"
        times = np.array([value[0] for value in window], dtype=np.int64)
        if np.diff(times).max() > 40_000_000:
            return None, "feedback gap exceeds 40 ms"
        spread = np.ptp(np.stack([value[2] for value in window]), axis=0)
        if spread.max() > 0.002:
            return None, "hold wrist still for 1-2 seconds"
        closest = min(window, key=lambda value: abs(value[0] - frame_ns))
        if abs(closest[0] - frame_ns) > 25_000_000:
            return None, "image/feedback mismatch exceeds 25 ms"
        return {
            "joint_time_ns": closest[0],
            "joint_sequence": int(closest[1]),
            "joints_rad": closest[2].tolist(),
            "joint_window_span_rad": spread.tolist(),
            "time_error_ns": closest[0] - frame_ns,
        }, "stationary"

    def close(self):
        self.stopped.set()
        if self.thread is not None:
            self.thread.join(2.0)
            if self.thread.is_alive():
                raise RuntimeError(
                    "SDK feedback thread did not stop; connection left untouched"
                )
        self.sdk.close()


def collect(args):
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise RuntimeError("The existing Tianji SDK requires the local Linux x86_64 PC")
    # Fail before creating any live driver when accidentally invoked on this simulation server.
    if not list(Path("/dev").glob("video*")):
        raise RuntimeError(
            "No local video devices. Run collect on the PC physically connected to the wrist cameras"
        )
    import pyrealsense2 as rs

    from bimanual_teleop.devices.tianji.sdk import ControlSDK

    sides = tuple(SERIALS) if args.side == "both" else (args.side,)
    serials = {"left": args.left_serial, "right": args.right_serial}
    board = CircleBoard(same_row_spacing_m=args.same_row_mm / 1000)
    board.points()  # Validate the scale before any connection.
    # The official SDK has a single UDP receiver; avoid taking over another teleop process.
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.bind(("0.0.0.0", 4730))
    except OSError as error:
        raise RuntimeError(
            "UDP 4730 is already in use. Close the local teleop/SDK process before calibration"
        ) from error
    finally:
        probe.close()
    context = rs.context()
    devices = {
        device.get_info(rs.camera_info.serial_number): device
        for device in context.query_devices()
    }
    if missing := [serials[side] for side in sides if serials[side] not in devices]:
        raise RuntimeError(
            f"Wrist cameras not connected: {missing}; found {list(devices)}"
        )
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    pipelines = {}
    pending = {side: deque(maxlen=20) for side in sides}
    last_q = {side: [] for side in sides}
    counts = dict.fromkeys(sides, 0)
    reasons = dict.fromkeys(sides, "warming up")
    document = {
        "schema": "wrist_handeye_capture_v1",
        "status": "capturing",
        "host": socket.gethostname(),
        "created_at": datetime.now().isoformat(),
        "board": {
            "columns": 4,
            "rows": 5,
            "same_row_spacing_m": board.same_row_spacing_m,
        },
        "robot_ip": args.robot_ip,
        "images": "raw RGB sensor orientation; no image rotation",
        "cameras": {},
        "stationary_only": True,
        "joint_units": "radians; measured SDK feedback, not commands",
    }
    reader = None
    ledger = None
    figure = None
    stop_ui = threading.Event()
    last_console = 0.0
    clock_wait_started = {}
    last_frame_at = {side: time.monotonic() for side in sides}
    try:
        wall_offset = time.time_ns() - time.monotonic_ns()
        for side in sides:
            device = devices[serials[side]]
            for sensor in device.query_sensors():
                if sensor.supports(rs.option.global_time_enabled):
                    sensor.set_option(rs.option.global_time_enabled, 1.0)
            pipeline = rs.pipeline(context)
            settings = rs.config()
            settings.enable_device(serials[side])
            settings.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            profile = pipeline.start(settings)
            pipelines[side] = pipeline
            value = (
                profile.get_stream(rs.stream.color)
                .as_video_stream_profile()
                .get_intrinsics()
            )
            document["cameras"][side] = {
                "serial": serials[side],
                "model": device.get_info(rs.camera_info.name),
                "resolution": [640, 480],
                "fps": 30,
                "output_rotation_deg": OUTPUT_ROTATIONS[side],
                "rgb_intrinsics": {
                    "width": value.width,
                    "height": value.height,
                    "fx": value.fx,
                    "fy": value.fy,
                    "ppx": value.ppx,
                    "ppy": value.ppy,
                    "model": str(value.model),
                    "coeffs": list(value.coeffs),
                },
            }
            (output / side).mkdir()
        # ControlSDK.open only establishes the feedback link. No profile, servo, state,
        # force reset, home, drag-mode or target-position API is called by this utility.
        reader = FeedbackReader(ControlSDK(), args.robot_ip)
        reader.start()
        ledger = (output / "samples.jsonl").open("x")
        (output / "session.json").write_text(json.dumps(document, indent=2) + "\n")
        artists = {}
        if not args.no_preview:
            import matplotlib.pyplot as plt

            if plt.get_backend().lower() == "agg":
                raise RuntimeError(
                    "No local desktop display. Use --no-preview, or run from the local desktop terminal"
                )
            figure, axes = plt.subplots(
                1, len(sides), squeeze=False, figsize=(6 * len(sides), 5)
            )
            figure.canvas.manager.set_window_title("Wrist hand-eye: read-only capture")
            for axis, side in zip(axes[0], sides):
                axis.set_axis_off()
                artists[side] = axis.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
            figure.canvas.mpl_connect("close_event", lambda event: stop_ui.set())
            figure.canvas.mpl_connect(
                "key_press_event",
                lambda event: (
                    stop_ui.set() if (event.key or "").lower() == "q" else None
                ),
            )
            plt.show(block=False)
        print(
            f"Read-only capture: {output}\nKeep board fixed. Move wrists with the pendant/drag mode; hold each pose 1-2 s.\n"
            f"Target {args.samples} distinct poses per wrist. Q / close window / Ctrl+C finishes and saves.",
            flush=True,
        )
        while not stop_ui.is_set() and any(
            counts[side] < args.samples for side in sides
        ):
            if reader.error:
                raise RuntimeError(reader.error)
            if abs((time.time_ns() - time.monotonic_ns()) - wall_offset) > 10_000_000:
                raise RuntimeError(
                    "Host clock changed by more than 10 ms; start a fresh capture"
                )
            now = time.monotonic_ns()
            for side in sides:
                if counts[side] >= args.samples:
                    continue
                frames = pipelines[side].poll_for_frames()
                if frames and (frame := frames.get_color_frame()):
                    last_frame_at[side] = time.monotonic()
                    rgb = np.asarray(frame.get_data()).copy()
                    if (
                        frame.get_frame_timestamp_domain()
                        != rs.timestamp_domain.global_time
                    ):
                        reasons[side] = "waiting for camera GLOBAL_TIME"
                        if artists:
                            artists[side].set_data(
                                rgb[::-1, ::-1] if OUTPUT_ROTATIONS[side] else rgb
                            )
                        clock_wait_started.setdefault(side, time.monotonic())
                        if time.monotonic() - clock_wait_started[side] > 20:
                            raise RuntimeError(
                                f"{side} camera did not enter GLOBAL_TIME within 20 s; check librealsense global-time support"
                            )
                        continue
                    clock_wait_started.pop(side, None)
                    timestamp = round(frame.get_timestamp() * 1_000_000) - wall_offset
                    pending[side].append(
                        (timestamp, int(frame.get_frame_number()), rgb)
                    )
                if time.monotonic() - last_frame_at[side] > 10:
                    raise RuntimeError(
                        f"No {side} RGB frames for 10 s; check USB and close other camera applications"
                    )
                candidate = None
                while pending[side] and pending[side][0][0] < now - 220_000_000:
                    candidate = pending[side].popleft()
                if candidate is None:
                    continue
                timestamp, frame_number, rgb = candidate
                if now - timestamp > 500_000_000 or timestamp > now:
                    reasons[side] = "stale or invalid camera timestamp"
                    continue
                centers = detect_board(rgb, board)
                feedback, reasons[side] = reader.stationary(side, timestamp)
                import cv2

                display = rgb.copy()
                if centers is not None:
                    cv2.drawChessboardCorners(
                        display,
                        (4, 5),
                        centers.astype(np.float32).reshape(-1, 1, 2),
                        True,
                    )
                else:
                    reasons[side] = "show all 20 circles"
                if artists:
                    artists[side].set_data(
                        display[::-1, ::-1] if OUTPUT_ROTATIONS[side] else display
                    )
                    artists[side].axes.set_title(
                        f"{side}: {counts[side]}/{args.samples} - {reasons[side]}"
                    )
                if centers is None or feedback is None:
                    continue
                if np.any(centers < 5) or np.any(centers > [634, 474]):
                    reasons[side] = "leave a margin around all circles"
                    continue
                q = np.asarray(feedback["joints_rad"])
                if any(np.max(np.abs(q - old)) < 0.07 for old in last_q[side]):
                    reasons[side] = "pose already saved; vary orientation"
                    continue
                try:
                    _, quality = estimate_board_pose(
                        centers, document["cameras"][side], board
                    )
                except ValueError as error:
                    reasons[side] = str(error)
                    continue
                name = f"{side}/{counts[side]:04d}.png"
                Image.fromarray(rgb).save(output / name)
                record = {
                    "side": side,
                    "image": name,
                    "frame_time_ns": timestamp,
                    "frame_number": frame_number,
                    **feedback,
                    **quality,
                }
                ledger.write(json.dumps(record) + "\n")
                ledger.flush()
                last_q[side].append(q)
                counts[side] += 1
                print(
                    f"Saved {side} {counts[side]}/{args.samples}: PnP {quality['pnp_rms_px']:.3f} px",
                    flush=True,
                )
            if figure is not None:
                for side in sides:
                    artists[side].axes.set_title(
                        f"{side}: {counts[side]}/{args.samples} - {reasons[side]}"
                    )
                figure.canvas.draw_idle()
                plt.pause(0.01)
            else:
                time.sleep(0.01)
            if time.monotonic() - last_console > 5:
                print(
                    " | ".join(
                        f"{side} {counts[side]}/{args.samples}: {reasons[side]}"
                        for side in sides
                    ),
                    flush=True,
                )
                last_console = time.monotonic()
    except KeyboardInterrupt:
        pass
    except BaseException as error:
        document["error"] = str(error)
        raise
    finally:
        document.update(
            status="captured" if "error" not in document else "failed", counts=counts
        )
        (output / "session.json").write_text(json.dumps(document, indent=2) + "\n")
        if ledger is not None:
            ledger.close()
        try:
            if reader is not None:
                reader.close()
        finally:
            for pipeline in pipelines.values():
                pipeline.stop()
        if figure is not None:
            plt.close(figure)
        print(f"Capture saved: {output}; {counts}", flush=True)
    return output
