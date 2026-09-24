"""One recording process, isolated from both robot control interpreters."""

from dataclasses import asdict, replace
from datetime import datetime
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import signal
import threading
import time
from uuid import uuid4

from .sink import CaptureChannel, RecorderSink, STATE_STREAMS, COMMAND_STREAMS


def _lower_priority(increment):
    """Let motion/control processes win CPU contention without requiring privileges.

    Do not also pin this process to a CPU subset. Encoder workers are spawned
    afterward and inherit this niceness, while still being allowed to run on
    idle cores. Motion control keeps the full affinity.
    """
    if not increment:
        return
    try:
        os.nice(increment)
    except (AttributeError, OSError):
        pass


def _worker(config, sdk_root, metadata, channel, connection, viewer, nice_increment=5):
    from bimanual_teleop.common.affinity import apply_recording_affinity
    from bimanual_teleop.devices.tianji.model import TianjiKinematics
    from .camera import CameraRig
    from .spool import RawEpisodeWriter

    # Terminal SIGINT belongs to the coordinator, which pauses motion and sends
    # bounded shutdown requests. Do not interrupt a child midway through I/O.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    apply_recording_affinity("background")
    _lower_priority(nice_increment)
    rig = CameraRig(config)
    writer = None
    preview = None
    failure = "录制未正常完成"
    generation = 0
    ending = None
    latest_images = {}
    camera_seen = {}
    required_cameras = [f"cameras/camera_{i}/rgb" for i in range(3)]
    if config.main_depth:
        required_cameras.append("cameras/camera_0/depth")
    try:
        kinematics = TianjiKinematics(sdk_root)
        if viewer:
            from .preview import RecordingPreview
            preview = RecordingPreview()
        rig.start()
        # Before C, only the optional 5 Hz RGB preview needs pixels. Avoid four
        # full-rate image copies while the robot is following but not recording.
        if viewer:
            rig.preview_delivery()
        else:
            rig.suspend_delivery()
        metadata = {**metadata, "recording": asdict(config), "cameras": rig.metadata,
                    "model_sha256": kinematics.model.digest,
                    "pose_source": "FK of recorded measured joints; xyz_m + quaternion_xyzw",
                    "arm_frames": {s: [f"tianji_{s}_base", f"tianji_{s}_flange"]
                                   for s in ("left", "right")},
                    "joint_order": {"arms": "left/right: J1..J7",
                                    "hands": "left/right: finger1..5, each joint1..4"},
                    "wrench": "Fx,Fy,Fz [N], Tx,Ty,Tz [Nm]; native sensor axes, no added compensation",
                    "state_time": "host SDK observation/dequeue monotonic_ns",
                    "command_time": "successful SDK submission; not physical execution",
                    "camera_time": "GLOBAL_TIME frame timestamp mapped to host monotonic; not exposure midpoint",
                    "episode_time": "host monotonic with disengaged intervals removed; "
                                    "resume continues from the pause"}
        connection.send(("ready", None))
        running = True
        holding = False
        pause_ns = None
        time_offset_ns = 0
        resume_after_ns = 0
        pause_requested = None
        stop_immediately = False

        def episode_record(record):
            if holding or record.time_ns < resume_after_ns:
                return None
            return replace(record, time_ns=record.time_ns - time_offset_ns)

        while running:
            parent = mp.parent_process()
            if parent is not None and not parent.is_alive():
                raise RuntimeError("遥操作主进程已退出")
            while connection.poll():
                message = connection.recv()
                operation = message[0]
                if operation == "start":
                    if writer is not None:
                        raise RuntimeError("Previous episode has not finished")
                    _, generation, path, start_ns = message
                    # Zarr and codec initialization may take longer than the
                    # camera queue's intentional backpressure window.  No
                    # episode is active yet, so keep validating live cameras
                    # but do not buffer frames that cannot belong to it.
                    rig.suspend_delivery()
                    try:
                        writer = RawEpisodeWriter(path, start_ns, metadata, kinematics,
                                                  frame_capacity=config.frame_capacity)
                        writer.prepare_rgb(rig.metadata)
                    finally:
                        rig.resume_delivery()
                    ending = None
                    camera_seen.clear()
                    holding = False
                    pause_ns = None
                    time_offset_ns = 0
                    resume_after_ns = 0
                    stop_immediately = False
                    channel.active.value = True
                    connection.send(("recording", str(path)))
                elif operation == "pause" and writer is not None and ending is None:
                    _, pause_requested = message
                elif operation == "resume" and writer is not None and holding and ending is None:
                    _, resume_ns = message
                    pause_episode = (pause_ns if pause_ns is not None else resume_ns) - time_offset_ns
                    last_values = list(writer.last_ns.values())
                    cut = max([pause_episode, *last_values]) if last_values else pause_episode
                    # Host time at resume maps to the pause cut, so the gap is removed.
                    time_offset_ns = resume_ns - cut
                    resume_after_ns = resume_ns
                    holding = False
                    pause_ns = None
                    rig.resume_delivery()
                    channel.active.value = True
                    connection.send(("resumed", None))
                elif operation == "stop" and writer is not None:
                    channel.active.value = False
                    _, end_ns, status, reason = message
                    end_episode = end_ns - time_offset_ns
                    if holding:
                        last_values = list(writer.last_ns.values())
                        pause_episode = (pause_ns if pause_ns is not None else end_ns) - time_offset_ns
                        end_episode = max([pause_episode, *last_values]) if last_values else pause_episode
                        ending = (end_episode, status, reason, time.monotonic())
                        stop_immediately = True
                        holding = False
                    else:
                        ending = (end_episode, status, reason, time.monotonic() + .15)
                        stop_immediately = False
                    pause_requested = None
                elif operation == "close":
                    running = False
            # Camera input has the smaller fixed queue and cannot be replayed.
            # Service it before numeric streams so a Zarr flush cannot leave
            # all four 30 Hz image streams waiting behind a large state batch.
            for camera, kind, image, record in rig.poll():
                if kind == "rgb":
                    latest_images[camera] = image
                mapped = episode_record(record)
                if mapped is None or writer is None:
                    continue
                camera_seen[mapped.stream] = mapped.time_ns
                if ending is None or mapped.time_ns <= ending[0]:
                    if kind == "rgb":
                        writer.write_rgb(camera, image, mapped)
                    else:
                        writer.append(replace(mapped, values={**mapped.values, "image": image}))
            for _ in range(128):
                try:
                    record_generation, record = channel.queue.get_nowait()
                except Empty:
                    break
                channel.consumed[(STATE_STREAMS + COMMAND_STREAMS).index(record.stream)] += 1
                mapped = episode_record(record)
                if writer is not None and mapped is not None and record_generation == generation and (
                        ending is None or mapped.time_ns <= ending[0]):
                    writer.append(mapped)
            if pause_requested is not None and writer is not None and ending is None:
                pause_ns = pause_requested
                pause_requested = None
                holding = True
                channel.active.value = False
                if viewer:
                    rig.preview_delivery()
                else:
                    rig.suspend_delivery()
                connection.send(("paused", pause_ns))
            if preview is not None:
                preview.update(latest_images)
            drained = (not any(channel.inflight) and list(channel.sent) == list(channel.consumed))
            past_end = stop_immediately or (ending is not None and all(
                camera_seen.get(s, 0) >= ending[0] for s in required_cameras))
            if ending is not None and time.monotonic() >= ending[3] and (
                    drained and past_end or time.monotonic() >= ending[3] + 2.):
                end_ns, status, reason, _ = ending
                missing = set(STATE_STREAMS + COMMAND_STREAMS + tuple(required_cameras)) - set(writer.counts)
                if not drained or not past_end or missing:
                    status = "failed"
                    reason = f"录制数据未完整收尾；缺少流：{sorted(missing)}"
                    channel.fail(reason)
                if channel.failed.is_set():
                    status, reason = "failed", reason or "采集通道失败"
                path = str(writer.path)
                # Final encoder and array flushes are allowed to block without
                # accumulating frames for the next, not-yet-started episode.
                rig.suspend_delivery()
                try:
                    status = writer.close(end_ns, status, reason)
                finally:
                    if viewer:
                        rig.preview_delivery()
                writer = None
                ending = None
                stop_immediately = False
                holding = False
                channel.active.value = False
                connection.send(("saved", (path, status)))
            time.sleep(.001)
    except BaseException as error:
        failure = f"录制进程失败：{error}"
        channel.fail(failure)
    finally:
        channel.active.value = False
        rig.close()
        if writer is not None:
            try:
                writer.close(time.monotonic_ns(), "failed", failure)
            except Exception:
                pass  # Disk failure leaves the manifest incomplete, never complete.
        if preview is not None:
            preview.close()
        connection.close()


class Recorder:
    """UI coordinator. begin/end are asynchronous during teleoperation."""

    def __init__(self, config, *, sdk_root=None, metadata=None, viewer=False):
        self.config, self.sdk_root = config, sdk_root
        self.metadata, self.viewer = metadata or {}, viewer
        self.context = mp.get_context("spawn")
        self.channel = CaptureChannel(self.context)
        self.sink = RecorderSink(self.channel, config.state_hz)
        self.session = Path(config.output_dir).resolve() / (
            datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8])
        self.process = self.connection = None
        self.ready = False
        self.state = "idle"
        self.error = None
        self.notices = []
        self._restart_required = False
        self._nvenc_checked = False

    @property
    def recording(self):
        return self.state in ("starting", "recording")

    @property
    def episode_open(self):
        return self.state in ("starting", "recording", "paused", "resuming")

    def status(self):
        process = self.process
        return {
            "state": self.state, "ready": self.ready, "error": self.error,
            "session": str(self.session), "viewer": self.viewer,
            "process_pid": process.pid if process is not None else None,
            "process_alive": process.is_alive() if process is not None else False,
            "channel_active": bool(self.channel.active.value),
            "channel_failed": self.channel.failed.is_set(),
            "generation": self.channel.generation.value,
            "sent": list(self.channel.sent), "consumed": list(self.channel.consumed),
            "inflight": list(self.channel.inflight),
            "numeric_occupancy": self.channel.occupancy(),
        }

    def _launch(self):
        self.channel.failed.clear()
        self.error = None
        # Keep cumulative counters and any committed old-generation records.
        # The replacement worker drains them but writes only its own generation.
        while True:
            try:
                self.channel.errors.get_nowait()
            except Empty:
                break
        parent, child = self.context.Pipe()
        self.connection = parent
        self.process = self.context.Process(target=_worker,
            args=(self.config, self.sdk_root, self.metadata, self.channel, child, self.viewer),
            name="demonstration-recorder")
        self.process.start()
        child.close()
        self.ready = False
        self.state = "idle"

    def start(self):
        """Camera preflight before teleoperation is engaged."""
        from .spool import preflight_nvenc
        preflight_nvenc()
        self._nvenc_checked = True
        self._launch()
        deadline = time.monotonic() + 15.
        while not self.ready:
            self.poll()
            if self.error:
                raise RuntimeError(self.error)
            if time.monotonic() >= deadline:
                raise RuntimeError("采集相机启动超时")
            time.sleep(.01)

    def begin(self):
        if self.state != "idle":
            raise RuntimeError("请等待当前录制保存完成")
        if self.error:
            raise ValueError("请暂停遥操作后恢复采集进程")
        if not self.ready:
            raise ValueError("采集相机尚未就绪")
        now = time.monotonic_ns()
        for stream, stamp in zip(STATE_STREAMS + COMMAND_STREAMS, self.channel.latest_ns):
            if stamp == 0 or now - stamp > 100_000_000:
                raise ValueError(f"等待新鲜的采集数据：{stream}")
        self.channel.generation.value += 1
        generation = self.channel.generation.value
        path = self.session / f"episode_{generation - 1:06d}"
        self.connection.send(("start", generation, str(path), now))
        self.state = "starting"

    def recover(self):
        """Called only while the UI has paused motion; joining cannot stall control."""
        self.channel.active.value = False
        self._stop_process()
        if self._restart_required:
            self.ready = False
            self.state = "idle"
            self.error = "采集进程被强制终止，通信队列可能损坏；请退出并重新启动遥操作"
            return
        from .spool import preflight_nvenc
        preflight_nvenc()
        self._nvenc_checked = True
        self._launch()

    def pause(self):
        """Keep the episode open and stop ingesting until resume."""
        if self.state not in ("starting", "recording", "resuming"):
            return
        now = time.monotonic_ns()
        self.channel.active.value = False
        try:
            self.connection.send(("pause", now))
        except (OSError, EOFError):
            self.state = "idle"
            self.error = "录制进程暂停通道已断开"
            self.channel.fail(self.error)
            return
        self.state = "paused"
        self.notices.append("录制已暂停，重新接合后继续本条。")

    def resume(self):
        """Continue the paused episode. Timestamps continue from the pause."""
        if self.state != "paused":
            raise RuntimeError("当前没有暂停的录制")
        if self.error:
            raise ValueError("请先脱离遥操作，再恢复采集进程")
        if not self.ready:
            raise ValueError("采集相机尚未就绪")
        now = time.monotonic_ns()
        for stream, stamp in zip(STATE_STREAMS + COMMAND_STREAMS, self.channel.latest_ns):
            if stamp == 0 or now - stamp > 100_000_000:
                raise ValueError(f"等待新鲜的采集数据：{stream}")
        try:
            self.connection.send(("resume", now))
        except (OSError, EOFError):
            self.state = "idle"
            self.error = "录制进程继续通道已断开"
            self.channel.fail(self.error)
            return
        self.state = "resuming"

    def end(self, *, status="complete", reason=None):
        if not self.episode_open:
            return
        self.channel.active.value = False
        if self.process is not None and self.process.is_alive():
            try:
                self.connection.send(("stop", time.monotonic_ns(), status, reason))
                self.state = "saving"
            except (OSError, EOFError):
                self.state = "idle"
                self.error = "录制进程停止通道已断开"
                self.channel.fail(self.error)
        else:
            self.state = "idle"

    def poll(self):
        if self.connection is not None:
            try:
                while self.connection.poll():
                    kind, value = self.connection.recv()
                    if kind == "ready":
                        self.ready = True
                        self.notices.append("三路相机采集已就绪。")
                    elif kind == "recording":
                        if self.state == "starting":
                            self.state = "recording"
                        self.notices.append(f"正在录制：{value}")
                    elif kind == "resumed":
                        if self.state == "resuming":
                            self.state = "recording"
                        self.notices.append("继续录制本条。")
                    elif kind == "saved":
                        self.state = "idle"
                        path, status = value
                        label = {"captured": "已采集，等待离线整理",
                                 "complete": "已保存", "discarded": "已作废"}.get(status, "不完整")
                        self.notices.append(f"录制{label}：{path}")
            except (EOFError, OSError):
                pass
        while True:
            try:
                self.error = self.channel.errors.get_nowait()
            except Empty:
                break
        if self.channel.failed.is_set():
            self.error = self.error or "采集通道失败"
        if self.process is not None and not self.process.is_alive():
            self.ready = False
            self.state = "idle"
            self.error = self.error or "录制进程已退出"
        if self.state == "recording" and not self.error:
            now = time.monotonic_ns()
            for stream, stamp in zip(STATE_STREAMS + COMMAND_STREAMS, self.channel.latest_ns):
                if now - stamp > 500_000_000:
                    self.channel.fail(f"采集数据断流：{stream}")
                    self.error = f"采集数据断流：{stream}"
                    break
        return self.error

    def _stop_process(self):
        if self.process is None:
            return
        if self.process.is_alive():
            try:
                self.connection.send(("close",))
            except (OSError, EOFError):
                pass
            self.process.join(3.)
        if self.process.is_alive():
            self._restart_required = True
            self.process.terminate()
            self.process.join(2.)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(1.)
        if self.process.exitcode not in (None, 0):
            self._restart_required = True
        self.connection.close()
        self.process = self.connection = None

    def close(self):
        self.end(status="failed", reason="退出时录制尚未结束")
        # Encoder and depth workers close concurrently, but MP4 trailer and
        # fsync completion can legitimately exceed the old three-second limit.
        deadline = time.monotonic() + 20.
        while self.state == "saving" and time.monotonic() < deadline:
            self.poll()
            time.sleep(.01)
        self._stop_process()
        self.channel.errors.close()
        self.channel.errors.cancel_join_thread()
