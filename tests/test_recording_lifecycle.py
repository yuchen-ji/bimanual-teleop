"""Recorder acknowledgements, shutdown and UI ordering without device connections."""

import json
import multiprocessing as mp
from pathlib import Path
from queue import Empty, Queue
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import zarr

from bimanual_teleop.recording.config import RecordingConfig
from bimanual_teleop.recording.convert import _read_stream
from bimanual_teleop.recording.recorder import Recorder, _lower_priority, _worker
from bimanual_teleop.recording.sink import COMMAND_STREAMS, STATE_STREAMS, Record
from bimanual_teleop.recording.storage import EpisodeWriter
from bimanual_teleop.recording.ui import RecordingUI
from bimanual_teleop.system import SystemState
from tests.test_recording_sink import _channel
from tests.test_recording_storage import _Kinematics


class _Rig:
    def __init__(self):
        self.metadata, self.closed = {}, threading.Event()
        self.frames = Queue()
        self.delivery_events = []

    def start(self):
        pass

    def poll(self):
        frames = []
        while True:
            try:
                frames.append(self.frames.get_nowait())
            except Empty:
                return frames

    def frame_set(self, stamp):
        for index in range(3):
            camera = f"camera_{index}"
            record = Record(f"cameras/{camera}/rgb", stamp, stamp, {"source_time_ms": stamp / 1e6})
            self.frames.put((camera, "rgb", np.zeros((480, 640, 3), dtype="u1"), record))

    def suspend_delivery(self):
        self.delivery_events.append("suspend")

    def resume_delivery(self):
        self.delivery_events.append("resume")

    def preview_delivery(self):
        self.delivery_events.append("preview")

    def close(self):
        self.closed.set()


class _DelayedValue:
    """Hold the real Queue feeder after put_nowait has already returned."""

    def __init__(self, entered, release):
        self.entered, self.release = entered, release

    def __reduce__(self):
        self.entered.set()
        if not self.release.wait(5.):
            raise RuntimeError("test feeder was not released")
        return float, (.1,)


class RecordingLifecycleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "episode"
        self.config = RecordingConfig(("main", "left", "right"), main_depth=False)

    def worker(self, *, delay=0., failure=False, channel=None):
        channel = _channel(512) if channel is None else channel
        rig, parent_alive = _Rig(), threading.Event()
        parent_alive.set()
        channel.active.value = False
        connection, child = mp.Pipe()
        seen = Queue()
        def factory(*args):
            writer = EpisodeWriter(*args)
            append = writer.append
            def observed(record):
                if failure:
                    raise OSError("simulated disk failure")
                if delay:
                    time.sleep(delay)
                append(record)
                seen.put(record)
            writer.append = observed
            return writer
        kine = _Kinematics()
        kine.model = SimpleNamespace(digest="test-model")
        for patcher in (patch("bimanual_teleop.recording.camera.CameraRig", return_value=rig),
                        patch("bimanual_teleop.devices.tianji.model.TianjiKinematics", return_value=kine),
                        patch("bimanual_teleop.recording.storage.EpisodeWriter", side_effect=factory),
                        patch("bimanual_teleop.recording.recorder.mp.parent_process",
                              return_value=SimpleNamespace(is_alive=parent_alive.is_set))):
            patcher.start()
            self.addCleanup(patcher.stop)
        thread = threading.Thread(target=_worker,
            args=(self.config, None, {}, channel, child, False, 0), daemon=True)
        thread.start()
        def cleanup():
            if thread.is_alive():
                parent_alive.clear()
                thread.join(2.)
            connection.close()
            self.assertFalse(thread.is_alive(), "worker cleanup exceeded two seconds")
        self.addCleanup(cleanup)
        self.assertEqual(self.receive(connection), ("ready", None))
        connection.send(("start", 7, str(self.path), 1000))
        self.assertEqual(self.receive(connection), ("recording", str(self.path)))
        self.assertEqual(rig.delivery_events, ["suspend", "suspend", "resume"])
        return channel, connection, thread, parent_alive, rig, seen

    def receive(self, connection):
        self.assertTrue(connection.poll(3.), "recorder acknowledgement timed out")
        return connection.recv()

    def test_recording_worker_priority_is_best_effort(self):
        with patch("bimanual_teleop.recording.recorder.os.nice") as nice:
            _lower_priority(5)
            nice.assert_called_once_with(5)
        with patch("bimanual_teleop.recording.recorder.os.nice",
                   side_effect=OSError("unsupported")):
            _lower_priority(5)

    def record(self, stamp, sequence=1, stream="hands/left"):
        arm = stream.startswith(("arms/", "arm_commands/"))
        values = {"joint_pos": (.1,) * (7 if arm else 20)}
        if stream.startswith("arms/"):
            values["wrench"] = (0.,) * 6
        if stream.startswith("arm_commands/"):
            values["eef_pose"] = (0.,) * 6 + (1.,)
        return Record(stream, stamp, sequence, values)

    def enqueue(self, channel, record, generation=7):
        channel.sent[(STATE_STREAMS + COMMAND_STREAMS).index(record.stream)] += 1
        channel.queue.put((generation, record))

    def seed(self, channel, rig, exclude=()):
        for stream in STATE_STREAMS + COMMAND_STREAMS:
            if stream not in exclude:
                self.enqueue(channel, self.record(1001, 0, stream))
        rig.frame_set(1002)

    def test_worker_save_ack_follows_flush_and_effective_end_excludes_late_data(self):
        channel, connection, thread, _, rig, seen = self.worker()
        self.seed(channel, rig)
        self.enqueue(channel, self.record(1001), generation=6)  # Ignore earlier episode.
        self.enqueue(channel, self.record(1020))  # Already written before stop arrives.
        while seen.get(timeout=2.).time_ns != 1020:
            pass
        connection.send(("stop", 1010, "complete", None))
        self.enqueue(channel, self.record(1030, 2))
        self.enqueue(channel, self.record(1005, 1, "hands/right"))
        rig.frame_set(3000)
        self.assertEqual(self.receive(connection), ("saved", (str(self.path), "complete")))
        document = json.loads((self.path / "episode.json").read_text())
        self.assertEqual((document["status"], document["end_ns"]), ("complete", 1010))
        raw = zarr.open_group(str(self.path / "raw.zarr"), mode="r")
        self.assertEqual(raw["hands/right/time_ns"][:].tolist(), [1001, 1005])
        # Raw may retain a previously written tail; conversion must honor the manifest window.
        selected = _read_stream(raw, "hands/left", {"joint_pos": 20}, 1000, 1010)
        self.assertEqual(selected.times.tolist(), [1001])
        self.assertNotIn(1030, raw["hands/left/time_ns"][:])
        connection.send(("close",))
        thread.join(2.)
        self.assertFalse(thread.is_alive())
        self.assertTrue(rig.closed.is_set())

    def test_slow_writer_cannot_ack_complete_while_valid_tail_records_are_lost(self):
        channel, connection, _, _, rig, _ = self.worker(delay=.002)
        self.seed(channel, rig, exclude=("hands/left",))
        connection.send(("stop", 2000, "complete", None))
        for index in range(260):
            self.enqueue(channel, self.record(1001 + index, index))
        rig.frame_set(3000)
        self.assertEqual(self.receive(connection), ("saved", (str(self.path), "complete")))
        document = json.loads((self.path / "episode.json").read_text())
        self.assertEqual(document["status"], "complete")
        raw = zarr.open_group(str(self.path / "raw.zarr"), mode="r")
        self.assertEqual(raw["hands/left/sequence"][:].tolist(), list(range(260)))

    def test_parent_death_and_writer_failure_close_resources_without_complete_episode(self):
        for failure in (False, True):
            with self.subTest(writer_failure=failure):
                self.path = self.path.parent / f"failure_{failure}"
                channel, _, thread, parent_alive, rig, _ = self.worker(failure=failure)
                if failure:
                    self.enqueue(channel, self.record(1001))
                else:
                    parent_alive.clear()
                thread.join(2.)
                self.assertFalse(thread.is_alive())
                self.assertTrue(channel.failed.is_set())
                self.assertFalse(channel.active.value)
                self.assertTrue(rig.closed.is_set())
                document = json.loads((self.path / "episode.json").read_text())
                self.assertEqual(document["status"], "failed")
                self.assertIn("simulated disk failure" if failure else "遥操作主进程已退出",
                              document["reason"])

    def test_missing_camera_tail_is_bounded_and_marks_episode_failed(self):
        channel, connection, _, _, rig, _ = self.worker()
        self.seed(channel, rig)
        connection.send(("stop", 2000, "complete", None))
        self.assertEqual(self.receive(connection), ("saved", (str(self.path), "failed")))
        document = json.loads((self.path / "episode.json").read_text())
        self.assertEqual(document["status"], "failed")
        self.assertTrue(channel.failed.is_set())
        self.assertFalse(channel.active.value)

    def coordinator(self):
        recorder = Recorder.__new__(Recorder)
        recorder.channel, recorder.connection, recorder.process = _channel(), Mock(), Mock()
        recorder.process.is_alive.return_value = True
        recorder.process.exitcode = 0
        recorder.state, recorder.error, recorder.ready = "recording", None, True
        recorder.notices, recorder.session = [], self.path
        recorder._restart_required = False
        return recorder

    def test_restart_keeps_counts_until_old_feeder_data_arrives_and_new_episode_saves(self):
        recorder = Recorder(self.config)
        channel = recorder.channel
        entered, release = threading.Event(), threading.Event()
        def cleanup_channels():
            release.set()
            for queue in (channel.queue, channel.errors):
                queue.close()
                queue.join_thread()
        self.addCleanup(cleanup_channels)
        old = Record("hands/left", 1001, 99, {"joint_pos": (_DelayedValue(entered, release),) * 20})
        index = (STATE_STREAMS + COMMAND_STREAMS).index(old.stream)
        channel.sent[index], channel.consumed[index] = 4, 3
        channel.queue.put_nowait((6, old))
        self.assertTrue(entered.wait(2.), "Queue feeder did not reach its delay")
        with patch.object(recorder.context, "Process", return_value=Mock()):
            recorder._launch()  # A replacement worker starts before the old feeder sends.
        recorder.connection.close()
        self.assertEqual((channel.sent[index], channel.consumed[index]), (4, 3))
        channel.generation.value = 7
        channel, connection, _, _, rig, _ = self.worker(channel=channel)
        self.seed(channel, rig)
        connection.send(("stop", 2000, "complete", None))
        rig.frame_set(3000)
        try:
            self.assertFalse(connection.poll(.2), "saved before the feeder delivered its tail")
        finally:
            release.set()
        self.assertEqual(self.receive(connection), ("saved", (str(self.path), "complete")))
        self.assertEqual(list(channel.sent), list(channel.consumed))
        self.assertEqual(channel.consumed[index], 5)
        raw = zarr.open_group(str(self.path / "raw.zarr"), mode="r")
        self.assertEqual(raw["hands/left/sequence"][:].tolist(), [0])
        self.assertEqual(json.loads((self.path / "episode.json").read_text())["status"], "complete")

    def test_late_recording_ack_does_not_cancel_saving_and_end_waits_for_saved_ack(self):
        recorder = self.coordinator()
        parent, child = mp.Pipe()
        self.addCleanup(parent.close)
        self.addCleanup(child.close)
        recorder.connection, recorder.state = parent, "idle"
        recorder.channel.latest_ns[:] = [time.monotonic_ns()] * 8
        recorder.begin()
        self.assertEqual(recorder.state, "starting")
        self.assertEqual(self.receive(child)[0], "start")
        recorder.end()
        self.assertEqual(self.receive(child)[0], "stop")
        child.send(("recording", str(self.path)))
        recorder.poll()
        self.assertEqual(recorder.state, "saving")
        child.send(("saved", (str(self.path), "complete")))
        recorder.poll()
        self.assertEqual(recorder.state, "idle")

    def test_manual_pause_saves_fault_pause_fails_and_recovery_pauses_before_join(self):
        for mode in ("keyboard", "gesture", "fault", "recover", "broken_pipe"):
            with self.subTest(mode=mode):
                recorder, runtime = self.coordinator(), SimpleNamespace(state=SystemState.ENGAGED, last_error=None)
                runtime.pause = lambda reason: setattr(runtime, "state", SystemState.PAUSED)
                ui = RecordingUI(runtime, None, recorder=recorder, emit=lambda _: None)
                recorder.recover = Mock(side_effect=lambda: self.assertEqual(runtime.state, SystemState.PAUSED))
                if mode == "keyboard":
                    ui.handle(" ")
                elif mode == "gesture":
                    ui.handle_gesture("pause")
                elif mode == "fault":
                    runtime.state, runtime.last_error = SystemState.PAUSED, "force feedback lost"
                    ui.report_runtime_pause()
                elif mode == "broken_pipe":
                    recorder.connection.send.side_effect = BrokenPipeError("writer exited")
                    ui.abort("writer exited")
                    self.assertEqual(runtime.state, SystemState.PAUSED)
                    self.assertTrue(recorder.channel.failed.is_set())
                else:
                    recorder.error = "writer unavailable"
                    ui.handle("c")
                    recorder.recover.assert_called_once()
                sent, = recorder.connection.send.call_args_list
                self.assertEqual(sent.args[0][2], "complete" if mode in ("keyboard", "gesture") else "failed")

    def test_stuck_process_shutdown_has_bounded_join_escalation(self):
        recorder = self.coordinator()
        process = recorder.process
        recorder._stop_process()
        self.assertEqual([args.args[0] for args in process.join.call_args_list], [3., 2., 1.])
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        self.assertIsNone(recorder.process)
        self.assertIsNone(recorder.connection)
        self.assertTrue(recorder._restart_required)

    def test_recording_failure_explains_recovery_once(self):
        recorder = self.coordinator()
        recorder.poll = Mock(return_value="Camera recording queue is full")
        runtime = SimpleNamespace(state=SystemState.ENGAGED, last_error=None)
        runtime.pause = lambda reason: setattr(runtime, "state", SystemState.PAUSED)
        messages = []
        ui = RecordingUI(runtime, None, recorder=recorder, emit=messages.append)
        ui.poll_operation()
        ui.poll_operation()
        self.assertEqual(sum("按 C 恢复采集" in message for message in messages), 1)
        self.assertEqual(runtime.state, SystemState.PAUSED)

    def test_forced_or_abnormal_exit_refuses_channel_reuse_and_leaves_motion_paused(self):
        for abnormal_exit in (False, True):
            with self.subTest(abnormal_exit=abnormal_exit):
                recorder = self.coordinator()
                recorder.error = "writer unavailable"
                if abnormal_exit:
                    recorder.process.is_alive.return_value = False
                    recorder.process.exitcode = -9
                recorder._launch = Mock()
                channel = recorder.channel
                runtime = SimpleNamespace(state=SystemState.ENGAGED, last_error=None)
                runtime.pause = lambda reason: setattr(runtime, "state", SystemState.PAUSED)
                messages = []
                ui = RecordingUI(runtime, None, recorder=recorder, emit=messages.append)
                ui.handle("c")
                self.assertEqual(runtime.state, SystemState.PAUSED)
                self.assertFalse(channel.active.value)
                self.assertTrue(recorder._restart_required)
                self.assertIn("重新启动遥操作", recorder.error)
                self.assertFalse(recorder.ready)
                self.assertIsNone(recorder.process)
                self.assertIn(recorder.error, messages)
                recorder._launch.assert_not_called()
                recorder.recover()  # Repeated recovery cannot accidentally reuse the channel.
                recorder._launch.assert_not_called()
