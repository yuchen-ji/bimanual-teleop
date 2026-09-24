"""Recorder acknowledgements, shutdown and UI ordering without device connections."""

import json
import multiprocessing as mp
import os
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

from bimanual_teleop.recording.config import RecordingConfig, load_config
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
        def factory(*args, **kwargs):
            del kwargs
            writer = EpisodeWriter(*args)
            append = writer.append
            close = writer.close
            def observed(record):
                if failure:
                    raise OSError("simulated disk failure")
                if delay:
                    time.sleep(delay)
                append(record)
                seen.put(record)
            writer.append = observed
            def close_with_status(end_ns, status="complete", reason=None):
                close(end_ns, status, reason)
                return status
            writer.close = close_with_status
            return writer
        kine = _Kinematics()
        kine.model = SimpleNamespace(digest="test-model")
        for patcher in (patch("bimanual_teleop.recording.camera.CameraRig", return_value=rig),
                        patch("bimanual_teleop.devices.tianji.model.TianjiKinematics", return_value=kine),
                        patch("bimanual_teleop.recording.spool.RawEpisodeWriter", side_effect=factory),
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

    def test_worker_does_not_emit_periodic_diagnostics(self):
        _channel_value, connection, _thread, _parent_alive, _rig, _seen = self.worker()
        self.assertFalse(connection.poll(.05))

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

    def test_pause_resume_continues_episode_time_from_the_pause(self):
        channel, connection, _, _, rig, seen = self.worker()
        self.seed(channel, rig)
        self.enqueue(channel, self.record(2_000, 1))
        while seen.get(timeout=2.).time_ns != 2_000:
            pass
        connection.send(("pause", 2_500))
        self.assertEqual(self.receive(connection)[0], "paused")
        self.enqueue(channel, self.record(8_000, 2))
        connection.send(("resume", 9_000))
        self.assertEqual(self.receive(connection)[0], "resumed")
        self.enqueue(channel, self.record(9_100, 3))
        mapped = None
        while mapped is None:
            item = seen.get(timeout=2.)
            if item.sequence == 3:
                mapped = item
            self.assertNotEqual(item.time_ns, 8_000)
        self.assertEqual(mapped.time_ns, 2_600)

    def test_engagement_starts_after_delay_and_reengagement_resumes(self):
        recorder = self.coordinator()
        recorder.poll = Mock(return_value=None)
        recorder.notices = []
        recorder.config = RecordingConfig(("a", "b", "c"), start_delay_s=5)
        recorder.state = "idle"
        recorder.begin = Mock(side_effect=lambda: setattr(recorder, "state", "starting"))
        recorder.pause = Mock(side_effect=lambda: setattr(recorder, "state", "paused"))
        recorder.resume = Mock(side_effect=lambda: setattr(recorder, "state", "resuming"))
        runtime = SimpleNamespace(state=SystemState.ENGAGED, last_error=None)
        clock = {"now": 0.}
        with patch("bimanual_teleop.recording.ui.time.monotonic", side_effect=lambda: clock["now"]):
            ui = RecordingUI(runtime, None, recorder=recorder, emit=lambda _: None)
            ui.poll_operation()
            recorder.begin.assert_not_called()
            clock["now"] = 5
            ui.poll_operation()
            recorder.begin.assert_called_once()
            runtime.state = SystemState.PAUSED
            ui.poll_operation()
            recorder.pause.assert_called_once()
            runtime.state = SystemState.ENGAGED
            ui.poll_operation()
            recorder.resume.assert_not_called()
            clock["now"] = 10
            ui.poll_operation()
            recorder.resume.assert_called_once()
        recorder.begin.assert_called_once()

    def test_recording_config_reads_keys_and_rejects_duplicates(self):
        config = load_config()
        self.assertEqual((config.save_key, config.discard_key, config.quit_key, config.recover_key),
                         ("s", "x", "q", "c"))
        self.assertEqual(config.start_delay_s, 0)
        self.assertEqual(config.frame_capacity, 256)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.yaml"
            path.write_text("cameras: [a, b, c]\ncontrols: {save_key: s, discard_key: s}\n",
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "distinct"):
                load_config(path)
            path.write_text("cameras: [a, b, c]\nframe_capacity: 8\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frame_capacity"):
                load_config(path)

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

    def test_restart_drains_old_generation_from_shared_ring_before_new_episode_saves(self):
        recorder = Recorder(self.config)
        channel = recorder.channel
        def cleanup_channels():
            for queue in (channel.queue, channel.errors):
                queue.close()
                queue.join_thread()
        self.addCleanup(cleanup_channels)
        old = Record("hands/left", 1001, 99, {"joint_pos": (.1,) * 20})
        index = (STATE_STREAMS + COMMAND_STREAMS).index(old.stream)
        channel.sent[index], channel.consumed[index] = 4, 3
        channel.queue.put_nowait((6, old))
        with patch.object(recorder.context, "Process", return_value=Mock()):
            recorder._launch()
        recorder.connection.close()
        self.assertEqual((channel.sent[index], channel.consumed[index]), (4, 3))
        channel.generation.value = 7
        channel, connection, _, _, rig, _ = self.worker(channel=channel)
        self.seed(channel, rig)
        connection.send(("stop", 2000, "complete", None))
        rig.frame_set(3000)
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

    def test_disengagement_pauses_the_episode_and_recovery_requires_disengagement(self):
        for mode in ("keyboard", "gesture", "fault", "recover", "broken_pipe"):
            with self.subTest(mode=mode):
                recorder, runtime = self.coordinator(), SimpleNamespace(state=SystemState.ENGAGED, last_error=None)
                runtime.pause = lambda reason: setattr(runtime, "state", SystemState.PAUSED)
                ui = RecordingUI(runtime, None, recorder=recorder, emit=lambda _: None)
                recorder.recover = Mock(side_effect=lambda: self.assertNotEqual(runtime.state, SystemState.ENGAGED))
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
                    runtime.state = SystemState.READY
                    recorder.error = "writer unavailable"
                    ui.handle("c")
                    recorder.recover.assert_called_once()
                    self.assertEqual(runtime.state, SystemState.READY)
                    continue
                self.assertEqual(recorder.connection.send.call_args.args[0][0], "pause")
                self.assertFalse(any(
                    call.args and call.args[0][0] == "stop"
                    for call in recorder.connection.send.call_args_list))

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

    def test_recording_failure_explains_recovery_once_without_pausing_motion(self):
        recorder = self.coordinator()
        recorder.poll = Mock(return_value="Camera recording queue is full")
        runtime = SimpleNamespace(state=SystemState.ENGAGED, last_error=None)
        runtime.pause = lambda reason: setattr(runtime, "state", SystemState.PAUSED)
        messages = []
        recorder.end = Mock(wraps=recorder.end)
        ui = RecordingUI(runtime, None, recorder=recorder, emit=messages.append)
        ui.poll_operation()
        ui.poll_operation()
        self.assertEqual(sum("按 C 恢复采集" in message for message in messages), 1)
        self.assertEqual(runtime.state, SystemState.ENGAGED)
        recorder.end.assert_called_once_with(status="failed", reason="Camera recording queue is full")

    def test_forced_or_abnormal_exit_refuses_channel_reuse_after_operator_disengages(self):
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
                process_before = recorder.process
                ui.handle("c")
                self.assertEqual(runtime.state, SystemState.ENGAGED)
                self.assertIs(recorder.process, process_before)
                runtime.state = SystemState.READY
                ui.handle("c")
                self.assertFalse(channel.active.value)
                self.assertTrue(recorder._restart_required)
                self.assertIn("重新启动遥操作", recorder.error)
                self.assertFalse(recorder.ready)
                self.assertIsNone(recorder.process)
                self.assertIn(recorder.error, messages)
                recorder._launch.assert_not_called()
                recorder.recover()  # Repeated recovery cannot accidentally reuse the channel.
                recorder._launch.assert_not_called()
