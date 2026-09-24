"""Process lifecycle tests with fake devices; no SDK imports or transport."""

from dataclasses import asdict
import multiprocessing as mp
import os
import signal
import threading
import time
import unittest
from unittest.mock import Mock, patch

from bimanual_teleop.control.hand.process import WujiProcess, _Snapshot, _create_runtime
from bimanual_teleop.system import SystemState
from bimanual_teleop.types import Health, Sample, SampleHeader, SampleRef


class _Device:
    def __init__(self, side, *, fixed=False, fixed_diagnostics=False):
        self.side, self.fixed, self.fixed_diagnostics = side, fixed, fixed_diagnostics
        self.received_ns = time.monotonic_ns()

    def get_latest(self):
        received = self.received_ns if self.fixed else time.monotonic_ns()
        return Sample(SampleHeader(SampleRef(self.side, "test", 7), received, True,
                                   source_sequence=123), (1., 2., 3.))

    def get_latest_stream(self, stream):
        received = self.received_ns if self.fixed_diagnostics else time.monotonic_ns()
        return Sample(SampleHeader(SampleRef(stream, "test", 7), received, True), ())


class _Runtime:
    def __init__(self, config, *, verbose=False):
        self.config = config
        self.verbose = verbose
        self.sides = ("left", "right")
        self.glove_timeout_ns = round(config.get("glove_timeout_s", .25)*1e9)
        self.hand_timeout_ns = round(config.get("hand_timeout_s", .5)*1e9)
        self.gloves = {side: _Device(side, fixed=config.get("fixed", False)) for side in self.sides}
        self.hands = {side: _Device(side, fixed_diagnostics=config.get("fixed_diagnostics", False))
                      for side in self.sides}
        self.state, self.last_error = SystemState.DISCONNECTED, None
        self.mode, self.threaded = None, True
        self.worker_completed = time.monotonic_ns()
        self.generation, self.prepared = 0, None

    def _event(self, name):
        if "events" in self.config:
            self.config["events"].put((name, os.getpid()))

    def start(self):
        self._event("started")
        if self.config.get("fail_start"):
            raise RuntimeError("fake startup failure")
        self.state = SystemState.READY

    def prepare_engage(self, *, cancelled=None):
        if "release_prepare_entry" in self.config:
            self._event("prepare_before_entry")
            self.config["release_prepare_entry"].wait(3.)
        if cancelled is not None and cancelled():
            raise RuntimeError("fake preparation cancelled before entry")
        generation = self.generation
        self._event("prepare_entered")
        if "release_prepare" in self.config:
            self.config["release_prepare"].wait(3.)
        if "prepare_delay" in self.config:
            time.sleep(self.config["prepare_delay"])
        if generation != self.generation:
            raise RuntimeError("fake preparation cancelled")
        self.prepared = generation
        self.mode = "hold"
        self._event("prepared")

    def begin_follow(self, *, cancelled=None):
        if self.prepared is None or self.prepared != self.generation:
            raise RuntimeError("prepare before following")
        if "release_follow" in self.config:
            self._event("follow_before_commit")
            self.config["release_follow"].wait(3.)
        if cancelled is not None and cancelled():
            raise RuntimeError("fake following cancelled")
        self.state, self.last_error = SystemState.ENGAGED, None
        self.mode = "follow"
        self._event("following")

    def pause(self, reason):
        self.generation += 1
        self.prepared = None
        self.state, self.last_error = SystemState.PAUSED, reason
        self.mode = "hold"
        self._event("paused")

    def glove_samples(self):
        return {side: glove.get_latest() for side, glove in self.gloves.items()}

    def status(self, *, include_target=True):
        return {"state": self.state.value, "last_error": self.last_error,
                "mode": self.mode,
                "worker_completed_monotonic_ns": (self.worker_completed
                    if self.config.get("fixed_worker") else time.monotonic_ns()),
                "health": asdict(Health(True, time.monotonic_ns(), "fake ready")),
                "pid": os.getpid(), "verbose": self.verbose, "hands": {
                    side: {"last_target": [1.]*20 if include_target else None} for side in self.sides}}

    def close(self):
        self.state = SystemState.CLOSED
        self._event("closed")


def _abandon_parent(events):
    runtime = WujiProcess({"events": events}, _runtime_factory=_Runtime)
    runtime.start()
    runtime.prepare_engage()
    runtime.begin_follow()
    events.put(("abandoning", runtime._process.pid))
    # Flush the test's report before simulating abrupt coordinator loss.
    events.close()
    events.join_thread()
    os._exit(0)


class WujiProcessTests(unittest.TestCase):
    def setUp(self):
        self.context = mp.get_context("spawn")
        self.runtimes = []

    def tearDown(self):
        for runtime in self.runtimes:
            runtime.close()

    def runtime(self, config=None, *, verbose=False):
        value = WujiProcess(config or {}, verbose=verbose,
                            _runtime_factory=_Runtime)
        self.runtimes.append(value)
        return value

    def test_output_mode_reaches_spawned_runtime(self):
        for verbose in (False, True):
            with self.subTest(verbose=verbose):
                runtime = self.runtime(verbose=verbose)
                runtime.start()
                self.assertEqual(runtime.status()["verbose"], verbose)
                runtime.close()

    def test_real_factory_configures_sdk_logging_before_device_creation(self):
        for verbose in (False, True):
            with self.subTest(verbose=verbose), \
                    patch("bimanual_teleop.common.console.configure_runtime_logging") as configure, \
                    patch("bimanual_teleop.control.hand.follow.create_wuji_teleop") as create:
                order = Mock()
                order.attach_mock(configure, "configure")
                order.attach_mock(create, "create")
                self.assertIs(_create_runtime({}, verbose=verbose),
                              create.return_value)
                configure.assert_called_once_with(wuji=True, verbose=verbose)
                self.assertEqual([call[0] for call in order.mock_calls], ["configure", "create"])

    def event(self, events, wanted, timeout=3.):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            name, pid = events.get(timeout=max(.001, deadline-time.monotonic()))
            if name == wanted:
                return pid
        self.fail(f"Missing event: {wanted}")

    def test_complete_runtime_is_created_in_child_and_follow_ack_is_current(self):
        runtime = self.runtime()
        runtime.start()
        self.assertNotEqual(runtime.status()["pid"], os.getpid())
        self.assertEqual(runtime.state, SystemState.READY)
        self.assertTrue(runtime.health().ready)
        runtime.prepare_engage()
        runtime.begin_follow()
        self.assertEqual(runtime.state, SystemState.ENGAGED)
        self.assertTrue(runtime.health().ready)
        status = runtime.status(include_target=False)
        self.assertIsNone(status["hands"]["left"]["last_target"])
        self.assertNotIn("process", status)
        self.assertIn("send_drops", status["observation_transport"])
        self.assertIn("sequence_gaps", status["observation_transport_parent"])

    def test_snapshot_keeps_original_source_identity_and_time(self):
        runtime = self.runtime({"fixed": True})
        runtime.start()
        first = runtime.glove_samples()["left"]
        time.sleep(.04)
        latest = runtime.glove_samples()["left"]
        self.assertEqual(first.header, latest.header)
        self.assertEqual(latest.header.source_sequence, 123)
        self.assertEqual(latest.payload, (1., 2., 3.))

    def test_missing_glove_samples_have_fixed_side_keys(self):
        self.assertEqual(self.runtime().glove_samples(), {"left": None, "right": None})

    def test_synchronous_preparation_can_outlast_hand_timeout(self):
        runtime = self.runtime({"prepare_delay": .65})
        runtime.start()
        runtime.prepare_engage()
        runtime.begin_follow()
        self.assertEqual(runtime.state, SystemState.ENGAGED)
        self.assertIsNone(runtime.last_error)

    def test_pause_during_preparation_is_nonblocking_and_cancels_follow(self):
        events = self.context.Queue()
        release = self.context.Event()
        runtime = self.runtime({"events": events, "release_prepare": release})
        runtime.start()
        errors = []

        def prepare():
            try:
                runtime.prepare_engage()
            except RuntimeError as error:
                errors.append(str(error))

        thread = threading.Thread(target=prepare)
        thread.start()
        self.event(events, "prepare_entered")
        before = time.monotonic()
        runtime.pause("Space")
        self.assertLess(time.monotonic()-before, .1)
        self.assertEqual(runtime.state, SystemState.PAUSED)
        self.event(events, "paused")
        release.set()
        thread.join(2.)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, ["fake preparation cancelled"])
        with self.assertRaisesRegex(RuntimeError, "prepare before"):
            runtime.begin_follow()
        self.assertEqual(runtime.state, SystemState.PAUSED)
        events.close()

    def test_old_snapshot_cannot_undo_local_pause(self):
        runtime = self.runtime()
        now = time.monotonic_ns()
        status = {"state": "engaged", "last_error": None,
                  "health": asdict(Health(True, now))}
        runtime._accept(_Snapshot(1, 0, now, now+1_000_000_000, status, {}))
        runtime.pause("Space")
        runtime._accept(_Snapshot(2, 0, now, now+1_000_000_000, status, {}))
        self.assertEqual(runtime.state, SystemState.PAUSED)
        self.assertEqual(runtime.last_error, "Space")

    def test_pause_before_prepare_generation_is_captured_cancels_operation(self):
        events = self.context.Queue()
        release = self.context.Event()
        runtime = self.runtime({"events": events, "release_prepare_entry": release})
        runtime.start()
        errors = []

        def prepare():
            try:
                runtime.prepare_engage()
            except RuntimeError as error:
                errors.append(str(error))

        thread = threading.Thread(target=prepare)
        thread.start()
        try:
            self.event(events, "prepare_before_entry")
            runtime.pause("Space")
            self.event(events, "paused")
            release.set()
            thread.join(2.)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, ["fake preparation cancelled before entry"])
            with self.assertRaisesRegex(RuntimeError, "prepare before"):
                runtime.begin_follow()
            self.assertEqual(runtime.state, SystemState.PAUSED)
        finally:
            release.set()
            thread.join(2.)
            events.close()

    def test_parent_control_stall_pauses_child_without_auto_resume(self):
        events = self.context.Queue()
        runtime = self.runtime({"events": events})
        runtime.start()
        runtime.prepare_engage()
        runtime.begin_follow()
        # No health/glove/tick reads: an unrelated parent thread must not renew it.
        self.event(events, "paused", timeout=2.)
        self.assertEqual(runtime.state, SystemState.PAUSED)
        self.assertIn("coordinator stopped", runtime.last_error)
        runtime.health()
        time.sleep(.04)
        self.assertEqual(runtime.state, SystemState.PAUSED)
        events.close()

    def test_timed_out_follow_cancels_itself_without_coordinator_pause(self):
        events = self.context.Queue()
        release = self.context.Event()
        runtime = self.runtime({"events": events, "release_follow": release,
                                "hand_timeout_s": .15})
        runtime.start()
        runtime.prepare_engage()
        try:
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                runtime.begin_follow()
            self.event(events, "follow_before_commit")
            self.event(events, "paused")
            release.set()
            time.sleep(.04)
            self.assertEqual(runtime.state, SystemState.PAUSED)
            runtime.health()
            time.sleep(.04)
            self.assertEqual(runtime.state, SystemState.PAUSED)
            # A fresh explicit prepare/follow sequence is still allowed.
            runtime.prepare_engage()
            runtime.begin_follow()
            self.assertEqual(runtime.state, SystemState.ENGAGED)
        finally:
            release.set()
            events.close()

    def test_original_glove_expiry_is_not_extended_by_fresh_process_snapshots(self):
        runtime = self.runtime({"fixed": True, "glove_timeout_s": .12})
        runtime.start()
        deadline = time.monotonic()+.2
        while time.monotonic() < deadline:
            runtime.health()
            time.sleep(.02)
        self.assertFalse(runtime.health().ready)
        self.assertIn("stale", runtime.health().detail)
        self.assertIn("glove", runtime.health().detail)

    def test_original_diagnostics_expiry_is_not_extended_by_fresh_joint_samples(self):
        runtime = self.runtime({"fixed_diagnostics": True, "hand_timeout_s": .12})
        runtime.start()
        deadline = time.monotonic()+.2
        while time.monotonic() < deadline:
            runtime.health()
            time.sleep(.02)
        self.assertFalse(runtime.health().ready)
        self.assertIn("stale", runtime.health().detail)
        self.assertIn("hand diagnostics", runtime.health().detail)

    def test_worker_stall_is_unhealthy_even_with_fresh_devices_and_snapshots(self):
        runtime = self.runtime({"fixed_worker": True, "hand_timeout_s": .12})
        runtime.start()
        runtime.prepare_engage()
        runtime.begin_follow()
        deadline = time.monotonic()+.2
        while time.monotonic() < deadline:
            runtime.health()
            time.sleep(.02)
        self.assertFalse(runtime.health().ready)
        self.assertIn("stale", runtime.health().detail)
        self.assertIn("control worker", runtime.health().detail)

    def test_terminal_interrupt_leaves_child_available_for_coordinated_close(self):
        events = self.context.Queue()
        runtime = self.runtime({"events": events})
        runtime.start()
        os.kill(runtime._process.pid, signal.SIGINT)
        runtime.prepare_engage()
        runtime.begin_follow()
        self.assertTrue(runtime.health().ready)
        runtime.close()
        self.event(events, "closed")
        self.assertEqual(runtime._process.exitcode, 0)
        events.close()

    def test_child_exit_is_unhealthy(self):
        runtime = self.runtime()
        runtime.start()
        runtime._process.terminate()
        runtime._process.join(2.)
        self.assertFalse(runtime.health().ready)
        self.assertIn("exited", runtime.last_error)

    def test_startup_failure_closes_child(self):
        runtime = self.runtime({"fail_start": True})
        with self.assertRaisesRegex(RuntimeError, "fake startup failure"):
            runtime.start()
        self.assertFalse(runtime._process.is_alive())

    def test_parent_exit_closes_child(self):
        events = self.context.Queue()
        parent = self.context.Process(target=_abandon_parent, args=(events,))
        parent.start()
        try:
            child_pid = self.event(events, "abandoning", timeout=5.)
            parent.join(2.)
            self.assertFalse(parent.is_alive())
            self.assertEqual(self.event(events, "closed", timeout=3.), child_pid)
        finally:
            if parent.is_alive():
                parent.terminate()
                parent.join()
            events.close()

if __name__ == "__main__":
    unittest.main()
