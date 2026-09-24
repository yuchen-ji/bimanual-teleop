"""Deterministic driver races with fake SDK calls; no device connection."""

from copy import deepcopy
import threading
import unittest
from unittest.mock import patch

from bimanual_teleop.devices.tianji import driver as driver_module
from tests.support import tianji as fixtures

from tests.support.clock import Clock


class WatchedLock:
    """Let the submitter observe when the watchdog attempts the real lock."""

    def __init__(self, attempted):
        self.lock = threading.RLock()
        self.attempted = attempted

    def __enter__(self):
        if threading.current_thread().name == "watchdog-regression":
            self.attempted.set()
        self.lock.acquire()
        return self

    def __exit__(self, *args):
        self.lock.release()


class TianjiWatchdogRegressionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TianjiFixture()
        self.addCleanup(self.fixture.driver.close)
        self.fixture.engage(("left", "right"))
        self.driver, self.native = self.fixture.driver, self.fixture.native

    def timed_state(self, now=1_049_000_000):
        clock = Clock(now)
        self.driver.watchdog_ns = 50_000_000
        self.driver._deadline_ns = 1_050_000_000
        self.driver._last_target_ns = 1_000_000_000
        self.driver._last_command_id = "previous-target"
        for side in ("left", "right"):
            self.driver._advanced[side] = 1_049_000_000
        return clock

    def test_watchdog_rechecks_deadline_after_inflight_submit_releases_lock(self):
        clock = self.timed_state()
        native_entered, watch_attempted = threading.Event(), threading.Event()
        self.driver._lock = WatchedLock(watch_attempted)
        native_call = self.native.call
        results, errors, polls = {}, [], []

        def call(name, *args):
            if name == "submit":
                # Submit begins at 49 ms with a 50 ms deadline. A native
                # call crosses that deadline while holding the driver lock.
                clock.now = 1_051_000_000
                native_entered.set()
                if not watch_attempted.wait(1):
                    raise RuntimeError("watchdog did not attempt the command lock")
                clock.now = 1_052_000_000
            return native_call(name, *args)

        def wait(_):
            polls.append(True)
            if len(polls) > 1:
                return True
            if not native_entered.wait(1):
                raise RuntimeError("submit did not enter its fake native call")
            return False

        def watch():
            try:
                self.driver._watch()
            except Exception as error:
                errors.append(error)

        with patch.object(driver_module.time, "monotonic_ns", clock), \
                patch.object(self.native, "call", side_effect=call), \
                patch.object(self.driver._stop, "wait", side_effect=wait):
            command = self.fixture.command("renewed-target")
            submitter = threading.Thread(target=lambda: results.update(result=self.driver.submit(command)))
            watcher = threading.Thread(target=watch, name="watchdog-regression")
            submitter.start()
            watcher.start()
            submitter.join(2)
            watcher.join(2)
            self.assertFalse(submitter.is_alive())
            self.assertFalse(watcher.is_alive())
            self.assertEqual(errors, [])
            self.assertTrue(results["result"].accepted)
            self.assertTrue(self.driver.engaged)
            self.assertEqual(self.driver._deadline_ns, 1_102_000_000)
            self.assertEqual(self.driver._last_target_ns, 1_052_000_000)
            self.assertIsNone(self.driver.motion_stop)
            self.assertFalse(any(name == "hold" for name, _ in self.native.calls))
            self.assertEqual(self.driver._last_sdk_call["duration_ms"], 3.)

    def test_genuinely_expired_watchdog_still_holds_with_original_50ms_budget(self):
        clock = self.timed_state(1_052_000_000)
        with patch.object(driver_module.time, "monotonic_ns", clock), \
                patch.object(self.driver._stop, "wait", side_effect=(False, True)):
            self.driver._watch()
        self.assertFalse(self.driver.engaged)
        stop = self.driver.motion_stop
        self.assertEqual(stop["schema"], "tianji.motion_stop.v1")
        self.assertEqual(stop["monotonic_ns"], 1_052_000_000)
        self.assertEqual(stop["last_target_ns"], 1_000_000_000)
        self.assertEqual(stop["deadline_ns"], 1_050_000_000)
        self.assertEqual(stop["target_age_ms"], 52.)
        self.assertEqual(stop["deadline_overrun_ms"], 2.)
        self.assertEqual(stop["feedback_age_ms"], {"left": 3., "right": 3.})
        self.assertEqual(stop["last_command_id"], "previous-target")
        self.assertIn("Accepted target expired", stop["reason"])
        self.assertTrue(any(name == "submit" for name, _ in self.native.calls))
        self.assertEqual(self.fixture.events("tianji.hold_sdk_returned")[-1].details["method"],
                         "cartesian_measured_hold")

    def test_submission_after_old_deadline_does_not_renew_or_send_it(self):
        clock = self.timed_state(1_051_000_000)
        before = sum(name == "submit" for name, _ in self.native.calls)
        with patch.object(driver_module.time, "monotonic_ns", clock):
            result = self.driver.submit(self.fixture.command("late-target"))
        self.assertFalse(result.accepted)
        self.assertIn("Previous target watchdog expired", result.reason)
        self.assertEqual(self.driver.motion_stop["reason"], result.reason)
        self.assertEqual(self.driver._deadline_ns, 1_050_000_000)
        submits = [args for name, args in self.native.calls if name == "submit"]
        self.assertEqual(len(submits), before + 1)
        self.assertTrue(self.driver._commands[self.driver._token].startswith("hold-"))

    def test_first_cause_is_published_before_stop_and_survives_retries_and_rejection(self):
        clock = self.timed_state(1_052_000_000)
        native_call, observations = self.native.call, []

        def call(name, *args):
            if name == "submit":
                observations.append(self.driver.motion_stop)
                if len(observations) == 1:
                    raise RuntimeError("datagram pending")
            return native_call(name, *args)

        with patch.object(driver_module.time, "monotonic_ns", clock), \
                patch.object(self.native, "call", side_effect=call):
            self.driver.request_hold("left feedback source stopped advancing")
            first = self.driver.motion_stop
            self.assertEqual(self.driver._hold_reason, first["reason"])
            clock.now += 7_000_000
            self.driver.request_hold("Cartesian engagement ended during planning")
            self.assertIsNone(self.driver._hold_reason)
            result = self.driver.submit(self.fixture.command("after-stop"))
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, first["reason"])
        self.assertEqual(self.driver.motion_stop, first)
        self.assertEqual(observations, [first, first])
        events = self.fixture.events("tianji.hold_requested")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].details["motion_stop"], first)

    def test_snapshot_is_detached_and_successful_reengagement_starts_a_new_stop_record(self):
        self.assertIsNone(self.driver.motion_stop)
        self.driver.request_hold("first engagement stopped")
        first = self.driver.motion_stop
        expected = deepcopy(first)
        first["reason"] = "modified by reader"
        first["feedback_age_ms"]["left"] = -123.
        first["last_native_call"]["duration_ms"] = -123.
        self.assertEqual(self.driver.motion_stop, expected)
        p = driver_module.deepcopy(self.driver._packet)
        p.received_ns = driver_module.time.monotonic_ns()
        p.sequence[:] = [q + 1 for q in p.sequence]
        p.low_speed[:] = (1, 1)
        self.driver._on_feedback(p)
        self.driver.engage()
        self.assertTrue(self.driver.engaged)
        self.assertIsNone(self.driver.motion_stop)
        self.driver.request_hold("new engagement stopped")
        self.assertEqual(self.driver.motion_stop["reason"], "new engagement stopped")

    def test_rejected_reengagement_seed_keeps_the_previous_stop_record(self):
        self.driver.request_hold("first cause")
        first = self.driver.motion_stop
        native_call = self.native.call

        def reject_seed(name, *args):
            if name == "engage":
                raise RuntimeError("seed rejected before acceptance")
            return native_call(name, *args)

        with patch.object(self.native, "call", side_effect=reject_seed):
            with self.assertRaisesRegex(RuntimeError, "seed rejected before acceptance"):
                self.driver.engage()
        self.assertEqual(self.driver.motion_stop, first)

    def test_accepted_reengagement_seed_gets_its_own_mode_confirmation_failure(self):
        self.driver.request_hold("previous engagement cause")
        first = self.driver.motion_stop
        self.native.echo = False
        self.driver.engagement_timeout_ns = 1_000_000
        with self.assertRaisesRegex(RuntimeError, "startup timeout"):
            self.driver.engage()
        second = self.driver.motion_stop
        self.assertNotEqual(second["reason"], first["reason"])
        self.assertIn("Engagement mode was not reported", second["reason"])
        self.assertNotEqual(second["last_command_id"], first["last_command_id"])
        self.assertGreaterEqual(second["monotonic_ns"], first["monotonic_ns"])
        self.assertFalse(self.driver.engaged)


if __name__ == "__main__":
    unittest.main()
