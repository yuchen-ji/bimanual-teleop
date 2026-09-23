"""Hand following and lifecycle tests; no SDK manager or hardware connections."""

from dataclasses import replace
import importlib.util
import math
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bimanual_teleop.devices.wuji.adapter import JOINT_LIMITS_RAD, JOINT_NAMES
from bimanual_teleop.system import SystemState
from bimanual_teleop.control.hand.follow import WujiHandRetargeter, WujiTeleop, create_wuji_teleop, preflight
from bimanual_teleop.types import ControlProfile, HandSkeleton, Health, Sample, SampleHeader, SampleRef


from tests.support.wuji import Sink, Device, Mapper
from tests.support.clock import Clock


class PreflightTests(unittest.TestCase):
    def test_checks_installation_without_importing_native_modules(self):
        with patch("bimanual_teleop.control.hand.follow.version", return_value="2026.8.31"), \
                patch("bimanual_teleop.control.hand.follow.find_spec", return_value=object()) as spec, \
                patch("builtins.__import__", side_effect=AssertionError("preflight imported a module")):
            preflight()
        self.assertEqual([call.args[0] for call in spec.call_args_list], ["numpy", "wuji_sdk"])

    def test_missing_package_fails_before_runtime_construction(self):
        with patch("bimanual_teleop.control.hand.follow.version", return_value="2026.8.31"), \
                patch("bimanual_teleop.control.hand.follow.find_spec", return_value=None):
            with self.assertRaisesRegex(ImportError, "numpy"):
                preflight()


class StartupTests(unittest.TestCase):
    def test_native_failure_closes_all_devices_and_preserves_both_errors(self):
        class WujiException(Exception):
            pass

        clock, sink = Clock(), Sink()
        gloves = {s: Device(s, clock, glove=True) for s in ("left", "right")}
        hands = {s: Device(s, clock) for s in gloves}
        cause = WujiException("right hand connection timeout")
        hands["left"].close_error = WujiException("left hand cleanup failed")
        runtime = WujiTeleop(gloves, hands, {s: Mapper() for s in gloves},
            profile=ControlProfile("test", "mit", {}), sink=sink, threaded=False)
        original_start = Device.start

        def start(device, sink=None):
            if device is hands["right"]:
                raise cause
            original_start(device, sink)

        with patch.object(Device, "start", start), self.assertRaisesRegex(
                RuntimeError, "right hand connection timeout.*left hand cleanup failed") as caught:
            runtime.start()
        self.assertIs(caught.exception.__cause__, cause)
        self.assertEqual(runtime.state, SystemState.CLOSED)
        self.assertEqual(runtime.last_error, str(caught.exception))
        self.assertTrue(all(d.closed for d in (*hands.values(), *gloves.values())))
        self.assertTrue(all(not h.enabled for h in hands.values()))
        failure = next(e for e in sink.events if e.kind == "wuji.start_failed")
        self.assertEqual(failure.details["error"], str(cause))
        runtime.close()


class RuntimeTests(unittest.TestCase):
    def make(self, *, sides=("left", "right"), threaded=False, clock=None):
        self.clock = clock or Clock()
        self.gloves = {s: Device(s, self.clock, glove=True) for s in sides}
        self.hands = {s: Device(s, self.clock) for s in sides}
        self.maps = {s: Mapper() for s in sides}
        self.sink = Sink()
        self.runtime = WujiTeleop(self.gloves, self.hands, self.maps,
            profile=ControlProfile("tested", "mit", {"kp": 5., "kd": .05, "current_limit_a": 1.5}),
            sink=self.sink, clock_ns=self.clock, threaded=threaded)
        self.addCleanup(self.runtime.close)
        self.runtime.start()
        return self.runtime

    def refresh(self):
        for device in (*self.gloves.values(), *self.hands.values()): device.emit()

    def join_disables(self):
        for thread in tuple(self.runtime._disable_threads.values()): thread.join(1)

    def test_start_is_acquisition_only_until_explicit_engagement(self):
        runtime = self.make()
        self.assertEqual(runtime.state, SystemState.READY)
        self.assertTrue(all(h.calls == ["start"] for h in self.hands.values()))
        runtime.engage()
        runtime._step()
        self.assertTrue(all(h.enabled and h.commands for h in self.hands.values()))
        self.assertEqual(runtime.mode, "follow")

    def test_realtime_observer_failure_pauses_before_hand_submission(self):
        runtime = self.make()
        runtime.engage()
        before = {side: len(hand.commands) for side, hand in self.hands.items()}
        self.sink.try_event = lambda _: False
        self.gloves["left"].emit(q=.4)
        runtime._step()
        self.assertEqual(runtime.state, SystemState.PAUSED)
        self.assertIn("observer", runtime.last_error)
        self.assertEqual({side: len(hand.commands) for side, hand in self.hands.items()}, before)

    def test_left_and_right_targets_with_exact_takeover_endpoints(self):
        runtime = self.make()
        self.gloves["left"].emit(q=.3)
        self.gloves["right"].emit(q=.5)
        runtime.engage()
        runtime._step()
        self.assertTrue(all(h.last_target.position_rad == (.1,)*20 for h in self.hands.values()))
        self.clock.advance_s(.375); self.refresh(); runtime._step()
        self.assertAlmostEqual(self.hands["left"].last_target.position_rad[0], .2)
        self.assertAlmostEqual(self.hands["right"].last_target.position_rad[0], .3)
        self.clock.advance_s(.375); self.refresh(); runtime._step()
        self.assertEqual(self.hands["left"].last_target.position_rad, (.3,)*20)
        self.assertEqual(self.hands["right"].last_target.position_rad, (.5,)*20)
        # Following after takeover has no added velocity limiter.
        self.gloves["left"].emit(q=.6); self.clock.advance_s(.008); runtime._step()
        self.assertEqual(self.hands["left"].last_target.position_rad, (.6,)*20)

    def test_repeated_latest_and_caller_tick_do_not_resolve_or_add_raw_samples(self):
        runtime = self.make()
        runtime.engage()
        count = len(self.sink.samples)
        for _ in range(5): runtime._step(); runtime.tick()
        self.assertEqual([m.calls for m in self.maps.values()], [1, 1])
        self.assertEqual(len(self.sink.samples), count)
        self.assertEqual(len(self.hands["left"].commands), 5)

    def test_healthy_hands_hold_while_waiting_for_arm_engagement(self):
        runtime = self.make()
        runtime.prepare_engage()
        runtime._step()
        self.assertEqual(runtime.mode, "hold")
        self.assertNotEqual(runtime.state, SystemState.ENGAGED)
        self.assertTrue(all(len(h.commands) == 1 for h in self.hands.values()))
        runtime.begin_follow()
        self.assertEqual(runtime.state, SystemState.ENGAGED)

    def test_pause_resume_starts_at_held_command_not_loaded_actual_position(self):
        runtime = self.make()
        self.gloves["left"].emit(q=.4)
        runtime.engage(); self.clock.advance_s(.75); self.refresh(); runtime._step()
        runtime.pause("Space")
        self.hands["left"].emit(q=.05)
        self.gloves["left"].emit(q=.7)
        runtime._step()
        self.assertEqual(self.hands["left"].last_target.position_rad, (.4,)*20)
        runtime.engage(); runtime._step()
        self.assertEqual(self.hands["left"].last_target.position_rad, (.4,)*20)
        self.assertEqual(self.hands["left"].calls.count("configure"), 1)

    def test_hold_failure_between_prepare_and_follow_preserves_the_cause(self):
        runtime = self.make()
        runtime.prepare_engage()
        self.hands["right"].accept = False
        runtime._step()
        with self.assertRaisesRegex(RuntimeError, "right send failed"):
            runtime.begin_follow()
        self.assertEqual(runtime.state, SystemState.PAUSED)
        self.join_disables()

    def test_resume_after_disable_references_new_actual_position(self):
        runtime = self.make()
        runtime.engage(); runtime.pause("fault")
        self.hands["left"].disable()
        self.hands["left"].emit(q=.02)
        runtime.engage(); runtime._step()
        self.assertEqual(self.hands["left"].last_target.position_rad, (.02,)*20)

    def test_transient_glove_failure_pauses_despite_valid_latest_then_manual_ack(self):
        runtime = self.make()
        runtime.engage()
        self.gloves["left"].emit(valid=False)
        self.gloves["left"].emit(valid=True)
        runtime._step()
        self.assertEqual(runtime.state, SystemState.PAUSED)
        self.assertTrue(all(h.enabled for h in self.hands.values()))
        self.assertTrue(runtime.health().ready)
        runtime.engage()
        self.assertIsNone(self.gloves["left"].fault)

    def test_stale_input_does_not_extend_its_own_deadline_through_hold(self):
        runtime = self.make()
        runtime.engage(); runtime._step()
        source = self.gloves["left"].latest.header
        self.clock.advance_s(.25)
        for hand in self.hands.values(): hand.emit()
        runtime._step(); runtime._step()
        self.assertEqual(runtime.state, SystemState.PAUSED)
        self.assertEqual(self.gloves["left"].latest.header, source)
        self.assertTrue(all(h.enabled for h in self.hands.values()))
        self.assertEqual(len(self.hands["left"].commands), 2)

    def test_hand_fault_disables_only_failed_side_and_other_side_holds(self):
        runtime = self.make()
        runtime.engage(); runtime._step()
        self.hands["left"].fault = "joint missing"
        runtime._step(); self.join_disables()
        self.assertEqual(runtime.state, SystemState.PAUSED)
        self.assertFalse(self.hands["left"].enabled)
        self.assertTrue(self.hands["right"].enabled)
        self.assertEqual(len(self.hands["right"].commands), 2)

    def test_slow_fault_disable_does_not_block_healthy_hand_or_pause(self):
        runtime = self.make()
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        self.hands["left"].disable_hook = lambda: (entered.set(), release.wait(1))
        runtime.engage()
        self.hands["left"].fault = "offline"
        runtime._step()
        self.assertTrue(entered.wait(1))
        runtime.pause("operator pause")
        runtime._step()
        self.assertEqual(len(self.hands["right"].commands), 2)
        release.set(); self.join_disables()

    def test_both_solutions_are_checked_before_either_side_is_submitted(self):
        runtime = self.make()
        runtime.engage()
        self.maps["right"].fail = True
        self.gloves["left"].emit(q=.2); self.gloves["right"].emit(q=.3)
        runtime._step()
        self.assertFalse(self.hands["left"].commands)
        self.assertFalse(self.hands["right"].commands)
        self.assertEqual(runtime.state, SystemState.PAUSED)

    def test_source_failure_during_other_side_solve_is_rechecked_before_submit(self):
        runtime = self.make()
        runtime.engage()
        self.maps["right"].hook = lambda: self.gloves["left"].emit(valid=False)
        self.gloves["right"].emit()
        runtime._step()
        self.assertFalse(self.hands["left"].commands)
        self.assertEqual(runtime.state, SystemState.PAUSED)

    def test_partial_submission_failure_is_not_claimed_atomic(self):
        runtime = self.make()
        runtime.engage()
        self.hands["right"].accept = False
        runtime._step(); self.join_disables()
        self.assertEqual(len(self.hands["left"].commands), 1)
        self.assertFalse(self.hands["right"].commands)
        self.assertFalse(self.hands["right"].enabled)
        self.assertEqual(runtime.state, SystemState.PAUSED)

    def test_bad_solver_is_rejected_before_any_enable(self):
        runtime = self.make()
        self.maps["right"].fail = True
        with self.assertRaises(ValueError): runtime.prepare_engage()
        self.assertTrue(all(not h.enabled and h.profile is None for h in self.hands.values()))

    def test_prepare_cancelled_during_enable_cannot_later_begin_follow(self):
        runtime = self.make()
        self.hands["right"].engage_hook = lambda: runtime.pause("Space")
        with self.assertRaises(RuntimeError): runtime.prepare_engage()
        with self.assertRaises(RuntimeError): runtime.begin_follow()
        self.assertEqual(runtime.state, SystemState.PAUSED)
        runtime._step()
        self.assertTrue(self.hands["left"].enabled)

    def test_cancelled_before_prepare_lock_never_establishes_new_generation(self):
        runtime = self.make()
        entered, cancelled = threading.Event(), threading.Event()
        errors = []

        def prepare():
            entered.set()
            try:
                runtime.prepare_engage(cancelled=cancelled.is_set)
            except RuntimeError as error:
                errors.append(str(error))

        # Hold the exact lock used before prepare captures its generation.
        # The caller's pause wins while the new worker is still outside it.
        with runtime._lock:
            thread = threading.Thread(target=prepare)
            thread.start()
            self.assertTrue(entered.wait(1.))
            cancelled.set()
            runtime.pause("Space")
            cancelled_generation = runtime._generation
        thread.join(2.)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, ["Wuji engagement cancelled"])
        self.assertEqual(runtime._generation, cancelled_generation)
        self.assertIsNone(runtime._prepared_generation)
        self.assertTrue(all(not hand.enabled and hand.profile is None for hand in self.hands.values()))
        self.assertTrue(all(mapper.calls == 0 for mapper in self.maps.values()))
        with self.assertRaises(RuntimeError):
            runtime.begin_follow()

    def test_external_cancel_prevents_enable_before_pause_changes_generation(self):
        for stage in ("before_configure", "after_configure", "during_enable"):
            with self.subTest(stage=stage):
                runtime = self.make()
                cancelled = threading.Event()
                expected_generation = runtime._generation + 1
                if stage == "before_configure":
                    self.maps["right"].hook = cancelled.set
                elif stage == "after_configure":
                    original_configure = self.hands["left"].configure

                    def configure(profile):
                        original_configure(profile)
                        cancelled.set()

                    self.hands["left"].configure = configure
                else:
                    self.hands["left"].engage_hook = cancelled.set
                with self.assertRaisesRegex(RuntimeError, "cancelled"):
                    runtime.prepare_engage(cancelled=cancelled.is_set)
                self.assertEqual(runtime._generation, expected_generation)
                self.assertTrue(all(not hand.enabled for hand in self.hands.values()))
                self.assertIsNone(self.hands["right"].profile)
                self.assertIsNone(runtime._prepared_generation)

    def test_external_cancel_during_final_locked_reads_prevents_prepare_commit(self):
        runtime = self.make()
        cancelled = threading.Event()
        expected_generation = runtime._generation + 1
        original_current = runtime._current

        def current(side, *, glove=False, **kwargs):
            sample = original_current(side, glove=glove, **kwargs)
            if side == "right" and not glove and all(hand.enabled for hand in self.hands.values()):
                # Both enables already finished. The command loop has cancelled,
                # but its pause is still waiting for this final preparation lock.
                cancelled.set()
            return sample

        with patch.object(runtime, "_current", side_effect=current):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                runtime.prepare_engage(cancelled=cancelled.is_set)
        self.assertEqual(runtime._generation, expected_generation)
        self.assertIsNone(runtime._prepared_generation)
        self.assertFalse(any(event.kind == "wuji.prepared" for event in self.sink.events))
        with self.assertRaises(RuntimeError):
            runtime.begin_follow()

    def test_pause_between_prepare_and_follow_invalidates_token(self):
        runtime = self.make()
        runtime.prepare_engage(); runtime.pause("Space")
        with self.assertRaises(RuntimeError): runtime.begin_follow()

    def test_single_side_and_close_all_even_after_one_failure(self):
        runtime = self.make(sides=("right",))
        runtime.engage(); runtime._step(); runtime.close()
        self.assertTrue(self.hands["right"].closed and self.gloves["right"].closed)
        self.assertEqual(runtime.state, SystemState.CLOSED)
        self.assertFalse(self.hands["right"].enabled)

    def test_close_error_is_reported_after_all_cleanup(self):
        runtime = self.make()
        self.hands["left"].close_error = RuntimeError("parameter restore failed")
        with self.assertRaisesRegex(RuntimeError, "parameter restore failed"): runtime.close()
        self.assertTrue(all(d.closed for d in (*self.hands.values(), *self.gloves.values())))

    def test_sdk_user_checked_at_engagement_without_control_loop_polling(self):
        runtime = self.make()
        query = Mock(return_value=Health(True, self.clock()))
        runtime.session = SimpleNamespace(health=query, close=lambda: None)
        runtime._step()
        query.assert_not_called()
        runtime.engage()
        self.assertEqual(query.call_count, 1)  # Preparation owns the SDK user check.
        query.side_effect = AssertionError("control loop queried SDK user")
        runtime._step()
        self.assertTrue(runtime.health().ready)
        runtime.status(include_target=False)
        runtime.pause("operator")
        runtime._step()
        self.assertEqual(query.call_count, 1)
        query.side_effect = None
        query.return_value = Health(False, self.clock(), "SDK user changed")
        with self.assertRaisesRegex(RuntimeError, "SDK user changed"):
            runtime.engage()
        self.assertEqual(runtime.state, SystemState.PAUSED)

    def test_health_and_status_do_not_wait_for_blocked_retarget_or_call_sdk(self):
        runtime = self.make()
        runtime.engage()
        runtime._step()
        self.refresh()
        entered, release, read_done = threading.Event(), threading.Event(), threading.Event()
        def blocked():
            entered.set()
            if not release.wait(2):
                raise RuntimeError("test did not release retarget")
        self.maps["left"].hook = blocked
        worker = threading.Thread(target=runtime._step)
        worker.start()
        result = {}
        try:
            self.assertTrue(entered.wait(1))
            self.clock.advance_s(.08)
            self.refresh()
            def read():
                try:
                    result["health"] = runtime.health()
                    result["status"] = runtime.status(include_target=False)
                finally:
                    read_done.set()
            reader = threading.Thread(target=read)
            reader.start()
            self.assertTrue(read_done.wait(.1), "UI waited for the hand control lock")
            reader.join(.1)
            self.assertTrue(result["health"].ready)
            self.assertEqual(result["status"]["worker_age_ns"], 80_000_000)
            self.assertEqual(runtime.state, SystemState.ENGAGED)
            self.clock.advance_s(.251)
            self.assertFalse(runtime.health().ready, "cached health concealed expired live input")
            self.refresh()
            self.gloves["left"].emit(valid=False)
            self.gloves["left"].emit()
            self.assertFalse(runtime.health().ready, "a fresh packet erased a latched fault")
        finally:
            release.set()
            worker.join(1)

    def test_following_checks_live_data_after_long_preparation(self):
        runtime = self.make()
        runtime.prepare_engage()
        self.clock.advance_s(2.)
        self.refresh()
        runtime.begin_follow()
        self.assertTrue(runtime.health().ready)

    def test_expired_solved_input_cannot_be_renewed_by_new_feedback(self):
        runtime = self.make()
        runtime.engage()
        self.refresh()
        before = {side: len(hand.commands) for side, hand in self.hands.items()}
        def delayed_solve():
            self.clock.advance_s(.251)
            self.refresh()
        self.maps["left"].hook = delayed_solve
        runtime._step()
        self.assertEqual(runtime.state, SystemState.PAUSED)
        self.assertIn("expired", runtime.last_error)
        self.assertEqual({side: len(hand.commands) for side, hand in self.hands.items()}, before)

    def test_begin_follow_has_no_repeated_sdk_user_round_trip(self):
        runtime = self.make()
        query = Mock(return_value=Health(True, self.clock()))
        runtime.session = SimpleNamespace(health=query, close=lambda: None)
        runtime.prepare_engage()
        query.side_effect = AssertionError("following queried SDK user")
        runtime.begin_follow()
        self.assertEqual(query.call_count, 1)
        self.assertEqual(runtime.state, SystemState.ENGAGED)

    def test_follow_cancellation_is_checked_while_pause_waits_for_control_lock(self):
        runtime = self.make()
        runtime.prepare_engage()
        entered, release = threading.Event(), threading.Event()
        cancelled, pause_entered = threading.Event(), threading.Event()
        original_current = runtime._current
        errors = []

        def current(*args, **kwargs):
            entered.set()
            if not release.wait(2.):
                raise RuntimeError("test did not release following")
            return original_current(*args, **kwargs)

        def follow():
            try:
                runtime.begin_follow(cancelled=cancelled.is_set)
            except RuntimeError as error:
                errors.append(str(error))

        def pause():
            # The process command loop updates its cancellation identity before
            # calling pause, which can itself be waiting on this same lock.
            cancelled.set()
            pause_entered.set()
            runtime.pause("Space")

        with patch.object(runtime, "_current", side_effect=current):
            follower = threading.Thread(target=follow)
            follower.start()
            self.assertTrue(entered.wait(1.))
            stopper = threading.Thread(target=pause)
            stopper.start()
            self.assertTrue(pause_entered.wait(1.))
            release.set()
            follower.join(2.)
            stopper.join(2.)
        self.assertFalse(follower.is_alive() or stopper.is_alive())
        self.assertEqual(errors, ["Wuji engagement cancelled"])
        self.assertEqual(runtime.state, SystemState.PAUSED)
        self.assertFalse(any(event.kind == "wuji.engaged" for event in self.sink.events))

    def test_glove_samples_are_the_latest_owned_samples(self):
        runtime = self.make()
        initial = runtime.glove_samples()
        self.assertEqual(initial, {side: glove.get_latest() for side, glove in self.gloves.items()})
        self.gloves["left"].emit()
        latest = runtime.glove_samples()
        self.assertIs(latest["left"], self.gloves["left"].get_latest())
        self.assertIsNot(latest["left"], initial["left"])
        self.assertIs(latest["right"], initial["right"])

    def test_unobserved_retarget_does_not_serialize_events(self):
        runtime = self.make()
        runtime.sink = None
        with patch("bimanual_teleop.control.hand.follow.asdict", side_effect=AssertionError("unused event")):
            runtime._map("left", self.gloves["left"].get_latest())

    def test_worker_runs_independently_and_skips_overdue_periods(self):
        runtime = self.make(clock=time.monotonic_ns, threaded=True)
        runtime.engage()
        self.maps["left"].hook = lambda: time.sleep(.025)
        self.gloves["left"].emit()
        time.sleep(.09)
        runtime.pause("done")
        self.assertGreater(runtime.missed_periods, 0)
        self.assertGreaterEqual(len(self.hands["left"].commands), 2)
        self.assertLess(len(self.hands["left"].commands), 14)

    def test_dead_worker_cannot_be_reengaged_with_healthy_feedback(self):
        runtime = self.make()
        runtime.engage()
        runtime._step = lambda: (_ for _ in ()).throw(TypeError("unexpected worker failure"))
        runtime._run()
        self.join_disables()
        self.assertFalse(runtime.health().ready)
        self.assertTrue(runtime._stop.is_set())
        with self.assertRaisesRegex(RuntimeError, "worker stopped"): runtime.prepare_engage()
        self.assertTrue(all(not h.enabled for h in self.hands.values()))

    def test_actual_rate_counts_only_worker_cycles(self):
        runtime = self.make()
        runtime._step()
        self.clock.advance_s(.01)
        runtime._step()
        for _ in range(50): runtime.tick()
        self.assertEqual(runtime.status()["control_hz_actual"], 100.)
        self.assertEqual(runtime.cycles, 2)

    def test_disable_rpc_failure_needs_new_explicit_enable(self):
        runtime = self.make()
        runtime.engage()
        self.hands["left"].disable_hook = lambda: (_ for _ in ()).throw(RuntimeError("network lost"))
        self.hands["left"].fault = "bad feedback"
        runtime._step(); self.join_disables()
        self.assertIn("left", runtime._blocked_hands)
        self.assertTrue(any(e.kind == "wuji.disable_failed" for e in self.sink.events))
        self.hands["left"].disable_hook = None
        self.refresh()
        runtime.engage()
        self.assertEqual(self.hands["left"].calls.count("engage"), 2)
        self.assertNotIn("left", runtime._blocked_hands)


class RetargetTests(unittest.TestCase):
    def sample(self):
        return Sample(SampleHeader(SampleRef("skeleton", "test", 1), 1, True),
            HandSkeleton("left_wrist", tuple(str(i) for i in range(21)),
                         ((0., 0., 0.),)*21, (1.,)*21, ()))

    def test_official_side_model_float32_order_and_mechanical_limits(self):
        import numpy as np
        calls = []
        class Session:
            def step(self, points):
                self.points = points
                return np.array([-100, 100]*10)
            def reset(self): calls.append("reset")
        def for_hand(model, *, side):
            calls.append((model, side))
            return Session()
        sdk = SimpleNamespace(RetargetSession=SimpleNamespace(for_hand=for_hand),
            HandModel=SimpleNamespace(WujiHand2="Hand2"),
            Handedness=SimpleNamespace(Left="L", Right="R"))
        for side, expected in (("left", "L"), ("right", "R")):
            mapper = WujiHandRetargeter(side, sdk=sdk)
            result = mapper.solve(self.sample())
            self.assertIn(("Hand2", expected), calls)
            self.assertEqual(result.joint_names, JOINT_NAMES)
            self.assertEqual(result.position_rad,
                tuple(limit[i % 2] for i, limit in enumerate(JOINT_LIMITS_RAD)))
            self.assertEqual(mapper._session.points.dtype, np.float32)
            mapper.reset()
        self.assertEqual(calls.count("reset"), 2)

    def test_invalid_input_and_native_output_are_not_fabricated(self):
        import numpy as np
        mapper = WujiHandRetargeter("left")
        sample = self.sample()
        for positions in (((0.,)*3,)*20, ((math.nan, 0., 0.),)*21):
            with self.assertRaises(ValueError):
                mapper.solve(replace(sample, payload=replace(sample.payload, positions_m=positions)))
        for q in (np.zeros(19), np.full(20, math.nan)):
            mapper._session = SimpleNamespace(step=lambda points: q)
            with self.assertRaises(ValueError): mapper.solve(sample)

    @unittest.skipUnless(importlib.util.find_spec("wuji_sdk"), "optional Wuji SDK not installed")
    def test_native_both_side_and_reset_smoke_without_manager(self):
        sample = self.sample()
        points = [(0., 0., 0.)]
        for finger in range(5):
            points.extend(((finger-2)*.018, .025 + joint*.023, 0.) for joint in range(4))
        sample = replace(sample, payload=replace(sample.payload, positions_m=tuple(points)))
        for side in ("left", "right"):
            mapper = WujiHandRetargeter(side)
            first = mapper.solve(sample)
            mapper.reset()
            second = mapper.solve(sample)
            self.assertEqual(len(first.position_rad), 20)
            self.assertTrue(all(math.isfinite(x) for x in first.position_rad))
            self.assertEqual(first.position_rad, second.position_rad)

    def test_factory_does_not_import_or_connect_sdk(self):
        config = {"profile_id": "test", "sdk_user_name": "yuchen",
                  "parameters": {"kp": 5, "kd": .05, "current_limit_a": 1.5},
                  "devices": {s: {"glove": "192.168.1.100:50001", "hand": "192.168.1.110:7447"}
                              for s in ("left", "right")}}
        runtime = create_wuji_teleop(config, sides=("left",))
        self.assertEqual(runtime.state, SystemState.DISCONNECTED)
        self.assertIsNone(runtime.session.manager)
        self.assertEqual(runtime.session.user_name, "yuchen")
        self.assertIsNone(runtime.retargeters["left"]._session)
        self.assertEqual(runtime.gloves["left"].streams, ("skeleton",))
        self.assertEqual(runtime.hands["left"].feedback_hz, 200)
        recorded = create_wuji_teleop(config, sides=("left",), sink=Sink())
        self.assertEqual(recorded.gloves["left"].streams, ("emf", "skeleton"))
        configured = create_wuji_teleop({**config, "feedback_hz": 250}, sides=("left",))
        self.assertEqual(configured.hands["left"].feedback_hz, 250)
        with self.assertRaisesRegex(ValueError, "feedback_hz"):
            create_wuji_teleop({**config, "feedback_hz": 0}, sides=("left",))
        for selection in ({"sdk_user_name": 123}, {"sdk_user_name": " "},
                          {"sdk_user_id": "ambiguous"}):
            with self.assertRaises(ValueError):
                create_wuji_teleop({**config, **selection}, sides=("left",))
        with self.assertRaisesRegex(ValueError, "sdk_user_name"):
            create_wuji_teleop({**config, "sdk_user_name": "", "sdk_user_id": "old"}, sides=("left",))


if __name__ == "__main__":
    unittest.main()
