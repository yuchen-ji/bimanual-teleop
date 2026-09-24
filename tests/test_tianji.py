"""Driver boundary tests with synthetic packets; never connect to a robot."""

from copy import deepcopy
from dataclasses import asdict
import errno
import json
import math
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bimanual_teleop.devices.tianji.driver import TianjiDriver, FeedbackSnapshot, decode_feedback
from bimanual_teleop.types import CommandEvent, CommandStatus


from tests.support.tianji import FakeSDK, Sink, packet, TianjiFixture


class TianjiDriverTests(unittest.TestCase, TianjiFixture):
    def setUp(self):
        TianjiFixture.__init__(self)
        self.addCleanup(self.driver.close)


    def test_selected_health_ignores_other_arm_fault_and_counter_reset(self):
        p = packet(1, (1, 1))
        p.error[1] = 13
        self.driver._on_feedback(p)
        self.assertTrue(self.driver.health(sides=("left",)).ready)
        self.assertFalse(self.driver.health().ready)
        self.configure(("left",))
        self.assertTrue(self.driver.health().ready)
        reset = packet(2, (2, 0xFFFFFFFF))
        self.driver._on_feedback(reset)
        self.assertIsNone(self.driver._motion_fault)
        self.assertTrue(self.driver.health().ready)


    def test_native_joint_move_uses_sdk_ratios_and_trajectory_target_completion(self):
        self.configure(("left", "right"))
        target = tuple(math.radians(q) for q in [35.12345, -55, 0, -65, 0, 0, 0])
        initial = tuple(self.driver._packet.q[7:])
        result = self.driver.move_joints("right", target)
        mode = next(args for name, args in self.native.calls if name == "move_joints")
        call = next(args for name, args in self.native.calls if name == "submit")
        self.assertEqual((mode[0], mode[2], mode[3]), (1, 100, 100))
        for actual, expected in zip(mode[1], initial):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(call[0], 2)
        self.assertAlmostEqual(call[1][7], 35.12345)
        self.assertIs(result, self.driver.get_latest())
        self.assertNotEqual(result.payload.arms["right"].joints.position_rad, target)
        self.assertFalse(self.driver.engaged)
        self.assertIsNone(self.driver._moving_side)
        self.assertEqual([name for name, _ in self.native.calls], ["configure", "move_joints", "submit"])

    def test_first_move_from_idle_or_cartesian_survives_mode_entry_target_overwrite(self):
        self.configure(("left", "right"))
        target = tuple(map(math.radians, (35, -55, 0, -65, 0, 0, 0)))
        for side, index in (("left", 0), ("right", 1)):
            for state in (0, 3):
                with self.subTest(side=side, state=state):
                    p = deepcopy(self.driver._packet)
                    p.state[index] = state
                    p.sequence[index] += 1
                    p.received_ns = time.monotonic_ns()
                    self.driver._on_feedback(p)
                    self.native.calls.clear()
                    result = self.driver.move_joints(side, target)
                    self.assertEqual([name for name, _ in self.native.calls], ["move_joints", "submit"])
                    self.assertEqual(result.payload.arms[side].state, 1)
                    self.assertEqual(tuple(map(lambda q: round(math.degrees(q)),
                                               result.payload.arms[side].controller_target_rad)),
                                     (35, -55, 0, -65, 0, 0, 0))

    def test_already_in_position_sends_only_the_final_target(self):
        self.configure()
        self.position_feedback()
        self.native.calls.clear()
        self.driver.move_joints("left", (0,)*7)
        self.assertEqual([name for name, _ in self.native.calls], ["submit"])

    def test_final_target_waits_for_mode_transition_feedback(self):
        self.configure()
        self.native.complete_joint_move = False
        native_call = self.native.call
        states = []

        def call(name, *args):
            if name == "submit":
                self.assertEqual(states, [101, 1])
                self.native.complete_joint_move = True
            return native_call(name, *args)

        def transition(_):
            p = deepcopy(self.driver._packet)
            p.state[0] = 101 if not states else 1
            states.append(p.state[0])
            p.sequence[0] += 1
            p.received_ns = time.monotonic_ns()
            self.driver._on_feedback(p)

        with patch.object(self.native, "call", side_effect=call), \
                patch.object(self.driver._stop, "wait", side_effect=transition):
            self.driver.move_joints("left", (0,)*7)
        self.assertEqual(states, [101, 1])

    def test_failed_position_transition_emits_no_final_motion_target(self):
        self.configure()
        self.native.complete_joint_move = False
        self.driver.engagement_timeout_ns = 10_000_000

        def transitioning(_):
            p = deepcopy(self.driver._packet)
            p.state[0] = 101
            p.sequence[0] += 1
            p.received_ns = time.monotonic_ns()
            self.driver._on_feedback(p)

        with patch.object(self.driver._stop, "wait", side_effect=transitioning):
            with self.assertRaisesRegex(RuntimeError, "position mode was not reported"):
                self.driver.move_joints("left", (0,)*7)
        self.assertFalse(any(name == "submit" for name, _ in self.native.calls))
        self.assertTrue(any(name == "hold" for name, _ in self.native.calls))
        self.assertFalse(self.events("tianji.joint_move_completed"))

    def test_position_transition_preserves_controller_fault_before_or_after_watchdog(self):
        for watchdog_first in (False, True):
            with self.subTest(watchdog_first=watchdog_first):
                fixture = TianjiFixture()
                driver, native = fixture.driver, fixture.native
                self.addCleanup(driver.close)
                fixture.configure()
                native.complete_joint_move = False

                def fault(_):
                    p = deepcopy(driver._packet)
                    p.packet_index += 1
                    p.sequence[0] += 1
                    p.received_ns = time.monotonic_ns()
                    p.state[0], p.error[0] = 100, 4
                    driver._on_feedback(p)
                    if watchdog_first:
                        with patch.object(driver._stop, "wait", side_effect=(False, True)):
                            driver._watch()
                        self.assertFalse(driver.engaged)

                with patch.object(driver._stop, "wait", side_effect=fault):
                    with self.assertRaisesRegex(RuntimeError, "left controller error 4") as raised:
                        driver.move_joints("left", (0.,) * 7)
                self.assertEqual(driver.motion_stop["reason"], str(raised.exception))
                self.assertFalse(any(name == "submit" for name, _ in native.calls))
                self.assertEqual([args[0] for name, args in native.calls if name == "hold"], [1])
                self.assertEqual(driver.get_latest().payload.arms["right"].error, 0)

    def test_joint_move_waits_for_final_internal_target_and_low_speed_without_stream_expiry(self):
        self.configure()
        self.position_feedback()
        self.native.complete_joint_move = False
        self.driver._deadline_ns = 0
        final_target = [35, -55, 0, -65, 0, 0, 0]
        steps = []

        def progress(_):
            p = deepcopy(self.driver._packet)
            p.sequence[0] += 1
            p.received_ns = time.monotonic_ns()
            p.state[0] = 1
            p.dq[:7] = [100]*7
            p.low_speed[0] = 0 if len(steps) == 1 else 1
            p.target[:7] = [0]*7 if not steps else final_target
            p.q[:7] = final_target
            self.driver._on_feedback(p)
            steps.append(p)

        with patch.object(self.driver._stop, "wait", side_effect=progress):
            result = self.driver.move_joints("left", tuple(map(math.radians, final_target)))
        self.assertEqual(len(steps), 3)
        self.assertEqual(result.payload.arms["left"].low_speed, 1)
        self.assertFalse(any(name == "hold" for name, _ in self.native.calls))




    def test_interrupted_native_joint_move_stops_only_moving_arm_with_position_fallback(self):
        self.configure(("left", "right"))
        self.native.stop_nack = True
        native_call = self.native.call

        def interrupt(name, *args):
            result = native_call(name, *args)
            if name == "move_joints" and len([1 for n, _ in self.native.calls if n == name]) == 1:
                raise KeyboardInterrupt
            return result

        with patch.object(self.native, "call", side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
            self.driver.move_joints("right", (0,)*7)
        self.assertEqual([args[0] for name, args in self.native.calls if name == "hold"], [2])
        fallback = [args for name, args in self.native.calls if name == "move_joints"][-1]
        self.assertEqual((fallback[0], fallback[2], fallback[3]), (1, 100, 100))
        for actual, expected in zip(fallback[1], self.driver._packet.q[7:]):
            self.assertAlmostEqual(actual, expected)
        self.assertIsNone(self.driver._hold_reason)
        self.assertFalse(self.driver.engaged)

    def test_interrupting_final_target_stops_after_successful_mode_transition(self):
        self.configure(("left", "right"))
        native_call = self.native.call

        def interrupt(name, *args):
            result = native_call(name, *args)
            if name == "submit":
                raise KeyboardInterrupt
            return result

        with patch.object(self.native, "call", side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
            self.driver.move_joints("right", (0,)*7)
        self.assertEqual([args[0] for name, args in self.native.calls if name == "hold"], [2])
        self.assertFalse(self.events("tianji.joint_move_completed"))
        self.assertFalse(self.driver.engaged)

    def test_native_joint_move_rejects_nonfinite_and_real_controller_fault(self):
        self.configure()
        for target in ((0,)*6, (float("nan"),)*7):
            with self.assertRaises(ValueError):
                self.driver.move_joints("left", target)
        self.driver._packet.error[0] = 13
        self.driver._on_feedback(self.driver._packet)
        with self.assertRaisesRegex(RuntimeError, "controller error 13"):
            self.driver.move_joints("left", (0,)*7)
        self.assertFalse(any(name == "move_joints" for name, _ in self.native.calls))

    def test_observer_rejection_latches_fault_and_blocks_next_target(self):
        self.engage()
        self.sink.accept = False
        self.sink.try_event = lambda _: False
        self.driver._on_feedback(packet(1))
        self.assertFalse(self.driver.health().ready)
        self.assertIn("observer_error", self.driver.metadata)
        result = self.driver.submit(self.command())
        self.assertFalse(result.accepted)
        self.assertFalse(any(name == "submit" for name, _ in self.native.calls))

    def test_position_to_cartesian_does_not_require_stationary_receipt(self):
        self.configure()
        self.driver.move_joints("left", self.driver.get_latest().payload.arms["left"].joints.position_rad)
        p = deepcopy(self.driver._packet)
        p.sequence[0] += 1
        p.received_ns = time.monotonic_ns()
        p.low_speed[0], p.dq[0] = 0, 2
        self.driver._on_feedback(p)
        self.driver.engage()
        self.assertEqual(self.driver._control_mode, "cartesian")
        self.assertTrue(self.driver.health().ready)


    def feed_latest_until(self, stop):
        while not stop.wait(.005):
            p = deepcopy(self.driver._packet)
            p.sequence[:] = [s+1 for s in p.sequence]
            p.received_ns = time.monotonic_ns()
            self.driver._on_feedback(p)

    def clear_with_feedback(self, **kwargs):
        stop = threading.Event()
        worker = threading.Thread(target=self.feed_latest_until, args=(stop,))
        worker.start()
        try:
            return self.driver.clear_errors(**kwargs)
        finally:
            stop.set()
            worker.join()

    def test_clear_controller_errors_including_emergency_without_enabling(self):
        self.set_emergency_feedback((2, 13))
        result = self.clear_with_feedback()
        self.assertEqual(result["requested_arms"], ["left", "right"])
        self.assertEqual([a[0] for n, a in self.native.calls if n == "clear_errors"], [0, 1])
        self.assertFalse(self.driver.engaged)
        self.assertIsNone(self.driver.profile)
        self.assertFalse(any(n in ("engage", "move_joints", "submit", "configure")
                             for n, _ in self.native.calls))

    def test_clear_servo_only_fault_and_skip_healthy_arm(self):
        self.set_emergency_feedback((0, 0))
        self.native.servo_errors["SERVO1ERR3"] = 42
        result = self.clear_with_feedback()
        self.assertEqual(result["requested_arms"], ["right"])
        self.assertEqual(result["initial_errors"]["right"]["servos"][3], 42)
        self.assertFalse(any(result["final_errors"]["right"]["servos"]))

    def test_clear_no_fault_is_noop_and_single_side_ignores_other_arm(self):
        self.set_emergency_feedback((0, 13))
        result = self.driver.clear_errors(("left",))
        self.assertEqual(result["requested_arms"], [])
        self.assertFalse(self.native.calls)
        self.assertEqual(self.driver.get_latest().payload.arms["right"].error, 13)

    def test_clear_rejects_invalid_stale_or_moving_feedback_before_reset(self):
        for problem in ("invalid", "stale", "moving", "engaged"):
            with self.subTest(problem=problem):
                self.set_emergency_feedback()
                if problem == "invalid":
                    p = deepcopy(self.driver._packet)
                    p.dq[0] = float("nan")
                    p.sequence[:] = [s+1 for s in p.sequence]
                    self.driver._on_feedback(p)
                elif problem == "stale":
                    self.driver._advanced["left"] = 0
                elif problem == "moving":
                    p = deepcopy(self.driver._packet)
                    p.low_speed[0] = 0
                    p.sequence[:] = [s+1 for s in p.sequence]
                    self.driver._on_feedback(p)
                else:
                    self.driver.engaged = True
                before = len(self.native.calls)
                try:
                    with self.assertRaises(RuntimeError):
                        self.driver.clear_errors()
                finally:
                    self.driver.engaged = False
                self.assertFalse(any(n == "clear_errors" for n, _ in self.native.calls[before:]))

    def test_clear_rejection_and_residual_error_are_not_retried(self):
        self.set_emergency_feedback()
        self.native.reset_rejected = True
        with self.assertRaisesRegex(RuntimeError, "returned 1"):
            self.driver.clear_errors()
        self.assertEqual(len([n for n, _ in self.native.calls if n == "clear_errors"]), 1)
        self.native.calls.clear()
        self.native.reset_rejected = False
        self.native.reset_without_state_change = True
        with self.assertRaisesRegex(RuntimeError, "errors remain"):
            self.clear_with_feedback()
        self.assertEqual(len([n for n, _ in self.native.calls if n == "clear_errors"]), 2)
        self.assertFalse(self.driver.engaged)

    def test_clear_query_failure_is_not_treated_as_zero(self):
        self.set_emergency_feedback()
        with patch.object(self.native, "get_int", side_effect=RuntimeError("query timeout")):
            with self.assertRaisesRegex(RuntimeError, "left J1.*query timeout"):
                self.driver.clear_errors()
        self.assertFalse(self.native.calls)

    def test_clear_does_not_discard_pending_stop_in_motion_mode(self):
        self.configure()
        self.set_emergency_feedback((0, 0))
        value = deepcopy(self.driver._packet)
        value.state[0] = 1
        value.sequence[:] = [s+1 for s in value.sequence]
        self.driver._on_feedback(value)
        self.driver._hold_reason = "pending stop"
        self.driver._hold_sides = ("left",)
        with self.assertRaisesRegex(RuntimeError, "stop is still pending"):
            self.driver.clear_errors(("left",))
        self.assertEqual(self.driver._hold_reason, "pending stop")
        self.assertFalse(any(n == "clear_errors" for n, _ in self.native.calls))

    def test_clear_ack_requires_new_feedback_and_supports_cancellation(self):
        self.set_emergency_feedback()
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            self.driver.clear_errors(timeout_s=.01)
        self.assertFalse(self.driver.engaged)
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            self.driver.clear_errors(cancel=cancelled)

    def test_feedback_units_partial_invalid_and_no_derived_or_fixed_fields(self):
        p = packet(8)
        p.dq[0], p.current[0], p.torque[0], p.external_torque[0] = 180, 240, 2, 3
        p.target[0], p.q[8] = 90, float("nan")
        sample = decode_feedback(p, "run")
        arm = sample.payload.arms["left"]
        self.assertAlmostEqual(arm.joints.velocity_rad_s[0], math.pi)
        self.assertEqual(arm.native_current_permille[0], 240)
        self.assertIsNone(arm.joints.motor_current_a)
        self.assertEqual(arm.joints.measured_torque_nm[0], 2)
        self.assertEqual(arm.joints.estimated_external_torque_nm[0], 3)
        self.assertAlmostEqual(arm.controller_target_rad[0], math.pi/2)
        self.assertIsNone(sample.payload.arms["right"].joints.position_rad[1])
        self.assertIsNone(sample.header.source_time)
        self.assertFalse(sample.header.valid)
        encoded = json.dumps(asdict(sample), allow_nan=False)
        self.assertNotIn("external_wrench", encoded)
        self.assertNotIn("stiffness", encoded)
        self.driver._on_feedback(p)
        self.assertEqual(len(self.events("tianji.invalid_feedback")), 1)

    def test_every_packet_published_latest_does_not_publish(self):
        self.native.feedback.extend([packet(i) for i in range(1, 7)])
        while self.driver._drain():
            pass
        feedback = [s for s in self.sink.samples if s.header.ref.stream == "tianji.feedback"]
        self.assertEqual([s.payload.packet_index for s in feedback], list(range(7)))
        before = len(self.sink.samples)
        a, b = self.driver.get_latest(), self.driver.get_latest()
        self.assertIs(a, b)
        self.assertEqual(len(self.sink.samples), before)
        self.assertEqual(len(self.events("tianji.reported_config")), 2)

    def test_independent_source_gap_wrap_stall_and_reset(self):
        self.driver._sequences = {"left": 999999, "right": 10}
        now = time.monotonic_ns()
        self.driver._advanced = {"left": now, "right": now - 1_000_000_000}
        self.driver._on_feedback(packet(1, (0, 10), now))
        self.assertFalse(self.driver.health().ready)
        self.assertIn("right", self.driver.health().detail)
        self.assertFalse(self.events("tianji.source_gap"))
        self.driver._on_feedback(packet(2, (3, 11)))
        self.assertEqual(self.events("tianji.source_gap")[0].details["missing"], 2)
        advanced = self.driver._advanced["left"]
        self.driver._on_feedback(packet(3, (1, 12)))
        self.assertEqual(self.driver._advanced["left"], advanced)
        self.assertIn("re-engagement", self.driver._motion_fault)
        self.assertEqual(len(self.events("tianji.source_reset")), 1)


    def test_engage_requires_applied_profile_without_idle_gate(self):
        with self.assertRaisesRegex(RuntimeError, "profile"):
            self.driver.engage()
        self.driver._packet.state[0] = 3
        self.driver._on_feedback(self.driver._packet)
        self.configure()
        self.driver.engage()
        self.assertTrue(self.driver.engaged)

    def test_configure_accepts_compatible_controller_version_but_requires_echo(self):
        self.native.version = (100343014, 100344001)
        self.configure()
        self.assertEqual([name for name, _ in self.native.calls], ["configure"])
        self.assertIsNotNone(self.driver.profile)


    def test_requested_configuration_is_recorded(self):
        profile = self.make_profile()
        self.driver.configure(profile)
        event = self.events("tianji.configure_requested")[0]
        self.assertEqual(event.details["profile"], asdict(profile))
        self.assertEqual(self.driver.metadata["control_profile"], asdict(profile))
        self.assertTrue(self.events("tianji.configure_sdk_returned"))

    def test_verified_empty_load_reaches_native_configuration_and_readback(self):
        profile = self.make_profile(("left", "right"))
        for arm in profile.parameters["arms"].values():
            arm["tool_dyn10"] = [0]*10
        self.driver.configure(profile)
        native_profiles = next(args[1] for name, args in self.native.calls if name == "configure")
        for index, side in enumerate(("left", "right")):
            self.assertEqual(tuple(native_profiles[index].tool_dyn10), (0.0,)*10)
            self.assertEqual(self.driver.profile.arms[side].tool_dyn10, (0.0,)*10)
        self.assertTrue(self.events("tianji.configure_sdk_returned"))
        self.assertFalse(self.driver.engaged)

    def test_measured_group_seed_exact_sample_and_acceptance_not_sent(self):
        seed = self.engage(("left", "right"))
        self.assertIs(self.driver.engagement_sample, seed)
        call = next(args for name, args in self.native.calls if name == "engage")
        self.assertEqual(call[0], 3)
        for actual, expected in zip(call[1], [12, -22, 31, -48, 17, 15, -12]*2):
            self.assertAlmostEqual(actual, expected)
        self.assertTrue(self.driver.engaged)
        self.assertTrue(any(isinstance(e, CommandEvent) and e.status == CommandStatus.ACCEPTED for e in self.sink.events))
        self.assertFalse(any(isinstance(e, CommandEvent) and e.command_id.startswith("engage-")
                             and e.status == CommandStatus.SENT for e in self.sink.events))

    def test_expired_target_not_native_accepted(self):
        self.engage()
        result = self.driver.submit(self.command(targets={"left": (1.,)*7},
                                                 expires=time.monotonic_ns()-1))
        self.assertFalse(result.accepted)
        # The expired command is rejected; the only submitted target is a fresh hold.
        hold, = [args for name, args in self.native.calls if name == "submit"]
        for actual, expected in zip(hold[1][:7], self.driver._packet.q[:7]):
            self.assertAlmostEqual(actual, expected)
        self.assertTrue(self.driver._commands[self.driver._token].startswith("hold-"))

    def test_cartesian_command_has_no_host_joint_speed_cap(self):
        self.engage()
        q = list(self.driver.get_latest().payload.arms["left"].joints.position_rad)
        q[0] += .2
        result = self.driver.submit(self.command(targets={"left": tuple(q)}))
        self.assertTrue(result.accepted, result.reason)
        self.assertTrue(any(name == "submit" for name, _ in self.native.calls))
        self.assertFalse(any(name == "hold" for name, _ in self.native.calls))

    def test_cartesian_command_rejects_nonfinite_joints(self):
        self.engage()
        q = list(self.driver.get_latest().payload.arms["left"].joints.position_rad)
        q[0] = math.nan
        result = self.driver.submit(self.command(targets={"left": tuple(q)}))
        self.assertFalse(result.accepted)
        self.assertIn("finite", result.reason)


    def test_command_payload_snapshot_and_sdk_submission_deadline(self):
        self.engage()
        targets = {"left": list(self.driver.get_latest().payload.arms["left"].joints.position_rad)}
        command = self.command(targets=targets, expires=time.monotonic_ns()+5_000_000_000)
        self.assertTrue(self.driver.submit(command).accepted)
        call = [args for name, args in self.native.calls if name == "submit"][-1]
        expected = tuple(math.degrees(q) for q in targets["left"])
        targets["left"][:] = (99,)*7
        self.assertEqual(tuple(call[1][:7]), expected)
        self.assertLess(call[2], command.expires_monotonic_ns)

    def test_hold_records_sdk_return_without_blocking_resume_on_stationary_receipt(self):
        self.engage()
        self.driver.request_hold("test pause")
        self.assertFalse(self.driver._held_feedback)
        self.assertFalse(self.events("tianji.hold_sdk_returned")[-1].details["physical_stop_confirmed"])
        self.driver.engage()
        self.assertTrue(self.driver.engaged)

    def test_cartesian_pause_replaces_target_without_rsta_and_allows_resume(self):
        self.configure(("left", "right"))
        self.driver.engage()
        self.native.stop_nack = True
        p = deepcopy(self.driver._packet)
        p.packet_index += 1
        p.sequence[:] = [s + 1 for s in p.sequence]
        p.received_ns = time.monotonic_ns()
        p.q[0] += 2.
        p.target[0] = p.q[0] + 10.
        self.driver._on_feedback(p)
        self.native.calls.clear()

        self.driver.request_hold("rock gesture")

        self.assertFalse(self.driver.engaged)
        self.assertTrue(self.driver.health().ready, self.driver.health().detail)
        self.assertEqual([name for name, _ in self.native.calls], ["submit"])
        mask, target, expiry = self.native.calls[0][1]
        token = self.driver._token
        self.assertEqual(mask, 3)
        for actual, expected in zip(target, p.q):
            self.assertAlmostEqual(actual, expected)
        self.assertTrue(any(isinstance(e, CommandEvent) and e.command_id == self.driver._commands[token]
                            and e.status == CommandStatus.SDK_SUBMITTED for e in self.sink.events))
        self.assertLessEqual(expiry - p.received_ns, self.driver.watchdog_ns + 10_000_000)
        self.driver.engage()
        self.assertTrue(self.driver.engaged)

    def test_sdk_rejected_cartesian_hold_stays_pending(self):
        self.engage()
        self.native.send_submit = False
        self.driver.watchdog_ns = 10_000_000
        self.driver.request_hold("rock gesture")
        self.assertFalse(self.driver.health().ready)
        self.assertEqual(self.driver._hold_reason, "rock gesture")
        self.assertFalse(self.events("tianji.hold_sdk_returned"))
        self.assertIn("SDK submit rejected", self.driver.health().detail)

    def test_cartesian_hold_uses_only_fresh_selected_arm_feedback(self):
        self.engage()
        p = deepcopy(self.driver._packet)
        p.sequence[0] += 1
        p.received_ns = time.monotonic_ns()
        p.error[1], p.q[7] = 13, math.nan
        self.driver._on_feedback(p)
        self.native.calls.clear()
        self.driver.request_hold("pause left")
        self.assertTrue(self.driver.health().ready)
        hold, = [args for name, args in self.native.calls if name == "submit"]
        self.assertEqual(hold[0], 1)
        self.assertEqual(tuple(hold[1][7:]), (0.,) * 7)

    def test_stale_feedback_cannot_be_used_as_cartesian_hold_target(self):
        self.engage()
        self.driver._advanced["left"] -= self.driver.watchdog_ns
        self.native.calls.clear()
        self.driver.request_hold("feedback lost")
        self.assertEqual([name for name, _ in self.native.calls], ["hold"])
        self.assertFalse(self.driver.health().ready)

    def test_engage_missing_mode_echo_times_out_and_holds(self):
        self.configure()
        self.native.echo = False
        self.driver.engagement_timeout_ns = 20_000_000
        with self.assertRaisesRegex(RuntimeError, "confirm.*engagement"):
            self.driver.engage()
        self.assertFalse(self.driver.engaged)
        self.assertTrue(any(name == "hold" for name, _ in self.native.calls))

    def test_startup_grace_accepts_delayed_mode_with_fresh_feedback(self):
        self.configure()
        self.native.echo = False
        self.driver.watchdog_ns = 30_000_000
        self.driver.engagement_timeout_ns = 500_000_000
        published = threading.Event()
        finish = threading.Event()

        def feedback():
            while not finish.is_set():
                if any(name == "engage" for name, _ in self.native.calls):
                    break
                finish.wait(.001)
            ready_at = time.monotonic() + .10
            while not finish.is_set():
                p = deepcopy(self.driver._packet)
                p.received_ns, p.packet_index = time.monotonic_ns(), p.packet_index + 1
                p.sequence[0] += 1
                p.state[0] = 3 if time.monotonic() >= ready_at else 103
                p.impedance_type[0] = 2
                self.driver._on_feedback(p)
                published.set()
                finish.wait(.002)

        worker = threading.Thread(target=feedback)
        worker.start()
        self.driver._watchdog = threading.Thread(target=self.driver._watch)
        self.driver._watchdog.start()
        try:
            self.driver.engage()
            self.assertTrue(published.is_set())
            self.assertTrue(self.driver.engaged)
            report = self.events("tianji.engagement_mode_reported")[-1]
            self.assertGreater(report.details["elapsed_ms"], 30)
            self.assertGreater(self.driver._deadline_ns - report.observed_monotonic_ns,
                               self.driver.watchdog_ns)
            self.assertLessEqual(self.driver._deadline_ns - report.observed_monotonic_ns,
                                 self.driver.engagement_timeout_ns)
            engage_call = next(args for name, args in self.native.calls if name == "engage")
            self.assertLessEqual(engage_call[2] - report.observed_monotonic_ns, 40_000_000)
            # Fresh feedback alone does not extend the first-target grace period.
            deadline = time.monotonic() + .7
            while self.driver.engaged and time.monotonic() < deadline:
                time.sleep(.002)
            self.assertFalse(self.driver.engaged)
            self.assertTrue(self.events("tianji.hold_requested"))
        finally:
            finish.set()
            worker.join()

    def test_startup_grace_does_not_allow_stale_feedback(self):
        self.configure()
        self.native.echo = False
        self.driver.watchdog_ns = 20_000_000
        self.driver.engagement_timeout_ns = 500_000_000
        self.driver._watchdog = threading.Thread(target=self.driver._watch)
        self.driver._watchdog.start()
        started = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "confirm.*engagement"):
            self.driver.engage()
        self.assertLess(time.monotonic() - started, .4)
        self.assertIn("source counter", self.events("tianji.hold_requested")[0].details["reason"])

    def test_transition_low_speed_flag_cannot_confirm_hold(self):
        self.engage()
        self.driver.request_hold("test pause during transition")
        for state in (103, 3, 103):
            p = deepcopy(self.driver._packet)
            p.received_ns, p.packet_index = time.monotonic_ns(), p.packet_index + 1
            p.sequence[0] += 1
            p.state[0], p.low_speed[0] = state, 1
            self.driver._on_feedback(p)
            self.assertEqual(self.driver._held_feedback, state == 3)
        self.assertEqual(len(self.events("tianji.hold_low_speed_observed")), 1)

    def test_reported_configuration_change_is_recorded_without_stopping(self):
        self.engage()
        self.sink.events.clear()
        p = deepcopy(self.driver._packet)
        p.received_ns, p.packet_index = time.monotonic_ns(), p.packet_index+1
        p.sequence[0] += 1
        p.cart_k[0] += 1
        self.driver._on_feedback(p)
        result = self.driver.submit(self.command())
        self.assertTrue(result.accepted)
        self.assertTrue(self.driver.engaged)
        event, = self.events("tianji.reported_config")
        self.assertEqual(event.details["side"], "left")
        self.assertEqual(event.details["packet_index"], p.packet_index)
        self.assertEqual(event.details["configuration"]["stiffness_native"][0], p.cart_k[0])

    def test_invalid_measured_velocity_prevents_another_target(self):
        self.engage()
        p = deepcopy(self.driver._packet)
        p.received_ns, p.packet_index = time.monotonic_ns(), p.packet_index + 1
        p.sequence[0] += 1
        p.dq[0] = float("nan")
        self.driver._on_feedback(p)
        result = self.driver.submit(self.command())
        self.assertFalse(result.accepted)
        self.assertIn("left invalid dq at J1", result.reason)
        self.assertFalse(self.driver.engaged)

    def test_hold_pending_is_retried_without_touching_other_arm(self):
        self.configure()
        self.driver.move_joints("left", self.driver.get_latest().payload.arms["left"].joints.position_rad)
        # Simulate a position move awaiting arrival when cancellation occurs.
        self.driver.engaged = True
        self.driver._moving_side = "left"
        self.native.hold_failures = 1
        self.driver.request_hold("test")
        self.assertIsNotNone(self.driver._hold_reason)
        self.driver.request_hold("test")
        self.assertIsNone(self.driver._hold_reason)
        self.assertEqual([args[0] for name, args in self.native.calls if name == "hold"], [1, 1])

    def test_close_disables_owned_arm_without_pause_before_native_close(self):
        self.engage()
        self.driver.watchdog_ns = 10_000_000
        self.driver.close()
        names = [name for name, args in self.native.calls]
        self.assertNotIn("hold", names)
        self.assertLess(names.index("disable"), names.index("close"))
        self.assertEqual([args[0] for name, args in self.native.calls if name == "disable"], [1])
        self.assertFalse(self.driver.engaged)
        self.assertTrue(self.events("tianji.disable_confirmed"))

    def test_worker_watchdog_sends_measured_hold_without_a_new_operator_target(self):
        self.engage()
        self.driver._deadline_ns = time.monotonic_ns() - 1
        self.driver._watchdog = threading.Thread(target=self.driver._watch)
        self.driver._watchdog.start()
        deadline = time.monotonic() + .2
        while self.driver.engaged and time.monotonic() < deadline:
            time.sleep(.001)
        # request_hold clears engaged before submitting the measured hold.
        # Wait for its critical section to finish before inspecting SDK calls.
        with self.driver._lock:
            self.assertFalse(self.driver.engaged)
            self.assertEqual([args[0] for name, args in self.native.calls if name == "submit"], [1])
            self.assertFalse(any(name == "hold" for name, _ in self.native.calls))




    def test_start_is_capture_only_and_worker_drains_packets(self):
        driver = TianjiDriver("192.0.2.2")
        native = FakeSDK(driver)
        native.feedback.extend(packet(i) for i in range(4))
        sink = Sink()
        with patch("bimanual_teleop.devices.tianji.driver.ControlSDK", return_value=native):
            driver.start(sink)
        deadline = time.monotonic() + .5
        while len(sink.samples) < 4 and time.monotonic() < deadline:
            time.sleep(.001)
        driver.close()
        self.assertEqual(len(sink.samples), 4)
        self.assertEqual([name for name, _ in native.calls], ["open", "close"])

    def test_failed_open_does_not_close_some_other_owner(self):
        driver = TianjiDriver("192.0.2.2")
        native = FakeSDK(driver)
        native.fail_open = True
        with patch("bimanual_teleop.devices.tianji.driver.ControlSDK", return_value=native):
            with self.assertRaisesRegex(RuntimeError, "owned"):
                driver.start()
        self.assertEqual([name for name, _ in native.calls], ["open"])


if __name__ == "__main__":
    unittest.main()
