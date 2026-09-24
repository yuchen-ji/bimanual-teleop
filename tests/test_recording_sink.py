"""Device-to-recorder contracts without opening SDK or camera connections."""

from dataclasses import replace
import math
from queue import Empty, Queue
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from bimanual_teleop.devices.tianji.driver import TianjiDriver, TianjiJointCommand, decode_feedback
from bimanual_teleop.devices.wuji.adapter import JOINT_NAMES, WujiHandDriver
from bimanual_teleop.recording.sink import CaptureChannel, Record, RecorderSink, STATE_STREAMS
from bimanual_teleop.types import (
    CommandEvent, CommandStatus, DeviceCommand, Event, JointState, JointTarget, Pose,
    Sample, SampleHeader, SampleRef,
)
from tests.support.tianji import packet
from tests.test_wuji_driver import feedback


class _LocalContext:
    """Use deterministic queues while retaining the production channel behavior."""

    Queue = staticmethod(Queue)
    Event = staticmethod(threading.Event)

    @staticmethod
    def Value(_kind, value, **_kwargs):
        return SimpleNamespace(value=value)

    @staticmethod
    def Array(_kind, size, **_kwargs):
        return [0] * size


def _channel(capacity=128):
    channel = CaptureChannel(_LocalContext(), capacity)
    channel.active.value = True
    channel.generation.value = 7
    return channel


def _drain(channel):
    records = []
    while True:
        try:
            generation, record = channel.queue.get_nowait()
        except Empty:
            return records
        if generation != 7:
            raise AssertionError(f"wrong recording generation: {generation}")
        records.append(record)


def _arm_packet(index, sequences, stamp):
    value = packet(index, sequences, stamp)
    value.force_tag[:] = [116, 216]
    value.wrench_raw[:] = [10000., -20000., 30000., 4000., -5000., 6000.] * 2
    return value


class RecordingSinkTests(unittest.TestCase):
    def test_arm_source_deduplication_and_200hz_sampling_do_not_change_device_io(self):
        channel = _channel()
        sink = RecorderSink(channel, state_hz=200.)
        driver = TianjiDriver("192.0.2.1")
        driver._sink, driver._sdk = sink, Mock()
        start = 1_000_000_000
        for index in range(11):
            value = _arm_packet(index, (index + 1, 20 + index // 2), start + index * 1_000_000)
            value.q[:7] = [90. + index] * 7
            value.q[7:] = [-180. + index // 2] * 7
            driver._on_feedback(value)
        # A fresh host read of an unchanged SDK snapshot is not a new measurement.
        driver._on_feedback(_arm_packet(12, (11, 25), start + 100_000_000))
        records = _drain(channel)
        left = [r for r in records if r.stream == "arms/left"]
        right = [r for r in records if r.stream == "arms/right"]
        self.assertEqual([r.time_ns - start for r in left], [0, 5_000_000, 10_000_000])
        self.assertEqual([r.sequence for r in left], [1, 6, 11])
        self.assertEqual([r.time_ns - start for r in right], [0, 6_000_000, 10_000_000])
        self.assertEqual([r.sequence for r in right], [20, 23, 25])
        self.assertEqual(left[0].values["joint_pos"], (math.pi / 2,) * 7)
        self.assertEqual(right[0].values["joint_pos"], (-math.pi,) * 7)
        self.assertEqual(left[0].values["wrench"], (1., -2., 3., .4, -.5, .6))
        self.assertEqual(channel.latest_ns[:2], [start + 10_000_000] * 2)
        self.assertEqual(driver._sdk.mock_calls, [])
        self.assertIsNone(driver._observer_error)
        self.assertFalse(channel.failed.is_set())

    def test_real_hand_stream_keeps_source_sequence_and_ignores_duplicates_and_other_streams(self):
        channel = _channel()
        sink = RecorderSink(channel)
        hand = WujiHandDriver("left", "fake")
        hand._sink = sink
        hand._consume("joints", feedback(seq=91), 1_000_000_000)
        hand._consume("joints", feedback(seq=91), 1_006_000_000)
        hand._consume("joints", feedback(seq=92), 1_007_000_000)
        expected_positions = tuple(.1 + index * .001 for index in range(20))
        for stream in ("wuji_left_glove/angles", "wuji_left_hand/diagnostics", "other_left/joints"):
            self.assertTrue(sink.try_publish(Sample(
                SampleHeader(SampleRef(stream, "other", 1), 1_020_000_000, True),
                JointState(JOINT_NAMES, expected_positions))))
        records = _drain(channel)
        self.assertEqual([r.stream for r in records], ["hands/left"] * 2)
        self.assertEqual([r.sequence for r in records], [91, 92])
        self.assertEqual([r.time_ns for r in records], [1_000_000_000, 1_007_000_000])
        self.assertEqual(records[0].values, {"joint_pos": expected_positions})
        self.assertEqual(channel.latest_ns[STATE_STREAMS.index("hands/left")], 1_007_000_000)
        self.assertIsNone(hand.manager)
        self.assertIsNone(hand._device)
        self.assertIsNone(hand._observer_error)

    def test_only_accepted_hand_targets_and_original_desired_arm_pose_are_recorded(self):
        channel = _channel()
        sink = RecorderSink(channel)
        hands = {}
        for side in ("left", "right"):
            hand = WujiHandDriver(side, "fake", clock=lambda: 1_005_000_000)
            hand._sink = sink
            hand.sdk = SimpleNamespace(JointCommand=lambda q, dq, effort: (q, dq, effort))
            hand._publisher = Mock()
            hands[side] = hand
        left_command = DeviceCommand("wuji_left_hand", "left-target",
            JointTarget(JOINT_NAMES, (.2,) * 20), (), 1_000_000_000, 1_050_000_000, "test")
        right_command = replace(left_command, device_id="wuji_right_hand", command_id="right-rejected")
        hands["left"]._send(left_command)
        hands["right"]._publisher.send.side_effect = RuntimeError("SDK rejected target")
        with self.assertRaisesRegex(RuntimeError, "SDK rejected"):
            hands["right"]._send(right_command)
        sink.try_event(CommandEvent("wuji_right_hand", "unmatched", CommandStatus.ACCEPTED, 1_006_000_000))
        # Duplicate acknowledgements cannot create another training action.
        sink.try_event(CommandEvent("wuji_left_hand", "left-target", CommandStatus.ACCEPTED, 1_006_000_000))
        desired = Pose("tianji_left_base", "tianji_left_flange", (.3, .2, .1), (0., 0., 0., 1.))
        limited = replace(desired, position_m=(.01, .02, .03))
        arm_command = DeviceCommand("tianji", "arm-target",
            TianjiJointCommand({"left": (.1,) * 7}, {"left": limited}, {"left": desired}),
            (), 1_000_000_000, 1_050_000_000, "test")
        sink.try_event(Event("tianji_command_submitted", 1_008_000_000, "tianji", {"command": arm_command}))
        records = _drain(channel)
        records = {record.stream: record for record in records}
        self.assertEqual(set(records), {"hand_commands/left", "arm_commands/left"})
        self.assertEqual(records["hand_commands/left"].time_ns, 1_005_000_000)
        self.assertEqual(records["arm_commands/left"].time_ns, 1_008_000_000)
        self.assertEqual(records["hand_commands/left"].values, {"joint_pos": (.2,) * 20})
        self.assertEqual(records["arm_commands/left"].values["eef_pose"],
                         (.3, .2, .1, 0., 0., 0., 1.))
        self.assertEqual(records["arm_commands/left"].values["joint_pos"], (.1,) * 7)
        self.assertFalse(channel.failed.is_set())

    def test_full_queue_marks_recording_failed_without_poisoning_driver_observer(self):
        channel = _channel(capacity=1)
        driver = TianjiDriver("192.0.2.1")
        driver._sink = RecorderSink(channel)
        driver._on_feedback(_arm_packet(1, (1, 1), 1_000_000_000))
        driver._on_feedback(_arm_packet(2, (2, 2), 1_006_000_000))
        self.assertTrue(channel.failed.is_set())
        self.assertIn("Full", channel.errors.get_nowait())
        self.assertIsNone(driver._observer_error)
        self.assertIsNone(driver._problem)
        self.assertTrue(driver.get_latest().header.valid)
        self.assertEqual(len(_drain(channel)), 2)

    def test_shared_rings_never_overwrite_unread_records(self):
        channel = _channel(capacity=2)
        sink = RecorderSink(channel, state_hz=1000.)
        for index in range(3):
            record = Record("hands/left", 1_000_000_000 + index * 1_000_000,
                            index + 10, {"joint_pos": (float(index),) * 20})
            sink._put(record)
        self.assertTrue(channel.failed.is_set())
        records = _drain(channel)
        self.assertEqual([record.sequence for record in records], [10, 11])
        self.assertEqual(records[0].values["joint_pos"], (0.,) * 20)
        self.assertEqual(records[1].values["joint_pos"], (1.,) * 20)

    def test_invalid_force_or_closed_channels_do_not_raise_from_observer(self):
        for failure in (None, OSError("closed pipe"), ValueError("closed queue")):
            with self.subTest(failure=failure):
                channel = _channel()
                sink = RecorderSink(channel)
                value = _arm_packet(1, (1, 1), 1_000_000_000)
                if failure is None:
                    value.wrench_raw[0] = math.nan
                else:
                    channel.put_nowait = Mock(side_effect=failure)
                    channel.errors = Mock()
                    channel.errors.put_nowait.side_effect = failure
                self.assertTrue(sink.try_publish(decode_feedback(value, "test")))
                self.assertTrue(channel.failed.is_set())
                # Later event delivery remains nonblocking and successful to the driver.
                self.assertTrue(sink.try_event(Event("irrelevant", 1_001_000_000, "test", {})))


if __name__ == "__main__":
    unittest.main()
