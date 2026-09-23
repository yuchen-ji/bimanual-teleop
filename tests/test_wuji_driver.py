"""SDK-shaped fakes exercise Wuji acquisition and activation without devices."""

from collections import deque
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bimanual_teleop.devices.wuji.adapter import (
    JOINT_NAMES, JOINT_LIMITS_RAD, SKELETON_NAMES, WujiGloveSource, WujiHandDriver, WujiSdkSession,
    WujiHandAngles, WujiTactileFrame, nid_to_index,
)
from bimanual_teleop.types import (
    CommandEvent, CommandStatus, ControlProfile, DeviceCommand, JointTarget,
)


def header(seq=0, timestamp=123):
    return NS(seq=seq, timestamp_us=timestamp, frame_id="l_wrist")


def skeleton(seq=0, *, valid=True):
    return NS(header=header(seq), joints=[NS(name=SKELETON_NAMES[i],
        pose=NS(position=[i * .001, 0., 0.]), confidence=.8)
        for i in range(21 if valid else 20)])


def emf(seq=0):
    return NS(header=header(seq), poses=[NS(pose=NS(position=[0., 0., 0.],
        orientation=NS(x=0., y=0., z=0., w=1.)), confidence=.9) for _ in range(5)])


def angles(seq=0, *, bad=False, padded=True):
    fingers = [NS(angles=[.1] * n + ([0.] * (5 - n) if padded else []),
                  confidence=.9) for n in (5, 4, 4, 4, 4)]
    if bad:
        fingers[0].angles[0] = float("nan")
    return NS(header=header(seq), fingers=fingers)


def tactile(seq=0, count=744):
    return NS(header=header(seq), data=[float(i) for i in range(count)])


def feedback(seq=0):
    return NS(header=header(seq), joints=[NS(nid=(i // 4) * 5 + i % 4 + 1,
        position=.1 + i * .001, velocity=i * .01, effort=i * .02) for i in range(20)])


def diagnostics(seq=0, enabled=False, code=0):
    return NS(header=header(seq), comm=NS(sdk_dropped=0, e2e_lost=0), joints=[
        NS(nid=(i // 4) * 5 + i % 4 + 1, error_code_current=code,
           status_word=NS(ext_state_name="Enabled" if enabled else "Disabled",
               position_limit_active=False, velocity_limit_active=False,
               current_limit_active=False)) for i in range(20)])


class Sink:
    def __init__(self):
        self.samples, self.events = [], []

    def try_publish(self, sample):
        self.samples.append(sample)
        return True

    def try_event(self, event):
        self.events.append(event)
        return True


class Subscription:
    def __init__(self):
        self.frames = deque()
        self.closed = False
        self.error = None
        self.rates = []

    def recv(self):
        if self.error:
            raise self.error
        return self.frames.popleft() if self.frames else None

    def close(self):
        self.closed = True

    def set_rate(self, frequency_hz):
        self.rates.append(frequency_hz)
        return frequency_hz


class Resource:
    def __init__(self, value=None, *, mit=False):
        self.value, self.mit = value, mit
        self.subscription = Subscription()
        self.writes = []

    def get(self):
        return self.value

    def set(self, value):
        self.writes.append(value)
        if self.mit:
            pairs = [value] * 20 if isinstance(value, tuple) else value
            self.value = [NS(kp=kp, kd=kd) for kp, kd in pairs]
        else:
            self.value = [value] * 20 if isinstance(value, (int, float)) else value

    def subscribe(self):
        return self.subscription


class FakeGlove:
    serial_number = "glove-test"
    info = NS(firmware_version="glove-fw")
    is_connected = True

    def __init__(self, side="left"):
        self.side = Resource(side)
        self.skeleton, self.emf = Resource(), Resource()
        self.angle_data, self.tactile_data, self.contact_data = Resource(), Resource(), Resource()
        self.model = Resource("")

    def hand_side(self):
        return self.side

    def hand_skeleton(self):
        return self.skeleton

    def emf_poses(self):
        return self.emf

    def hand_joint_angles(self):
        return self.angle_data

    def tactile(self):
        return self.tactile_data

    def tactile_binary(self):
        return self.contact_data

    def hand_model_path(self):
        return self.model


class Publisher:
    def __init__(self):
        self.commands = []
        self.closed = False
        self.error = None

    def send(self, command):
        if self.error:
            raise self.error
        self.commands.append(command)

    def close(self):
        self.closed = True


class FakeHand:
    serial_number = "hand-test"
    info = NS(firmware_version="2.3.0")
    is_connected = True

    def __init__(self, side="left"):
        self.side = Resource(side)
        self.state, self.diag = Resource(), Resource()
        self.effort = Resource([1.] * 20)
        self.mit = Resource([NS(kp=3., kd=.03) for _ in range(20)], mit=True)
        self.publisher = Publisher()
        self.enables = self.disables = 0

    def handedness(self):
        return self.side

    def online_joints_count(self):
        return Resource(20)

    def hw_version(self):
        return Resource(NS(major=2, minor=0, patch=0))

    def joint_states(self):
        return self.state

    def joint_diagnostics(self):
        return self.diag

    def effort_limit(self):
        return self.effort

    def mit_params(self):
        return self.mit

    def joint_command(self):
        return NS(publish=lambda: self.publisher)

    def enable(self):
        self.enables += 1
        self.diag.subscription.frames.append(diagnostics(1, True))

    def disable(self):
        self.disables += 1
        self.diag.subscription.frames.append(diagnostics(2, False))

    @staticmethod
    def describe_error(code):
        return {"severity": "Warning"} if code == 1 else None


class Manager:
    def __init__(self, device):
        self.device = device
        self.disconnects = []
        self.user = "previous"

    def connect(self, **kwargs):
        return self.device

    def disconnect(self, device_name):
        self.disconnects.append(device_name)

    def current_user(self):
        return {"user_id": self.user, "is_default": self.user == ""}

    def switch_to_default_user(self):
        self.user = ""

    def switch_user(self, user_id):
        self.user = user_id


SDK = NS(WujiGlove=FakeGlove, WujiHand2=FakeHand,
         ConnectOptions=lambda **kwargs: NS(**kwargs),
         JointCommand=lambda q, dq, current: NS(position=q, velocity=dq, effort=current))
PROFILE = ControlProfile("test-mit", "mit", {"kp": 5., "kd": .05, "current_limit_a": 1.5})


def wait_until(predicate):
    deadline = time.monotonic() + 1
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for fake SDK receiver")
        time.sleep(.001)


class WujiAcquisitionTests(unittest.TestCase):
    def test_health_uses_receiver_faults_and_live_age_without_sdk_getter(self):
        class GuardedHand(FakeHand):
            @property
            def is_connected(self):
                raise AssertionError("health queried the SDK")
        now = [1_000_000_000]
        device = GuardedHand()
        source = WujiHandDriver("left", "test", sdk=SDK, clock=lambda: now[0])
        source._device = device
        source._consume("joints", feedback(), now[0])
        source._consume("diagnostics", diagnostics(), now[0])
        self.assertTrue(source.health().ready)
        for _ in range(20):
            self.assertTrue(source.health(check_latch=True).ready)
            source.statistics
        now[0] += 501_000_000
        source._consume("joints", feedback(1), now[0])
        self.assertIn("diagnostics timed out", source.health().detail)
        source._consume("diagnostics", diagnostics(1), now[0])
        self.assertTrue(source.health().ready)
        self.assertFalse(source.health(check_latch=True).ready)
        source.clear_fault()
        subscription = Subscription()
        subscription.error = RuntimeError("device disconnected")
        source._subscriptions = {"joints": subscription}
        source._receive()
        self.assertIn("SDK receive ended: device disconnected", source.health().detail)
        self.assertIn("device disconnected", source.fault)

    def test_diagnostic_status_read_once_and_fixed_severity_cached(self):
        reads = []
        class Joint:
            def __init__(self, nid):
                self.nid, self.error_code_current = nid, 7
            @property
            def status_word(self):
                reads.append(self.nid)
                return NS(ext_state_name="Enabled", position_limit_active=False,
                          velocity_limit_active=False, current_limit_active=False)
        describe = Mock(side_effect=lambda code: {"severity": "Warning"} if code == 7 else None)
        source = WujiHandDriver("left", "test", sdk=NS(WujiHand2=NS(describe_error=describe)))
        frame = diagnostics()
        frame.joints = [Joint(j.nid) for j in frame.joints]
        for _ in range(2):
            payload, valid, _ = source._decode("diagnostics", frame, None)
            self.assertTrue(valid)
            self.assertEqual(payload.states, ("Enabled",) * 20)
        self.assertEqual(len(reads), 40)
        describe.assert_called_once_with(7)
        frame.joints[0].error_code_current = 999
        for _ in range(2):
            self.assertFalse(source._decode("diagnostics", frame, None)[1])
        self.assertEqual(describe.call_count, 2)

    def test_statistics_reader_does_not_take_receiver_lock(self):
        source = WujiGloveSource("left", "test", sdk=SDK, clock=lambda: 50)
        source._consume("skeleton", skeleton(), 10)
        source._lock = Mock()
        source._lock.__enter__ = Mock(side_effect=AssertionError("reader acquired receiver lock"))
        self.assertEqual(source.statistics["skeleton"]["count"], 1)

    def test_contact_stream_uses_sdk_binary_and_requires_complete_model_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {"dir": directory, "safetensors": f"{directory}/contact.safetensors",
                     "npz": f"{directory}/contact.npz"}
            for name in ("contact.safetensors", "contact.npz"):
                Path(directory, name).touch()
            for committed in (False, True):
                if committed:
                    Path(directory, "contact.json").touch()
                device = FakeGlove()
                manager = Manager(device)
                manager.tactile_model_paths = Mock(return_value=paths)
                source = WujiGloveSource("left", "test", sdk=SDK, manager=manager,
                                         streams=("skeleton", "tactile", "contact"))
                source.start()
                try:
                    self.assertEqual(source.metadata["tactile_contact_model_present"], committed)
                    self.assertIs(source._subscriptions["contact"], device.contact_data.subscription)
                    self.assertNotIn("angles", source._subscriptions)
                    data = tactile()
                    data.data = [-1., 0., 1.] * 248
                    source._consume("contact", data, 100)
                    sample = source.get_latest_stream("contact")
                    self.assertTrue(sample.header.valid)
                    self.assertEqual(sample.payload.values[:3], (-1., 0., 1.))
                    data.header.seq = 1
                    data.data[2] = .5
                    source._consume("contact", data, 101)
                    self.assertFalse(source.get_latest_stream("contact").header.valid)
                finally:
                    source.close()

    def test_angle_slots_decode_to_21_dofs_in_finger_order(self):
        for padded in (True, False):
            with self.subTest(padded=padded):
                source = WujiGloveSource("left", "test", sdk=SDK)
                data = angles(padded=padded)
                expected = []
                for i, (finger, dofs) in enumerate(zip(data.fingers, (5, 4, 4, 4, 4))):
                    values = [(i * 5 + j) / 100 for j in range(dofs)]
                    finger.angles[:dofs] = values
                    expected.extend(values)
                source._consume("angles", data, time.monotonic_ns())
                sample = source.get_latest_stream("angles")
                self.assertTrue(sample.header.valid)
                self.assertIsNone(source.fault)
                self.assertEqual(sample.payload.position_rad, tuple(expected))
                self.assertEqual(sample.payload.joint_names, tuple(
                    f"{name}_{j + 1}" for name, dofs in zip(
                        ("thumb", "index", "middle", "ring", "pinky"), (5, 4, 4, 4, 4))
                    for j in range(dofs)))
                data.fingers[0].angles[0] = 99.
                self.assertEqual(sample.payload.position_rad[0], expected[0])

    def test_malformed_finger_angle_groups_are_invalid(self):
        for counts in ((4, 5, 4, 4, 4), (5, 3, 4, 4, 5), (6, 4, 4, 4, 4)):
            with self.subTest(counts=counts):
                source = WujiGloveSource("left", "test", sdk=SDK)
                data = angles()
                for finger, count in zip(data.fingers, counts):
                    finger.angles = [.1] * count
                source._consume("angles", data, time.monotonic_ns())
                self.assertFalse(source.get_latest_stream("angles").header.valid)
                self.assertIn("angle", source.fault)

    def test_live_observer_failure_latches_source_fault(self):
        source = WujiGloveSource("left", "test", sdk=SDK, clock=lambda: 50)
        source._device = FakeGlove()
        sink = source._sink = Sink()
        sink.try_publish = Mock(return_value=False)
        source._consume("skeleton", skeleton(), 40)
        self.assertFalse(source.health().ready)
        self.assertIn("observer rejected", source.fault)
        with self.assertRaisesRegex(RuntimeError, "observer rejected"):
            source.clear_fault()

    def test_native_connect_failure_has_device_context_and_preserves_cleanup_error(self):
        # Match the real SDK hierarchy; RuntimeError fakes missed this regression.
        class WujiException(Exception):
            pass

        for cleanup_fails in (False, True):
            with self.subTest(cleanup_fails=cleanup_fails):
                cause = WujiException("Connection timeout")
                manager = Manager(None)
                manager.connect = Mock(side_effect=cause)
                if cleanup_fails:
                    manager.disconnect = Mock(side_effect=WujiException("disconnect failed"))
                sink = Sink()
                source = WujiHandDriver("right", "192.168.1.111:7447", manager=manager, sdk=SDK)
                with self.assertRaisesRegex(RuntimeError, r"wuji_right_hand.*192\.168\.1\.111:7447.*connect failed.*Connection timeout") as caught:
                    source.start(sink)
                self.assertIs(caught.exception.__cause__, cause)
                self.assertEqual("cleanup failed" in str(caught.exception), cleanup_fails)
                self.assertTrue(source._closed)
                self.assertFalse(source._enable_attempted)
                options = manager.connect.call_args.kwargs["options"]
                self.assertEqual((options.timeout_ms, options.retry_count), (2000, 3))
                self.assertFalse(options.enable_bridge)
                failure = next(e for e in sink.events if e.kind == "wuji_start_failed")
                self.assertEqual(failure.details["stage"], "connect")
                self.assertEqual(failure.details["error_type"], "WujiException")

    def test_all_queued_samples_owned_and_invalid_latched(self):
        device, sink = FakeGlove(), Sink()
        source = WujiGloveSource("left", "test", manager=Manager(device), sdk=SDK)
        device.emf.subscription.frames.append(emf())
        invalid = skeleton(1, valid=False)
        final = skeleton(2)
        device.skeleton.subscription.frames.extend((skeleton(), invalid, final))
        source.start(sink)
        self.addCleanup(source.close)
        self.assertEqual(tuple(source._subscriptions), ("emf", "skeleton"))
        wait_until(lambda: len(sink.samples) == 4)
        latest = source.get_latest()
        self.assertEqual(latest.header.source_sequence, 2)
        self.assertEqual(latest.payload.source_refs, (sink.samples[0].header.ref,))
        self.assertTrue(source.fault)
        self.assertTrue(source.health().ready)
        self.assertFalse(source.health(check_latch=True).ready)
        final.joints[0].pose.position[0] = 100
        self.assertEqual(latest.payload.positions_m[0][0], 0.)
        self.assertIs(source.get_latest(), latest)
        self.assertIn("capture completed", latest.header.source_time.meaning)
        json.dumps(asdict(latest))
        source.clear_fault()
        self.assertIsNone(source.fault)

    def test_opt_in_glove_streams_do_not_change_legacy_control_subscriptions(self):
        device = FakeGlove()
        source = WujiGloveSource("left", "test", manager=Manager(device), sdk=SDK,
                                 streams=("emf", "skeleton", "angles", "tactile"))
        source.start()
        self.addCleanup(source.close)
        self.assertEqual(tuple(source._subscriptions), ("emf", "skeleton", "angles", "tactile"))
        self.assertEqual(source.metadata["streams"], source.streams)

    def test_counters_wrap_gap_and_duplicate_do_not_extend_life(self):
        source = WujiGloveSource("left", "test", sdk=SDK, clock=lambda: 50)
        source._device = FakeGlove()
        sink = source._sink = Sink()
        for seq, now in ((2**32 - 1, 10), (0, 20), (3, 30), (3, 40)):
            source._consume("skeleton", skeleton(seq), now)
        self.assertEqual(source.get_latest().header.received_monotonic_ns, 30)
        self.assertEqual(source.statistics["skeleton"]["source_gaps"], 2)
        self.assertEqual(len(sink.samples), 4)
        self.assertEqual(len(sink.events), 2)
        source._consume("skeleton", skeleton(1), 45)
        self.assertEqual(source.get_latest().header.source_sequence, 3)
        self.assertEqual(source.get_latest().header.received_monotonic_ns, 30)
        self.assertIn("backwards", source.fault)

    def test_bad_side_and_receive_end_are_reported(self):
        device = FakeGlove("right")
        manager = Manager(device)
        source = WujiGloveSource("left", "test", manager=manager, sdk=SDK)
        with self.assertRaisesRegex(RuntimeError, "expected left"):
            source.start()
        self.assertEqual(manager.disconnects, [source.device_id])
        device = FakeGlove()
        device.skeleton.subscription.error = RuntimeError("connection ended")
        source = WujiGloveSource("left", "test", manager=Manager(device), sdk=SDK)
        source.start()
        self.addCleanup(source.close)
        wait_until(lambda: source.fault is not None)
        self.assertIn("connection ended", source.fault)
        self.assertFalse(source.health().ready)
        with self.assertRaisesRegex(RuntimeError, "connection ended"):
            source.clear_fault()

    def test_default_user_restored(self):
        manager = Manager(None)
        session = WujiSdkSession(manager=manager, sdk=SDK).open()
        self.assertEqual(manager.user, "")
        self.assertTrue(session.health().ready)
        manager.user = "changed"
        self.assertFalse(session.health().ready)
        session.close()
        self.assertEqual(manager.user, "previous")

    def test_named_user_is_selected_monitored_and_restored(self):
        manager = Manager(None)
        manager.list_users = Mock(return_value=[{"user_id": "person-1", "display_name": "Alice"}])
        session = WujiSdkSession(user_name="Alice", manager=manager, sdk=SDK).open()
        self.assertEqual(manager.user, "person-1")
        self.assertEqual(session.metadata["sdk_user_id"], "person-1")
        self.assertTrue(session.health().ready)
        manager.user = ""
        self.assertFalse(session.health().ready)
        session.close()
        self.assertEqual(manager.user, "previous")

    def test_user_name_resolves_before_device_connection_and_restores_previous_user(self):
        manager = Manager(None)
        manager.list_users = Mock(return_value=[
            {"user_id": "u_123", "display_name": "yuchen", "is_default": False}])
        manager.create_user = Mock()
        manager.current_user = lambda: {"user_id": manager.user, "display_name": "yuchen"}
        session = WujiSdkSession(user_name="yuchen", manager=manager, sdk=SDK).open()
        self.assertEqual(manager.user, "u_123")
        self.assertEqual(session.metadata["sdk_user_name"], "yuchen")
        self.assertTrue(session.health().ready)
        manager.user = "changed"
        self.assertFalse(session.health().ready)
        session.close()
        self.assertEqual(manager.user, "previous")
        manager.create_user.assert_not_called()

        for users in ([], [
                {"user_id": "a", "display_name": "yuchen"},
                {"user_id": "b", "display_name": "yuchen"}]):
            manager.list_users.return_value = users
            failed = WujiSdkSession(user_name="yuchen", manager=manager, sdk=SDK)
            with self.assertRaises(ValueError):
                failed.open()
            failed.close()
            self.assertEqual(manager.user, "previous")
        manager.create_user.assert_not_called()

    def test_angles_and_both_tactile_layouts_are_owned(self):
        source = WujiGloveSource("left", "test", sdk=SDK)
        source._device = FakeGlove()
        for stream, frame in (("angles", angles()), ("tactile", tactile()),
                              ("tactile", tactile(1, 768))):
            source._consume(stream, frame, time.monotonic_ns())
        joint_sample = source.get_latest_stream("angles")
        self.assertTrue(joint_sample.header.valid)
        self.assertIsInstance(joint_sample.payload, WujiHandAngles)
        self.assertEqual(len(joint_sample.payload.position_rad), 21)
        tactile_sample = source.get_latest_stream("tactile")
        self.assertTrue(tactile_sample.header.valid)
        self.assertIsInstance(tactile_sample.payload, WujiTactileFrame)
        self.assertEqual((tactile_sample.payload.rows, tactile_sample.payload.columns), (24, 32))
        source._consume("angles", angles(1, bad=True), time.monotonic_ns())
        self.assertFalse(source.get_latest_stream("angles").header.valid)
        self.assertTrue(source.fault)

    def test_wrong_skeleton_axes_or_landmark_order_is_invalid(self):
        source = WujiGloveSource("left", "test", sdk=SDK)
        source._device = FakeGlove()
        frame = skeleton()
        frame.header.frame_id = "r_wrist"
        source._consume("skeleton", frame, time.monotonic_ns())
        self.assertFalse(source.get_latest().header.valid)
        frame = skeleton(1)
        frame.joints.reverse()
        source._consume("skeleton", frame, time.monotonic_ns())
        self.assertFalse(source.get_latest().header.valid)


class WujiHandTests(unittest.TestCase):
    def opened(self):
        device, sink = FakeHand(), Sink()
        device.state.subscription.frames.append(feedback())
        device.diag.subscription.frames.append(diagnostics())
        driver = WujiHandDriver("left", "test", manager=Manager(device), sdk=SDK)
        driver.start(sink)
        self.addCleanup(driver.close)
        wait_until(lambda: driver.health().ready)
        return driver, device, sink

    def command(self, driver, **changes):
        now = time.monotonic_ns()
        values = dict(device_id=driver.device_id, command_id="command", payload=JointTarget(
            JOINT_NAMES, (.2,) * 20), source_refs=(), created_monotonic_ns=now,
            expires_monotonic_ns=now + 50_000_000, control_profile_id=PROFILE.profile_id)
        values.update(changes)
        return DeviceCommand(**values)

    def test_reordered_feedback_preserves_current_units_and_missing(self):
        driver, _, _ = self.opened()
        frame = feedback(1)
        frame.joints.reverse()
        driver._consume("joints", frame, time.monotonic_ns())
        actual = driver.get_latest()
        self.assertEqual(actual.payload.position_rad[3], .10300000000000001)
        self.assertEqual(actual.payload.motor_current_a[3], .06)
        self.assertIsNone(actual.payload.measured_torque_nm)
        self.assertIn("firmware send", actual.header.source_time.meaning)
        frame = feedback(2)
        frame.joints.pop(7)
        driver._consume("joints", frame, time.monotonic_ns())
        self.assertFalse(driver.get_latest().header.valid)
        self.assertIsNone(driver.get_latest().payload.position_rad[7])
        self.assertTrue(driver.fault)

    def test_readonly_start_and_verified_parameters_restored(self):
        driver, device, _ = self.opened()
        self.assertEqual(driver.feedback_hz_actual, {"joints": 200, "diagnostics": 200})
        self.assertEqual(device.state.subscription.rates, [200])
        self.assertEqual(device.diag.subscription.rates, [200])
        self.assertEqual(device.enables, 0)
        self.assertEqual(device.effort.writes, [])
        driver.configure(PROFILE)
        self.assertEqual(device.effort.get(), [1.5] * 20)
        self.assertEqual(device.mit.get()[0].kp, 5.)
        driver.close()
        self.assertEqual(device.effort.get(), [1.] * 20)
        self.assertEqual(device.mit.get()[0].kp, 3.)
        self.assertEqual(device.disables, 0)

    def test_missing_original_parameter_prevents_write(self):
        driver, device, _ = self.opened()
        device.effort.value[0] = None
        with self.assertRaisesRegex(RuntimeError, "all 20"):
            driver.configure(PROFILE)
        self.assertEqual(device.effort.writes, [])

    def test_externally_enabled_hand_is_not_reconfigured(self):
        driver, device, _ = self.opened()
        driver._consume("diagnostics", diagnostics(1, enabled=True), time.monotonic_ns())
        with self.assertRaisesRegex(RuntimeError, "other controller"):
            driver.configure(PROFILE)
        self.assertEqual(device.effort.writes, [])
        self.assertEqual(device.disables, 0)

    def test_new_fault_during_configuration_is_not_cleared_by_engage(self):
        driver, device, _ = self.opened()
        driver.configure(PROFILE)
        driver._consume("diagnostics", diagnostics(1, code=99), time.monotonic_ns())
        driver._consume("diagnostics", diagnostics(2), time.monotonic_ns())
        self.assertTrue(driver.health().ready)
        with self.assertRaisesRegex(RuntimeError, "fault"):
            driver.engage()
        self.assertEqual(device.enables, 0)

    def test_engage_seed_local_acceptance_hold_and_close(self):
        driver, device, sink = self.opened()
        driver.configure(PROFILE)
        driver.engage()
        self.assertTrue(driver.enabled)
        self.assertEqual(device.publisher.commands[0][0].position, .1)
        self.assertTrue(driver.submit(self.command(driver)).accepted)
        target = driver.last_target
        driver.request_hold("space")
        self.assertIs(driver.last_target, target)
        self.assertEqual(device.disables, 0)
        receipts = [event for event in sink.events if isinstance(event, CommandEvent)]
        self.assertTrue(receipts)
        self.assertTrue(all(event.status == CommandStatus.ACCEPTED for event in receipts))
        driver.close()
        self.assertEqual(device.disables, 1)
        self.assertEqual(device.effort.get(), [1.] * 20)
        self.assertTrue(any(getattr(e, "kind", "") == "wuji_disabled_observed" for e in sink.events))

    def test_unconfirmed_disable_does_not_restore_live_parameters(self):
        driver, device, _ = self.opened()
        driver.configure(PROFILE)
        driver.engage()
        device.disable = lambda: None  # SDK returns without new disabled feedback.
        device.is_connected = False
        with self.assertRaisesRegex(RuntimeError, "disable unconfirmed; parameters not restored"):
            driver.close()
        self.assertEqual(device.effort.get(), [1.5] * 20)
        self.assertEqual(device.mit.get()[0].kp, 5.)
        self.assertTrue(device.publisher.closed)
        self.assertEqual(driver.manager.disconnects, [driver.device_id])

    def test_engagement_seed_near_limit_survives_the_next_hold_submission(self):
        for index, measured in ((18, -1.0501492023468018),
                                (0, JOINT_LIMITS_RAD[0][1] + math.radians(.2))):
            with self.subTest(index=index):
                driver, device, _ = self.opened()
                driver.configure(PROFILE)
                frame = feedback(1)
                frame.joints[index].position = measured
                driver._consume("joints", frame, time.monotonic_ns())
                driver.engage()
                seed = driver.last_target
                low, high = JOINT_LIMITS_RAD[index]
                self.assertEqual(seed.position_rad[index], min(high, max(low, measured)))
                self.assertEqual(device.publisher.commands[0][index].position, seed.position_rad[index])
                self.assertTrue(driver.submit(self.command(driver, payload=seed)).accepted)
                # Only the measured engagement seed may be corrected; ordinary
                # out-of-range commands must still be rejected.
                self.assertFalse(driver.submit(self.command(driver, payload=JointTarget(
                    JOINT_NAMES, tuple(j.position for j in frame.joints)))).accepted)
                driver.close()

    def test_engagement_rejects_large_limit_error_before_enabling_or_sending(self):
        driver, device, _ = self.opened()
        driver.configure(PROFILE)
        frame = feedback(1)
        frame.joints[18].position = JOINT_LIMITS_RAD[18][0] - math.radians(.6)
        driver._consume("joints", frame, time.monotonic_ns())
        with self.assertRaisesRegex(RuntimeError, "left.*finger5_joint3.*0.5"):
            driver.engage()
        self.assertEqual(device.enables, 0)
        self.assertEqual(device.publisher.commands, [])

    def test_expired_bad_limit_and_failed_sdk_calls(self):
        driver, device, sink = self.opened()
        driver.configure(PROFILE)
        driver.engage()
        before = len(device.publisher.commands)
        self.assertFalse(driver.submit(self.command(driver, expires_monotonic_ns=0)).accepted)
        self.assertFalse(driver.submit(self.command(driver,
            payload=JointTarget(JOINT_NAMES, (100.,) * 20))).accepted)
        self.assertEqual(len(device.publisher.commands), before)
        device.publisher.error = RuntimeError("publisher failed")
        self.assertFalse(driver.submit(self.command(driver)).accepted)
        self.assertIn("publisher failed", driver.fault)
        self.assertFalse(any(isinstance(e, CommandEvent) and e.status == CommandStatus.SENT
                             for e in sink.events))

    def test_warning_recorded_but_unknown_fault_and_drop_latched(self):
        driver, _, sink = self.opened()
        driver._consume("diagnostics", diagnostics(1, code=1), time.monotonic_ns())
        self.assertIsNone(driver.fault)
        frame = diagnostics(2, code=99)
        frame.comm.sdk_dropped = 5
        driver._consume("diagnostics", frame, time.monotonic_ns())
        self.assertTrue(driver.fault)
        self.assertTrue(any(getattr(e, "kind", "") == "wuji_sdk_dropped" for e in sink.events))

    def test_cancel_before_enable_and_duplicate_joint(self):
        driver, device, _ = self.opened()
        driver.configure(PROFILE)
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            driver.engage(cancelled=lambda: True)
        self.assertEqual(device.enables, 0)
        frame = feedback(1)
        frame.joints[1] = frame.joints[0]
        driver._consume("joints", frame, time.monotonic_ns())
        self.assertFalse(driver.get_latest().header.valid)
        self.assertIn("duplicate", driver.fault)
        with self.assertRaises(ValueError):
            nid_to_index(5)


if __name__ == "__main__":
    unittest.main()
