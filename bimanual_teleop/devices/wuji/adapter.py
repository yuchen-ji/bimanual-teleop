"""Wuji SDK boundary: owned samples, continuous reception and local submission.

The SDK is imported only when opening a session/device. Host timestamps mark
SDK dequeue, not network reception; each device stream states its source meaning.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from importlib.metadata import version
import math
from pathlib import Path
import threading
import time
from typing import Callable
from uuid import uuid4

from bimanual_teleop.devices.interfaces import RealtimeObserver
from bimanual_teleop.types import (
    CommandEvent, CommandStatus, ControlProfile, DeviceCommand, Event,
    HandSkeleton, Health, JointState, JointTarget, Sample, SampleHeader,
    SampleRef, SourceTime, Submission,
)

SDK_VERSION = "2026.8.31"
JOINT_NAMES = tuple(f"finger{finger}_joint{joint}" for finger in range(1, 6)
                    for joint in range(1, 5))
SKELETON_NAMES = ("wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip",
    "middle_finger_mcp", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip",
    "ring_finger_mcp", "ring_finger_pip", "ring_finger_dip", "ring_finger_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip")
JOINT_LIMITS_RAD = tuple((math.radians(low), math.radians(high)) for low, high in
                        [(-68, 74), (-85, 40), (-60, 90), (-60, 90)]
                        + [(-60, 90), (-40, 40), (-60, 120), (-60, 90)] * 4)


def _sdk_module():
    if version("wuji-sdk") != SDK_VERSION:
        raise RuntimeError(f"Wuji requires wuji-sdk=={SDK_VERSION}")
    import wuji_sdk
    return wuji_sdk


def resolve_user_id(manager, *, user_name, create=False):
    """Resolve a unique SDK user; only calibration may create a missing name."""
    if not isinstance(user_name, str) or not user_name.strip():
        raise ValueError("SDK 用户名不能为空")
    matches = [user for user in manager.list_users() if user.get("display_name") == user_name]
    if len(matches) > 1:
        raise ValueError(f"SDK 用户名 {user_name!r} 对应多个用户；请先在 Wuji 用户管理中将用户名改为唯一名称。")
    if matches:
        if create and matches[0].get("is_default", False):
            raise ValueError("calibration requires a named SDK user")
        return matches[0]["user_id"]
    if create:
        return manager.create_user(user_name)["user_id"]
    raise ValueError(f"未找到 SDK 用户名 {user_name!r}；请先用该用户名完成标定。")


class WujiSdkSession:
    """Own the SDK user selection; close after all devices are disconnected."""

    def __init__(self, *, user_name: str = "", manager=None, sdk=None):
        if not isinstance(user_name, str) or (user_name and not user_name.strip()):
            raise ValueError("SDK user name must be a nonblank string")
        self.user_name = user_name
        self.user_id = ""
        self.manager, self.sdk = manager, sdk
        self._previous_user = None
        self.metadata = {"sdk_version": SDK_VERSION, "sdk_user_name": user_name,
                         "sdk_user_id": ""}

    def open(self):
        if self._previous_user is not None:
            raise RuntimeError("Wuji SDK session already open")
        self.sdk = self.sdk or _sdk_module()
        self.manager = self.manager or self.sdk.SdkManager.instance()
        if self.user_name:
            self.user_id = resolve_user_id(self.manager, user_name=self.user_name)
        self._previous_user = self.manager.current_user()["user_id"]
        try:
            user = (self.manager.switch_user(self.user_id) if self.user_id
                    else self.manager.switch_to_default_user())
            user = self.manager.current_user() if user is None else user
            if user["user_id"] != self.user_id:
                raise RuntimeError(f"SDK selected user {user['user_id']!r}, expected {self.user_id!r}")
            self.metadata.update(sdk_user_id=user["user_id"],
                                 sdk_user_name=user.get("display_name", "Default"))
        except Exception:
            try:
                self.manager.switch_user(self._previous_user)
            finally:
                self._previous_user = None
            raise
        return self

    def close(self):
        if self._previous_user is not None:
            self.manager.switch_user(self._previous_user)
            self._previous_user = None

    def health(self):
        ready = (self._previous_user is not None
                 and self.manager.current_user()["user_id"] == self.user_id)
        return Health(ready, time.monotonic_ns(), None if ready else "SDK user changed")


@dataclass(frozen=True)
class WujiEmfFrame:
    frame: str
    positions_m: tuple[tuple[float, ...], ...]
    orientations_xyzw: tuple[tuple[float, ...], ...]
    confidences: tuple[float, ...]


@dataclass(frozen=True)
class WujiHandAngles:
    """Five SDK finger groups, retaining their original order and confidence."""

    joint_names: tuple[str, ...]
    position_rad: tuple[float, ...]
    finger_confidences: tuple[float, ...]


@dataclass(frozen=True)
class WujiTactileFrame:
    """Full row-major SDK tactile matrix; no firmware column is dropped."""

    rows: int
    columns: int
    values: tuple[float | None, ...]


@dataclass(frozen=True)
class WujiDiagnostics:
    states: tuple[str | None, ...]
    error_codes: tuple[int | None, ...]
    limit_flags: tuple[tuple[bool, bool, bool] | None, ...]
    sdk_dropped: int
    e2e_lost: int


@dataclass(frozen=True)
class InvalidWujiFrame:
    error: str


def nid_to_index(nid: int) -> int:
    bus, node = divmod(int(nid) - 1, 5)
    if not (0 <= bus < 5 and 0 <= node < 4):
        raise ValueError(f"invalid Hand2 joint nid {nid}")
    return bus * 4 + node


def _ordered(joints):
    result = [None] * 20
    for joint in joints:
        index = nid_to_index(joint.nid)
        if result[index] is not None:
            raise ValueError(f"duplicate Hand2 joint nid {joint.nid}")
        result[index] = joint
    return result


def _finite(values):
    return all(value is not None and math.isfinite(value) for value in values)


class _WujiSource:
    """Shared Wuji connection and queue consumption, without a transport layer."""

    def __init__(self, side, address, *, manager=None, sdk=None, timeout_s,
                 clock: Callable[[], int] = time.monotonic_ns):
        if side not in ("left", "right") or not address:
            raise ValueError("Wuji requires left/right and an explicit address")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Wuji timeout must be positive")
        self.side, self.address = side, address
        self.manager, self.sdk, self._clock = manager, sdk, clock
        self.timeout_ns = int(timeout_s * 1e9)
        self.device_id = f"wuji_{side}_{self.kind}"
        self.metadata = {}
        self.fault = None
        self._latest = None
        self._latest_by_stream = {}
        self._device = None
        self._sink = None
        self._observer_error = None
        self._subscriptions = {}
        self._thread = None
        self._stop = threading.Event()
        self._epoch = uuid4().hex
        self._sequences = {}
        self._source_sequences = {}
        self._stats = {}
        self._statistics_snapshot = {}
        self._lock = threading.RLock()
        self._attempted = False
        self._closed = False
        self._receive_error = None

    def _event(self, kind, **details):
        self._notify_event(Event(kind, self._clock(), self.device_id, details))

    def _observer_failure(self, error=None):
        if self._observer_error is not None:
            return
        self._observer_error = (f"realtime observer failed: {error}" if error else
                                "realtime observer rejected live data")
        self.fault = self.fault or self._observer_error

    def _notify_event(self, event):
        if self._sink is None or self._observer_error is not None:
            return
        try:
            if self._sink.try_event(event):
                return
            self._observer_failure()
        except Exception as error:
            self._observer_failure(error)

    def _notify_sample(self, sample):
        if self._sink is None or self._observer_error is not None:
            return
        try:
            if self._sink.try_publish(sample):
                return
            self._observer_failure()
        except Exception as error:
            self._observer_failure(error)

    def _fault(self, reason):
        if self.fault is None:
            self.fault = reason
            self._event("wuji_fault", reason=reason)

    def start(self, sink: RealtimeObserver | None = None):
        if self._attempted or self._closed:
            raise RuntimeError("Wuji source cannot be started twice")
        self._sink = sink
        self.sdk = self.sdk or _sdk_module()
        self.manager = self.manager or self.sdk.SdkManager.instance()
        self._attempted = True
        stage = "connect"
        self._event("wuji_connecting", address=self.address, timeout_ms=2000, retry_count=3)
        try:
            self._device = self.manager.connect(
                address=self.address, device_name=self.device_id,
                options=self.sdk.ConnectOptions(timeout_ms=2000, retry_count=3,
                    enable_bridge=False, auto_time_sync_interval_ms=None))
            stage = "identity"
            expected = self.sdk.WujiGlove if self.kind == "glove" else self.sdk.WujiHand2
            if not isinstance(self._device, expected):
                raise RuntimeError(f"{self.address} is not {expected.__name__}")
            side = (self._device.hand_side() if self.kind == "glove"
                    else self._device.handedness()).get()
            reported = str(getattr(side, "name", side)).lower().rsplit(".", 1)[-1]
            if reported != self.side:
                raise RuntimeError(f"expected {self.side}, device reports {side}")
            info = self._device.info
            user = self.manager.current_user()
            self.metadata = {
                "device_id": self.device_id, "address": self.address, "side": self.side,
                "sdk_version": SDK_VERSION, "serial_number": self._device.serial_number,
                "firmware_version": None if info is None else str(info.firmware_version),
                "source_timestamp": self._source_time_meaning(),
                "host_timestamp": "monotonic SDK dequeue observation",
                "time_sync": "connect-time SDK sync; periodic sync disabled; epoch unverified",
                "sdk_user_id": user["user_id"],
                "sdk_user_name": user.get("display_name", ""),
                "human_model": (("builtin" if user["is_default"]
                                 else "user_model_or_builtin_fallback")
                                if self.kind == "glove" else None),
            }
            stage = "subscribe"
            self._open_streams()
            self._event("wuji_metadata", **self.metadata)
            if self._observer_error:
                raise RuntimeError(self._observer_error)
            self._thread = threading.Thread(target=self._receive, name=self.device_id,
                                            daemon=True)
            self._thread.start()
        except Exception as error:
            # WujiException inherits Exception directly, not RuntimeError. Keep
            # native failures inside this boundary and identify the failed device.
            message = f"{self.device_id} ({self.address}) {stage} failed: {type(error).__name__}: {error}"
            self._event("wuji_start_failed", address=self.address, stage=stage,
                        error_type=type(error).__name__, error=str(error))
            try:
                self.close()
            except Exception as cleanup_error:
                message += f"; cleanup failed: {cleanup_error}"
            raise RuntimeError(message) from error

    def _source_time_meaning(self, stream=None):
        if self.kind == "glove":
            meaning = ("EMF capture completed; skeleton inherits EMF time"
                       if stream in (None, "emf", "skeleton") else f"{stream} SDK frame timestamp")
        else:
            meaning = "firmware send"
        return meaning + "; us, uptime or synchronized UTC, epoch unverified"

    def _receive(self):
        try:
            while not self._stop.is_set():
                received = False
                for stream, subscription in self._subscriptions.items():
                    # Bound each visit so one busy stream cannot starve diagnostics.
                    for _ in range(128):
                        frame = subscription.recv()
                        if frame is None:
                            break
                        received = True
                        self._consume(stream, frame, self._clock())
                if not received:
                    self._stop.wait(.001)
        except Exception as error:
            if not self._stop.is_set():
                self._receive_error = str(error)
                self._fault(f"SDK receive ended: {error}")

    def _consume(self, stream, frame, now):
        with self._lock:
            self._consume_frame(stream, frame, now)

    def _consume_frame(self, stream, frame, now):
        sequence = self._sequences.get(stream, 0)
        self._sequences[stream] = sequence + 1
        header = frame.header
        source_sequence = int(header.seq)
        previous = self._source_sequences.get(stream)
        delta = None if previous is None else (source_sequence - previous) % (1 << 32)
        advances = previous is None or 0 < delta < (1 << 31)
        stats = self._stats.setdefault(stream, {"count": 0, "source_gaps": 0,
                                               "first_ns": now, "latest_ns": now})
        stats["count"] += 1
        stats["latest_ns"] = now
        if previous is not None and delta != 1:
            stats["source_gaps"] += 1
            self._event("wuji_source_gap", stream=stream, previous=previous,
                        current=source_sequence, counter_bits=32)
        if advances:
            self._source_sequences[stream] = source_sequence
        elif delta:
            self._fault(f"{stream}: source sequence moved backwards")
        ref = SampleRef(f"{self.device_id}/{stream}", self._epoch, sequence)
        try:
            payload, valid, reason = self._decode(stream, frame, ref)
        except (ValueError, TypeError, AttributeError, OverflowError) as error:
            payload, valid, reason = InvalidWujiFrame(str(error)), False, str(error)
        sample = Sample(SampleHeader(ref, now, valid,
            SourceTime(int(header.timestamp_us), "us", f"{self.device_id}/firmware",
                       self._source_time_meaning(stream)),
            source_sequence), payload)
        if not valid:
            self._fault(f"{stream}: {reason}")
        if stream == self.main_stream:
            # A repeated source counter cannot extend input life.
            if advances:
                self._latest = sample
        if advances:
            self._latest_by_stream[stream] = sample
            self._observe(stream, sample)
        self._statistics_snapshot = {name: {"count": values["count"], "source_gaps": values["source_gaps"],
            "observed_hz": ((values["count"]-1)*1e9/(values["latest_ns"]-values["first_ns"])
                            if values["latest_ns"] > values["first_ns"] else 0.)}
            for name, values in self._stats.items()}
        self._notify_sample(sample)

    def _observe(self, stream, sample):
        pass

    def get_latest(self):
        return self._latest

    def get_latest_stream(self, stream):
        with self._lock:
            return self._latest_by_stream.get(stream)

    @property
    def statistics(self):
        return {stream: dict(values) for stream, values in self._statistics_snapshot.items()}

    def _current_problem(self):
        if self._observer_error is not None:
            return self._observer_error
        if self._receive_error is not None:
            return f"SDK receive ended: {self._receive_error}"
        if self._closed or self._device is None:
            return "device disconnected"
        if self._latest is None:
            return "waiting for feedback"
        if not self._latest.header.valid:
            return "invalid latest feedback"
        if self._clock() - self._latest.header.received_monotonic_ns > self.timeout_ns:
            return "feedback timed out"
        return None

    def health(self, *, check_latch=False):
        """Read receiver faults and live sample age without an SDK query or lock."""
        problem = self._current_problem()
        if problem and self._latest is not None:
            self._fault(problem)
        reason = (self.fault if check_latch else None) or problem
        return Health(reason is None, self._clock(), reason)

    def clear_fault(self):
        with self._lock:
            problem = self._current_problem()
            if problem:
                raise RuntimeError(f"{self.device_id}: {problem}")
            self.fault = None

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        errors = []
        for subscription in self._subscriptions.values():
            try:
                subscription.close()
            except Exception as error:
                errors.append(str(error))
        if self._attempted:
            try:
                self.manager.disconnect(device_name=self.device_id)
            except Exception as error:
                errors.append(str(error))
        if errors:
            self._event("wuji_close_error", errors=tuple(errors))
            raise RuntimeError("; ".join(errors))


class WujiGloveSource(_WujiSource):
    kind, main_stream = "glove", "skeleton"

    def __init__(self, side, address, *, manager=None, sdk=None, timeout_s=.25,
                 clock=time.monotonic_ns, streams=("emf", "skeleton")):
        super().__init__(side, address, manager=manager, sdk=sdk, timeout_s=timeout_s,
                         clock=clock)
        if len(set(streams)) != len(streams) or set(streams) - {
                "emf", "skeleton", "angles", "tactile", "contact"}:
            raise ValueError("unsupported or repeated Wuji glove stream")
        self.streams = tuple(streams)
        self._emf_refs = OrderedDict()

    def _open_streams(self):
        resources = {"emf": "emf_poses", "skeleton": "hand_skeleton",
                     "angles": "hand_joint_angles", "tactile": "tactile",
                     "contact": "tactile_binary"}
        if "contact" in self.streams:
            paths = self.manager.tactile_model_paths(
                self.metadata["sdk_user_id"], self._device.serial_number)
            # The SDK publishes all-zero binary frames even without a model.
            # Check its documented artifacts so the viewer does not call that "no contact".
            self.metadata["tactile_contact_model_present"] = all(path.is_file() for path in (
                Path(paths["safetensors"]), Path(paths["npz"]), Path(paths["dir"]) / "contact.json"))
        for stream in self.streams:
            self._subscriptions[stream] = getattr(self._device, resources[stream])().subscribe()
        self.metadata["streams"] = self.streams
        try:
            model = self._device.hand_model_path().get()
            self.metadata["human_model"] = model or "builtin"
            self.metadata["human_model_source"] = "sdk_user" if model else "builtin"
        except Exception as error:
            self.metadata["human_model"] = "unknown"
            self.metadata["human_model_error"] = str(error)

    def _decode(self, stream, frame, ref):
        if stream == "emf":
            payload = WujiEmfFrame(str(frame.header.frame_id),
                tuple(tuple(float(v) for v in entry.pose.position) for entry in frame.poses),
                tuple(tuple(float(getattr(entry.pose.orientation, axis)) for axis in "xyzw")
                      for entry in frame.poses),
                tuple(float(entry.confidence) for entry in frame.poses))
            self._emf_refs[int(frame.header.timestamp_us)] = ref
            if len(self._emf_refs) > 256:
                self._emf_refs.popitem(last=False)
            valid = len(payload.positions_m) == 5 and all(len(p) == 3 and _finite(p)
                for p in payload.positions_m) and all(_finite(q) for q in payload.orientations_xyzw)
            valid = valid and _finite(payload.confidences)
        elif stream == "skeleton":
            header, joints = frame.header, frame.joints
            source = self._emf_refs.get(int(header.timestamp_us))
            payload = HandSkeleton(str(header.frame_id),
                tuple(str(j.name) for j in joints),
                tuple(tuple(float(v) for v in j.pose.position) for j in joints),
                tuple(float(j.confidence) for j in joints),
                () if source is None else (source,))
            valid = (payload.joint_names == SKELETON_NAMES
                     and payload.frame == f"{self.side[0]}_wrist")
            valid = valid and all(len(p) == 3 and _finite(p) for p in payload.positions_m)
            valid = valid and _finite(payload.confidences)
        elif stream == "angles":
            names = ("thumb", "index", "middle", "ring", "pinky")
            dofs = (5, 4, 4, 4, 4)
            if len(frame.fingers) != 5:
                raise ValueError("expected five SDK finger angle groups")
            # SDK arrays have five slots per finger; non-thumb slot 5 is padding.
            # Also accept compact arrays, but validate each finger's DoF count.
            if any(len(finger.angles) not in (count, 5)
                   for finger, count in zip(frame.fingers, dofs)):
                raise ValueError("expected 5 thumb angles and 4 angles (or 5 slots) per finger")
            positions = tuple(float(value) for finger, count in zip(frame.fingers, dofs)
                              for value in finger.angles[:count])
            joint_names = tuple(f"{name}_{index + 1}" for name, count in zip(names, dofs)
                                for index in range(count))
            confidences = tuple(float(finger.confidence) for finger in frame.fingers)
            payload = WujiHandAngles(joint_names, positions, confidences)
            valid = (len(positions) == 21 and _finite(positions) and _finite(confidences))
        elif stream in ("tactile", "contact"):
            raw = tuple(float(value) for value in frame.data)
            if len(raw) not in (744, 768):
                raise ValueError(f"unsupported tactile layout: {len(raw)} values")
            payload = WujiTactileFrame(24, len(raw) // 24,
                tuple(value if math.isfinite(value) else None for value in raw))
            valid = all(value is not None for value in payload.values)
            if stream == "contact":
                valid = valid and all(value in (-1., 0., 1.) for value in payload.values)
        else:
            raise ValueError(f"unknown Wuji glove stream {stream}")
        return payload, valid, f"invalid {stream} data"


class WujiHandDriver(_WujiSource):
    kind, main_stream = "hand", "joints"

    def __init__(self, side, address, *, manager=None, sdk=None, timeout_s=.5,
                 feedback_hz=200, clock=time.monotonic_ns):
        if (isinstance(feedback_hz, bool) or not isinstance(feedback_hz, int)
                or not 1 <= feedback_hz <= 1000):
            raise ValueError("Hand2 feedback_hz must be an integer in [1, 1000]")
        super().__init__(side, address, manager=manager, sdk=sdk, timeout_s=timeout_s,
                         clock=clock)
        self.feedback_hz = feedback_hz
        self.feedback_hz_actual = {}
        self.enabled = False
        self.profile = None
        self.last_target = None
        self._diagnostics = None
        self._publisher = None
        self._original = None
        self._parameters_touched = False
        self._enable_attempted = False
        self._arming = False
        self._last_dropped = 0
        self._error_warnings = {}

    def _open_streams(self):
        if self._device.online_joints_count().get() != 20:
            raise RuntimeError("Hand2 requires all 20 joints online")
        hardware = self._device.hw_version().get()
        self.metadata["hardware_version"] = (hardware.major, hardware.minor, hardware.patch)
        self.metadata["command_velocity_rad_s"] = 0.
        self.metadata["command_feedforward_current_a"] = 0.
        self._subscriptions["joints"] = self._device.joint_states().subscribe()
        self._subscriptions["diagnostics"] = self._device.joint_diagnostics().subscribe()
        # Hand2 publishes both streams at 1 kHz by default. Decoding four such
        # streams in the hand process starves its 120 Hz controller and creates
        # system-wide scheduling pressure. Both requests intentionally use the
        # same value because the device rate is shared and last-writer-wins.
        for stream, subscription in self._subscriptions.items():
            actual = subscription.set_rate(self.feedback_hz)
            if isinstance(actual, bool) or not isinstance(actual, int) or actual <= 0:
                raise RuntimeError(f"Hand2 {stream} returned an invalid feedback rate: {actual!r}")
            self.feedback_hz_actual[stream] = actual
        self.metadata.update(feedback_hz_requested=self.feedback_hz,
                             feedback_hz_actual=dict(self.feedback_hz_actual))

    def _decode(self, stream, frame, ref):
        joints = _ordered(frame.joints)
        if stream == "joints":
            channels = [tuple(None if j is None else float(getattr(j, key)) for j in joints)
                        for key in ("position", "velocity", "effort")]
            return JointState(JOINT_NAMES, *channels), all(_finite(c) for c in channels), \
                "missing or non-finite joint feedback"
        status_words = tuple(None if j is None else j.status_word for j in joints)
        states = tuple(None if word is None else str(word.ext_state_name) for word in status_words)
        codes = tuple(None if j is None else int(j.error_code_current) for j in joints)
        flags = tuple(None if word is None else tuple(bool(getattr(word, key)) for key in
            ("position_limit_active", "velocity_limit_active", "current_limit_active"))
            for word in status_words)
        comm = frame.comm
        payload = WujiDiagnostics(states, codes, flags, int(comm.sdk_dropped), int(comm.e2e_lost))
        valid = all(j is not None for j in joints)
        for code in set(codes) - {None, 0}:
            # Fault severity belongs to the fixed SDK catalog, not to a frame.
            if code not in self._error_warnings:
                info = self.sdk.WujiHand2.describe_error(code)
                self._error_warnings[code] = info is not None and str(info["severity"]).lower() == "warning"
            if not self._error_warnings[code]:
                valid = False
        return payload, valid, "offline joint or Hand2 device fault"

    def _observe(self, stream, sample):
        if stream != "diagnostics":
            return
        self._diagnostics = sample
        if isinstance(sample.payload, WujiDiagnostics):
            count = sample.payload.sdk_dropped
            if count != self._last_dropped:
                self._event("wuji_sdk_dropped", previous=self._last_dropped, current=count)
                self._last_dropped = count
            if self.enabled and not self._arming and not all(
                    state == "Enabled" for state in sample.payload.states):
                self._fault("Hand2 left Enabled state")

    def _current_problem(self):
        problem = super()._current_problem()
        if problem:
            return problem
        diag = self._diagnostics
        if diag is None or not diag.header.valid:
            return "waiting for valid Hand2 diagnostics"
        if self._clock() - diag.header.received_monotonic_ns > self.timeout_ns:
            return "Hand2 diagnostics timed out"
        return None

    def _read_parameters(self):
        effort = tuple(self._device.effort_limit().get())
        mit = tuple(self._device.mit_params().get())
        if len(effort) != 20 or len(mit) != 20 or not _finite(effort) or any(p is None for p in mit):
            raise RuntimeError("cannot read all 20 Hand2 control parameters")
        pairs = tuple((float(p.kp), float(p.kd)) for p in mit)
        if not all(_finite(pair) for pair in pairs):
            raise RuntimeError("non-finite Hand2 MIT parameters")
        return tuple(float(v) for v in effort), pairs

    def configure(self, profile: ControlProfile):
        if self.enabled:
            raise RuntimeError("disable Hand2 before changing parameters")
        health = self.health(check_latch=True)
        if not health.ready:
            raise RuntimeError(health.detail)
        if any(state == "Enabled" for state in self._diagnostics.payload.states):
            raise RuntimeError("Hand2 is already enabled outside this driver; "
                               "exit the other controller before restarting")
        if profile.mode != "mit":
            raise ValueError("Hand2 profile mode must be mit")
        kp, kd, current = (float(profile.parameters[key]) for key in
                           ("kp", "kd", "current_limit_a"))
        if not _finite((kp, kd, current)) or min(kp, kd) < 0 or current <= 0:
            raise ValueError("invalid Hand2 MIT parameters")
        if self._original is None:
            self._original = self._read_parameters()
        self._parameters_touched = True
        self._device.effort_limit().set(current)
        self._device.mit_params().set((kp, kd))
        effort, pairs = self._read_parameters()
        if not all(math.isclose(v, current, rel_tol=1e-4, abs_tol=1e-5) for v in effort) or not all(
                math.isclose(a, kp, rel_tol=1e-4, abs_tol=1e-5) and
                math.isclose(b, kd, rel_tol=1e-4, abs_tol=1e-5) for a, b in pairs):
            raise RuntimeError("Hand2 parameter readback does not match requested profile")
        self.profile = profile
        self._event("wuji_profile_effective", profile=profile, original=self._original)

    def engage(self, *, cancelled: Callable[[], bool] | None = None):
        health = self.health(check_latch=True)
        if not health.ready:
            raise RuntimeError(health.detail)
        if self.profile is None:
            raise RuntimeError("configure Hand2 before engagement")
        if self.enabled:
            return
        measured = self._latest.payload.position_rad
        seed = tuple(min(high, max(low, q)) for q, (low, high) in zip(measured, JOINT_LIMITS_RAD))
        # A measured angle can sit just outside a nominal limit. Bound the
        # takeover correction, but keep every transmitted target within limits.
        for name, q, target in zip(JOINT_NAMES, measured, seed):
            if abs(q - target) > math.radians(.5):
                raise RuntimeError(f"{self.side} Hand2 {name}: measured {math.degrees(q):.3f}° "
                                   "is outside joint limits by more than 0.5°; engagement refused")
        self.last_target = JointTarget(JOINT_NAMES, seed)
        if self._publisher is None:
            self._publisher = self._device.joint_command().publish()
        self._enable_attempted = True
        self._arming = True
        try:
            if cancelled and cancelled():
                raise RuntimeError("Hand2 engagement cancelled")
            enable_started_ns = self._clock()
            self._device.enable()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if cancelled and cancelled():
                    raise RuntimeError("Hand2 engagement cancelled")
                if not self.health(check_latch=True).ready:
                    raise RuntimeError(self.health(check_latch=True).detail)
                now = self._clock()
                self._send(DeviceCommand(self.device_id, uuid4().hex, self.last_target, (),
                           now, now + 50_000_000, self.profile.profile_id))
                diag = self._diagnostics
                if diag.header.received_monotonic_ns >= enable_started_ns and all(
                        state == "Enabled" for state in diag.payload.states):
                    self.enabled = True
                    self._event("wuji_enabled", feedback_ref=diag.header.ref)
                    return
                time.sleep(1 / 120)
            raise TimeoutError("Hand2 did not enable all 20 joints within 5 s")
        except Exception:
            self.disable("engagement failed")
            raise
        finally:
            self._arming = False

    def _send(self, command):
        now = self._clock()
        self._event("wuji_command", command=command, sdk_call_started_ns=now)
        try:
            self._publisher.send([self.sdk.JointCommand(q, 0., 0.)
                                  for q in command.payload.position_rad])
        except Exception as error:
            self._fault(f"SDK submission failed: {error}")
            self._notify_event(CommandEvent(self.device_id, command.command_id,
                CommandStatus.REJECTED, self._clock(), f"SDK call failed: {error}"))
            raise
        self.last_target = command.payload
        self._notify_event(CommandEvent(self.device_id, command.command_id,
            CommandStatus.ACCEPTED, self._clock(),
            "SDK send() returned; transport delivery and execution unconfirmed"))
        if self._observer_error:
            raise RuntimeError(self._observer_error)

    def submit(self, command: DeviceCommand[JointTarget]) -> Submission:
        reason = None
        target = command.payload
        if not self.enabled or not self.health(check_latch=True).ready:
            reason = self.fault or "Hand2 is not enabled and healthy"
        elif command.device_id != self.device_id or command.control_profile_id != self.profile.profile_id:
            reason = "wrong Hand2 device or control profile"
        elif command.expires_monotonic_ns <= self._clock():
            reason = "Hand2 command expired"
        elif target.joint_names != JOINT_NAMES or len(target.position_rad) != 20 or not _finite(target.position_rad):
            reason = "invalid Hand2 joint target"
        elif any(not low <= q <= high for q, (low, high) in zip(target.position_rad, JOINT_LIMITS_RAD)):
            reason = "Hand2 target exceeds mechanical limits"
        if reason is None:
            try:
                self._send(command)
                return Submission(command.command_id, True)
            except Exception as error:
                reason = str(error)
        else:
            self._notify_event(CommandEvent(self.device_id, command.command_id,
                CommandStatus.REJECTED, self._clock(), reason))
        return Submission(command.command_id, False, reason)

    def request_hold(self, reason: str):
        self._event("wuji_hold_requested", reason=reason, target=self.last_target)

    def disable(self, reason="requested"):
        self.enabled = False
        if self._enable_attempted and self._device is not None:
            self._event("wuji_disable_requested", reason=reason)
            requested_ns = self._clock()
            self._device.disable()
            self._event("wuji_disable_returned", execution_confirmed=False)
            deadline = time.monotonic() + .5
            while self._device.is_connected and time.monotonic() < deadline:
                diag = self._diagnostics
                if (diag is not None and diag.header.received_monotonic_ns >= requested_ns
                        and isinstance(diag.payload, WujiDiagnostics)
                        and all(state is not None and state != "Enabled"
                                for state in diag.payload.states)):
                    self._enable_attempted = False
                    self._event("wuji_disabled_observed", feedback_ref=diag.header.ref,
                                states=diag.payload.states)
                    return
                time.sleep(.005)
            self._event("wuji_disable_unconfirmed", reason="no fresh disabled feedback within 500 ms")

    def close(self):
        if self._closed:
            return
        errors = []
        try:
            self.disable("close")
        except Exception as error:
            errors.append(str(error))
        if self._enable_attempted:
            errors.append("Hand2 disable unconfirmed; parameters not restored")
        elif self._parameters_touched:
            effort, mit = self._original
            for resource, value in ((self._device.effort_limit(), list(effort)),
                                    (self._device.mit_params(), list(mit))):
                try:
                    resource.set(value)
                except Exception as error:
                    errors.append(str(error))
            try:
                restored = self._read_parameters()
                if any(not math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-5)
                       for left, right in zip(restored, self._original)
                       for a, b in zip(_flatten(left), _flatten(right))):
                    errors.append("Hand2 parameter restoration readback mismatch")
            except Exception as error:
                errors.append(str(error))
        if self._publisher is not None:
            try:
                self._publisher.close()
            except Exception as error:
                errors.append(str(error))
        try:
            super().close()
        except Exception as error:
            errors.append(str(error))
        if errors:
            raise RuntimeError("; ".join(errors))


def _flatten(values):
    for value in values:
        yield from value if isinstance(value, tuple) else (value,)
