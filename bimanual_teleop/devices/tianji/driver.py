"""One owner for Tianji's two arms, using the official Python SDK.

The watchdog is a host-side stop request, not a controller safety guarantee.
"""

from __future__ import annotations

import ctypes as ct
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
import logging
import math
from pathlib import Path
import threading
import time
from typing import Mapping
import uuid

from bimanual_teleop.devices.interfaces import RealtimeObserver
from bimanual_teleop.types import (
    CommandEvent, CommandStatus, ControlProfile, DeviceCommand, Event, Health,
    JointState, Pose, Sample, SampleHeader, SampleRef, Side, Submission,
)
from bimanual_teleop.devices.tianji.model import MotionProfile
from bimanual_teleop.devices.tianji.sdk import ControlSDK, FeedbackSnapshot, sdk_root as resolve_sdk_root

LOGGER = logging.getLogger(__name__)
SIDES: tuple[Side, Side] = ("left", "right")
SOURCE_MODULUS = 1_000_000
SHUTDOWN_TIMEOUT_S = 2.0


@dataclass(frozen=True)
class TianjiArmState:
    joints: JointState
    source_sequence: int
    input_sequence: int
    state: int
    commanded_state: int
    error: int
    impedance_type: int
    low_speed: int
    controller_target_rad: tuple[float | None, ...]
    native_current_permille: tuple[float | None, ...]
    wrench: tuple[float, ...] | None = None  # Fx, Fy, Fz in N; Tx, Ty, Tz in Nm, sensor axes.
    force_tag: float | None = None


@dataclass(frozen=True)
class TianjiFrame:
    packet_index: int
    arms: Mapping[Side, TianjiArmState]


@dataclass(frozen=True)
class TianjiFeedbackIssue:
    side: Side
    channel: str
    joint_indices: tuple[int, ...] = ()  # One-based, as on the controller.
    error_code: int | None = None


@dataclass(frozen=True)
class TianjiFeedbackAssessment:
    """Motion faults and unavailable observations have different consequences.

    q, dq and controller target must contain seven finite joint values. Torque,
    external torque and native current are observations unused by motion control;
    their missing values remain None. SampleHeader.valid describes the required
    numeric motion channels, not completeness of every observation. Controller
    errors, control mode and freshness still require separate checks.
    """

    packet_index: int
    received_monotonic_ns: int
    source_ref: SampleRef
    arm_counters: Mapping[Side, Mapping[str, int]]
    control_issues: tuple[TianjiFeedbackIssue, ...]
    observation_issues: tuple[TianjiFeedbackIssue, ...]

    def control_problem_for(self, sides: tuple[Side, ...]) -> str | None:
        """Select already classified issues without inspecting joint arrays again."""
        if not self.control_issues:
            return None
        issues = tuple(issue for issue in self.control_issues if issue.side in sides)
        if not issues:
            return None
        return replace(self, control_issues=issues,
                       arm_counters={side: self.arm_counters[side] for side in sides}).control_problem

    @property
    def control_problem(self) -> str | None:
        if not self.control_issues:
            return None
        issues = []
        for issue in self.control_issues:
            if issue.error_code is not None:
                issues.append(f"{issue.side} controller error {issue.error_code} "
                              f"(0x{issue.error_code & 0xffffffff:08X})")
            else:
                joints = ",".join(f"J{index}" for index in issue.joint_indices)
                issues.append(f"{issue.side} invalid {issue.channel} at {joints}")
        counters = "; ".join(f"{side} sequence={values['source_sequence']} "
                             f"input_sequence={values['input_sequence']}"
                             for side, values in self.arm_counters.items())
        ref = self.source_ref
        return ("Tianji feedback: " + "; ".join(issues) +
                f"; packet_index={self.packet_index}, received_monotonic_ns={self.received_monotonic_ns}, "
                f"ref={ref.stream}/{ref.epoch}/{ref.sequence}; {counters}")


def _invalid_joint_indices(channel: tuple[float | None, ...] | None) -> tuple[int, ...]:
    if channel is None:
        return tuple(range(1, 8))
    return tuple(index + 1 for index in range(max(7, len(channel)))
                 if index >= len(channel) or index >= 7 or channel[index] is None or
                 not math.isfinite(channel[index]))


def classify_feedback(sample: Sample[TianjiFrame], sides: tuple[Side, ...] = SIDES) -> TianjiFeedbackAssessment:
    """Classify selected arms without consulting global validity or issuing I/O."""
    control, observation, counters = [], [], {}
    for side in sides:
        arm = sample.payload.arms[side]
        counters[side] = {"source_sequence": arm.source_sequence, "input_sequence": arm.input_sequence}
        if arm.error:
            control.append(TianjiFeedbackIssue(side, "controller_error", error_code=arm.error))
        for name, channel in (("q", arm.joints.position_rad), ("dq", arm.joints.velocity_rad_s),
                              ("target", arm.controller_target_rad)):
            if indices := _invalid_joint_indices(channel):
                control.append(TianjiFeedbackIssue(side, name, indices))
        for name, channel in (("torque", arm.joints.measured_torque_nm),
                              ("external_torque", arm.joints.estimated_external_torque_nm),
                              ("current", arm.native_current_permille)):
            if indices := _invalid_joint_indices(channel):
                observation.append(TianjiFeedbackIssue(side, name, indices))
    return TianjiFeedbackAssessment(sample.payload.packet_index, sample.header.received_monotonic_ns,
                                   sample.header.ref, counters, tuple(control), tuple(observation))


@dataclass(frozen=True)
class TianjiJointCommand:
    targets: Mapping[Side, tuple[float, ...]]
    cartesian_targets: Mapping[Side, Pose]
    requested_cartesian_targets: Mapping[Side, Pose] = field(default_factory=dict)


def decode_feedback(packet: FeedbackSnapshot, epoch: str) -> Sample[TianjiFrame]:
    arms = {}
    for index, side in enumerate(SIDES):
        def values(name: str) -> tuple[float | None, ...]:
            return tuple(x if math.isfinite(x) else None for x in getattr(packet, name)[index * 7:(index + 1) * 7])

        radians = lambda name: tuple(math.radians(x) if x is not None else None for x in values(name))
        joints = JointState(
            tuple(f"{side}_joint_{j + 1}" for j in range(7)), radians("q"), radians("dq"),
            measured_torque_nm=values("torque"), estimated_external_torque_nm=values("external_torque"),
        )
        force_tag = packet.force_tag[index]
        force_tag = force_tag if math.isfinite(force_tag) else None
        wrench_raw = packet.wrench_raw[index * 6:(index + 1) * 6]
        wrench = (tuple(value / 10_000.0 for value in wrench_raw)
                  if force_tag == (116, 216)[index] and len(wrench_raw) == 6
                  and all(math.isfinite(value) for value in wrench_raw) else None)
        arms[side] = TianjiArmState(
            joints, packet.sequence[index], packet.input_sequence[index], packet.state[index],
            packet.commanded_state[index], packet.error[index], packet.impedance_type[index],
            packet.low_speed[index], radians("target"), values("current"), wrench, force_tag,
        )
    # Optional observation loss does not invalidate usable numeric motion state.
    # Errors, mode and freshness are checked independently before any command.
    valid = all(not _invalid_joint_indices(channel) for arm in arms.values() for channel in (
        arm.joints.position_rad, arm.joints.velocity_rad_s, arm.controller_target_rad))
    return Sample(SampleHeader(
        SampleRef("tianji.feedback", epoch, packet.packet_index), packet.received_ns, valid,
    ), TianjiFrame(packet.packet_index, arms))


class TianjiDriver:
    """Live feedback start; apply a profile before explicit engage.

    Observer methods must be nonblocking. Freshness means a side's source counter
    advanced recently at this host; no device timestamp or clock mapping exists.
    """

    def __init__(self, controller_ip: str, sdk_root: str | Path | None = None,
                 *, model_path: str | Path | None = None, watchdog_s: float = 0.08,
                 engagement_timeout_s: float = 1.0, record_force: bool = False):
        if not math.isfinite(watchdog_s) or watchdog_s <= 0:
            raise ValueError("watchdog_s must be positive and finite")
        if not math.isfinite(engagement_timeout_s) or engagement_timeout_s <= 0:
            raise ValueError("engagement_timeout_s must be positive and finite")
        self.controller_ip = controller_ip
        self.sdk_root = resolve_sdk_root(sdk_root)
        self.model_path = model_path
        self.record_force = record_force
        # 80 ms is one extra slow control cycle over the previous 50 ms budget.
        # The same interval bounds feedback-counter stalls.
        self.watchdog_ns = int(watchdog_s * 1e9)
        self.engagement_timeout_ns = int(engagement_timeout_s * 1e9)
        self.profile: MotionProfile | None = None
        self.engaged = False
        self.engagement_sample: Sample[TianjiFrame] | None = None
        self._sdk: ControlSDK | None = None
        self._sink: RealtimeObserver | None = None
        self._lock = threading.RLock()
        self._close_lock = threading.Lock()
        self._stop = threading.Event()
        self._closing = threading.Event()
        self._command_cancel = None
        self._motion_sides: set[Side] = set()
        self._receiver: threading.Thread | None = None
        self._watchdog: threading.Thread | None = None
        self._latest_feedback: tuple[Sample[TianjiFrame], TianjiFeedbackAssessment] | None = None
        self._packet: FeedbackSnapshot | None = None
        self._advanced: dict[Side, int] = {}
        self._sequences: dict[Side, int] = {}
        self._reordered_sides: set[Side] = set()
        self._reported_config: dict[Side, object] = {}
        self._record_reported_config = True
        self._problem: str | None = None
        self._observer_error: str | None = None
        self._hold_reason: str | None = None
        self._hold_returned_ns: int | None = None
        self._held_feedback = False
        self._mode_confirmed = False
        self._control_mode = "cartesian"
        self._moving_side: Side | None = None
        self._clearing_errors = False
        self._hold_sides: tuple[Side, ...] = ()
        self._motion_fault: str | None = None
        self._control_issues: tuple[TianjiFeedbackIssue, ...] = ()
        self._observation_issues: tuple[TianjiFeedbackIssue, ...] = ()
        self._deadline_ns = 0
        self._last_target_ns = 0
        self._last_command_id: str | None = None
        self._last_sdk_call: dict[str, object] | None = None
        self._motion_stop: dict[str, object] | None = None
        self._token = 0
        self._commands: dict[int, str] = {}
        self._epoch = uuid.uuid4().hex
        self._metadata: dict[str, object] = {
            "controller_ip": controller_ip, "sdk_root": str(self.sdk_root),
            "host_watchdog_s": watchdog_s, "source_clock": None,
            "engagement_timeout_s": engagement_timeout_s,
            "feedback_clock": "host monotonic time when a new SDK feedback snapshot is observed",
            "freshness": "host elapsed since each arm's source counter advanced; not source age",
            "current_unit": "native permille; not amperes",
            "torque_unit": "Nm; sensor torque and vendor disturbance estimate are separate",
            "controller_target": "controller-reported m_FB_Joint_Cmd, converted from degrees to radians",
            "feedback_validity": "seven finite q/dq/target values per arm; errors, mode and freshness checked separately",
            "feedback_observation_issues": [],
            "tool_frame": "flange identity; each arm has its own base frame",
            "command_deadline": "checked before SDK submission; vendor send timing is unobservable",
            "udp_send_receipt": "unavailable; SDK_SUBMITTED means only that the official API accepted the command",
        }

    @property
    def metadata(self) -> Mapping[str, object]:
        packet = self._packet
        reported = ({side: self._packet_config(packet, index) for index, side in enumerate(SIDES)}
                    if packet is not None else {})
        return dict(self._metadata, epoch=self._epoch, reported_config=reported,
                    problem=self._problem)

    @property
    def motion_stop(self) -> dict[str, object] | None:
        """First stop of this engagement, retained after SDK stop/retries.

        Published snapshots are never mutated. Readers can obtain a detached
        copy without waiting for a command or stop operation's lock.
        """
        return deepcopy(self._motion_stop)

    def start(self, sink: RealtimeObserver | None = None, *, record_reported_config: bool = True) -> None:
        """Start feedback; the flag controls config-change observation events only."""
        if self._sdk is not None:
            raise RuntimeError("Tianji driver is already started")
        if self._stop.is_set() or self._closing.is_set():
            raise RuntimeError("Create a new driver after close")
        self._sink = sink
        self._record_reported_config = record_reported_config
        sdk = ControlSDK(self.sdk_root)
        sdk.cancelled = lambda: (self._closing.is_set() or
                                 self._command_cancel is not None and self._command_cancel.is_set())
        opened = False
        try:
            sdk.open(self.controller_ip)
            opened = True
            self._sdk = sdk
            if self.record_force:
                for arm, channel in (("A", 116), ("B", 216)):
                    if not sdk.robot.set_user_specified_data(arm, channel):
                        raise RuntimeError(f"failed to select arm {arm} six-axis force data (channel {channel})")
            sdk_version, controller = sdk.versions()
            self._metadata.update(sdk_version=sdk_version, controller_version=controller)
            self._motion_stop = None
            self._receiver = threading.Thread(target=self._receive, name="tianji-receive", daemon=True)
            self._watchdog = threading.Thread(target=self._watch, name="tianji-watchdog", daemon=True)
            self._receiver.start()
            self._watchdog.start()
            self._event("tianji.started", self.metadata)
            if self._problem:
                raise RuntimeError(self._problem)
        except Exception:
            self._stop.set()
            if self._receiver and self._receiver.is_alive():
                self._receiver.join(timeout=2)
            if opened:
                sdk.close()
            self._sdk = None
            raise

    def _connected(self) -> ControlSDK:
        if self._sdk is None or self._stop.is_set() or self._closing.is_set():
            raise RuntimeError("Tianji driver is not running")
        return self._sdk

    def _publish(self, sample: Sample[object]) -> None:
        if self._sink is None or self._observer_error is not None:
            return
        try:
            if self._sink.try_publish(sample):
                return
            self._observer_failure()
        except Exception as error:
            self._observer_failure(error)

    def _event(self, kind: str, details: Mapping[str, object], stamp: int | None = None) -> None:
        self._emit_event(Event(kind, time.monotonic_ns() if stamp is None else stamp, "tianji", details))

    def _emit_event(self, event: Event | CommandEvent) -> None:
        if self._sink is None or self._observer_error is not None:
            return
        try:
            if self._sink.try_event(event):
                return
            self._observer_failure()
        except Exception as error:
            self._observer_failure(error)

    def _observer_failure(self, error: Exception | None = None) -> None:
        if self._observer_error is not None:
            return
        detail = (f"Tianji realtime observer failed: {error}" if error else
                  "Tianji realtime observer rejected live data")
        self._observer_error = self._problem = detail
        self._metadata["observer_error"] = detail
        LOGGER.error(detail)

    def get_latest(self) -> Sample[TianjiFrame] | None:
        feedback = self._latest_feedback
        return feedback[0] if feedback is not None else None

    def _feedback_problem(self, sides: tuple[Side, ...], now: int, *, idle: bool = False) -> str | None:
        feedback = self._latest_feedback
        if feedback is None:
            return "No Tianji feedback"
        sample, assessment = feedback
        for side in sides:
            arm = sample.payload.arms[side]
            if side in self._reordered_sides:
                return f"{side} source counter reset/reordered; await a subsequent advancing sample"
            if now - self._advanced.get(side, 0) >= self.watchdog_ns:
                return f"{side} source counter has not advanced within the host watchdog interval"
            if problem := assessment.control_problem_for((side,)):
                return problem
            if self.engaged and self._mode_confirmed and not self._in_control_mode(arm):
                return f"{side} is not reporting the engaged {self._control_mode} mode"
        return None

    def _in_control_mode(self, arm: TianjiArmState) -> bool:
        return (arm.state == 1 if self._control_mode == "position"
                else arm.state == 3 and arm.impedance_type == 2)

    def health(self, *, sides: tuple[Side, ...] | None = None) -> Health:
        now = time.monotonic_ns()
        sides = (self.profile.active_arms if self.profile else SIDES) if sides is None else tuple(sides)
        if not sides or len(set(sides)) != len(sides) or any(side not in SIDES for side in sides):
            raise ValueError("Tianji health sides must be left, right, or both")
        problem = self._problem or (self._motion_fault if self.engaged else None) or self._feedback_problem(sides, now)
        if self._sdk is None or self._stop.is_set():
            problem = "Tianji driver is closed"
        elif self._hold_reason:
            problem = f"Hold request pending: {self._hold_reason}"
            if detail := self._metadata.get("last_hold_error"):
                problem += f" ({detail})"
        return Health(problem is None, now, problem)

    def _on_feedback(self, packet: FeedbackSnapshot) -> None:
        sample = decode_feedback(packet, self._epoch)
        assessment = classify_feedback(sample)
        for side, arm in sample.payload.arms.items():
            previous = self._sequences.get(side)
            delta = None if previous is None else (arm.source_sequence - previous) % SOURCE_MODULUS
            reset = delta is not None and delta >= SOURCE_MODULUS // 2
            if reset:
                if self.profile is None or side in self.profile.active_arms:
                    self._motion_fault = f"{side} source sequence reset/reordered; explicit re-engagement required"
                self._reordered_sides.add(side)
            elif previous is None or previous != arm.source_sequence:
                self._advanced[side] = packet.received_ns
                self._reordered_sides.discard(side)
            if previous is not None:
                if delta > 1:
                    self._event("tianji.source_gap" if delta < SOURCE_MODULUS // 2 else "tianji.source_reset",
                                {"side": side, "previous": previous, "current": arm.source_sequence,
                                 "missing": delta - 1 if delta < SOURCE_MODULUS // 2 else None,
                                 "meaning": "unobserved source frames; SDK cache sampling can skip frames; not UDP loss",
                                 "packet_index": packet.packet_index}, packet.received_ns)
            self._sequences[side] = arm.source_sequence
            if self._sink is not None and self._record_reported_config:
                config = self._packet_config(packet, SIDES.index(side))
                if self._reported_config.get(side) != config:
                    self._reported_config[side] = config
                    self._event("tianji.reported_config", {"side": side, "configuration": config,
                                "packet_index": packet.packet_index}, packet.received_ns)
        self._packet = packet
        # Publish the sample and its classification together, so health readers
        # never pair a new sample with the preceding packet's assessment.
        self._latest_feedback = sample, assessment
        stable_low_speed = self.profile is not None and all(
            arm.low_speed == 1 and (arm.state == 0 or self._in_control_mode(arm)) and
            all(q is not None for q in arm.joints.velocity_rad_s)
            for side, arm in sample.payload.arms.items() if side in (self._hold_sides or self.profile.active_arms))
        if self._held_feedback and not stable_low_speed:
            self._held_feedback = False
        if self._hold_returned_ns is not None and not self._held_feedback and self.profile:
            if stable_low_speed and all(self._advanced.get(side, 0) > self._hold_returned_ns
                                        for side in (self._hold_sides or self.profile.active_arms)):
                self._held_feedback = True
                self._event("tianji.hold_low_speed_observed", {
                    "packet_index": packet.packet_index,
                    "joint_velocity_rad_s": {side: sample.payload.arms[side].joints.velocity_rad_s
                                             for side in self.profile.active_arms},
                    "meaning": "fresh controller low-speed flag in idle/owned control mode; not an independent physical-stop measurement",
                }, packet.received_ns)
        if self.engaged:
            sides = ((self._moving_side,) if self._moving_side else
                     self.profile.active_arms if self.profile else SIDES)
            self._motion_fault = self._motion_fault or assessment.control_problem_for(sides)
            if self._mode_confirmed and self._motion_fault is None:
                for side in sides:
                    arm = sample.payload.arms[side]
                    if not self._in_control_mode(arm):
                        self._motion_fault = (
                            f"{side} is not reporting the engaged {self._control_mode} mode "
                            f"(state={arm.state}, impedance_type={arm.impedance_type}, "
                            f"packet_index={packet.packet_index}, received_monotonic_ns={packet.received_ns}); "
                            "explicit re-engagement required")
                        self._event("tianji.control_mode_changed", {
                            "side": side, "expected_mode": self._control_mode, "state": arm.state,
                            "impedance_type": arm.impedance_type, "packet_index": packet.packet_index,
                            "source_sequence": arm.source_sequence, "input_sequence": arm.input_sequence,
                        }, packet.received_ns)
                        break
        if assessment.control_issues != self._control_issues:
            self._control_issues = assessment.control_issues
            self._event("tianji.invalid_feedback" if assessment.control_issues else "tianji.feedback_recovered",
                        asdict(assessment), packet.received_ns)
        if assessment.observation_issues != self._observation_issues:
            self._observation_issues = assessment.observation_issues
            self._metadata["feedback_observation_issues"] = [asdict(issue) for issue in self._observation_issues]
            self._event("tianji.observation_unavailable" if assessment.observation_issues else
                        "tianji.observation_recovered", asdict(assessment), packet.received_ns)
        self._publish(sample)

    @staticmethod
    def _packet_config(packet: FeedbackSnapshot, index: int) -> dict[str, object]:
        def array(name, width):
            return tuple(x if math.isfinite(x) else None
                         for x in getattr(packet, name)[index * width:(index + 1) * width])
        return {"stiffness_native": array("cart_k", 7), "damping_native": array("cart_d", 7),
                "tool_pose_mm_deg": array("tool_pose", 6), "tool_dynamics_native": array("tool_dynamics", 10),
                "velocity_ratio_percent": packet.velocity_ratio[index],
                "acceleration_ratio_percent": packet.acceleration_ratio[index],
                "force_type": packet.force_type[index], "impedance_rotation_native": array("impedance_rotation", 7)}

    def _drain(self) -> int:
        if self._sdk is None:
            return 0
        packet = self._sdk.poll_feedback()
        if packet is None:
            return 0
        self._on_feedback(packet)
        return 1

    def _receive(self) -> None:
        try:
            while not self._stop.is_set():
                self._drain()
                self._stop.wait(0.001)
        except Exception as error:
            self._problem = f"Tianji receiver failed: {error}"
            self._event("tianji.receiver_failed", {"error": str(error)})

    def _watch(self) -> None:
        while not self._stop.wait(min(0.005, self.watchdog_ns / 4e9)):
            # A submission can renew its deadline while the sdk call holds
            # this lock. Inspect the deadline after acquiring it, or an expired
            # old deadline can stop a newly accepted, still-valid command.
            with self._lock:
                if self._closing.is_set():
                    return
                now = time.monotonic_ns()
                if self.engaged and self.profile:
                    sides = (self._moving_side,) if self._moving_side else self.profile.active_arms
                    reason = self._problem or self._motion_fault or self._feedback_problem(sides, now)
                    if self._moving_side is None and now >= self._deadline_ns:
                        reason = reason or "Accepted target expired or no target arrived within host watchdog interval"
                    if reason:
                        self.request_hold(reason)
                elif self._hold_reason and not self._clearing_errors:
                    self.request_hold(self._hold_reason)

    def _next_token(self, command_id: str) -> int:
        self._token += 1
        self._commands[self._token] = command_id
        # Bound command IDs retained for observation events.
        if len(self._commands) > 2048:
            oldest = min(self._commands)
            del self._commands[oldest]
        return self._token

    @staticmethod
    def _mask(sides: tuple[Side, ...]) -> int:
        return sum(1 << SIDES.index(side) for side in sides)

    def _require_feedback(self, sides: tuple[Side, ...], *, idle: bool = False) -> None:
        reason = self._problem or self._hold_reason or (self._motion_fault if self.engaged else None) or self._feedback_problem(sides, time.monotonic_ns(), idle=idle)
        if reason:
            raise RuntimeError(reason)

    def clear_errors(self, sides=SIDES, *, cancel=None, timeout_s=2.) -> dict:
        """Reset each faulty stationary arm once, then verify controller and servos.

        The caller has confirmed physical emergency release. No enable or target
        command is issued. Queries run only outside the realtime motion loop.
        """
        sides = tuple(sides)
        if not sides or len(set(sides)) != len(sides) or any(s not in SIDES for s in sides):
            raise ValueError("Select left, right, or both arms to clear errors")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Clear-errors timeout must be positive and finite")

        def snapshot():
            if self._closing.is_set() or self._stop.is_set() or cancel is not None and cancel.is_set():
                raise RuntimeError("Tianji clear errors cancelled")
            now, sample = time.monotonic_ns(), self.get_latest()
            if self._problem or sample is None:
                raise RuntimeError(self._problem or "No feedback for clearing errors")
            for side in sides:
                arm = sample.payload.arms[side]
                if side in self._reordered_sides or now-self._advanced.get(side, 0) >= self.watchdog_ns:
                    raise RuntimeError(f"{side} clear-errors feedback is stale/reordered")
                if (arm.low_speed != 1 or arm.state not in (0, 1, 3, 100)
                        or any(_invalid_joint_indices(channel) for channel in
                               (arm.joints.position_rad, arm.joints.velocity_rad_s))):
                    raise RuntimeError(f"{side} clear errors requires valid stationary feedback "
                                       f"(state={arm.state}, error={arm.error})")
            return sample

        def errors():
            result = {}
            for side in sides:
                servos = []
                for joint in range(7):
                    with self._lock:
                        snapshot()
                        try:
                            servos.append(sdk.get_int(f"SERVO{SIDES.index(side)}ERR{joint}"))
                        except RuntimeError as error:
                            raise RuntimeError(f"{side} J{joint+1} servo error query failed: {error}") from error
                arm = snapshot().payload.arms[side]
                result[side] = {"state": arm.state, "controller": arm.error, "servos": tuple(servos)}
            return result

        with self._lock:
            sdk = self._connected()
            if self.engaged or self._moving_side is not None or self._clearing_errors:
                raise RuntimeError("Pause before clearing Tianji errors")
            snapshot()
            if self._hold_reason and not set(self._hold_sides).issubset(sides):
                raise RuntimeError("Stop all pending hold arms before clearing errors")
            self._clearing_errors = True
        try:
            # Never retire a pending stop from a low-speed packet that predates
            # it: the controller may still have an unobserved motion target.
            stop_ns = self._hold_returned_ns or 0
            if self._hold_reason and self._motion_stop:
                stop_ns = max(stop_ns, self._motion_stop["monotonic_ns"])
            deadline = time.monotonic() + timeout_s
            while stop_ns and any(self._advanced.get(side, 0) <= stop_ns for side in sides):
                snapshot()
                if time.monotonic() >= deadline:
                    raise RuntimeError("No fresh stationary feedback after Tianji stop")
                self._stop.wait(.005)
            with self._lock:
                sample = snapshot()
                if self._hold_reason:
                    if any(sample.payload.arms[side].state not in (0, 100) for side in sides):
                        raise RuntimeError("Tianji stop is still pending; cannot clear errors")
                    # A fresh stationary disabled/fault state supersedes a
                    # rejected stop, while motion modes still require its ACK.
                    self._hold_reason = None
            initial = errors()
            requested, sent = [], {}
            for side, faults in initial.items():
                if not (faults["controller"] or faults["state"] == 100 or any(faults["servos"])):
                    continue
                with self._lock:
                    snapshot()
                    token = self._next_token(f"clear-errors-{side}-{uuid.uuid4().hex}")
                    try:
                        sent[side] = self._invoke("clear_errors", SIDES.index(side), token=token)
                    except RuntimeError as error:
                        raise RuntimeError(f"{side} clear errors failed ({faults}): {error}") from error
                requested.append(side)
            deadline = time.monotonic() + timeout_s
            final = initial
            while requested:
                sample = snapshot()
                # The SDK recommends allowing 200 ms for reset. Require a newer
                # source frame too; an ACK alone never establishes recovery.
                if all(self._advanced.get(side, 0) > stamp + 200_000_000 for side, stamp in sent.items()):
                    final = errors()
                    if all(not e["controller"] and e["state"] != 100 and not any(e["servos"])
                           for e in final.values()):
                        break
                    raise RuntimeError(f"Tianji errors remain after one reset: {final}")
                if time.monotonic() >= deadline:
                    remaining = {side: {"state": sample.payload.arms[side].state,
                                       "controller": sample.payload.arms[side].error,
                                       "servos": final[side]["servos"]} for side in sides}
                    raise RuntimeError(f"Tianji clear errors timed out: {remaining}")
                self._stop.wait(.005)
            snapshot()
            result = {"requested_arms": requested, "initial_errors": initial, "final_errors": final}
            self._event("tianji.errors_cleared", result)
            return result
        finally:
            with self._lock:
                self._clearing_errors = False

    def configure(self, profile: ControlProfile) -> None:
        parsed = MotionProfile.from_control_profile(profile, self.model_path)
        with self._lock:
            self._connected()
            if self._clearing_errors or self._moving_side is not None:
                raise RuntimeError("Tianji maintenance is in progress")
            profiles = [parsed.arms.get(side) for side in SIDES]
            token = self._next_token(f"configure-{uuid.uuid4().hex}")
            self._event("tianji.configure_requested", {"sdk_token": token, "profile": asdict(profile)})
            submitted = self._invoke("configure", self._mask(parsed.active_arms), profiles, token=token)
            expected = {}
            for side, arm in parsed.arms.items():
                expected[side] = {
                    "cart_k": (*arm.stiffness, arm.nullspace_stiffness),
                    "cart_d": (*arm.damping, arm.nullspace_damping),
                    "tool_pose": (0.,) * 6, "tool_dynamics": arm.tool_dyn10,
                    "impedance_rotation": (0.,) * 7,
                }
            deadline = time.monotonic_ns() + self.engagement_timeout_ns
            while True:
                self._connected()
                self._require_feedback(parsed.active_arms)
                packet = self._packet
                confirmed = all(self._advanced.get(side, 0) > submitted for side in parsed.active_arms)
                for side, channels in expected.items():
                    index, arm = SIDES.index(side), parsed.arms[side]
                    confirmed &= (packet.velocity_ratio[index] == arm.velocity_ratio
                                  and packet.acceleration_ratio[index] == arm.acceleration_ratio
                                  and packet.force_type[index] == 1)
                    for name, values in channels.items():
                        width = len(values)
                        confirmed &= tuple(getattr(packet, name)[index*width:(index+1)*width]) == tuple(
                            ct.c_float(value).value for value in values)
                if confirmed:
                    break
                if time.monotonic_ns() >= deadline:
                    raise RuntimeError("Controller did not report the requested configuration")
                self._stop.wait(.001)
            self.profile = parsed
            self._metadata["control_profile"] = asdict(profile)
            self._event("tianji.configure_sdk_returned", {"sdk_token": token, "profile_id": profile.profile_id})

    def move_joints(self, side: Side, target_rad: tuple[float, ...], *,
                    cancel=None, timeout_s=60.) -> Sample[TianjiFrame]:
        """Run a cancellable controller-planned move; confirm measured arrival."""
        target = tuple(target_rad)
        if len(target) != 7 or any(q is None or not math.isfinite(q) for q in target):
            raise ValueError("Joint move requires seven finite joints")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Joint move timeout must be positive and finite")
        deadline = time.monotonic() + timeout_s

        def check():
            if self._closing.is_set() or self._stop.is_set() or cancel is not None and cancel.is_set():
                raise RuntimeError("Tianji joint move cancelled")
            if time.monotonic() >= deadline:
                raise RuntimeError(f"{side} joint move timed out after {timeout_s:g} s")
            if not self.engaged:
                raise RuntimeError(self._motion_stop["reason"] if self._motion_stop
                                   else "Tianji joint move stopped")
            self._require_feedback((side,))

        with self._lock:
            self._connected()
            if self.profile is None or side not in self.profile.active_arms:
                raise RuntimeError("Joint move requires a configured arm")
            if self.engaged or self._moving_side is not None or self._clearing_errors:
                raise RuntimeError("Tianji is already engaged or clearing errors")
            if cancel is not None and cancel.is_set():
                raise RuntimeError("Tianji joint move cancelled")
            self._require_feedback((side,))
            angles = tuple(math.degrees(q) for q in target)
            encoded_target = tuple(ct.c_float(q).value for q in angles)
            arm_profile = self.profile.arms[side]
            self._command_cancel = cancel
            self._moving_side, self._control_mode = side, "position"
            self.engaged, self._mode_confirmed = True, False
            self._motion_fault = None
            self._hold_returned_ns, self._held_feedback = None, False
        try:
            index = SIDES.index(side)
            with self._lock:
                check()
                mode_sent_ns = None
                if self.get_latest().payload.arms[side].state != 1:
                    seed = tuple(math.degrees(q) for q in self.get_latest().payload.arms[side].joints.position_rad)
                    token = self._next_token(f"position-mode-{side}-{uuid.uuid4().hex}")
                    now = time.monotonic_ns()
                    self._event("tianji.position_mode_requested", {"side": side, "seed_deg": seed,
                                                                 "sdk_token": token})
                    self._motion_sides.add(side)
                    mode_sent_ns = self._invoke("move_joints", index, seed,
                                arm_profile.velocity_ratio, arm_profile.acceleration_ratio,
                                now+self.watchdog_ns, token=token)
                    self._motion_stop = None
                    mode_deadline = now+self.engagement_timeout_ns
            while mode_sent_ns is not None:
                with self._lock:
                    check()
                    if (self._advanced.get(side, 0) > mode_sent_ns
                            and self.get_latest().payload.arms[side].state == 1):
                        break
                    if time.monotonic_ns() >= mode_deadline:
                        raise RuntimeError(f"{side} position mode was not reported before the startup deadline")
                self._stop.wait(.005)
            with self._lock:
                check()
                self._mode_confirmed = True
                token = self._next_token(f"joint-move-{side}-{uuid.uuid4().hex}")
                now = time.monotonic_ns()
                self._event("tianji.joint_move_requested", {"side": side, "target_rad": target,
                            "sdk_token": token, "velocity_ratio": arm_profile.velocity_ratio,
                            "acceleration_ratio": arm_profile.acceleration_ratio})
                command = [0.] * 14
                command[index*7:index*7+7] = angles
                self._motion_sides.add(side)
                sent_ns = self._invoke("submit", 1 << index, command, now+self.watchdog_ns, token=token)
                self._motion_stop = None
            while True:
                with self._lock:
                    check()
                    sample = self.get_latest()
                    arm = sample.payload.arms[side]
                    reported_target = tuple(ct.c_float(math.degrees(q)).value for q in arm.controller_target_rad)
                    if (self._advanced.get(side, 0) > sent_ns
                            and arm.state == 1 and arm.low_speed == 1
                            and reported_target == encoded_target
                            and all(abs(q-goal) <= math.radians(.5)
                                    for q, goal in zip(arm.joints.position_rad, target))):
                        self.engaged = False
                        self._event("tianji.joint_move_completed", {"side": side, "feedback": asdict(sample)})
                        return sample
                self._stop.wait(.005)
        except BaseException as error:
            self.request_hold(str(error) or type(error).__name__)
            raise
        finally:
            with self._lock:
                self._moving_side = None
                self._command_cancel = None

    def engage(self) -> None:
        """Seed Cartesian impedance control from fresh measured joints."""
        with self._lock:
            self._connected()
            if self.profile is None:
                raise RuntimeError("A configured profile is required before engage")
            if self.engaged or self._clearing_errors or self._moving_side is not None:
                raise RuntimeError("Tianji is already engaged or performing maintenance")
            self._require_feedback(self.profile.active_arms)
            sample = self.get_latest()
            targets = {side: sample.payload.arms[side].joints.position_rad for side in self.profile.active_arms}
            now = time.monotonic_ns()
            command = DeviceCommand("tianji", f"engage-{uuid.uuid4().hex}", TianjiJointCommand(targets, {}),
                                    (sample.header.ref,), now, now + self.engagement_timeout_ns, self.profile.profile_id)
            self._motion_fault = None
            self._mode_confirmed = False
            self._control_mode = "cartesian"
            result = self._send_command(command, "engage")
            if not result.accepted:
                raise RuntimeError(result.reason)
            # The accepted seed begins a new motion episode. A later mode
            # confirmation failure must record this episode's own stop cause.
            self._motion_stop = None
            self.engaged = True
            self.engagement_sample = sample
            self._hold_returned_ns, self._held_feedback = None, False
            self._hold_sides = ()
            self._mode_confirmed = False
            while time.monotonic_ns() < self._deadline_ns:
                if self._closing.is_set():
                    raise RuntimeError("Tianji engagement cancelled by shutdown")
                # engage owns the command lock while waiting, so the watchdog
                # thread cannot issue hold here. Check feedback in this loop too.
                reason = self._problem or self._motion_fault or self._feedback_problem(
                    self.profile.active_arms, time.monotonic_ns())
                if reason:
                    self.request_hold(reason)
                    raise RuntimeError(f"Cannot confirm cartesian engagement: {reason}")
                latest = self.get_latest()
                if all(self._advanced.get(side, 0) >= now and self._in_control_mode(latest.payload.arms[side])
                       for side in self.profile.active_arms):
                    # The measured seed holds the current pose, so retain the
                    # startup grace period until the first planned target is
                    # accepted.  Starting the runtime watchdog here made one
                    # dual-arm IK cycle consume most or all of its budget.
                    confirmed_ns = time.monotonic_ns()
                    self._mode_confirmed = True
                    self._event("tianji.engagement_mode_reported", {"packet_index": latest.payload.packet_index,
                                                                 "command_id": command.command_id,
                                                                 "control_mode": "cartesian",
                                                                 "elapsed_ms": (confirmed_ns - now) / 1e6,
                                                                 "first_target_deadline_ns": self._deadline_ns})
                    return
                self._stop.wait(0.001)
            self.request_hold("Engagement mode was not reported before the startup deadline")
            raise RuntimeError("Controller did not confirm cartesian engagement before startup timeout")

    def submit(self, command: DeviceCommand[TianjiJointCommand]) -> Submission:
        with self._lock:
            try:
                self._connected()
                if not self.engaged or self.profile is None:
                    raise RuntimeError(self._motion_stop["reason"] if self._motion_stop else "Tianji is not engaged")
                if not self._mode_confirmed:
                    raise RuntimeError("Engagement mode has not been reported")
                self._require_feedback(self.profile.active_arms)
                return self._send_command(command, "submit")
            except (ValueError, RuntimeError) as error:
                self._emit_event(CommandEvent("tianji", command.command_id, CommandStatus.REJECTED,
                                              time.monotonic_ns(), str(error)))
                if self.engaged:
                    self.request_hold(str(error))
                return Submission(command.command_id, False, str(error))

    def _send_command(self, command: DeviceCommand[TianjiJointCommand], operation: str) -> Submission:
        now, profile = time.monotonic_ns(), self.profile
        if command.device_id != "tianji" or command.control_profile_id != profile.profile_id:
            raise ValueError("Device or control profile does not match Tianji")
        if command.expires_monotonic_ns <= now or command.created_monotonic_ns > now:
            raise ValueError("Target expired or has a future creation time")
        if operation == "submit" and now >= self._deadline_ns:
            raise ValueError("Previous target watchdog expired; re-engage before submitting")
        if set(command.payload.targets) != set(profile.active_arms):
            raise ValueError("Targets must cover exactly the configured active arms")
        targets = {side: tuple(positions) for side, positions in command.payload.targets.items()}
        q = [0.] * 14
        for side, positions in targets.items():
            if len(positions) != 7 or any(value is None or not math.isfinite(value) for value in positions):
                raise ValueError(f"{side} target must contain seven finite joints")
            q[SIDES.index(side) * 7:(SIDES.index(side) + 1) * 7] = tuple(math.degrees(value) for value in positions)
        token = self._next_token(command.command_id)
        expires = min(command.expires_monotonic_ns, now + self.watchdog_ns)
        if self._problem:
            raise RuntimeError(self._problem)
        self._connected()
        self._motion_sides.update(profile.active_arms)
        submitted = self._invoke(operation, self._mask(profile.active_arms), q, expires, token=token)
        accepted_ns = time.monotonic_ns()
        if operation == "submit":
            self._event("tianji_command_submitted", {"command": command}, submitted)
        # Target freshness is enforced above and by the deadline passed to the
        # native SDK.  Once that fresh target is accepted, the host liveness
        # watchdog gets its full interval; IK time must not shorten the interval
        # in which the next target is allowed to arrive.
        self._deadline_ns = (min(command.expires_monotonic_ns, now + self.engagement_timeout_ns)
                             if operation == "engage" else accepted_ns + self.watchdog_ns)
        self._last_target_ns = accepted_ns
        self._last_command_id = command.command_id
        return Submission(command.command_id, True)

    def _invoke(self, operation: str, *args, token: int) -> int:
        started = time.monotonic_ns()
        try:
            submitted = getattr(self._sdk, operation)(*args)
            if type(submitted) is not int or submitted <= 0:
                raise RuntimeError(f"Official Tianji SDK did not accept {operation}")
            command_id = self._commands[token]
            self._emit_event(CommandEvent("tianji", command_id, CommandStatus.ACCEPTED, submitted))
            self._emit_event(CommandEvent("tianji", command_id, CommandStatus.SDK_SUBMITTED, submitted,
                                          "official API accepted; UDP delivery and execution unobserved"))
            return submitted
        finally:
            completed = time.monotonic_ns()
            self._last_sdk_call = {
                "operation": operation, "started_monotonic_ns": started,
                "completed_monotonic_ns": completed, "duration_ms": (completed - started) / 1e6,
            }

    def _retain_motion_stop(self, reason: str, sides: tuple[Side, ...]) -> None:
        if self._motion_stop is not None:
            return
        now, sample = time.monotonic_ns(), self.get_latest()
        deadline = self._deadline_ns if self._moving_side is None else 0
        self._motion_stop = {
            "schema": "tianji.motion_stop.v1", "reason": reason, "monotonic_ns": now,
            "last_target_ns": self._last_target_ns or None, "deadline_ns": deadline or None,
            "target_age_ms": max(0, now - self._last_target_ns) / 1e6 if self._last_target_ns else None,
            "deadline_overrun_ms": max(0, now - deadline) / 1e6 if deadline else None,
            "feedback_age_ms": {side: max(0, now - self._advanced[side]) / 1e6
                                if side in self._advanced else None for side in sides},
            "feedback_packet_index": sample.payload.packet_index if sample else None,
            "feedback_received_monotonic_ns": sample.header.received_monotonic_ns if sample else None,
            "last_command_id": self._last_command_id,
            "last_native_call": deepcopy(self._last_sdk_call),
        }

    def request_hold(self, reason: str) -> None:
        with self._lock:
            if self._closing.is_set():
                return
            if not self.engaged and not self._hold_reason:
                return
            sides = (self._moving_side,) if self._moving_side else self.profile.active_arms
            self._retain_motion_stop(reason, sides)
            reason = self._motion_stop["reason"]
            self.engaged = False
            self._command_cancel = None
            self._mode_confirmed = False
            if self._hold_reason is None:
                self._hold_sides = (self._moving_side,) if self._moving_side else self.profile.active_arms
                self._event("tianji.hold_requested", {"reason": reason, "active_arms": self._hold_sides,
                                                     "motion_stop": self.motion_stop})
            sides = self._hold_sides
            self._hold_reason = reason
            token = self._next_token(f"hold-{uuid.uuid4().hex}")
            method = "sdk_stop"
            try:
                sample, assessment = self._latest_feedback or (None, None)
                if (self._control_mode == "cartesian" and sample is not None
                        and assessment.control_problem_for(sides) is None
                        and not self._feedback_problem(sides, time.monotonic_ns())
                        and all(self._in_control_mode(sample.payload.arms[s]) for s in sides)):
                    # Cartesian following holds a joint reference in torque mode.
                    # Freeze it at measured q; RSTA can be rejected in this mode.
                    # Do not switch to position mode or retain a distant old goal.
                    q = [0.] * 14
                    for side in sides:
                        index = SIDES.index(side)
                        q[index*7:index*7+7] = tuple(math.degrees(v) for v in
                                                  sample.payload.arms[side].joints.position_rad)
                    expiry = min(sample.header.received_monotonic_ns + self.watchdog_ns,
                                 *(self._advanced[s] + self.watchdog_ns for s in sides))
                    self._invoke("submit", self._mask(sides), q, expiry, token=token)
                    method = "cartesian_measured_hold"
                else:
                    self._invoke("hold", self._mask(sides), token=token)
            except RuntimeError as error:
                # A prior datagram can still be pending; the watchdog retries.
                self._metadata["last_hold_error"] = str(error)
                # A controller may reject RSTA on an already completed position
                # stream. Explicitly replace that target with the measured pose,
                # only in fresh, error-free position mode. No PVT or torque-mode
                # stop is replaced by this fallback. Feedback must still confirm.
                if (self._control_mode != "position" or
                        "returned 1" not in str(error) or self._feedback_problem(sides, time.monotonic_ns()) or
                        any(self.get_latest().payload.arms[s].state != 1 for s in sides)):
                    return
                sample = self.get_latest()
                try:
                    for side in sides:
                        token = self._next_token(f"position-hold-{side}-{uuid.uuid4().hex}")
                        q = tuple(math.degrees(v) for v in sample.payload.arms[side].joints.position_rad)
                        arm = self.profile.arms[side]
                        self._invoke("move_joints", SIDES.index(side), q,
                                     arm.velocity_ratio, arm.acceleration_ratio,
                                     time.monotonic_ns()+self.watchdog_ns, token=token)
                except (RuntimeError, ValueError) as fallback_error:
                    self._metadata["position_hold_error"] = str(fallback_error)
                    return
                self._event("tianji.position_hold_target_replaced", {"rejected_stop": str(error),
                            "active_arms": sides, "feedback": asdict(sample), "physical_stop_confirmed": False})
                token, method = self._token, "measured_position_target"
            self._hold_reason = None
            self._metadata.pop("last_hold_error", None)
            self._hold_returned_ns = time.monotonic_ns()
            self._held_feedback = False
            self._event("tianji.hold_sdk_returned", {"sdk_token": token, "reason": reason,
                                                   "method": method,
                                                   "physical_stop_confirmed": False})

    def _disable_for_close(self, sides: tuple[Side, ...]) -> None:
        deadline = time.monotonic() + SHUTDOWN_TIMEOUT_S
        token = self._next_token(f"disable-{uuid.uuid4().hex}")
        submitted = None
        detail = "官方 SDK 未接受下伺服指令"
        self._event("tianji.disable_requested", {"active_arms": sides, "sdk_token": token})
        while time.monotonic() < deadline:
            if submitted is None:
                try:
                    submitted = self._invoke("disable", self._mask(sides), token=token)
                except RuntimeError as error:
                    # Retry only a busy send slot; never enable, clear errors or
                    # replace the target as a shutdown fallback.
                    if "pending" not in str(error):
                        raise
                    detail = str(error)
            if self._receiver is None or not self._receiver.is_alive():
                self._drain()
            if submitted is not None:
                detail = "未收到所控机械臂的新鲜 IDLE / 低速反馈"
                sample = self.get_latest()
                now = time.monotonic_ns()
                if (sample is not None and sample.header.received_monotonic_ns > submitted and all(
                    submitted < self._advanced.get(side, 0) <= now
                    and now - self._advanced[side] < self.watchdog_ns
                    and side not in self._reordered_sides
                    # m_CmdState can be -1 after disable; m_CurState is the arm mode.
                    and sample.payload.arms[side].state == 0
                    and sample.payload.arms[side].error == 0
                    and sample.payload.arms[side].low_speed == 1
                    and all(v is not None and abs(v) <= math.radians(.5)
                            for v in sample.payload.arms[side].joints.velocity_rad_s)
                    for side in sides
                )):
                    self._event("tianji.disable_confirmed", {
                        "active_arms": sides, "sdk_token": token,
                        "feedback": asdict(sample), "physical_stop_confirmed": False})
                    return
            time.sleep(.002)
        raise RuntimeError(detail)

    def close(self) -> None:
        # Latch before taking the command lock so in-flight engagement/moves
        # exit their waits. The receiver stays alive until disable is checked.
        self._closing.set()
        with self._close_lock:
            self._close()

    def _close(self) -> None:
        error = None
        with self._lock:
            sdk = self._sdk
            if sdk is None:
                return
            self.engaged = False
            self._mode_confirmed = False
            sides = tuple(side for side in SIDES if side in self._motion_sides)
            try:
                if sides:
                    self._disable_for_close(sides)
            except Exception as problem:
                error = RuntimeError(f"天机退出停机未确认，请按实体急停并检查机器人：{problem}")
                LOGGER.error("%s", error)
                self._event("tianji.disable_unconfirmed", {"error": str(problem), "active_arms": sides})
            finally:
                self._stop.set()
        # Do not hold the command lock while joining the watchdog.
        for thread in (self._watchdog, self._receiver):
            if thread and thread is not threading.current_thread():
                thread.join(timeout=2)
        try:
            sdk.close()
        except Exception as problem:
            if error is not None:
                raise RuntimeError(f"{error}; connection release failed: {problem}") from error
            raise
        finally:
            self._sdk = None
        if error is not None:
            raise error
