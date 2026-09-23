"""Wuji finger retargeting and an independent 120 Hz hand controller.

Device receivers own sample continuity. This loop uses the latest shape while
checking their latched faults, including faults hidden by a newer valid frame.
"""

from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import version
from importlib.util import find_spec
import math
import threading
import time
import uuid

from bimanual_teleop.devices.wuji.adapter import (
    JOINT_LIMITS_RAD, JOINT_NAMES, WujiGloveSource, WujiHandDriver, WujiSdkSession,
)
from bimanual_teleop.devices.wuji.config import sdk_user_name, validate_control_config
from bimanual_teleop.system import SystemState
from bimanual_teleop.types import ControlProfile, DeviceCommand, Event, Health, JointTarget


def preflight():
    """Check installation before motion; the hand process loads native modules."""
    if version("wuji-sdk") != "2026.8.31":
        raise RuntimeError("Install wuji-sdk==2026.8.31 in bimanual-teleop")
    for package in ("numpy", "wuji_sdk"):
        if find_spec(package) is None:
            raise ImportError(f"No module named {package!r}")


class WujiHandRetargeter:
    """One official, stateful solver per side; no additional coordinate mapping."""

    def __init__(self, side, *, sdk=None):
        if side not in ("left", "right"):
            raise ValueError("Hand side must be left or right")
        self.side, self.sdk = side, sdk
        self._session = None

    def reset(self):
        if self._session is not None:
            self._session.reset()

    def solve(self, sample):
        import numpy as np

        skeleton = sample.payload
        positions = np.asarray(skeleton.positions_m, dtype=np.float32)
        if (not sample.header.valid or positions.shape != (21, 3)
                or not np.isfinite(positions).all()):
            raise ValueError(f"{self.side} glove skeleton is invalid")
        if self._session is None:
            if self.sdk is None:
                import wuji_sdk
                self.sdk = wuji_sdk
            side = self.sdk.Handedness.Left if self.side == "left" else self.sdk.Handedness.Right
            self._session = self.sdk.RetargetSession.for_hand(self.sdk.HandModel.WujiHand2, side=side)
        q = np.asarray(self._session.step(positions))
        if q.shape != (20,) or not np.isfinite(q).all():
            raise ValueError(f"{self.side} retargeter returned invalid joint positions")
        values = tuple(float(max(low, min(high, value)))
                       for value, (low, high) in zip(q, JOINT_LIMITS_RAD))
        return JointTarget(JOINT_NAMES, values)


class WujiTeleop:
    """Explicit engagement, healthy-hand holding, and per-side fault shutdown.

prepare_engage enables hands at their current pose. begin_follow is separate so
a coordinator can engage the arms while this worker keeps both hand targets.
"""

    def __init__(self, gloves, hands, retargeters, *, profile, sink=None, hand_sink=None,
                 control_hz=120., transition_s=.75,
                 glove_timeout_s=.25, hand_timeout_s=.5, session=None,
                 clock_ns=time.monotonic_ns, threaded=True, metadata=None):
        self.sides = tuple(gloves)
        if (not self.sides or set(self.sides) - {"left", "right"}
                or set(hands) != set(self.sides) or set(retargeters) != set(self.sides)):
            raise ValueError("Gloves, hands, and retargeters must have matching selected sides")
        if any(not math.isfinite(x) or x <= 0 for x in
               (control_hz, transition_s, glove_timeout_s, hand_timeout_s)):
            raise ValueError("Wuji rates and timeouts must be positive")
        self.gloves, self.hands, self.retargeters = gloves, hands, retargeters
        self.profile, self.sink = profile, sink
        self.hand_sink = hand_sink
        self.session, self.clock_ns, self.threaded = session, clock_ns, threaded
        self.metadata = dict(metadata or {})
        self.period_ns = round(1e9 / control_hz)
        self.transition_ns = round(1e9 * transition_s)
        self.glove_timeout_ns, self.hand_timeout_ns = round(1e9 * glove_timeout_s), round(1e9 * hand_timeout_s)
        self._state, self.mode, self.last_error = SystemState.DISCONNECTED, None, None
        self._lock, self._stop = threading.RLock(), threading.Event()
        self._thread = None
        self._worker_error = None
        self._observer_error = None
        self._generation = 0
        self._prepared_generation = None
        self._preparing = False
        self._blocked_hands = set()
        self._disable_threads = {}
        self._origin, self._mapped, self._refs = {}, {}, {}
        self._transition_start = 0
        self._epoch, self._sequence = uuid.uuid4().hex, 0
        self.cycles = self.last_compute_ns = self.missed_periods = 0
        self._first_cycle_ns = self._last_cycle_ns = None
        self._cycle_snapshot = (0, None, None, 0, None)

    @property
    def state(self):
        return self._state

    def _event(self, kind, details):
        if self.sink is None or self._observer_error is not None:
            return
        try:
            if self.sink.try_event(Event(f"wuji.{kind}", self.clock_ns(), "wuji_teleop", details)):
                return
            self._observer_failure()
        except Exception as error:
            self._observer_failure(error)

    def _observer_failure(self, error=None):
        if self._observer_error is not None:
            return
        self._observer_error = (f"Wuji realtime observer failed: {error}" if error else
                                "Wuji realtime observer rejected live data")
        self._worker_error = self._observer_error
        if self.state == SystemState.ENGAGED:
            self.pause(self._observer_error)

    def start(self):
        if self.state != SystemState.DISCONNECTED:
            raise RuntimeError("Create a new runtime after start/close")
        try:
            if self.session:
                self.session.open()
                self._check_session()
                for device in (*self.gloves.values(), *self.hands.values()):
                    device.manager, device.sdk = self.session.manager, self.session.sdk
                for retargeter in self.retargeters.values():
                    retargeter.sdk = self.session.sdk
            for side in self.sides:
                self.gloves[side].start(sink=self.sink)
                self.hands[side].start(sink=self.sink if self.hand_sink is None else self.hand_sink)
            self._state = SystemState.READY
            self._event("started", {                        "profile": asdict(self.profile), "control_period_ns": self.period_ns,
                        "configuration": self.metadata,
                        "session": self.session.metadata if self.session else None,
                        "devices": {s: {"glove": self.gloves[s].metadata,
                                        "hand": self.hands[s].metadata} for s in self.sides}})
            if self._observer_error:
                raise RuntimeError(self._observer_error)
            if self.threaded:
                self._thread = threading.Thread(target=self._run, name="wuji-control", daemon=True)
                self._thread.start()
        except BaseException as error:
            # Preserve the startup cause even if rolling back another device
            # fails, and make SDK session errors reportable by both CLI entries.
            message = str(error)
            self._event("start_failed", {"error_type": type(error).__name__, "error": message})
            try:
                self.close()
            except Exception as cleanup_error:
                message += f"; cleanup failed: {cleanup_error}"
            if not isinstance(error, Exception):
                raise
            self.last_error = message
            raise RuntimeError(message) from error

    def glove_samples(self):
        """Return the latest owned skeleton samples without querying the SDK."""
        return {side: glove.get_latest() for side, glove in self.gloves.items()}

    def _current(self, side, *, glove=False, check_latch=True):
        device = self.gloves[side] if glove else self.hands[side]
        sample = device.get_latest()
        timeout = self.glove_timeout_ns if glove else self.hand_timeout_ns
        kind = "glove" if glove else "hand"
        if check_latch and device.fault:
            raise RuntimeError(f"{side} {kind}: {device.fault}")
        health = device.health(check_latch=check_latch)
        if not health.ready:
            raise RuntimeError(health.detail or f"{side} {kind} is unavailable")
        if (sample is None or not sample.header.valid
                or self.clock_ns() - sample.header.received_monotonic_ns >= timeout):
            raise RuntimeError(f"{side} {kind} feedback is missing, invalid, or stale")
        return sample

    def _check_session(self):
        """Confirm our SDK user at lifecycle boundaries, outside following."""
        if self.session:
            try:
                health = self.session.health()
            except Exception as error:
                raise RuntimeError(f"Wuji SDK session query failed: {error}") from error
            if not health.ready:
                raise RuntimeError(health.detail or "Wuji SDK user changed")

    def _map(self, side, sample):
        if self._refs.get(side) != sample.header.ref:
            self._mapped[side] = self.retargeters[side].solve(sample)
            self._refs[side] = sample.header.ref
            if self.sink is not None:
                self._event("retargeted", {"side": side, "source_ref": asdict(sample.header.ref),
                            "target": asdict(self._mapped[side])})

    def _cancelled(self, generation, cancelled=None):
        return (self._stop.is_set() or generation != self._generation
                or (cancelled is not None and cancelled()))

    def prepare_engage(self, *, cancelled=None):
        with self._lock:
            # An asynchronous caller can cancel before this thread is first
            # scheduled. Check before establishing a new engagement generation.
            if cancelled is not None and cancelled():
                raise RuntimeError("Wuji engagement cancelled")
            if self._worker_error:
                raise RuntimeError(self._worker_error)
            if self.state not in (SystemState.READY, SystemState.PAUSED) or self._preparing:
                raise RuntimeError("Pause before engaging again")
            self._generation += 1
            generation = self._generation
            self._prepared_generation, self._preparing = None, True
        try:
            self._check_session()
            for thread in tuple(self._disable_threads.values()):
                thread.join()
            for side in self.sides:
                self.gloves[side].clear_fault()
                self.hands[side].clear_fault()
                self._current(side, glove=True)
                self._current(side)
            self._mapped, self._refs = {}, {}
            # Warm up and validate both native solvers before enabling any hand.
            for side in self.sides:
                self.retargeters[side].reset()
                self._map(side, self._current(side, glove=True))
            for side in self.sides:
                if self._cancelled(generation, cancelled):
                    raise RuntimeError("Wuji engagement cancelled")
                hand = self.hands[side]
                if not hand.enabled or side in self._blocked_hands:
                    hand.configure(self.profile)
                    if self._cancelled(generation, cancelled):
                        raise RuntimeError("Wuji engagement cancelled")
                    hand.engage(cancelled=lambda: self._cancelled(generation, cancelled))
                    self._blocked_hands.discard(side)
            with self._lock:
                self._origin = {}
                feedback_refs = {}
                for side in self.sides:
                    self._current(side, glove=True)
                    feedback = self._current(side)
                    hand = self.hands[side]
                    feedback_refs[side] = asdict(feedback.header.ref)
                    # A held target can intentionally differ from measured q under load.
                    self._origin[side] = hand.last_target if hand.enabled else JointTarget(
                        JOINT_NAMES, tuple(feedback.payload.position_rad))
                if self._cancelled(generation, cancelled):
                    raise RuntimeError("Wuji engagement cancelled")
                self._prepared_generation = generation
                self.mode = "hold"
                self.last_error = None
                self._event("prepared", {"origins": {s: asdict(q) for s, q in self._origin.items()},
                            "feedback_refs": feedback_refs,
                            "input_refs": {s: asdict(ref) for s, ref in self._refs.items()}})
                if self._observer_error:
                    raise RuntimeError(self._observer_error)
        except BaseException as error:
            if not self._cancelled(generation, cancelled):
                self.pause(str(error))
            raise
        finally:
            self._preparing = False

    def begin_follow(self, *, cancelled=None):
        with self._lock:
            if self._prepared_generation != self._generation or self._prepared_generation is None:
                raise RuntimeError(self.last_error or "Prepare the hands before starting following")
            try:
                for side in self.sides:
                    self._current(side, glove=True)
                    self._current(side)
                    if not self.hands[side].enabled:
                        raise RuntimeError(f"{side} hand lost enable before following")
                self._transition_start = self.clock_ns()
                if cancelled is not None and cancelled():
                    raise RuntimeError("Wuji engagement cancelled")
                self._state = SystemState.ENGAGED
                self.mode = "follow"
                self._prepared_generation = None
                self._event("engaged", {"transition_ns": self.transition_ns, "mode": self.mode})
                if self._observer_error:
                    raise RuntimeError(self._observer_error)
            except BaseException as error:
                self.pause(str(error))
                raise

    def engage(self, profile=None):
        if profile is not None and profile != self.profile:
            raise ValueError("Restart to change the Wuji control profile")
        self.prepare_engage()
        self.begin_follow()

    def pause(self, reason):
        with self._lock:
            if self.state == SystemState.CLOSED:
                return
            self._generation += 1
            self._prepared_generation = None
            self._state, self.mode, self.last_error = SystemState.PAUSED, "hold", reason
            for hand in self.hands.values():
                if hand.enabled:
                    hand.request_hold(reason)
            self._event("paused", {"reason": reason, "hold_targets": {
                s: asdict(h.last_target) if h.last_target else None for s, h in self.hands.items()}})

    def _disable(self, side, reason):
        self._blocked_hands.add(side)
        previous = self._disable_threads.get(side)
        if previous and previous.is_alive():
            return

        def disable():
            try:
                self.hands[side].disable(reason)
            except Exception as error:
                self._event("disable_failed", {"side": side, "reason": str(error),
                                               "physical_stop_confirmed": False})

        # A device RPC must not stall healthy-hand holding or the coordinator's pause.
        thread = threading.Thread(target=disable, name=f"wuji-disable-{side}", daemon=True)
        self._disable_threads[side] = thread
        thread.start()

    def _submit(self, side, target, refs, now, deadline):
        self._sequence += 1
        hand = self.hands[side]
        command = DeviceCommand(hand.device_id, f"{self._epoch}:{self._sequence}", target,
                                refs, now, deadline, self.profile.profile_id)
        try:
            result = hand.submit(command)
        except (RuntimeError, ValueError) as error:
            self._disable(side, str(error))
            raise
        if not result.accepted:
            self._disable(side, result.reason or "SDK submission rejected")
            raise RuntimeError(result.reason or f"{side} hand SDK submission failed")

    def _step(self):
        """One control step; deliberately exposed only to deterministic tests."""
        with self._lock:
            if self.state in (SystemState.DISCONNECTED, SystemState.CLOSED):
                return
            started = self.clock_ns()
            failed_hands = []
            for side, hand in self.hands.items():
                if hand.enabled and side not in self._blocked_hands:
                    try:
                        self._current(side)
                    except (RuntimeError, ValueError) as error:
                        failed_hands.append((side, str(error)))
            if failed_hands:
                self.pause(failed_hands[0][1])
                for side, reason in failed_hands:
                    self._disable(side, reason)
            if self.state == SystemState.ENGAGED:
                try:
                    samples = {s: self._current(s, glove=True) for s in self.sides}
                    for side, sample in samples.items():
                        self._map(side, sample)
                    if self._worker_error or self.state != SystemState.ENGAGED:
                        raise RuntimeError(self._worker_error or "Wuji following was paused")
                    alpha = min(1., max(0., (started - self._transition_start) / self.transition_ns))
                    targets = {s: JointTarget(JOINT_NAMES, tuple((1-alpha)*a + alpha*b for a, b in zip(
                        self._origin[s].position_rad, self._mapped[s].position_rad))) for s in self.sides}
                    # Recheck after both solves; either source may fault during the SDK call.
                    for side in self.sides:
                        self._current(side, glove=True)
                        self._current(side)
                    for side in self.sides:
                        now = self.clock_ns()
                        deadline = min(now + 50_000_000,
                            samples[side].header.received_monotonic_ns + self.glove_timeout_ns)
                        self._submit(side, targets[side], (self._refs[side],), now, deadline)
                except Exception as error:
                    self.pause(str(error))
            else:
                for side, hand in self.hands.items():
                    if hand.enabled and side not in self._blocked_hands and hand.last_target is not None:
                        try:
                            now = self.clock_ns()
                            refs = (self._refs[side],) if side in self._refs else ()
                            self._submit(side, hand.last_target, refs, now, now + 50_000_000)
                        except (RuntimeError, ValueError) as error:
                            self.pause(str(error))
            self.cycles += 1
            if self._first_cycle_ns is None:
                self._first_cycle_ns = started
            self._last_cycle_ns = started
            self.last_compute_ns = self.clock_ns() - started
            self._cycle_snapshot = (self.cycles, self._first_cycle_ns, self._last_cycle_ns,
                                    self.last_compute_ns, self.clock_ns())
            self._event("cycle", {"compute_ns": self.last_compute_ns, "missed_periods": self.missed_periods})

    def _run(self):
        deadline = self.clock_ns()
        try:
            while not self._stop.is_set():
                self._step()
                deadline += self.period_ns
                now = self.clock_ns()
                if deadline <= now:
                    skipped = (now - deadline) // self.period_ns + 1
                    deadline += skipped * self.period_ns
                    self.missed_periods += skipped
                self._stop.wait(max(0, deadline - self.clock_ns()) / 1e9)
        except BaseException as error:
            self._worker_error = f"Wuji control worker stopped: {error}"
            self.pause(self._worker_error)
            self._stop.set()
            # A stopped worker cannot maintain a hold command stream.
            for side, hand in self.hands.items():
                if hand.enabled:
                    self._disable(side, str(error))

    def tick(self, now_monotonic_ns=None):
        """The independent worker owns scheduling; caller ticks never send twice."""
        return None

    def health(self):
        try:
            if self.state in (SystemState.DISCONNECTED, SystemState.CLOSED):
                raise RuntimeError(self.state.value)
            if self._worker_error:
                raise RuntimeError(self._worker_error)
            for side in self.sides:
                self._current(side, glove=True, check_latch=self.state == SystemState.ENGAGED)
                self._current(side, check_latch=self.state == SystemState.ENGAGED)
            return Health(True, self.clock_ns(), self.mode or "ready; press Enter to engage")
        except (RuntimeError, ValueError) as error:
            return Health(False, self.clock_ns(), str(error))

    def status(self, *, include_target=True):
        # The control lock spans native retarget/send calls. UI readers must not
        # wait for it while the arm group's command deadline is running.
        cycles, first, last, compute, completed = self._cycle_snapshot
        elapsed_ns = last-first if cycles > 1 else 0
        return {"state": self.state.value, "mode": self.mode,
                "health": asdict(self.health()), "last_error": self.last_error, "cycles": cycles,
                "control_hz_actual": (cycles-1)*1e9/elapsed_ns if elapsed_ns > 0 else None,
                "last_compute_ns": compute, "missed_periods": self.missed_periods,
                "worker_completed_monotonic_ns": completed,
                "worker_age_ns": self.clock_ns()-completed if completed is not None else None,
                "gloves": {s: {"health": asdict(g.health(check_latch=False)),
                                "statistics": g.statistics} for s, g in self.gloves.items()},
                "hands": {s: {"enabled": h.enabled, "health": asdict(h.health()),
                    "statistics": h.statistics,
                    **({"feedback_hz": {"requested": h.feedback_hz,
                                        "actual": dict(h.feedback_hz_actual)}}
                       if hasattr(h, "feedback_hz") else {}),
                    "last_target": asdict(h.last_target) if include_target and h.last_target else None}
                    for s, h in self.hands.items()}}

    def close(self):
        with self._lock:
            if self.state == SystemState.CLOSED:
                return
            self._stop.set()
            self._generation += 1
            self._prepared_generation = None
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join()
        for thread in tuple(self._disable_threads.values()):
            thread.join()
        errors = []
        for device in (*self.hands.values(), *self.gloves.values()):
            try:
                device.close()
            except Exception as error:
                errors.append(str(error))
        if self.session:
            try:
                self.session.close()
            except Exception as error:
                errors.append(str(error))
        self._state, self.mode = SystemState.CLOSED, None
        self._event("closed", {"errors": errors, "physical_stop_confirmed": False})
        if errors:
            self.last_error = "; ".join(errors)
            raise RuntimeError(self.last_error)


def create_wuji_teleop(config, sides=("left", "right"), *, sink=None, hand_sink=None):
    """Construct without connecting; hand_sink does not enable glove telemetry."""
    validate_control_config(config)
    if not sides or len(set(sides)) != len(sides) or set(sides) - {"left", "right"}:
        raise ValueError("Select left, right, or both hands")
    gloves = {s: WujiGloveSource(s, config["devices"][s]["glove"],
              streams=("emf", "skeleton") if sink is not None else ("skeleton",),
              timeout_s=config.get("glove_timeout_s", .25)) for s in sides}
    hands = {s: WujiHandDriver(s, config["devices"][s]["hand"],
             timeout_s=config.get("hand_timeout_s", .5),
             feedback_hz=config.get("feedback_hz", 200)) for s in sides}
    profile = ControlProfile(config.get("profile_id", "wuji-hand2"), "mit", dict(config["parameters"]))
    return WujiTeleop(gloves, hands, {s: WujiHandRetargeter(s) for s in sides},
        profile=profile, sink=sink, hand_sink=hand_sink,
        session=WujiSdkSession(user_name=sdk_user_name(config)),
        control_hz=config.get("control_hz", 120), transition_s=config.get("transition_s", .75),
        glove_timeout_s=config.get("glove_timeout_s", .25), hand_timeout_s=config.get("hand_timeout_s", .5),
        metadata=config)
