"""Run the complete Wuji device/control domain in its own interpreter.

Lifecycle requests and sampled observations use separate channels. An optional
recording sink receives original hand samples inside the child, independently of
the observation snapshots. The parent never owns a Wuji SDK handle or sends hand
joint commands.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import multiprocessing as mp
import pickle
import select
import signal
import socket
import threading
import time

from bimanual_teleop.system import SystemState
from bimanual_teleop.types import Health


_SNAPSHOT_PERIOD_S = .02
_PACKET_BYTES = 262144


@dataclass(frozen=True)
class _Snapshot:
    sequence: int
    command_id: int
    created_ns: int
    valid_until_ns: int
    status: dict
    gloves: dict
    deadlines: dict = field(default_factory=dict)


def _send(channel, message):
    # Unix datagrams preserve whole messages. A full lifecycle channel fails
    # explicitly instead of blocking the arm thread or retrying an old action.
    channel.send(pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL))


def _receive(channel):
    return pickle.loads(channel.recv(_PACKET_BYTES))


def _create_runtime(config, *, verbose=False, record_sink=None):
    from bimanual_teleop.common.console import configure_runtime_logging
    configure_runtime_logging(wuji=True, verbose=verbose)
    from .follow import create_wuji_teleop
    if record_sink is None:
        return create_wuji_teleop(config)
    return create_wuji_teleop(config, hand_sink=record_sink)


def _serve(config, control, observations, heartbeat, runtime_factory, verbose, record_sink=None):
    # The coordinator handles terminal Ctrl+C, pauses both domains, then asks
    # this child to close. SIGTERM remains available for bounded escalation.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    runtime = None
    sequence = command_id = 0
    operation = None
    operation_result = None
    closing = False
    coordinator_expired = False
    timeout_ns = round(config.get("hand_timeout_s", .5) * 1e9)

    def snapshot():
        nonlocal sequence
        now = time.monotonic_ns()
        gloves = runtime.glove_samples()
        status = runtime.status(include_target=False)
        deadlines = {"process snapshot": now + timeout_ns}
        if runtime.threaded and (runtime.state == SystemState.ENGAGED or status.get("mode") == "hold"):
            completed = status.get("worker_completed_monotonic_ns")
            deadlines["control worker"] = completed + timeout_ns if completed is not None else now
        for side in runtime.sides:
            glove, hand = gloves.get(side), runtime.hands[side].get_latest()
            diagnostics = runtime.hands[side].get_latest_stream("diagnostics")
            for name, sample, timeout in (("glove", glove, runtime.glove_timeout_ns),
                                          ("hand joints", hand, runtime.hand_timeout_ns),
                                          ("hand diagnostics", diagnostics, runtime.hand_timeout_ns)):
                deadlines[f"{side} {name}"] = (sample.header.received_monotonic_ns + timeout
                                                if sample else now)
        sequence += 1
        return _Snapshot(sequence, command_id, now, min(deadlines.values()),
                         status, gloves, deadlines)

    def reply(request_id, error=None):
        _send(control, (request_id, error, snapshot() if runtime is not None else None))

    def operate(request_id, action):
        nonlocal operation_result
        error = None
        try:
            cancelled = lambda: request_id != command_id or closing or coordinator_expired
            if action == "prepare":
                runtime.prepare_engage(cancelled=cancelled)
            else:
                runtime.begin_follow(cancelled=cancelled)
        except Exception as fault:
            error = str(fault)
        operation_result = (request_id, error)

    try:
        options = {"verbose": verbose}
        if record_sink is not None:
            options["record_sink"] = record_sink
        runtime = runtime_factory(config, **options)
        runtime.start()
        reply(0)
        next_snapshot = time.monotonic()
        parent = mp.parent_process()
        while True:
            if operation is not None and not operation.is_alive():
                operation.join()
                request_id, error = operation_result
                operation = operation_result = None
                reply(request_id, error)
            if closing and operation is None:
                break
            now = time.monotonic_ns()
            if not closing and parent is not None and not parent.is_alive():
                closing = True
                runtime.pause("Wuji coordinator exited")
            elif (not closing and not coordinator_expired and now - heartbeat.value >= timeout_ns
                  and (operation is not None or runtime.state == SystemState.ENGAGED)):
                coordinator_expired = True
                runtime.pause("Wuji coordinator stopped advancing within the hand timeout")

            now_s = time.monotonic()
            if now_s >= next_snapshot:
                try:
                    _send(observations, snapshot())
                except BlockingIOError:
                    pass  # Observations may be superseded; source times never change.
                next_snapshot = now_s + _SNAPSHOT_PERIOD_S
            wait_s = max(0, next_snapshot-time.monotonic())
            if operation is not None:
                wait_s = min(wait_s, .001)  # Lifecycle ACKs do not wait for the next observation.
            readable, _, _ = select.select([control], [], [], wait_s)
            if not readable:
                continue
            request_id, action, reason = _receive(control)
            if request_id <= command_id:
                if action in ("prepare", "follow"):
                    reply(request_id, "Wuji lifecycle request was cancelled")
                continue
            command_id = request_id
            if action == "pause":
                runtime.pause(reason)
            elif action == "close":
                closing = True
                runtime.pause("Wuji process closing")
            elif closing:
                reply(request_id, "Wuji process is closing")
            elif action in ("prepare", "follow"):
                if operation is not None:
                    reply(request_id, "A Wuji lifecycle operation is already running")
                else:
                    if action == "prepare":
                        coordinator_expired = False
                    operation = threading.Thread(target=operate, args=(request_id, action),
                                                 name="wuji-lifecycle", daemon=True)
                    operation.start()
            else:
                raise RuntimeError(f"Unknown Wuji lifecycle action: {action}")
    except Exception as fault:
        try:
            _send(control, (-1, str(fault), None))
        except OSError:
            pass
    finally:
        if runtime is not None:
            runtime.close()
        control.close()
        observations.close()


class WujiProcess:
    """Fixed hand-runtime interface with nonblocking cached observations."""

    def __init__(self, config, *, verbose=False, record_sink=None,
                 _runtime_factory=_create_runtime):
        self.config = dict(config)
        self.verbose = verbose
        self._record_sink = record_sink
        self.glove_timeout_ns = round(config.get("glove_timeout_s", .25) * 1e9)
        self.hand_timeout_ns = round(config.get("hand_timeout_s", .5) * 1e9)
        self._factory = _runtime_factory
        self._context = mp.get_context("spawn")
        self._heartbeat = self._context.RawValue("Q", 0)
        self._process = self._control = self._observations = None
        self._latest = None
        self._request_id = self._pause_id = 0
        self._pause_reason = self._failure = None
        self._closed = False
        self._requests = threading.Lock()
        self._send_lock = threading.Lock()
        self._read_lock = threading.RLock()

    def _touch(self):
        if not self._closed and not self._failure:
            self._heartbeat.value = time.monotonic_ns()

    def _accept(self, value):
        with self._read_lock:
            if value is not None and (self._latest is None or value.sequence > self._latest.sequence):
                self._latest = value

    def _refresh(self):
        if self._observations is None or not self._read_lock.acquire(blocking=False):
            return
        try:
            # Limit work in a control tick even after the parent was delayed.
            for _ in range(32):
                try:
                    self._accept(_receive(self._observations))
                except BlockingIOError:
                    break
        except (OSError, EOFError, pickle.UnpicklingError) as error:
            self._failure = self._failure or f"Wuji observation channel failed: {error}"
        finally:
            self._read_lock.release()

    @property
    def state(self):
        self._refresh()
        if self._closed:
            return SystemState.CLOSED
        if self._failure or (self._pause_id and
                             (self._latest is None or self._latest.command_id < self._pause_id)):
            return SystemState.PAUSED
        return SystemState(self._latest.status["state"]) if self._latest else SystemState.DISCONNECTED

    @property
    def last_error(self):
        if self._failure:
            return self._failure
        if self._pause_id and (self._latest is None or self._latest.command_id < self._pause_id):
            return self._pause_reason
        return self._latest.status.get("last_error") if self._latest else None

    def _send_request(self, action, reason=None):
        with self._send_lock:
            self._request_id += 1
            request_id = self._request_id
            if action == "pause":
                self._pause_id, self._pause_reason = request_id, reason
            _send(self._control, (request_id, action, reason))
            return request_id

    def _wait_reply(self, request_id, timeout_s, *, preparing=False):
        deadline = time.monotonic() + timeout_s
        while True:
            if preparing:
                # Preparing holds the hands still before arm engagement. A
                # synchronous caller is advancing setup while it waits here.
                self._touch()
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                reason = "Wuji lifecycle request timed out"
                self.pause(reason)
                raise RuntimeError(reason)
            readable, _, _ = select.select([self._control], [], [], min(.05, remaining))
            if readable:
                response_id, error, observation = _receive(self._control)
                self._accept(observation)
                if response_id == -1 or response_id == request_id:
                    if error:
                        raise RuntimeError(error)
                    return
            if not self._process.is_alive():
                raise RuntimeError(f"Wuji process exited (code {self._process.exitcode})")

    def start(self):
        if self._process is not None or self._closed:
            raise RuntimeError("Create a new Wuji process after start/close")
        self._control, child_control = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._observations, child_observations = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        for channel in (self._control, child_control, self._observations, child_observations):
            channel.setblocking(False)
        self._touch()
        self._process = self._context.Process(target=_serve,
            args=(self.config, child_control, child_observations,
                  self._heartbeat, self._factory, self.verbose, self._record_sink), name="wuji-teleop")
        self._process.start()
        child_control.close()
        child_observations.close()
        try:
            self._wait_reply(0, 30.)
        except Exception as error:
            self._failure = str(error)
            try:
                self.close()
            except RuntimeError as cleanup:
                raise RuntimeError(f"{error}; cleanup failed: {cleanup}") from error
            raise

    def _request(self, action, timeout_s):
        with self._requests:
            if self._closed or self._control is None or self._failure:
                raise RuntimeError(self._failure or "Wuji process is not available")
            self._touch()
            request_id = self._send_request(action)
            self._wait_reply(request_id, timeout_s, preparing=action == "prepare")

    def prepare_engage(self):
        self._request("prepare", 30.)

    def begin_follow(self):
        self._request("follow", self.hand_timeout_ns / 1e9)

    def pause(self, reason):
        if self._closed:
            return
        if self._control is None:
            self._pause_id, self._pause_reason = self._request_id + 1, reason
            return
        try:
            self._send_request("pause", reason)
        except OSError as error:
            self._failure = self._failure or f"Wuji pause channel failed: {error}"

    def health(self):
        self._touch()
        self._refresh()
        now = time.monotonic_ns()
        if self._closed or self._process is None:
            return Health(False, now, self.state.value)
        if not self._process.is_alive():
            self._failure = self._failure or f"Wuji process exited (code {self._process.exitcode})"
        if self._failure:
            return Health(False, now, self._failure)
        if self._latest is None or now >= self._latest.valid_until_ns:
            detail = "Wuji process observation is missing or stale"
            if self._latest is not None:
                expired = [name for name, deadline in self._latest.deadlines.items() if now >= deadline]
                if expired:
                    detail += ": " + ", ".join(expired)
            return Health(False, now, detail)
        health = self._latest.status["health"]
        return Health(bool(health["ready"]), now, health.get("detail"))

    def status(self, *, include_target=True):
        health = self.health()
        status = dict(self._latest.status) if self._latest else {}
        status.update(state=self.state.value, last_error=self.last_error,
                      health=asdict(health),
                      process_pid=self._process.pid if self._process is not None else None,
                      process_alive=self._process.is_alive() if self._process is not None else False,
                      snapshot_sequence=self._latest.sequence if self._latest else None,
                      snapshot_created_ns=self._latest.created_ns if self._latest else None,
                      snapshot_valid_until_ns=self._latest.valid_until_ns if self._latest else None,
                      snapshot_deadlines=dict(self._latest.deadlines) if self._latest else {})
        return status

    def glove_samples(self):
        self._touch()
        self._refresh()
        return dict(self._latest.gloves) if self._latest else {"left": None, "right": None}

    def close(self):
        if self._closed:
            return
        error = None
        if self._process is not None and self._process.is_alive():
            try:
                self._send_request("close")
            except OSError:
                pass  # A child finishing cleanup can already have closed its socket.
            self._process.join(timeout=5.)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.)
                error = "Wuji process did not finish closing; physical stop is unconfirmed"
            elif self._process.exitcode:
                error = f"Wuji process exited during close (code {self._process.exitcode})"
        self._closed = True
        for channel in (self._control, self._observations):
            if channel is not None:
                channel.close()
        if error:
            raise RuntimeError(error)
