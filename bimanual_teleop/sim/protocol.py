"""Versioned sim-only transport, clock mapping and bounded image exchange."""

from __future__ import annotations

import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from bimanual_teleop.sim._dependencies import msgpack
from bimanual_teleop.sim._dependencies import np

VERSION = 1
MAX_MESSAGE_BYTES = 1024 * 1024


def pack(message):
    return msgpack.packb({"version": VERSION, **message}, use_bin_type=True)


def unpack(payload):
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("Simulation message is too large")
    message = msgpack.unpackb(payload, raw=False, strict_map_key=True)
    if not isinstance(message, dict) or message.get("version") != VERSION:
        raise ValueError("Unsupported simulation message")
    return message


def finite_vector(value, size, name):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} requires {size} values")
    array = np.array(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def validate_input(message):
    if message.get("kind") != "input" or type(message.get("sequence")) is not int or message["sequence"] < 0:
        raise ValueError("Invalid operator input sequence")
    if type(message.get("sample_ns")) is not int or type(message.get("send_ns")) is not int:
        raise ValueError("Operator input requires original host monotonic timestamps")
    if not isinstance(message.get("session"), str) or not message["session"]:
        raise ValueError("Operator session is required")
    if type(message.get("valid")) is not bool:
        raise ValueError("Input validity is required")
    for side in ("left", "right"):
        pose = finite_vector(message["controllers"][side], 7, f"{side} controller")
        if abs(np.linalg.norm(pose[3:]) - 1) > .001:
            raise ValueError("Controller quaternion must be unit XYZW")
        finite_vector(message["hands"][side], 20, f"{side} hand")
    return message


@dataclass
class ClockEstimate:
    offset_ns: int  # server minus client monotonic clock
    uncertainty_ns: int
    updated_ns: int

    @classmethod
    def from_exchange(cls, client_send, server_receive, server_send, client_receive):
        if client_receive < client_send or server_send < server_receive:
            raise ValueError("Invalid clock exchange")
        delay = (client_receive - client_send) - (server_send - server_receive)
        if delay < 0:
            raise ValueError("Negative network round-trip time")
        offset = ((server_receive - client_send) + (server_send - client_receive)) // 2
        return cls(offset, delay // 2, server_receive)

    def maximum_age_ns(self, client_sample, server_now):
        return server_now - (client_sample + self.offset_ns) + self.uncertainty_ns


class InputGuard:
    def __init__(self, timeout_ns=100_000_000, uncertainty_ns=10_000_000):
        self.timeout_ns, self.uncertainty_ns = timeout_ns, uncertainty_ns
        self.clock = None
        self.session = None
        self.origin = None
        self.sequence = -1
        self.latest = None
        self.fault = None
        self.fault_generation = None

    def update_clock(self, estimate):
        if estimate.uncertainty_ns > self.uncertainty_ns:
            raise ValueError("Operator clock mapping is too uncertain")
        self.clock = estimate

    def receive(self, message, now_ns=None):
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        validate_input(message)
        if self.clock is None or now_ns - self.clock.updated_ns > 10_000_000_000:
            self.fault = "Clock synchronization unavailable"; return False
        session, origin = message["session"], message.get("origin")
        if self.session is not None and (session != self.session or origin != self.origin):
            self.fault = "Operator reference changed; re-engage"
            self.sequence = -1
        if message["sequence"] <= self.sequence:
            return False
        age = self.clock.maximum_age_ns(message["sample_ns"], now_ns)
        if age < -self.uncertainty_ns or age >= self.timeout_ns:
            self.fault = "Operator input expired"; return False
        if self.latest is not None and session == self.session:
            previous_time = self.latest["sample_ns"]
            if message["sample_ns"] < previous_time:
                self.fault = "Operator timestamp moved backwards"; return False
            # Latest-only transport deliberately skips intermediate samples.
            # A producer continuity counter preserves faults even when their
            # invalid packet was conflated away. Expiry is checked separately
            # by the host polling loop, independently of slow rendering.
            if "fault_generation" not in message and message["sample_ns"] - previous_time >= self.timeout_ns:
                self.fault = "Operator input stream gap; re-engage"
            if self.fault_generation is not None and message.get("fault_generation") != self.fault_generation:
                self.fault = "Operator tracking fault occurred between published frames; re-engage"
        self.session, self.origin, self.sequence, self.latest = session, origin, message["sequence"], message
        self.fault_generation = message.get("fault_generation")
        if not message["valid"]:
            self.fault = message.get("problem", "Operator tracking unavailable")
        return self.fault is None

    def healthy(self, now_ns=None):
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        if self.latest is None or self.clock is None:
            return False
        if self.clock.maximum_age_ns(self.latest["sample_ns"], now_ns) >= self.timeout_ns:
            self.fault = "Operator input expired"
        return self.fault is None and self.latest["valid"]

    def reengage(self, now_ns=None):
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        if self.latest is None or self.clock is None or not self.latest["valid"]:
            return False
        if self.clock.maximum_age_ns(self.latest["sample_ns"], now_ns) >= self.timeout_ns:
            return False
        self.fault = None
        return True


class ImageRing:
    """One host-owned /dev/shm mapping, engine producer and host consumer.

    Opening the mapping via memmap avoids Python resource-tracker interference
    between the container's Python 3.11 and the host's Python 3.10.
    """
    HEADER_SIZE = 4096

    def __init__(self, path, capacity, shape, create=False):
        self.path, self.capacity, self.shape = Path(path), capacity, tuple(shape)
        container_shm = os.environ.get("BIMANUAL_SIM_SHM_DIRECTORY")
        if container_shm and self.path.parent == Path("/dev/shm"):
            self.path = Path(container_shm) / self.path.name
        size = self.HEADER_SIZE + capacity * math.prod(self.shape)
        if create:
            with self.path.open("xb") as file:
                file.truncate(size)
            os.chmod(self.path, 0o600)
        elif self.path.stat().st_size != size:
            raise ValueError("Image ring dimensions do not match")
        self.mapping = np.memmap(self.path, dtype=np.uint8, mode="r+", shape=(size,))
        self.counters = self.mapping[:16].view(np.uint64)
        self.frames = self.mapping[self.HEADER_SIZE:].reshape((capacity, *self.shape))
        if create:
            self.counters[:] = 0

    @classmethod
    def create(cls, capacity, width, height):
        fd, path = tempfile.mkstemp(prefix="bimanual-sim-", dir="/dev/shm")
        os.close(fd); Path(path).unlink()
        return cls(path, capacity, (3, height, width, 3), create=True)

    def put(self, frames):
        written, read = map(int, self.counters)
        if written-read >= self.capacity:
            raise BufferError("Simulation image ring is full")
        if np.asarray(frames).shape != self.shape or np.asarray(frames).dtype != np.uint8:
            raise ValueError("Image ring requires three uint8 RGB images")
        self.frames[written % self.capacity] = frames
        self.counters[0] = written+1
        return written

    def take(self, index):
        written, read = map(int, self.counters)
        if index != read or not read <= index < written:
            raise ValueError("Image sequence gap or stale image ring slot")
        result = self.frames[index % self.capacity].copy()
        self.counters[1] = index+1
        return result

    def descriptor(self):
        return {"path": str(self.path), "capacity": self.capacity, "shape": list(self.shape)}

    def close(self, owner=False):
        self.frames = None; self.counters = None
        self.mapping._mmap.close()
        if owner:
            self.path.unlink(missing_ok=True)
