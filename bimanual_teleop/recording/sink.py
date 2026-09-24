"""Small nonblocking device observer; no encoding, FK, or disk access here."""

from dataclasses import dataclass
import math
from queue import Empty, Full

from bimanual_teleop.types import CommandEvent, CommandStatus, Event, JointState

SIDES = ("left", "right")
STATE_STREAMS = tuple(f"{kind}/{side}" for kind in ("arms", "hands") for side in SIDES)
COMMAND_STREAMS = tuple(f"{kind}/{side}" for kind in ("arm_commands", "hand_commands") for side in SIDES)
STREAMS = STATE_STREAMS + COMMAND_STREAMS
STREAM_FIELDS = {
    "arms/left": (("joint_pos", 7), ("wrench", 6)),
    "arms/right": (("joint_pos", 7), ("wrench", 6)),
    "hands/left": (("joint_pos", 20),),
    "hands/right": (("joint_pos", 20),),
    "arm_commands/left": (("joint_pos", 7), ("eef_pose", 7)),
    "arm_commands/right": (("joint_pos", 7), ("eef_pose", 7)),
    "hand_commands/left": (("joint_pos", 20),),
    "hand_commands/right": (("joint_pos", 20),),
}


@dataclass(frozen=True)
class Record:
    stream: str
    time_ns: int
    sequence: int
    values: dict


class _SharedRecordRing:
    """A fixed-schema SPSC queue backed only by shared numeric arrays.

    Slots are reused only after the consumer advances ``read_count``.  A full
    ring therefore fails the episode instead of overwriting an unread record.
    Each stream has one producer in the runtime architecture, so neither side
    needs a lock on the motion-control path.
    """

    def __init__(self, context, stream, capacity):
        self.stream = stream
        self.fields = STREAM_FIELDS[stream]
        self.width = sum(size for _name, size in self.fields)
        self.capacity = capacity
        self.time_ns = context.Array("q", capacity, lock=False)
        self.sequence = context.Array("q", capacity, lock=False)
        self.generation = context.Array("q", capacity, lock=False)
        self.values = context.Array("d", capacity * self.width, lock=False)
        # ``committed`` is written last by the producer.  Its expected value is
        # the absolute queue position + 1, which also rejects stale wrapped slots.
        self.committed = context.Array("q", capacity, lock=False)
        self.write_count = context.Value("q", 0, lock=False)
        self.read_count = context.Value("q", 0, lock=False)

    def put_nowait(self, generation, record):
        position = self.write_count.value
        if position - self.read_count.value >= self.capacity:
            raise Full
        slot = position % self.capacity
        flattened = []
        for name, size in self.fields:
            value = record.values.get(name)
            if value is None or len(value) != size:
                raise ValueError(f"{record.stream}.{name} must contain {size} values")
            flattened.extend(value)
        offset = slot * self.width
        self.time_ns[slot] = record.time_ns
        self.sequence[slot] = record.sequence
        self.generation[slot] = generation
        self.values[offset:offset + self.width] = flattened
        self.committed[slot] = position + 1
        self.write_count.value = position + 1

    def get_nowait(self):
        position = self.read_count.value
        if position >= self.write_count.value:
            raise Empty
        slot = position % self.capacity
        if self.committed[slot] != position + 1:
            raise Empty
        offset = slot * self.width
        flattened = self.values[offset:offset + self.width]
        values, cursor = {}, 0
        for name, size in self.fields:
            values[name] = tuple(flattened[cursor:cursor + size])
            cursor += size
        record = Record(self.stream, self.time_ns[slot], self.sequence[slot], values)
        generation = self.generation[slot]
        # The returned Record owns a process-local copy of every numeric value.
        # Releasing the slot here is safe; a later writer crash still leaves the
        # whole episode non-complete and can never publish partial data as valid.
        self.read_count.value = position + 1
        return generation, record

    def size(self):
        return self.write_count.value - self.read_count.value


class _SharedRecordQueue:
    """Round-robin compatibility facade used by the recorder consumer."""

    def __init__(self, rings):
        self.rings = rings
        self._next = 0

    def get_nowait(self):
        for offset in range(len(STREAMS)):
            index = (self._next + offset) % len(STREAMS)
            ring = self.rings[STREAMS[index]]
            try:
                value = ring.get_nowait()
            except Empty:
                continue
            self._next = (index + 1) % len(STREAMS)
            return value
        raise Empty

    def put_nowait(self, item):
        generation, record = item
        self.rings[record.stream].put_nowait(generation, record)

    def put(self, item, block=True, timeout=None):
        del block, timeout
        self.put_nowait(item)

    def qsize(self):
        return sum(ring.size() for ring in self.rings.values())

    def close(self):
        pass

    def cancel_join_thread(self):
        pass

    def join_thread(self):
        pass


def pose_values(pose):
    return (*pose.position_m, *pose.orientation_xyzw)


def finite(values, size):
    return values is not None and len(values) == size and all(
        v is not None and math.isfinite(v) for v in values)


class CaptureChannel:
    """Spawn-compatible shared state; failure signaling is separate from data."""

    def __init__(self, context, capacity=4096):
        self.rings = {stream: _SharedRecordRing(context, stream, capacity) for stream in STREAMS}
        self.queue = _SharedRecordQueue(self.rings)
        self.errors = context.Queue(16)
        self.failed = context.Event()
        self.active = context.Value("b", False, lock=False)
        self.generation = context.Value("q", 0, lock=False)
        self.latest_ns = context.Array("q", 8, lock=False)
        self.sent = context.Array("q", 8, lock=False)
        self.consumed = context.Array("q", 8, lock=False)
        self.inflight = context.Array("b", 8, lock=False)

    def put_nowait(self, generation, record):
        self.rings[record.stream].put_nowait(generation, record)

    def occupancy(self):
        return {stream: ring.size() for stream, ring in self.rings.items()}

    def fail(self, reason):
        if not self.failed.is_set():
            try:
                self.errors.put_nowait(str(reason))
            except (Full, OSError, ValueError):
                pass
            self.failed.set()


class RecorderSink:
    def __init__(self, channel, state_hz=200.):
        self.channel = channel
        self.period_ns = round(1e9 / state_hz)
        self._next_ns = {}
        self._sequences = {}
        self._pending = {}
        self._command_sequence = 0

    def _put(self, record, *, state=False):
        index = STREAMS.index(record.stream)
        self.channel.latest_ns[index] = record.time_ns
        if not self.channel.active.value or self.channel.failed.is_set():
            return
        if state:
            next_ns = self._next_ns.get(record.stream, record.time_ns)
            if record.time_ns < next_ns:
                return
            self._next_ns[record.stream] = next_ns + (
                (record.time_ns - next_ns) // self.period_ns + 1) * self.period_ns
        self.channel.inflight[index] = True
        try:
            if not self.channel.active.value:
                return
            self.channel.put_nowait(self.channel.generation.value, record)
            self.channel.sent[index] += 1
        except (Full, OSError, ValueError) as error:
            self.channel.fail(f"录制队列无法接收数据：{type(error).__name__}")
        finally:
            self.channel.inflight[index] = False

    def _invalid(self, stream):
        self.channel.latest_ns[STATE_STREAMS.index(stream)] = 0
        if self.channel.active.value:
            self.channel.fail(f"采集数据无效：{stream}")

    def try_publish(self, sample):
        try:
            stream = sample.header.ref.stream
            now = sample.header.received_monotonic_ns
            if stream == "tianji.feedback":
                for side, arm in sample.payload.arms.items():
                    name = f"arms/{side}"
                    sequence = arm.source_sequence
                    if self._sequences.get(name) == sequence:
                        continue
                    self._sequences[name] = sequence
                    if not finite(arm.joints.position_rad, 7) or not finite(arm.wrench, 6):
                        self._invalid(name)
                        continue
                    self._put(Record(name, now, sequence, {
                        "joint_pos": tuple(arm.joints.position_rad), "wrench": tuple(arm.wrench)}), state=True)
            elif isinstance(sample.payload, JointState) and stream in (
                    "wuji_left_hand/joints", "wuji_right_hand/joints"):
                side = "left" if stream == "wuji_left_hand/joints" else "right"
                name = f"hands/{side}"
                identity = (sample.header.ref.epoch, sample.header.source_sequence
                            if sample.header.source_sequence is not None else sample.header.ref.sequence)
                if self._sequences.get(name) == identity:
                    return True
                self._sequences[name] = identity
                if not finite(sample.payload.position_rad, 20) or not sample.header.valid:
                    self._invalid(name)
                else:
                    self._put(Record(name, now, sample.header.source_sequence
                        if sample.header.source_sequence is not None else sample.header.ref.sequence,
                        {"joint_pos": tuple(sample.payload.position_rad)}), state=True)
        except Exception as error:
            if self.channel.active.value:
                self.channel.fail(f"采集反馈失败：{error}")
        # Recorder failures stop the episode via the UI, never latch a device observer fault.
        return True

    def try_event(self, event):
        try:
            if isinstance(event, Event) and event.kind == "tianji_command_submitted":
                command = event.details["command"]
                self._command_sequence += 1
                for side, joints in command.payload.targets.items():
                    pose = command.payload.requested_cartesian_targets[side]
                    self._put(Record(f"arm_commands/{side}", event.observed_monotonic_ns,
                        self._command_sequence, {"joint_pos": tuple(joints), "eef_pose": pose_values(pose)}))
            elif isinstance(event, Event) and event.kind == "wuji_command":
                command = event.details["command"]
                self._pending[command.command_id] = command
                if len(self._pending) > 32:
                    self._pending.pop(next(iter(self._pending)))
            elif isinstance(event, CommandEvent):
                command = self._pending.pop(event.command_id, None)
                if command is not None and event.status == CommandStatus.ACCEPTED:
                    side = next(s for s in SIDES if s in command.device_id)
                    self._command_sequence += 1
                    self._put(Record(f"hand_commands/{side}", event.observed_monotonic_ns,
                        self._command_sequence, {"joint_pos": tuple(command.payload.position_rad)}))
        except Exception as error:
            if self.channel.active.value:
                self.channel.fail(f"采集控制目标失败：{error}")
        return True
