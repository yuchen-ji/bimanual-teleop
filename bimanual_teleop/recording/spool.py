"""Minimal online spool: shared frame pools, NVENC workers and raw numeric files."""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty, Full
import signal
import struct
import subprocess
import tempfile
import time

from .sink import Record, STREAM_FIELDS
from .storage import write_json


# One slot across the three RGB rings and the depth ring is 3,379,200 bytes
# (480×640×3 × 3 + 480×640×2). RawArray zeroes that storage when an episode
# starts, so the whole capacity stays resident. 256 frames is 825 MiB and
# 8.5 s at 30 Hz. This machine has 16 GiB and no swap; staying under 1 GiB
# leaves the rest for the control processes and the page cache.
FRAME_CAPACITY = 256
CAMERA_META = struct.Struct("<qqd")
NUMERIC_STRUCTS = {
    stream: struct.Struct("<qq" + "d" * sum(size for _name, size in fields))
    for stream, fields in STREAM_FIELDS.items()
}


class SharedFrameRing:
    """Single-producer/single-consumer image queue in shared memory."""

    def __init__(self, context, shape, dtype, capacity=None):
        if capacity is None:
            capacity = FRAME_CAPACITY
        import numpy as np
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype).str
        self.capacity = capacity
        elements = capacity
        for value in shape:
            elements *= value
        typecode = "B" if np.dtype(dtype) == np.dtype("u1") else "H"
        self.pixels = context.RawArray(typecode, elements)
        self.time_ns = context.Array("q", capacity, lock=False)
        self.sequence = context.Array("q", capacity, lock=False)
        self.source_time_ms = context.Array("d", capacity, lock=False)
        self.generation = context.Array("q", capacity, lock=False)
        self.committed = context.Array("q", capacity, lock=False)
        self.write_count = context.Value("q", 0, lock=False)
        self.read_count = context.Value("q", 0, lock=False)
        self.processed = context.Value("q", 0, lock=False)
        self.failed = context.Event()
        self.closing = context.Event()
        self.ready = context.Event()
        self.wake = context.Event()
        self.errors = context.Queue(4)

    def _array(self):
        import numpy as np
        return np.frombuffer(self.pixels, dtype=np.dtype(self.dtype)).reshape(
            self.capacity, *self.shape)

    def fail(self, reason):
        if not self.failed.is_set():
            try:
                self.errors.put_nowait(str(reason))
            except (Full, OSError, ValueError):
                pass
            self.failed.set()
            self.wake.set()

    def error(self):
        try:
            return self.errors.get_nowait()
        except Empty:
            return "帧处理进程失败"

    def put_nowait(self, generation, image, record):
        if self.failed.is_set():
            raise RuntimeError(self.error())
        position = self.write_count.value
        if position - self.read_count.value >= self.capacity:
            raise Full
        slot = position % self.capacity
        target = self._array()[slot]
        if image.shape != target.shape or image.dtype != target.dtype:
            raise ValueError(
                f"Frame layout mismatch: expected {target.shape}/{target.dtype}, "
                f"got {image.shape}/{image.dtype}")
        target[...] = image
        self.time_ns[slot] = record.time_ns
        self.sequence[slot] = record.sequence
        self.source_time_ms[slot] = float(record.values["source_time_ms"])
        self.generation[slot] = generation
        self.committed[slot] = position + 1
        self.write_count.value = position + 1
        self.wake.set()

    def peek(self):
        position = self.read_count.value
        if position >= self.write_count.value:
            return None
        slot = position % self.capacity
        if self.committed[slot] != position + 1:
            return None
        record = Record("", self.time_ns[slot], self.sequence[slot],
                        {"source_time_ms": self.source_time_ms[slot]})
        return position, self.generation[slot], self._array()[slot], record

    def acknowledge(self, position):
        if position != self.read_count.value:
            raise RuntimeError("Frame acknowledgement is out of order")
        self.read_count.value = position + 1
        self.processed.value += 1

    def size(self):
        return self.write_count.value - self.read_count.value

    def close_queues(self):
        self.errors.close()
        self.errors.cancel_join_thread()


class NVENCVideo:
    """One online H.264 hardware encoder owned by one worker process."""

    def __init__(self, path):
        import av
        self.container = av.open(str(path), "w")
        self.stream = self.container.add_stream("h264_nvenc", rate=30)
        self.stream.width, self.stream.height = 640, 480
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"preset": "p4", "cq": "21"}
        self.count = 0

    def write(self, rgb):
        import av
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts = self.count
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.count += 1

    def close(self):
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()


def _ignore_interrupt():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from bimanual_teleop.common.affinity import apply_recording_affinity
    apply_recording_affinity("background")


def _release(ring):
    """Copy one frame out and free its slot before the slow write."""
    item = ring.peek()
    if item is None:
        return None
    position, generation, image, record = item
    owned = image.copy()
    ring.acknowledge(position)
    return generation, owned, record


def _rgb_worker(ring, path, metadata_path):
    _ignore_interrupt()
    encoder = metadata = None
    try:
        encoder = NVENCVideo(path)
        metadata = Path(metadata_path).open("wb", buffering=1024 * 1024)
        ring.ready.set()
        while not ring.closing.is_set() or ring.size():
            released = _release(ring)
            if released is None:
                ring.wake.wait(.02)
                ring.wake.clear()
                continue
            _generation, image, record = released
            encoder.write(image)
            metadata.write(CAMERA_META.pack(
                record.time_ns, record.sequence, record.values["source_time_ms"]))
    except BaseException as error:
        ring.fail(f"RGB 编码失败：{error}")
    finally:
        if metadata is not None:
            try:
                metadata.flush()
                os.fsync(metadata.fileno())
            except OSError as error:
                ring.fail(f"RGB 元数据写入失败：{error}")
            metadata.close()
        if encoder is not None:
            try:
                encoder.close()
            except BaseException as error:
                ring.fail(f"RGB 编码收尾失败：{error}")


def _depth_worker(ring, image_path, metadata_path):
    _ignore_interrupt()
    image_file = metadata = None
    try:
        image_file = Path(image_path).open("wb", buffering=4 * 1024 * 1024)
        metadata = Path(metadata_path).open("wb", buffering=1024 * 1024)
        ring.ready.set()
        while not ring.closing.is_set() or ring.size():
            released = _release(ring)
            if released is None:
                ring.wake.wait(.02)
                ring.wake.clear()
                continue
            _generation, image, record = released
            image_file.write(image.tobytes(order="C"))
            metadata.write(CAMERA_META.pack(
                record.time_ns, record.sequence, record.values["source_time_ms"]))
    except BaseException as error:
        ring.fail(f"深度写入失败：{error}")
    finally:
        for output, label in ((image_file, "深度图像"), (metadata, "深度元数据")):
            if output is None:
                continue
            try:
                output.flush()
                os.fsync(output.fileno())
            except OSError as error:
                ring.fail(f"{label}收尾失败：{error}")
            output.close()


def _preflight_worker(path, result):
    try:
        import numpy as np
        video = NVENCVideo(path)
        frame = np.zeros((480, 640, 3), dtype="u1")
        for _ in range(3):
            video.write(frame)
        video.close()
        import av
        with av.open(str(path)) as container:
            decoded = sum(1 for _frame in container.decode(video=0))
        if decoded != 3:
            raise RuntimeError(f"预检视频应有 3 帧，实际 {decoded} 帧")
        result.put(None)
    except BaseException as error:
        result.put(f"{type(error).__name__}: {error}")


def preflight_nvenc(count=3, timeout_s=15.):
    """Prove that the configured number of encoders can run concurrently."""
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                                text=True, timeout=5., check=False)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"NVIDIA 驱动检查失败：{error}") from error
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"NVIDIA 驱动不可用：{detail or f'退出码 {result.returncode}'}")
    context = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="bimanual-nvenc-") as directory:
        result = context.Queue(count)
        processes = [context.Process(target=_preflight_worker,
            args=(str(Path(directory) / f"camera_{index}.mp4"), result),
            name=f"nvenc-preflight-{index}") for index in range(count)]
        for process in processes:
            process.start()
        deadline = time.monotonic() + timeout_s
        for process in processes:
            process.join(max(0., deadline - time.monotonic()))
        errors = []
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(1.)
                errors.append(f"{process.name} 超时")
            elif process.exitcode:
                errors.append(f"{process.name} 退出码 {process.exitcode}")
        while True:
            try:
                error = result.get_nowait()
            except Empty:
                break
            if error:
                errors.append(error)
        result.close()
        result.cancel_join_thread()
        if errors:
            raise RuntimeError("NVENC 三路并发预检失败：" + "; ".join(errors))


class RawEpisodeWriter:
    """Online writer with no FK, Zarr compression or cross-camera serialization."""

    def __init__(self, path, start_ns, metadata, _kinematics=None, *, context=None,
                 frame_capacity=None):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.raw = self.path / "raw_spool"
        (self.raw / "streams").mkdir(parents=True)
        (self.raw / "cameras").mkdir()
        self.document = dict(schema_version=2, status="capturing", start_ns=start_ns,
                             end_ns=None, metadata=metadata)
        write_json(self.path / "episode.json", self.document)
        self.context = context or mp.get_context("spawn")
        self.frame_capacity = FRAME_CAPACITY if frame_capacity is None else frame_capacity
        self.generation = None
        self.rings = {}
        self.processes = {}
        self.files = {}
        self.last_ns = {}
        self.counts = {}
        self._timing = {"append": {"count": 0, "total_ns": 0, "last_ns": 0, "max_ns": 0}}
        self._closed = False

    def _timed(self, started):
        elapsed = time.monotonic_ns() - started
        values = self._timing["append"]
        values["count"] += 1
        values["total_ns"] += elapsed
        values["last_ns"] = elapsed
        values["max_ns"] = max(values["max_ns"], elapsed)

    def prepare_rgb(self, cameras):
        for camera in cameras:
            ring = SharedFrameRing(self.context, (480, 640, 3), "u1", self.frame_capacity)
            metadata = self.raw / "cameras" / f"{camera}_rgb.bin"
            process = self.context.Process(target=_rgb_worker,
                args=(ring, str(self.path / f"{camera}.mp4"), str(metadata)),
                name=f"record-{camera}-nvenc")
            process.start()
            self.rings[f"cameras/{camera}/rgb"] = ring
            self.processes[f"cameras/{camera}/rgb"] = process
        if self.document["metadata"]["recording"]["main_depth"]:
            stream = "cameras/camera_0/depth"
            ring = SharedFrameRing(self.context, (480, 640), "u2", self.frame_capacity)
            process = self.context.Process(target=_depth_worker, args=(ring,
                str(self.raw / "cameras" / "camera_0_depth.raw"),
                str(self.raw / "cameras" / "camera_0_depth.bin")),
                name="record-camera_0-depth")
            process.start()
            self.rings[stream] = ring
            self.processes[stream] = process
        deadline = time.monotonic() + 10.
        for stream, ring in self.rings.items():
            while not ring.ready.wait(.01):
                if ring.failed.is_set():
                    raise RuntimeError(f"{stream}: {ring.error()}")
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"{stream}: 写入进程启动超时")

    def _accept(self, record):
        if record.time_ns < self.document["start_ns"]:
            return False
        previous = self.last_ns.get(record.stream)
        if previous is not None and record.time_ns <= previous:
            raise ValueError(f"Non-increasing timestamps: {record.stream}")
        self.last_ns[record.stream] = record.time_ns
        return True

    def write_rgb(self, camera, image, record):
        started = time.monotonic_ns()
        try:
            if not self._accept(record):
                return
            ring = self.rings[record.stream]
            try:
                ring.put_nowait(0, image, record)
            except Full as error:
                raise RuntimeError(
                    f"{record.stream}: 正式帧池已满 "
                    f"({ring.size()}/{ring.capacity})") from error
            self.counts[record.stream] = self.counts.get(record.stream, 0) + 1
        finally:
            self._timed(started)

    def append(self, record):
        started = time.monotonic_ns()
        try:
            if not self._accept(record):
                return
            if record.stream == "cameras/camera_0/depth":
                ring = self.rings[record.stream]
                image = record.values["image"]
                metadata = Record(record.stream, record.time_ns, record.sequence,
                                  {"source_time_ms": record.values["source_time_ms"]})
                try:
                    ring.put_nowait(0, image, metadata)
                except Full as error:
                    raise RuntimeError(
                        f"{record.stream}: 正式帧池已满 "
                        f"({ring.size()}/{ring.capacity})") from error
            else:
                output = self.files.get(record.stream)
                if output is None:
                    name = record.stream.replace("/", "__") + ".bin"
                    output = self.files[record.stream] = (self.raw / "streams" / name).open(
                        "wb", buffering=1024 * 1024)
                flattened = []
                for name, size in STREAM_FIELDS[record.stream]:
                    value = record.values[name]
                    if len(value) != size:
                        raise ValueError(f"{record.stream}.{name} must contain {size} values")
                    flattened.extend(value)
                output.write(NUMERIC_STRUCTS[record.stream].pack(
                    record.time_ns, record.sequence, *flattened))
            self.counts[record.stream] = self.counts.get(record.stream, 0) + 1
        finally:
            self._timed(started)

    def status(self):
        return {
            "counts": dict(self.counts),
            "pending": {stream: ring.size() for stream, ring in self.rings.items()},
            "timing": {name: dict(values) for name, values in self._timing.items()},
            "workers": {stream: {"pid": process.pid, "alive": process.is_alive(),
                                  "failed": self.rings[stream].failed.is_set(),
                                  "processed": self.rings[stream].processed.value}
                        for stream, process in self.processes.items()},
        }

    def close(self, end_ns, status="complete", reason=None):
        if self._closed:
            return self.document.get("status", status)
        error = None
        for output in self.files.values():
            try:
                output.flush()
                os.fsync(output.fileno())
            except OSError as problem:
                error = error or problem
            output.close()
        self.files.clear()
        for ring in self.rings.values():
            ring.closing.set()
            ring.wake.set()
        for stream, process in self.processes.items():
            process.join(15.)
            ring = self.rings[stream]
            if process.is_alive():
                process.terminate()
                process.join(2.)
                error = error or RuntimeError(f"{stream}: 写入进程收尾超时")
            if process.exitcode not in (0, None):
                error = error or RuntimeError(f"{stream}: 写入进程退出码 {process.exitcode}")
            if ring.failed.is_set():
                error = error or RuntimeError(f"{stream}: {ring.error()}")
            expected = self.counts.get(stream, 0)
            if ring.processed.value != expected:
                error = error or RuntimeError(
                    f"{stream}: 生产 {expected}，持久化 {ring.processed.value}")
            ring.close_queues()
        published = "failed" if error else "captured" if status == "complete" else status
        self.document.update(end_ns=end_ns, status=published,
                             reason=str(error) if error else reason,
                             counts=dict(self.counts),
                             spool={"numeric_format": "little-endian int64,int64,float64[]",
                                    "camera_meta_format": "little-endian int64,int64,float64",
                                    "depth_shape": [480, 640], "depth_dtype": "uint16"})
        write_json(self.path / "episode.json", self.document)
        self._closed = True
        if error:
            raise error
        return published
