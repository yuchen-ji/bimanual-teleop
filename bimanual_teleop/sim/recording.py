"""Bounded simulation recorder with continuous, unnamed boundary events."""

from __future__ import annotations

import copy
import json
import queue
import threading
import time
import uuid
from pathlib import Path

from bimanual_teleop.sim._dependencies import np

CAMERAS = ("camera_0", "camera_1", "camera_2")


def atomic_json(path, document):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def append_array(group, key, values):
    from numcodecs import Blosc
    values = np.asarray(values)
    if key not in group:
        group.create_dataset(key, shape=(0, *values.shape[1:]), dtype=values.dtype,
                             chunks=(128, *values.shape[1:]),
                             compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE))
    group[key].append(values)


class VideoWriter:
    def __init__(self, path, width, height, codec="h264_nvenc"):
        import av
        self.av = av
        self.container = av.open(str(path), "w")
        self.stream = self.container.add_stream(codec, rate=30)
        self.stream.width, self.stream.height, self.stream.pix_fmt = width, height, "yuv420p"
        self.stream.options = {"preset": "p4", "cq": "21"} if codec == "h264_nvenc" else {"preset": "veryfast", "crf": "18"}
        self.count = 0

    def append(self, image):
        frame = self.av.VideoFrame.from_ndarray(image, format="rgb24")
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


def state_row(state, block):
    row = {
        "tick": np.int64(state["tick"]), "time_ns": np.int64(state["time_ns"]),
        "block_id": np.int32(block), "robot_joint": np.asarray(state["q"][:14], dtype=np.float64),
        "hand_joint": np.asarray(state["q"][14:], dtype=np.float64),
        "eef_pose_xyzw": np.asarray(state["eef"], dtype=np.float64).reshape(2, 7),
        "arm_command_pose_xyzw": np.asarray(state["eef_target"], dtype=np.float64).reshape(2, 7),
        "hand_command": np.asarray(state["q_target"][14:], dtype=np.float64),
        "arm_joint_command": np.asarray(state["q_target"][:14], dtype=np.float64),
        "wrench": np.asarray(state["wrench"], dtype=np.float64).reshape(12),
        "command_tick": np.int64(state["command_tick"]),
        "rail_pose_xyzw": np.asarray(state["rail"], dtype=np.float64),
    }
    if "task_metrics" in state:
        metrics = state["task_metrics"]
        row["task_metrics"] = np.asarray([metrics[key] for key in ("pin_error_m", "axis_error_rad", "support_gap_m", "insertion_depth_m",
                                                                  "hands_released", "fixture_seated", "fixture_completed", "task_completed")], dtype=np.float64)
    if "hand_contact_force" in state:
        row["hand_contact_force"] = np.asarray(state["hand_contact_force"], dtype=np.float64)
    return row


class SimRecorder:
    def __init__(self, directory, metadata, width=640, height=480, capacity=1024, codec="h264_nvenc"):
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=False)
        self.document = {"schema_version": 1, "format": "bimanual_sim_raw", "status": "capturing",
                         "source_demo_id": str(uuid.uuid4()), "created_unix_ns": time.time_ns(),
                         "metadata": copy.deepcopy(metadata), "frames": 0, "states": 0}
        atomic_json(self.path / "episode.json", self.document)
        self.width, self.height, self.codec = width, height, codec
        self.queue = queue.Queue(capacity)
        self.failure = None
        self.frames, self.block, self.paused = 0, 0, False
        self.boundaries = []
        self.seen_events = set()
        self.closed = False
        self._closing = threading.Event()
        self.thread = threading.Thread(target=self._write, name="sim-recorder", daemon=True)
        self.thread.start()

    def _put(self, kind, value):
        if self.closed or self.failure or self._closing.is_set():
            return False
        try:
            self.queue.put_nowait((kind, value))
            return True
        except queue.Full:
            self.failure = "Simulation recording queue overflow"
            self._closing.set()
            return False

    def state(self, state):
        if self.paused:
            return False
        return self._put("state", state_row(state, self.block))

    def frame(self, state, images, boundary_events=()):
        if self.paused:
            return []
        if np.asarray(images).shape != (3, self.height, self.width, 3) or np.asarray(images).dtype != np.uint8:
            raise ValueError("Recorded camera images must be three uint8 RGB frames")
        row = state_row(state, self.block)
        row["frame_index"] = np.int64(self.frames)
        points = []
        for event in boundary_events:
            if event["event_id"] in self.seen_events:
                continue
            point = {"kind": "boundary", "boundary_id": len(self.boundaries)+len(points)+1,
                     "source_demo_id": self.document["source_demo_id"], "frame_index": self.frames,
                     "tick": int(state["tick"]), "time_ns": int(state["time_ns"]),
                     "block_id": self.block, **event}
            points.append(point)
        if not self._put("frame", (row, np.asarray(images).copy(), points)):
            return []
        self.frames += 1
        self.boundaries.extend(points)
        self.seen_events.update(p["event_id"] for p in points)
        return points

    def pause(self, tick, reason="operator"):
        if not self.paused:
            self._put("event", {"kind": "pause", "tick": tick, "reason": reason, "block_id": self.block})
            self.paused = True

    def resume(self, tick):
        if self.paused:
            self.block += 1; self.paused = False
            self._put("event", {"kind": "resume", "tick": tick, "block_id": self.block})

    def begin_close(self, discard=False):
        if not self.closed:
            self.closed = True
            self._discard = discard
            self._closing.set()

    def close(self, discard=False):
        self.begin_close(discard)
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError("Simulation recorder did not finish within 30 seconds")
        return self.document

    def _write(self):
        import zarr
        writers = []
        buffered = {"state": [], "frame": []}
        counters = {"state": 0, "frame": 0}
        root = None

        def flush(kind):
            rows = buffered[kind]
            if not rows:
                return
            group = root.require_group("states" if kind == "state" else "frames")
            for key in rows[0]:
                append_array(group, key, np.asarray([r[key] for r in rows]))
            counters[kind] += len(rows); rows.clear()

        try:
            root = zarr.open_group(str(self.path / "raw.zarr"), mode="w")
            root.attrs.update(format="bimanual_sim_raw", schema_version=1, metadata=self.document["metadata"])
            writers = [VideoWriter(self.path / f"{camera}.mp4", self.width, self.height, self.codec) for camera in CAMERAS]
            with (self.path / "events.jsonl").open("a", encoding="utf-8") as events:
                while not self._closing.is_set() or not self.queue.empty():
                    try:
                        kind, value = self.queue.get(timeout=.05)
                    except queue.Empty:
                        continue
                    if kind == "event":
                        events.write(json.dumps(value, ensure_ascii=False)+"\n"); events.flush()
                    elif kind == "state":
                        buffered[kind].append(value)
                    elif kind == "frame":
                        row, images, points = value
                        for writer, image in zip(writers, images):
                            writer.append(image)
                        buffered[kind].append(row)
                        for point in points:
                            events.write(json.dumps(point, ensure_ascii=False)+"\n")
                        events.flush()
                    for name in buffered:
                        if len(buffered[name]) >= 64:
                            flush(name)
                for kind in buffered:
                    flush(kind)
        except Exception as error:
            self.failure = str(error)
            self._closing.set()
        finally:
            for writer in writers:
                try:
                    writer.close()
                except Exception as error:
                    self.failure = self.failure or str(error)
            self.document.update(frames=counters["frame"], states=counters["state"], boundaries=len(self.boundaries),
                                 status="failed" if self.failure else "discarded" if getattr(self, "_discard", False) else "complete",
                                 failure=self.failure)
            try:
                atomic_json(self.path / "episode.json", self.document)
            except Exception as error:
                self.failure = self.failure or str(error)


def discover(directory):
    path = Path(directory)
    candidates = [path] if (path / "episode.json").is_file() else sorted(p.parent for p in path.rglob("episode.json"))
    result = []
    for candidate in candidates:
        document = json.loads((candidate / "episode.json").read_text())
        if document.get("format") != "bimanual_sim_raw":
            raise ValueError(f"Not a simulation recording: {candidate}")
        result.append((candidate, document))
    return result


def read_boundaries(directory):
    path = Path(directory) / "events.jsonl"
    if not path.exists():
        return []
    return [row for row in (json.loads(line) for line in path.read_text().splitlines()) if row.get("kind") == "boundary"]
