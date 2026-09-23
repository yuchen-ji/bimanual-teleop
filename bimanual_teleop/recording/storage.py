"""One raw episode: append-only numeric streams and one video per RGB camera."""

import json
from pathlib import Path

from .sink import pose_values


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class RGBVideo:
    def __init__(self, path):
        import av
        self.container = av.open(str(path), "w")
        self.stream = self.container.add_stream("libx264", rate=30)
        self.stream.width, self.stream.height = 640, 480
        self.stream.pix_fmt = "yuv420p"
        # Three cameras are encoded serially in the lower-priority recording
        # process. Two codec threads provide enough measured headroom for
        # three RGB streams plus depth while leaving control processes ahead
        # of this worker under scheduler contention.
        self.stream.options = {"crf": "21", "preset": "ultrafast"}
        self.stream.thread_count = 2
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


class EpisodeWriter:
    def __init__(self, path, start_ns, metadata, kinematics):
        import zarr
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.document = dict(schema_version=1, status="recording", start_ns=start_ns,
                             end_ns=None, metadata=metadata)
        write_json(self.path / "episode.json", self.document)
        self.root = zarr.open_group(str(self.path / "raw.zarr"), mode="w")
        self.root.attrs.update(schema_version=1, metadata=metadata)
        self.kinematics = kinematics
        self.pending = {}
        self.videos = {}
        self.last_ns = {}
        self.counts = {}

    def prepare_rgb(self, cameras):
        """Open every encoder before live camera delivery starts."""
        for camera in cameras:
            if camera not in self.videos:
                self.videos[camera] = RGBVideo(self.path / f"{camera}.mp4")

    def append(self, record):
        if record.time_ns < self.document["start_ns"]:
            return
        previous = self.last_ns.get(record.stream)
        if previous is not None and record.time_ns <= previous:
            raise ValueError(f"Non-increasing timestamps: {record.stream}")
        self.last_ns[record.stream] = record.time_ns
        values = dict(record.values)
        if record.stream.startswith("arms/"):
            side = record.stream.split("/")[1]
            values["eef_pose"] = pose_values(self.kinematics.fk(side, values["joint_pos"]))
        row = {"time_ns": record.time_ns, "sequence": record.sequence, **values}
        pending = self.pending.setdefault(record.stream, [])
        pending.append(row)
        self.counts[record.stream] = self.counts.get(record.stream, 0) + 1
        if len(pending) >= (8 if "image" in row else 128):
            self.flush(record.stream)

    def write_rgb(self, camera, image, record):
        if record.time_ns < self.document["start_ns"]:
            return
        if record.time_ns <= self.last_ns.get(record.stream, -1):
            raise ValueError(f"Non-increasing timestamps: {record.stream}")
        video = self.videos.get(camera)
        if video is None:
            video = self.videos[camera] = RGBVideo(self.path / f"{camera}.mp4")
        video.write(image)
        self.append(record)

    def flush(self, stream):
        import numpy as np
        from numcodecs import Blosc
        rows = self.pending.get(stream)
        if not rows:
            return
        group = self.root.require_group(stream)
        for key in rows[0]:
            dtype = "i8" if key in ("time_ns", "sequence") else "u2" if key == "image" else "f8"
            data = np.asarray([row[key] for row in rows], dtype=dtype)
            if key not in group:
                group.create_dataset(key, shape=(0, *data.shape[1:]), dtype=dtype,
                    chunks=(1 if key == "image" else 256, *data.shape[1:]),
                    compressor=Blosc(cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE))
            group[key].append(data)
        rows.clear()

    def close(self, end_ns, status="complete", reason=None):
        error = None
        for video in self.videos.values():
            try:
                video.close()
            except Exception as problem:
                error = problem
        self.videos.clear()
        try:
            for stream in self.pending:
                self.flush(stream)
        except Exception as problem:
            error = problem
        self.document.update(end_ns=end_ns, status="failed" if error else status,
                             reason=str(error) if error else reason, counts=self.counts)
        write_json(self.path / "episode.json", self.document)
        if error:
            raise error
