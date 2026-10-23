"""Episode directories, manifest statuses and recursive discovery."""

import fcntl
import json
import os
from pathlib import Path
import re

MANIFEST = "episode.json"
EPISODE_COUNTER = ".next_episode"
TASK_LOCK = ".recording.lock"
_EPISODE_NAME = re.compile(r"\.?episode_(\d+)(?:\.|$)")

# Manifest lifecycle: capturing -> captured -> finalizing -> complete. The
# offline EpisodeWriter starts at recording. Failed and discarded are final.
RECORDING = "recording"
CAPTURING = "capturing"
CAPTURED = "captured"
FINALIZING = "finalizing"
COMPLETE = "complete"
FAILED = "failed"
DISCARDED = "discarded"


def read_manifest(episode):
    return json.loads((Path(episode) / MANIFEST).read_text(encoding="utf-8"))


def read_phase_markers(document):
    """Read optional, ordered markers on the episode's recording timeline."""
    markers = document.get("phase_markers", [])
    if not isinstance(markers, list):
        raise ValueError("phase_markers must be a list")
    result = []
    previous = document.get("start_ns")
    end = document.get("end_ns")
    for index, marker in enumerate(markers, 1):
        if (not isinstance(marker, dict) or type(marker.get("index")) is not int
                or marker["index"] != index or type(marker.get("time_ns")) is not int):
            raise ValueError("phase_markers must contain consecutive integer index and integer time_ns")
        stamp = marker["time_ns"]
        if (type(previous) is not int or stamp < previous or not 0 <= stamp < 2**63
                or (end is not None and (type(end) is not int or stamp >= end))):
            raise ValueError("phase_markers times must be ordered and inside [start_ns, end_ns)")
        result.append({"index": index, "time_ns": stamp})
        previous = stamp
    return result


def find_episodes(path):
    """Episode directories at any depth below ``path``, including ``path`` itself.

    An episode's own contents are not searched, and hidden directories such as
    ``.episode_000000.finalizing-<pid>`` left by an interrupted run are skipped.
    """
    source = Path(path).resolve()
    if not source.is_dir():
        raise ValueError(f"输入不是目录：{source}")
    episodes = []
    for directory, children, files in os.walk(source):
        if MANIFEST in files:
            episodes.append(Path(directory))
            children.clear()
            continue
        children[:] = sorted(name for name in children if not name.startswith("."))
    return sorted(episodes)


def episode_label(episode, root):
    """Name relative to the searched root; an episode given directly keeps its own name."""
    episode, root = Path(episode), Path(root).resolve()
    return episode.name if episode == root else episode.relative_to(root).as_posix()


class EpisodeNumbers:
    """Episode paths in one task directory, numbered continuously across runs.

    The lock is held until ``close`` so only one recorder writes a task at a
    time. The persisted counter keeps numbers from being reused after finalize
    deletes discarded or failed episodes, or an episode is removed by hand.
    """

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = open(self.directory / TASK_LOCK, "a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            raise RuntimeError(f"另一个采集进程正在写入同一任务目录：{self.directory}") from None
        try:
            counter = self.directory / EPISODE_COUNTER
            stored = int(counter.read_text(encoding="utf-8")) if counter.exists() else 0
            used = [int(match.group(1)) + 1 for name in os.listdir(self.directory)
                    if (match := _EPISODE_NAME.match(name))]
            self.next = max([stored, *used])
        except (OSError, ValueError):
            self.close()
            raise

    def allocate(self):
        index = self.next
        counter = self.directory / EPISODE_COUNTER
        temporary = counter.with_name(counter.name + ".tmp")
        temporary.write_text(f"{index + 1}\n", encoding="utf-8")
        temporary.replace(counter)
        self.next = index + 1
        return self.directory / f"episode_{index:06d}"

    def close(self):
        self._lock.close()
