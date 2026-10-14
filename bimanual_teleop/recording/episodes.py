"""Episode directories, manifest statuses and recursive discovery."""

import json
import os
from pathlib import Path

MANIFEST = "episode.json"

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
