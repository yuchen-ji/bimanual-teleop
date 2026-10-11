"""Session metadata and a rebuildable index; episode.json owns dynamic status."""

from datetime import datetime
import json
import os
from pathlib import Path

from .episodes import MANIFEST as EPISODE_MANIFEST, find_episodes
from .schema import CAMERAS, MAIN_CAMERA
from .storage import write_json

MANIFEST = "dataset.json"
SCHEMA_VERSION = 1
FORMAT = "tianji_raw_dataset"


def _metadata(config, cameras):
    if set(cameras) != set(CAMERAS):
        raise ValueError("Dataset metadata requires all three started cameras")
    resolved = {}
    for camera, serial in zip(CAMERAS, config.cameras):
        if cameras[camera].get("serial") != serial:
            raise ValueError(f"Started {camera} serial differs from recording configuration")
        streams = ["rgb"]
        if camera == MAIN_CAMERA and config.main_depth:
            streams.append("depth")
        resolved[camera] = {"role": config.dataset.camera_roles[camera],
                            "serial": cameras[camera]["serial"], "streams": streams}
    devices = []
    for device in config.dataset.collection_devices:
        entry = {"model": device.model, "role": device.role, "sides": list(device.sides)}
        if device.components:
            entry["components"] = list(device.components)
        devices.append(entry)
    return {"schema_version": SCHEMA_VERSION, "format": FORMAT,
            "robot": {"name": "Tianji"}, "task": config.dataset.task,
            "cameras": resolved, "collection_devices": devices}


def read_dataset_manifest(session):
    path = Path(session) / MANIFEST
    document = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(document, dict) or type(document.get("schema_version")) is not int
            or document["schema_version"] != SCHEMA_VERSION or document.get("format") != FORMAT):
        raise ValueError(f"Unsupported dataset manifest: {path}")
    if any(not isinstance(document.get(key), str) or not document[key].strip()
           for key in ("dataset_id", "created_at")):
        raise ValueError(f"Dataset manifest needs dataset_id and created_at: {path}")
    return document


def ensure_dataset_manifest(session, config, cameras, *, created_at=None):
    """Create once, or reject changed session metadata without overwriting it."""
    session = Path(session)
    metadata = _metadata(config, cameras)
    if (session / MANIFEST).exists():
        document = read_dataset_manifest(session)
        for key, value in metadata.items():
            if document.get(key) != value:
                raise ValueError(f"Dataset {key} changed; start a new recording session: {session}")
        return document
    document = {"schema_version": SCHEMA_VERSION, "format": FORMAT,
                "dataset_id": session.name,
                "created_at": created_at or datetime.now().astimezone().isoformat(timespec="seconds"),
                **metadata, "episodes": []}
    session.mkdir(parents=True, exist_ok=True)
    write_json(session / MANIFEST, document)
    return document


def sync_dataset_episodes(session):
    """Rebuild relative episode entries, retaining failures until actually deleted.

    Legacy sessions without a dataset.json remain untouched. An unchanged index
    does not need another write; this is called only at lifecycle boundaries.
    """
    session = Path(session).resolve()
    if not (session / MANIFEST).exists():
        return False
    document = read_dataset_manifest(session)
    episodes = [{"episode_id": episode.name, "path": episode.relative_to(session).as_posix()}
                for episode in find_episodes(session)]
    if document.get("episodes") != episodes:
        document["episodes"] = episodes
        write_json(session / MANIFEST, document)
    return True


def find_dataset_sessions(path):
    """Find session manifests without descending into large episode payloads.

    Also include empty sessions so retrying finalization can repair an index
    after the last discarded episode was deleted but its index write failed.
    """
    source = Path(path).resolve()
    if (source / EPISODE_MANIFEST).is_file():
        source = source.parent
    sessions = []
    for directory, children, files in os.walk(source):
        if EPISODE_MANIFEST in files:
            children.clear()
            continue
        if MANIFEST in files:
            sessions.append(Path(directory))
        children[:] = sorted(name for name in children if not name.startswith("."))
    return sorted(sessions)
