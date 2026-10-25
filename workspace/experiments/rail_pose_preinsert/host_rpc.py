"""Single-owner RPC service with a heartbeat independent of planner execution."""
from __future__ import annotations

import fcntl
import json
import os
import threading
import time
from pathlib import Path

from common import write_json


def process_identity(pid):
    """Linux process start time prevents a reused PID from extending a service."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


class WorkerLease:
    def __init__(self, output, scene_pid=None):
        self.directory = Path(output) / "rpc"
        self.scene_pid = scene_pid
        self.state = {"status": "idle", "request_id": None}
        self.stop = threading.Event()

    def publish(self):
        write_json(self.directory / "worker.json", {"pid": os.getpid(), "scene_pid": self.scene_pid,
                   "updated_at": time.time(), **self.state})

    def heartbeat(self):
        while not self.stop.wait(1.):
            self.publish()

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / "worker.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError(f"An RPC worker already owns {self.directory}") from None
        self.publish()
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.state = {"status": "stopped", "request_id": None}
        self.publish()
        self.lock.close()


class RequestServer:
    def __init__(self, output, handler, lease):
        self.output, self.handler, self.lease = Path(output), handler, lease
        self.previous = None

    def poll(self):
        path = self.output / "rpc/request.json"
        if not path.is_file():
            return False
        request = json.loads(path.read_text())
        request_id = request["id"]
        if request_id == self.previous:
            return False
        response_path = self.output / "rpc/response.json"
        if response_path.is_file() and json.loads(response_path.read_text()).get("id") == request_id:
            self.previous = request_id
            return False
        self.lease.state = {"status": "busy", "request_id": request_id}
        try:
            response = self.handler(request)
        except Exception as error:  # noqa: BLE001 -- Report planner errors without killing the service.
            response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        write_json(self.output / "trials" / f"trial_{request.get('trial', 0):03d}" / "host_requests" / f"{request_id}.json",
                   {"request": request, "response": response})
        # A reset/reload may replace a request while its old solve is running.
        if json.loads(path.read_text()).get("id") == request_id:
            write_json(response_path, {"id": request_id, **response})
        self.previous = request_id
        self.lease.state = {"status": "idle", "request_id": None}
        print(request["stage"], response.get("error", "ok"), flush=True)
        return True
