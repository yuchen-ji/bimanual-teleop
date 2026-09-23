"""Low-overhead structured runtime logs for post-fault timing analysis."""

from __future__ import annotations

from datetime import datetime
import json
import logging
import math
import os
from pathlib import Path
import platform
from queue import Full, Queue
import resource
import sys
import threading
import time


def default_run_log_path(name="teleop_quest_tianji", directory="logs"):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return Path(directory) / f"{name}_{stamp}_{os.getpid()}.jsonl"


def process_snapshot():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    try:
        load_average = os.getloadavg()
    except OSError:
        load_average = None
    try:
        niceness = os.nice(0)
    except (AttributeError, OSError):
        niceness = None
    return {
        "pid": os.getpid(), "thread": threading.current_thread().name,
        "process_time_ns": time.process_time_ns(),
        "thread_time_ns": time.thread_time_ns(),
        "user_cpu_s": usage.ru_utime, "system_cpu_s": usage.ru_stime,
        "max_rss_kib": usage.ru_maxrss,
        "voluntary_context_switches": usage.ru_nvcsw,
        "involuntary_context_switches": usage.ru_nivcsw,
        "load_average": load_average, "niceness": niceness,
    }


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    return repr(value)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class RuntimeLog:
    """One line-buffered JSON stream; high-rate cycles remain in memory until a fault."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("x", encoding="utf-8", buffering=1)
        self._queue = Queue(maxsize=2048)
        self._closed = False
        self._write_error = None
        self._dropped = 0
        self._handler = None
        self._writer = threading.Thread(target=self._write,
                                        name="runtime-log-writer", daemon=True)
        self._writer.start()

    def _write(self):
        while True:
            row = self._queue.get()
            if row is None:
                break
            try:
                line = json.dumps(_json_safe(row), ensure_ascii=False,
                                  separators=(",", ":"), allow_nan=False,
                                  default=_json_default)
                self._stream.write(line + "\n")
                self._stream.flush()
            except Exception as error:
                self._write_error = self._write_error or repr(error)

    def event(self, event, **details):
        if self._closed:
            return
        row = {
            "schema": "bimanual_teleop.runtime_log.v1",
            "event": event,
            "wall_time": datetime.now().astimezone().isoformat(),
            "monotonic_ns": time.monotonic_ns(),
            "pid": os.getpid(),
            "thread": threading.current_thread().name,
            **details,
        }
        try:
            self._queue.put_nowait(row)
        except Full:
            self._dropped += 1

    def start(self, **details):
        self.event("session_start", python=sys.version, platform=platform.platform(),
                   cpu_count=os.cpu_count(), cwd=str(Path.cwd()),
                   process=process_snapshot(), **details)

    def attach_python_logging(self, logger):
        self._handler = _RuntimeLogHandler(self)
        self._handler.setLevel(logging.DEBUG)
        logger.addHandler(self._handler)

    def close(self, **details):
        if self._closed:
            return
        self.event("session_end", process=process_snapshot(),
                   dropped_events=self._dropped, write_error=self._write_error, **details)
        if self._handler is not None:
            logger = logging.getLogger("bimanual_teleop")
            logger.removeHandler(self._handler)
            self._handler = None
        self._closed = True
        self._queue.put(None)
        self._writer.join(timeout=10.)
        self._stream.close()


class _RuntimeLogHandler(logging.Handler):
    def __init__(self, run_log):
        super().__init__()
        self.run_log = run_log

    def emit(self, record):
        try:
            self.run_log.event("python_log", logger=record.name,
                               level=record.levelname, message=record.getMessage())
        except Exception:
            self.handleError(record)
