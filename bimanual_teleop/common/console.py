"""Concise terminal status and optional colored device logging."""

import sys
import logging
import json
import re
import time


_LEVELS = {
    "info": ("提示", "bright_black"),
    "ready": ("就绪", "cyan"),
    "warning": ("警告", "yellow"),
    "error": ("错误", "red"),
    "done": ("完成", "cyan"),
}


def format_message(message, level="info", *, stream=None):
    from rich.console import Console
    from rich.text import Text

    stream = sys.stderr if stream is None else stream
    label, colour = _LEVELS[level]
    console = Console(file=stream, highlight=False)
    with console.capture() as capture:
        console.print(Text.assemble((f"[{label}]", colour), f" {message}"),
                      end="", soft_wrap=True)
    return capture.get()


def print_message(message, level="info", *, stream=None):
    stream = sys.stderr if stream is None else stream
    if not getattr(stream, "isatty", lambda: False)():
        print(format_message(message, level, stream=stream), file=stream, flush=True)
        return

    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    label, colour = _LEVELS[level]
    table = Table.grid(padding=(0, 1))
    table.add_column(style=colour, no_wrap=True)
    table.add_row(f"[{label}]", Text(str(message)))
    Console(file=stream, highlight=False).print(table)


def runtime_message(message, *, verbose=False):
    """Hide structured diagnostic attachments while retaining the fault reason."""
    if verbose:
        return message

    def remove_attachment(match):
        try:
            _, end = json.JSONDecoder().raw_decode(match[1])
        except ValueError:
            return match[0]  # Keep unrecognized content so no fault text is lost.
        return match[1][end:]

    return re.sub(r"(?:^|\n)\[(?:IK诊断|控制诊断|停机诊断)\] ([^\n]*)",
                  remove_attachment, message).strip()


class StatusConsole:
    """Show state changes once and rate-limit repeated device warnings."""

    def __init__(self, *, stream=None, warning_interval_s=5.):
        self.stream = sys.stderr if stream is None else stream
        self.warning_interval_s = warning_interval_s
        self._last_state = None
        self._warnings = {}
        self._errors = {}

    def state(self, message, level="info"):
        current = (level, message)
        if current != self._last_state:
            print_message(message, level, stream=self.stream)
            self._last_state = current

    def warning(self, message, *, key=None):
        now = time.monotonic()
        key = message if key is None else key
        if now - self._warnings.get(key, float("-inf")) >= self.warning_interval_s:
            print_message(message, "warning", stream=self.stream)
            self._warnings[key] = now

    def error(self, message):
        now = time.monotonic()
        if now - self._errors.get(message, float("-inf")) >= self.warning_interval_s:
            print_message(message, "error", stream=self.stream)
            self._errors[message] = now


class LiveProgress:
    """A throttled, replace-in-place line for interactive target feedback."""

    def __init__(self, *, stream=None, interval_s=.2):
        self.stream = sys.stderr if stream is None else stream
        self.interval_s = interval_s
        self._last_at = float("-inf")
        self._width = 0

    def update(self, message):
        if not getattr(self.stream, "isatty", lambda: False)():
            return
        now = time.monotonic()
        if now - self._last_at < self.interval_s:
            return
        self._last_at = now
        from rich.text import Text

        formatted = format_message(message, "info", stream=self.stream)
        width = Text.from_ansi(formatted).cell_len
        print("\r" + formatted + " " * max(0, self._width - width),
              end="", file=self.stream, flush=True)
        self._width = width

    def clear(self):
        if self._width:
            print(file=self.stream, flush=True)
            self._width = 0


class _ConciseLogHandler(logging.Handler):
    def __init__(self, console, *, verbose=False):
        super().__init__()
        self.console = console
        self.verbose = verbose

    def emit(self, record):
        message = runtime_message(record.getMessage(), verbose=self.verbose)
        if not message:
            return
        if record.levelno >= logging.ERROR:
            self.console.error(message)
        elif record.levelno >= logging.WARNING:
            overflow = re.match(r"Tianji native (\S+) queue overflow:", message)
            key = f"Tianji native {overflow.group(1)} queue overflow" if overflow else message
            self.console.warning(message, key=key)
        else:
            self.console.state(message)


def configure_runtime_logging(*, wuji=False, verbose=False, log_file=None):
    """Select concise or verbose Python/SDK logging before device creation."""
    from .runlog import RuntimeLog, _RuntimeLogHandler

    logger = logging.getLogger("bimanual_teleop")
    logger.setLevel(logging.DEBUG if verbose or log_file is not None else logging.WARNING)
    for handler in list(logger.handlers):
        if isinstance(handler, (_ConciseLogHandler, _RuntimeLogHandler)):
            logger.removeHandler(handler)
            if isinstance(handler, _RuntimeLogHandler):
                handler.run_log.close(reconfigured=True)
    console_handler = _ConciseLogHandler(StatusConsole(), verbose=verbose)
    console_handler.setLevel(logging.DEBUG if verbose else logging.WARNING)
    logger.addHandler(console_handler)
    logger.propagate = False
    run_log = RuntimeLog(log_file) if log_file is not None else None
    if run_log is not None:
        # Routine debug/info messages are useful only in an explicitly verbose
        # run.  Normal runtime logs retain warnings and errors, avoiding
        # background JSON work for successful control cycles.
        run_log.attach_python_logging(
            logger, level=logging.DEBUG if verbose else logging.WARNING)
    if wuji:
        import wuji_sdk
        wuji_sdk.set_log_level("debug" if verbose else "error")
    return run_log
