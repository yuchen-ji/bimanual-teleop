"""Recording keys added to the existing terminal UI, not a second control loop."""

import time

from bimanual_teleop.cli.runtime import TeleopUI


def recording_help(config):
    delay = config.start_delay_s
    when = "立即" if not delay else f"{delay:g} 秒后"
    return (
        f"接合后{when}开始录制；脱离只暂停本条，再次接合后继续 · "
        f"{config.save_key.upper()} 保存 · {config.discard_key.upper()} 作废 · "
        f"{config.quit_key.upper()} 保存并退出 · "
        f"采集失败时先脱离，再按 {config.recover_key.upper()} 恢复进程"
    )


def _setting(recorder, name, default):
    config = getattr(recorder, "config", None)
    value = getattr(config, name, default) if config is not None else default
    if isinstance(default, str):
        return value if isinstance(value, str) and len(value) == 1 else default
    if isinstance(default, float) and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


class RecordingUI(TeleopUI):
    def __init__(self, *args, recorder, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorder = recorder
        self._reported_recording_error = None
        self.save_key = _setting(recorder, "save_key", "s")
        self.discard_key = _setting(recorder, "discard_key", "x")
        self.quit_key = _setting(recorder, "quit_key", "q")
        self.recover_key = _setting(recorder, "recover_key", "c")
        self.start_delay_s = _setting(recorder, "start_delay_s", 0.0)
        self._engaged_since = None
        self._was_engaged = False

    def handle(self, key):
        key = key.lower()
        if key == self.recover_key:
            if self.recorder.error:
                if getattr(self.runtime.state, "value", self.runtime.state) == "engaged":
                    self.say(f"采集已停止，但遥操作仍在继续。请先主动脱离，再按 {self.recover_key.upper()} 恢复采集。", "warning")
                    return
                self.recorder.recover()
                self.say(self.recorder.error or "正在恢复采集；相机就绪后重新接合即可继续。")
                return
            return
        if key == self.save_key:
            self.recorder.end(status="complete")
            return
        if key == self.discard_key:
            self.recorder.end(status="discarded")
            return
        if key == self.quit_key or key in ("\x04", "\x03"):
            self.recorder.end(status="complete")
            super().handle("q" if key == self.quit_key else key)
            return
        super().handle(key)

    def handle_gesture(self, command):
        if not self.gesture_engagement_enabled:
            return "ignored"
        return super().handle_gesture(command)

    def abort(self, reason):
        try:
            super().abort(reason)
        finally:
            self._pause_open_episode()

    def _pause_open_episode(self):
        if self.recorder.state in ("starting", "recording", "resuming"):
            self.recorder.pause()
            self.say("录制已暂停，重新接合后继续本条。")
        self._engaged_since = None
        self._was_engaged = False

    def _sync_recording(self):
        state = getattr(self.runtime.state, "value", self.runtime.state)
        engaged = state == "engaged"
        recorder = self.recorder
        if not engaged:
            self._pause_open_episode()
            return
        if recorder.error:
            return
        if not self._was_engaged and recorder.state in ("idle", "paused"):
            self._engaged_since = time.monotonic() + self.start_delay_s
            if self.start_delay_s:
                self.say(f"已接合，{self.start_delay_s:g} 秒后{'继续' if recorder.state == 'paused' else '开始'}录制。")
        self._was_engaged = True
        if self._engaged_since is None or time.monotonic() < self._engaged_since:
            return
        if recorder.state in ("starting", "recording", "saving", "resuming"):
            self._engaged_since = None
            return
        try:
            if recorder.state == "paused":
                recorder.resume()
                self._engaged_since = None
            elif recorder.state == "idle":
                recorder.begin()
                self._engaged_since = None
        except (OSError, RuntimeError, ValueError) as error:
            self.say(str(error), "warning")

    def poll_operation(self):
        error = self.recorder.poll()
        if error and error != self._reported_recording_error:
            self._reported_recording_error = error
            self.recorder.end(status="failed", reason=error)
            if self.runtime_log is not None:
                self.runtime_log.event("recording_failure", error=error,
                                       recorder=self.recorder.status())
            self.say(
                "采集已停止，当前条不完整；遥操作继续。请先主动脱离，排除原因后按 "
                f"{self.recover_key.upper()} 恢复采集，再重新接合。", "warning")
        elif not error:
            self._reported_recording_error = None
        while self.recorder.notices:
            self.say(self.recorder.notices.pop(0))
        super().poll_operation()
        self._sync_recording()

    def log_runtime_status(self, status):
        return None

    def report_runtime_pause(self):
        super().report_runtime_pause()
        self._sync_recording()

    def close(self):
        self.recorder.end(status="failed", reason="遥操作退出")
        try:
            super().close()
        finally:
            self.recorder.close()
