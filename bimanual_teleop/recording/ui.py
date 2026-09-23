"""Recording keys added to the existing terminal UI, not a second control loop."""

from bimanual_teleop.cli.runtime import TeleopUI

HELP = "C 开始录制 · S 保存 · X 作废当前条（录制要求双臂双手已接合）"


class RecordingUI(TeleopUI):
    def __init__(self, *args, recorder, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorder = recorder
        self._reported_recording_error = None

    def handle(self, key):
        key = key.lower()
        if key == "c":
            if self.runtime_log is not None:
                self.runtime_log.event("record_key", key="c", recorder=self.recorder.status())
            if self.recorder.error:
                self.abort(self.recorder.error)
                self.recorder.recover()
                self.say(self.recorder.error or "正在恢复采集；相机就绪后重新接合，再按 C。")
                return
            if getattr(self.runtime.state, "value", self.runtime.state) != "engaged":
                self.say("请先接合遥操作，再按 C 开始录制。", "warning")
                return
            try:
                self.recorder.begin()
            except (OSError, RuntimeError, ValueError) as error:
                self.say(str(error), "warning")
            return
        if key in ("s", "x"):
            if self.runtime_log is not None:
                self.runtime_log.event("record_key", key=key, recorder=self.recorder.status())
            self.recorder.end(status="complete" if key == "s" else "discarded")
            return
        if (self.is_pause_key(key) or key in ("q", "\x04", "\x03")
                or (key == self.ready_pose_key and self.home_enabled
                    and self._operation_thread is None and not self.quit)):
            self.recorder.end()
        super().handle(key)

    def handle_gesture(self, command):
        if not self.gesture_engagement_enabled:
            return "ignored"
        if command == "pause":
            self.recorder.end()
        return super().handle_gesture(command)

    def abort(self, reason):
        try:
            self.recorder.end(status="failed", reason=reason)
        finally:
            super().abort(reason)

    def poll_operation(self):
        error = self.recorder.poll()
        if error and error != self._reported_recording_error:
            self._reported_recording_error = error
            self.abort(error)
            self.say("采集已停止，当前条不完整。排除原因后按 C 恢复采集；相机就绪后重新接合，再按 C 开新条。", "warning")
        elif not error:
            self._reported_recording_error = None
        while self.recorder.notices:
            self.say(self.recorder.notices.pop(0))
        super().poll_operation()

    def log_runtime_status(self, status):
        if self.runtime_log is not None:
            self.runtime_log.event("runtime_status", status=status,
                                   host_loop=dict(self.loop_timing),
                                   retained_cycles=len(self._cycle_history),
                                   recorder=self.recorder.status())

    def report_runtime_pause(self):
        if getattr(self.runtime.state, "value", self.runtime.state) == "paused":
            self.recorder.end(status="failed", reason=self.runtime.last_error or "设备暂停")
        super().report_runtime_pause()

    def close(self):
        self.recorder.end(status="failed", reason="遥操作退出")
        try:
            super().close()
        finally:
            self.recorder.close()
