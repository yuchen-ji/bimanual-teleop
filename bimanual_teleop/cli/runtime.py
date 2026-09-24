"""Terminal interaction and scheduling shared by arm and hand teleoperation."""

from dataclasses import asdict
from collections import deque
import json
import math
import threading
import time

from bimanual_teleop.common.console import print_message, runtime_message
from bimanual_teleop.common.runlog import process_snapshot
from bimanual_teleop.control.arm.cartesian import PERIOD_NS


TRACKING_NOTICE_NS = 500_000_000
HELP = "Enter 开始/恢复 · Space 暂停/取消 · Q 退出"


def brief_reason(detail, *, verbose=False):
    detail = runtime_message(detail or "等待设备反馈", verbose=verbose)
    if verbose:
        return detail
    for prefix, message in (
        ("waiting for quest frames", "等待 Quest 数据"),
        ("quest left tracking unavailable", "左手柄未被追踪，请唤醒并放在头显可见范围"),
        ("quest right tracking unavailable", "右手柄未被追踪，请唤醒并放在头显可见范围"),
        ("quest xr session not focused", "请关闭 Quest 系统菜单，返回采集应用"),
        ("quest reference space is changing", "Quest 坐标系正在切换，等待新坐标系数据后重新接合"),
        ("quest reference changed", "Quest 坐标系已改变，请重新接合建立跟随基准"),
        ("quest input stream gap reached", "Quest 数据流曾中断达到超时阈值，数据恢复后请重新接合"),
        ("quest input is silent or additionally queued", "Quest 输入已超过 100 毫秒未更新或排队超时"),
        ("quest additional input backlog", "Quest 输入数据额外排队已超过 100 毫秒"),
        ("no quest frame arrived for 1 second", "Quest 数据流已连续 1 秒没有新帧"),
        ("no tianji feedback", "等待机器人反馈"),
    ):
        if detail.lower().startswith(prefix):
            return message
    return detail


class TeleopUI:
    """Only UI state lives here; the runtime validates and executes each request."""

    def __init__(self, runtime, profile, *, emit=None,
                 background_engage=False, following_message="已接合，双臂正在跟随手柄。", gesture=None,
                 verbose=False, home_enabled=False, toggle_engagement_key=None,
                 ready_pose_key="h", gesture_engagement_enabled=True, runtime_log=None):
        self.runtime, self.profile = runtime, profile
        self.emit = emit
        self.verbose = verbose
        self.quit = False
        self.engage_pending = False
        self._last_error = None
        self.last_motion_error = None
        self.motion_pauses = 0
        self._last_status = None
        self._last_message = None
        self.background_engage = background_engage
        self.following_message = following_message
        self._operation_thread = None
        self._operation_error = None
        self._operation_cancelled = False
        self.home_enabled = home_enabled
        self._operation = "engage"
        self._operation_cancel = threading.Event()
        self.gesture = gesture if gesture_engagement_enabled else None
        self.toggle_engagement_key = toggle_engagement_key
        self.ready_pose_key = ready_pose_key
        self.gesture_engagement_enabled = gesture_engagement_enabled
        self.runtime_log = runtime_log
        self._cycle_history = deque(maxlen=400)
        self._pause_logged = False
        self.loop_timing = {}
        self._reset_tracking_notice()

    @property
    def start_hint(self):
        hint = "按 Enter"
        if self.toggle_engagement_key not in (None, "enter"):
            hint += f" / {self.toggle_engagement_key.upper()}"
        if self.gesture is not None and self.gesture_engagement_enabled:
            hint += " 或双手重新比 V"
        return hint

    @property
    def help_text(self):
        text = HELP
        if self.toggle_engagement_key == "enter":
            text = "Enter 接合/脱离（操作中取消） · Space 暂停/取消 · Q 退出"
        elif self.toggle_engagement_key is not None:
            text += f" · {self.toggle_engagement_key.upper()} 接合/脱离（操作中取消）"
        if self.home_enabled:
            text += f" · {self.ready_pose_key.upper()} 停止跟随并回位，到位后脱离"
        if self.gesture is not None:
            text += "\n双手同时比 V 保持 0.3 秒：开始/恢复"
            text += "\n任一手摇滚保持 0.3 秒：暂停"
            if self.home_enabled:
                text += " · 暂停后双手张开保持 1 秒：清错并回位；请先释放实体急停"
        return text

    def is_toggle_key(self, key):
        if self.toggle_engagement_key == "enter":
            return key in ("\n", "\r")
        return key == self.toggle_engagement_key

    def is_pause_key(self, key):
        state = getattr(self.runtime, "state", None)
        return key == " " or (self.is_toggle_key(key) and (
            getattr(state, "value", state) in ("engaged", "homing")
            or self.engage_pending or self._operation_thread is not None))

    def say(self, message, level="info"):
        if message == self._last_message:
            return
        self._last_message = message
        if self.emit is None:
            print_message(message, level)
        else:
            self.emit(message)

    def report_status(self, status):
        health = status.get("health") or asdict(self.runtime.health())
        state = status.get("state", "").lower()
        current = (state, status.get("mode"), health["ready"],
                   health.get("detail") if not health["ready"] else None,
                   self.engage_pending)
        if current == self._last_status:
            return
        previous, self._last_status = self._last_status, current
        if not health["ready"]:
            detail = health.get("detail") or "等待设备反馈"
            if detail != self._last_error:
                self.say(self.waiting_message(detail), "warning")
        elif (state != "engaged" and self._operation_thread is None
              and (previous is None or not previous[2])):
            self.say(f"设备已就绪，{self.start_hint}开始遥操作。", "ready")

    def waiting_message(self, detail):
        message = brief_reason(detail, verbose=self.verbose)
        return f"{message}；就绪后自动接合，Space 可取消。" if self.engage_pending else message

    def abort(self, reason):
        self._operation_cancel.set()
        self._reset_tracking_notice()
        self.engage_pending = False
        self._operation_cancelled = True
        if self.gesture is not None:
            self.gesture.inhibit()
        self.runtime.pause(reason)
        self._log_pause(reason, origin="ui_abort")
        if reason != self._last_error:
            self.say(brief_reason(reason, verbose=self.verbose), "warning")
        self._last_error = reason

    def handle(self, key):
        if self.quit:
            return
        key = key.lower()
        try:
            if key in ("q", "\x04", "\x03"):
                self._operation_cancel.set()
                self._reset_tracking_notice()
                self.engage_pending = False
                self._operation_cancelled = True
                self.quit = True
            elif self.is_pause_key(key):
                self.abort(f"键盘暂停；恢复须{self.start_hint}重新接合")
            elif key in ("\n", "\r") or self.is_toggle_key(key):
                self.request_engage(wait_until_ready=True)
            elif key == self.ready_pose_key and self.home_enabled:
                self.request_home(stop_follow=True)
        except (OSError, RuntimeError, ValueError) as error:
            self.last_motion_error = str(error)
            self.abort(str(error))

    def request_engage(self, *, wait_until_ready=False):
        state = getattr(self.runtime, "state", None)
        if (getattr(state, "value", state) in ("engaged", "homing", "closed")
                or self._operation_thread is not None or self.quit):
            return False
        health = self.runtime.health()
        self.engage_pending = wait_until_ready and not health.ready
        if not health.ready:
            self.say(self.waiting_message(health.detail), "warning")
            return False
        if self.background_engage:
            self.say("正在接合；Space / Q 可取消。")
            self._start_operation("engage")
        else:
            self.runtime.engage(self.profile)
            self._engaged()
        return True

    def request_home(self, *, stop_follow=False):
        if not self.home_enabled or self._operation_thread is not None or self.quit:
            return False
        state = getattr(self.runtime.state, "value", self.runtime.state)
        if state == "engaged" and stop_follow:
            # Stop both runtimes before starting any ready-pose movement.
            # If stopping fails, propagate the error without launching homing.
            self.abort("键盘请求回位，已停止遥操作跟随")
            state = getattr(self.runtime.state, "value", self.runtime.state)
        if state not in ("ready", "paused"):
            home_hint = f"按 {self.ready_pose_key.upper()}"
            if self.gesture is not None:
                home_hint += " 或双手张开保持 1 秒"
            self.say(f"请先暂停遥操作，再{home_hint}回位。", "warning")
            return False
        self.engage_pending = False
        if self.gesture is not None:
            self.gesture.inhibit()
        cancel_hint = "Space" + (" / 摇滚手势" if self.gesture is not None else "")
        self.say(f"正在清错并回位，无需再次按回车；{cancel_hint}可中止，Q 退出。")
        self._start_operation("home")
        return True

    def _start_operation(self, operation):
        self._operation = operation
        self._operation_cancelled = False
        self._operation_error = None
        self._operation_cancel = threading.Event()
        self._operation_thread = threading.Thread(
            target=self._run_operation, name=f"teleop-{operation}", daemon=True)
        self._operation_thread.start()

    def handle_gesture(self, command):
        if not self.gesture_engagement_enabled:
            return "ignored"
        if command == "home":
            return "home" if self.request_home() else "ignored"
        if command == "pause":
            self.abort(f"摇滚手势暂停；{self.start_hint}恢复")
            return "pause"
        if command != "engage":
            raise ValueError(f"Unknown gesture command: {command}")
        state = self.runtime.state
        if getattr(state, "value", state) == "engaged" or self._operation_thread is not None:
            return "ignored"
        return "engage" if self.request_engage() else "not_ready"

    def _engaged(self):
        self._reset_tracking_notice()
        self._last_error = None
        self._pause_logged = False
        self._cycle_history.clear()
        self.say(self.following_message, "ready")

    def _run_operation(self):
        try:
            if self._operation == "home":
                self.runtime.home(self._operation_cancel)
            else:
                self.runtime.engage(self.profile)
        except Exception as error:
            self._operation_error = error

    def poll_operation(self):
        thread = self._operation_thread
        if thread is None or thread.is_alive():
            return
        thread.join()
        self._operation_thread = None
        if self.quit:
            return
        if self._operation_cancelled:
            self.runtime.pause("回位已取消" if self._operation == "home" else "接合已取消")
        elif self._operation_error is not None:
            self.last_motion_error = str(self._operation_error)
            self.abort(self.last_motion_error)
        elif self._operation == "home":
            if self.gesture is not None:
                self.gesture.inhibit()
            self._last_error = None
            self.say(f"已到达 ready pose，保持脱离；仅继续遥操作时才需{self.start_hint}接合。", "ready")
        else:
            self._engaged()

    def close(self):
        self.quit = True
        self.engage_pending = False
        self._operation_cancel.set()
        self._operation_cancelled = True
        error = None
        try:
            # Close the runtime before joining a slow engagement/home worker,
            # so it cannot keep the arms enabled or restart them after exit.
            self.runtime.close()
        except Exception as problem:
            error = problem
        finally:
            if self._operation_thread is not None:
                self._operation_thread.join(timeout=10.)
                if self._operation_thread.is_alive():
                    raise RuntimeError(f"{str(error) + '; ' if error else ''}后台操作未在关闭期限内退出")
                self._operation_thread = None
        if error is not None:
            raise error

    def report_runtime_pause(self):
        state = getattr(self.runtime, "state", None)
        reason = getattr(self.runtime, "last_error", None)
        if getattr(state, "value", state) != "engaged":
            self._reset_tracking_notice()
        if getattr(state, "value", state) == "paused" and reason and reason != self._last_error:
            if self.gesture is not None:
                self.gesture.inhibit()
            self.last_motion_error = self._last_error = reason
            self.motion_pauses += 1
            diagnostic, has_retained_diagnostic = self._log_pause(reason, origin="runtime")
            self.say(f"遥操作已暂停：{brief_reason(reason, verbose=self.verbose)}；恢复时{self.start_hint}。", "warning")
            if self.verbose and has_retained_diagnostic:
                self.say("[停机诊断] " + json.dumps(
                    diagnostic, ensure_ascii=False, separators=(",", ":")), "warning")

    def _log_pause(self, reason, *, origin):
        arms = getattr(self.runtime, "arms", self.runtime)
        retained_value = getattr(arms, "last_pause_diagnostic", None)
        retained = retained_value if isinstance(retained_value, dict) else {}
        diagnostic = {**retained, "reason": reason,
                      "host_loop": dict(self.loop_timing)}
        hands = getattr(self.runtime, "hands", None)
        if hands is not None:
            try:
                hand_status = hands.status(include_target=False)
                diagnostic["hands"] = hand_status if isinstance(hand_status, dict) else {}
            except Exception as error:
                diagnostic["hands"] = {"status_error": str(error)}
        if self.runtime_log is not None and not self._pause_logged:
            self._pause_logged = True
            try:
                status = self.runtime.status(include_target=False)
                if not isinstance(status, dict):
                    status = {}
            except Exception as error:
                status = {"status_error": str(error)}
            self.runtime_log.event(
                "motion_pause", origin=origin, reason=reason, diagnostic=diagnostic,
                runtime_status=status, preceding_cycles=list(self._cycle_history),
                process=process_snapshot())
        return diagnostic, bool(retained)

    def record_cycle(self, runtime, cycle_index):
        if self.runtime_log is None:
            return
        arms = getattr(runtime, "arms", runtime)
        executor = getattr(arms, "executor", None)
        self._cycle_history.append({
            "cycle_index": cycle_index,
            "state": getattr(getattr(runtime, "state", None), "value", getattr(runtime, "state", None)),
            "host_loop": dict(self.loop_timing),
            "arm_cycle": dict(getattr(arms, "cycle_timing", {})),
            "executor": dict(getattr(executor, "timing_status", {})) if executor else {},
        })

    def log_runtime_status(self, status):
        # The UI still checks live health once per second, but normal operation
        # does not serialize a full runtime snapshot.  The retained cycle ring
        # and device state are emitted by _log_pause only after a fault.
        return None

    def _reset_tracking_notice(self):
        self._tracking_limited_since = {}
        self._tracking_notice_active = False
        self._tracking_recovered_since = None

    def report_tracking(self, now_ns=None):
        """Report servo limitations only in verbose mode, using the small snapshot."""
        if not self.verbose:
            return
        state = getattr(self.runtime, "state", None)
        if getattr(state, "value", state) != "engaged":
            self._reset_tracking_notice()
            return
        arms = getattr(self.runtime, "arms", self.runtime)
        tracking = getattr(arms, "tracking_status", {})
        if not tracking:
            self._reset_tracking_notice()
            return
        now = time.monotonic_ns() if now_ns is None else now_ns
        limited = {side for side, values in tracking.items() if values.get("limited") and
                   (values.get("position_error_m", 0.) >= .001 or
                    values.get("orientation_error_rad", 0.) >= math.radians(.5))}
        self._tracking_limited_since = {
            side: self._tracking_limited_since.get(side, now) for side in limited}
        if limited:
            self._tracking_recovered_since = None
            sustained = {side for side, since in self._tracking_limited_since.items()
                         if now-since >= TRACKING_NOTICE_NS}
            if sustained and not self._tracking_notice_active:
                self._tracking_notice_active = True
                names = "、".join("左臂" if side == "left" else "右臂"
                                  for side in ("left", "right") if side in sustained)
                self.say(f"目标推进受限（{names}）；遥操作仍处于接合状态。"
                         "放慢手柄移动，或将手柄位置/姿态移回可达范围可继续跟随。", "warning")
        elif self._tracking_notice_active:
            if self._tracking_recovered_since is None:
                self._tracking_recovered_since = now
            elif now-self._tracking_recovered_since >= TRACKING_NOTICE_NS:
                self._tracking_notice_active = False
                self._tracking_recovered_since = None
                self.say("目标推进已恢复正常，遥操作继续跟随。", "ready")


def run_loop(runtime, ui, terminal, *, period_ns=PERIOD_NS):
    next_tick = time.monotonic_ns()
    started = next_tick
    next_report = next_tick
    cycles = skipped = 0
    while not ui.quit:
        now = time.monotonic_ns()
        if now >= next_tick:
            ui.loop_timing.update(tick_started_ns=now, wake_lateness_ns=now-next_tick)
            missed = (now - next_tick) // period_ns
            skipped += missed
            # Rebase a late cycle instead of squeezing it next to the next one.
            next_tick = now + period_ns
            try:
                ui.poll_operation()
                ui.report_runtime_pause()
                gesture = (ui.gesture.poll(start_ready=ui.gesture_engagement_enabled and runtime.health().ready,
                           home_ready=ui.home_enabled and ui._operation_thread is None
                           and getattr(runtime.state, "value", runtime.state) == "paused")
                           if ui.gesture is not None else None)
                if gesture is not None:
                    command, _sides = gesture
                    ui.handle_gesture(command)
                if ui.engage_pending and runtime.health().ready:
                    # Readiness completes the request; it is not another toggle keypress.
                    ui.request_engage(wait_until_ready=True)
                ui.loop_timing["pre_tick_ns"] = time.monotonic_ns() - now
                tick_started = time.monotonic_ns()
                tick_process_cpu = time.process_time_ns()
                tick_thread_cpu = time.thread_time_ns()
                runtime.tick(now)
                ui.loop_timing["runtime_tick_ns"] = time.monotonic_ns() - tick_started
                ui.loop_timing["runtime_tick_process_cpu_ns"] = time.process_time_ns() - tick_process_cpu
                ui.loop_timing["runtime_tick_thread_cpu_ns"] = time.thread_time_ns() - tick_thread_cpu
            except (OSError, RuntimeError, ValueError) as error:
                ui.last_motion_error = str(error)
                ui.record_cycle(runtime, cycles)
                ui.abort(str(error))
            else:
                ui.record_cycle(runtime, cycles)
            ui.report_runtime_pause()
            ui.report_tracking()
            cycles += 1
            finished = time.monotonic_ns()
            if finished >= next_tick:
                missed = (finished - next_tick) // period_ns + 1
                skipped += missed
                next_tick += missed * period_ns
        if now >= next_report:
            report_started = time.monotonic_ns()
            status = runtime.status(include_target=False)
            ui.report_status(status)
            ui.loop_timing["last_status_ns"] = time.monotonic_ns() - report_started
            ui.loop_timing["last_status_started_ns"] = report_started
            ui.log_runtime_status(status)
            next_report = now + 1_000_000_000
        wait_started = time.monotonic_ns()
        wait_ns = max(0, next_tick - wait_started)
        keys = terminal.read(wait_ns / 1e9)
        ui.loop_timing["last_terminal_wait_ns"] = time.monotonic_ns() - wait_started
        ui.loop_timing["requested_terminal_wait_ns"] = wait_ns
        if keys == "":
            ui.handle("q")
        elif keys:
            for key in keys:
                ui.handle(key)
                if ui.quit:
                    break
    return {"cycles": cycles, "skipped_deadlines": skipped, "target_hz": round(1e9 / period_ns),
            "last_motion_error": ui.last_motion_error, "motion_pauses": ui.motion_pauses,
            "elapsed_s": (time.monotonic_ns() - started) / 1e9}
