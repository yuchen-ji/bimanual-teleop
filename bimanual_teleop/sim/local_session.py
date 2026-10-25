"""In-process actions, optional recorder and native UI for desktop simulation."""

from __future__ import annotations

import copy
import json
import time
import uuid
from datetime import datetime

from . import ROOT
from ._dependencies import np
from .actions import JointTrajectory
from .config import sim_path
from .recording import atomic_json
from .task import TaskMonitor


class LocalSession:
    def __init__(self, config, model, record=False, codec="h264_nvenc", trajectory=None, gui=False, status_path=None,
                 action_source=None):
        self.config, self.model, self.codec, self.gui = config, model, codec, gui
        if action_source is not None and trajectory is not None:
            raise ValueError("Choose either an action source or a recorded trajectory")
        self.action_source = action_source
        self.capture_enabled = record
        self.capturing = record
        self.program = JointTrajectory(trajectory, model, config["runtime"]) if trajectory else None
        self.ready = np.r_[np.deg2rad(config["robot"]["ready_deg"]["left"]),
                           np.deg2rad(config["robot"]["ready_deg"]["right"]), np.zeros(40)]
        if action_source is not None and hasattr(action_source, "ready"):
            self.ready = np.asarray(action_source.ready).copy()
        if self.program and not np.allclose(self.program.sample(0), self.ready, atol=1e-6):
            raise ValueError("The first trajectory waypoint must equal the configured ready pose")
        self.manifest = json.loads((ROOT / "assets/generated/manifest.json").read_text())
        self.task = TaskMonitor(config, self.manifest["fixture_geometry"])
        self.directory = sim_path(config["paths"]["recordings"])/(
            datetime.now().strftime("%Y%m%d_%H%M%S")+"_local_"+uuid.uuid4().hex[:6])
        self.recorder = None
        self.pending = []
        self.index = 0
        self.frames_saved = 0
        self.last_completed = None
        self.last_state = None
        self.reset_requested = False
        self.label = None
        self.last_b = 0.
        from pathlib import Path
        self.status_path = Path(status_path) if status_path is not None else ROOT / "data/local/status.json"
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.keyboard_subscription = None
        self.run_id = uuid.uuid4().hex
        self.program_origin_tick = 0
        atomic_json(self.status_path, {"state": "loading", "run_id": self.run_id, "gui": gui})

    def targets(self, tick):
        if self.action_source is not None and self.last_state is not None:
            target = np.asarray(self.action_source.step(self.last_state, 1/self.config["runtime"]["control_hz"]), dtype=float)
            if target.shape != (54,) or not np.isfinite(target).all():
                raise ValueError("Action source must return 54 finite absolute joint targets")
            if np.any(target < self.model.lower-1e-8) or np.any(target > self.model.upper+1e-8):
                raise ValueError("Action source exceeded joint limits")
            previous = np.asarray(self.last_state["q_target"])
            limits = np.r_[np.full(14, self.config["runtime"]["joint_speed_rad_s"]),
                           np.full(40, self.config["runtime"]["hand_speed_rad_s"])]
            if np.any(np.abs(target-previous) > limits/self.config["runtime"]["control_hz"]+1e-8):
                raise ValueError("Action source exceeded joint speed limits")
            return target
        return self.program.sample((tick-self.program_origin_tick)/self.config["runtime"]["physics_hz"]) if self.program else self.ready.copy()

    def on_reset(self, tick):
        self.program_origin_tick = tick
        self.task.reset()
        self.last_state = None
        if self.action_source is not None:
            if getattr(self.action_source, "reset_requested", False):
                self.capturing = self.capture_enabled
            self.action_source.reset()

    def message(self, text):
        if self.label is not None:
            self.label.text = text
        print(text, flush=True)

    def _start(self):
        from .recording import SimRecorder
        metadata = {"backend": "isaac", "input_source": "joint_trajectory" if self.program else "synthetic",
                    "test_data": self.program.test_data if self.program else True,
                    "local_action_source": "joint_trajectory" if self.program else "hold",
                    "config": copy.deepcopy(self.config), "assets": self.manifest,
                    "asset_signature": self.manifest.get("robot_signature", self.manifest["robot_urdf_sha256"]),
                    "joint_names": list(self.model.names), "clock": "simulation nanoseconds; no pause compression",
                    "eef_frames": "each shoulder joint-origin base -> flange; XYZW",
                    "wrench": "incoming fixed-joint reaction at force-sensor input (legacy: hand mount); local link frame, N and N.m; includes gravity/inertia and tool masses"}
        metadata["config"]["network"]["token"] = "[redacted]"
        if self.action_source is not None:
            metadata.update(input_source=self.action_source.name, local_action_source=self.action_source.name,
                            test_data=True)
            if hasattr(self.action_source, "recording_metadata"):
                metadata["action_source"] = self.action_source.recording_metadata()
        self.recorder = SimRecorder(self.directory/f"episode_{self.index:06d}", metadata,
                                   self.config["cameras"]["width"], self.config["cameras"]["height"], codec=self.codec)
        if self.action_source is not None and hasattr(self.action_source, "recording_started"):
            self.action_source.recording_started(self.recorder.path, self.recorder.document["source_demo_id"])
        self.index += 1
        self.message(f"Recording: {self.recorder.path.name}")

    def toggle_recording(self):
        if not self.capture_enabled:
            self.message("Restart with --record to enable RGB capture")
            return
        self.capturing = not self.capturing
        if self.recorder and self.last_state:
            if self.capturing:
                self.recorder.resume(self.last_state["tick"])
            else:
                self.recorder.pause(self.last_state["tick"], "operator")
                self.pending.clear()
        self.message("Recording resumed" if self.capturing else "Recording paused")

    def boundary(self):
        now = time.monotonic()
        if now-self.last_b < .75:
            return
        self.last_b = now
        if not self.capturing or not self.recorder:
            self.message("B rejected: recording is not active")
            return
        self.pending.append({"event_id": uuid.uuid4().hex, "client_press_ns": time.monotonic_ns(),
                             "server_receive_ns": time.monotonic_ns()})

    def save(self):
        self.capturing = False
        self.pending.clear()
        if self.recorder:
            recorder, self.recorder = self.recorder, None
            document = recorder.close()
            self.frames_saved += document["frames"]
            self.last_completed = {"path": str(recorder.path), "status": document["status"],
                                   "frames": document["frames"], "failure": document["failure"]}
            self.message(f"Saved: {document['status']} ({document['frames']} frames)")
            if document["status"] != "complete":
                raise RuntimeError(f"Recording failed: {document['failure']}")

    def request_reset(self):
        self.save()
        self.task.reset()
        self.reset_requested = True
        self.message("Reset requested; recording stopped")

    def build_ui(self, frame_scene, view_head_camera=None):
        import omni.ui as ui
        import carb.input
        import omni.appwindow
        self.window = ui.Window("Simulation", width=350 if self.action_source is not None else 330,
                                height=430 if self.action_source is not None else 265)
        self.window.deferred_dock_in("Property", ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)
        with self.window.frame, ui.ScrollingFrame():
            with ui.VStack(spacing=6, height=0):
                ui.Label("Local Isaac simulation | no external devices")
                self.label = ui.Label("Recording enabled" if self.capture_enabled else "View only | use --record for RGB")
                ui.Button("Record / Pause recording", clicked_fn=self.toggle_recording)
                ui.Button("B: unnamed segment marker", clicked_fn=self.boundary)
                ui.Button("Save and stop recording", clicked_fn=self.save)
                ui.Button("Reset scene", clicked_fn=self.request_reset)
                if view_head_camera is not None:
                    ui.Button("View head camera", clicked_fn=view_head_camera)
                ui.Button("Frame entire scene", clicked_fn=frame_scene)
                if self.action_source is not None:
                    self.action_source.build_ui(ui)
        keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        interface = carb.input.acquire_input_interface()

        def on_key(event, *_):
            if event.type == carb.input.KeyboardEventType.KEY_PRESS and event.input == carb.input.KeyboardInput.B:
                self.boundary()
            return True

        self.keyboard_interface, self.keyboard = interface, keyboard
        self.keyboard_callback = on_key
        self.keyboard_subscription = interface.subscribe_to_keyboard_events(keyboard, on_key)

    def accept(self, state, images=None):
        self.last_state = state
        state["task_metrics"] = self.task.update(state)
        if self.action_source is not None:
            self.action_source.observe(state, images)
        if self.capturing:
            if self.recorder is None:
                self._start()
            self.recorder.state(state)
            if images is not None:
                points = self.recorder.frame(state, images, self.pending)
                if points:
                    self.pending.clear()
                    self.message(f"B confirmed at recorded frame {points[-1]['frame_index']}")
            if self.recorder.failure:
                raise RuntimeError(self.recorder.failure)
        if state["tick"] % 60 == 0:
            self.write_status("running")

    def write_status(self, status, error=None):
        state = self.last_state or {}
        atomic_json(self.status_path, {"state": status, "run_id": self.run_id, "gui": self.gui,
                    "tick": state.get("tick"), "q": state.get("q"), "rail": state.get("rail"),
                    "action_source": self.action_source.name if self.action_source is not None else ("joint_trajectory" if self.program else "hold"),
                    "capturing": self.capturing, "recording": str(self.recorder.path) if self.recorder else None,
                    "frames": self.recorder.frames if self.recorder else 0,
                    "last_completed": self.last_completed, "frames_saved": self.frames_saved, "error": error})

    def close(self, error=None):
        try:
            if self.keyboard_subscription is not None:
                self.keyboard_interface.unsubscribe_to_keyboard_events(self.keyboard, self.keyboard_subscription)
            self.save()
        except Exception as close_error:
            error = error or f"{type(close_error).__name__}: {close_error}"
            raise
        finally:
            try:
                if self.action_source is not None:
                    self.action_source.close(error)
            finally:
                if getattr(self, "window", None) is not None:
                    self.window.destroy()
                    self.window = None
                self.write_status("failed" if error else "closed", error)
