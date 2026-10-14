"""Host bridge: input validation, IK, unnamed markers and recording.

The engine is a separate process/container. This module does not create a real
arm, Hand2 or camera driver and never submits commands to robot hardware.
"""

from __future__ import annotations

import argparse
import copy
import json
import signal
import threading
import time
import uuid
from datetime import datetime

from bimanual_teleop.sim._dependencies import np
from bimanual_teleop.sim._dependencies import zmq
from bimanual_teleop.sim._dependencies import Rotation

from . import ROOT
from .config import load_config, sim_path
from .kinematics import RobotModel, pose_matrix, pose_vector
from .preview import Preview
from .protocol import ClockEstimate, ImageRing, InputGuard, pack, unpack
from .recording import SimRecorder
from .task import TaskMonitor


class SimServer:
    def __init__(self, config, codec="h264_nvenc"):
        # Load codec/storage modules before an operator can engage; their first
        # imports can otherwise hold the GIL past the input freshness deadline.
        import av
        import zarr
        import numcodecs
        from PIL import Image
        self.config, self.codec = config, codec
        from fractions import Fraction
        encoder = av.CodecContext.create(codec, "w")
        encoder.width, encoder.height = config["cameras"]["width"], config["cameras"]["height"]
        encoder.pix_fmt = "yuv420p"; encoder.time_base = Fraction(1, 30)
        encoder.open()
        blank = av.VideoFrame(encoder.width, encoder.height, "yuv420p")
        for plane in blank.planes:
            plane.update(bytes(plane.buffer_size))
        encoder.encode(blank); encoder.encode(None)
        del encoder
        self.model = RobotModel(sim_path(config["robot"]["urdf"]), config["robot"]["world_yaw_deg"])
        self.context = zmq.Context()
        self.input = self.context.socket(zmq.PULL); self.input.setsockopt(zmq.RCVHWM, 1)
        self.input.setsockopt(zmq.CONFLATE, 1)
        self.events = self.context.socket(zmq.REP)
        self.engine = self.context.socket(zmq.ROUTER)
        for socket in (self.input, self.events, self.engine):
            socket.setsockopt(zmq.LINGER, 0)
        network = config["network"]
        self.input.bind(f"tcp://{network['bind']}:{network['input_port']}")
        self.events.bind(f"tcp://{network['bind']}:{network['event_port']}")
        self.engine.bind(f"tcp://127.0.0.1:{network['engine_port']}")
        self.poller = zmq.Poller()
        for socket in (self.input, self.events, self.engine):
            self.poller.register(socket, zmq.POLLIN)
        self.preview = Preview(network["bind"], network["preview_port"])
        self.ring = ImageRing.create(config["runtime"]["frame_capacity"], config["cameras"]["width"], config["cameras"]["height"])
        self.guard = InputGuard(config["runtime"]["input_timeout_ms"]*1_000_000,
                                config["runtime"]["max_clock_uncertainty_ms"]*1_000_000)
        self.state = None; self.engaged = False; self.running = True
        self.engine_identity = None; self.engine_backend = "isaac"
        self.recorder = None; self.pending_boundary = None
        self.event_cache = {}; self.anchor = None
        self.joint_targets = None
        self.last_engine_ns = 0; self.last_preview_ns = 0
        self.command_sequence = 0
        self.recording_session = datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:6]
        self.recording_index = 0
        self.latest_boundary = None
        self.finish_threads = []
        self.last_heartbeat_ns = 0
        geometry = json.loads((ROOT / "assets/generated/manifest.json").read_text())["fixture_geometry"]
        self.task_monitor = TaskMonitor(config, geometry)

    def _finish_recording(self, discard=False):
        recorder = self.recorder
        self.recorder = None
        recorder.begin_close(discard)

        def finish():
            try:
                document = recorder.close(discard)
                self.preview.update(last_completed={"path": str(recorder.path), "status": document["status"], "failure": document["failure"]})
            except Exception as error:
                self.preview.update(recording_error=str(error))

        worker = threading.Thread(target=finish, name="sim-finalize", daemon=True)
        self.finish_threads.append(worker); worker.start()
        return str(recorder.path)

    def _send_engine(self, message):
        if self.engine_identity:
            self.engine.send_multipart([self.engine_identity, pack(message)])

    def _reply(self, message, event_id=None):
        if event_id:
            self.event_cache[event_id] = message
        self.events.send(pack(message))

    def _pause(self, reason):
        self.engaged = False
        if self.state:
            self._send_engine({"kind": "target", "q": self.state["q"], "eef_target": self.state["eef"],
                               "expires_ns": 0, "command_id": "pause_hold"})
        if self.recorder:
            self.recorder.pause(self.state["tick"] if self.state else 0, reason)
        if self.pending_boundary:
            event = self.pending_boundary; self.pending_boundary = None
            self._reply({"kind": "result", "ok": False, "error": "Boundary rejected: recording paused", "event_id": event["event_id"]}, event["event_id"])
        self.preview.update(state="paused", reason=reason)

    def _start_recording(self):
        if self.recorder:
            self.recorder.resume(self.state["tick"]); return
        path = sim_path(self.config["paths"]["recordings"]) / self.recording_session / f"episode_{self.recording_index:06d}"
        self.recording_index += 1
        manifest = json.loads((ROOT / "assets/generated/manifest.json").read_text())
        metadata = {"backend": self.engine_backend, "input_source": self.guard.latest.get("input_source", "unknown"),
                    "config": copy.deepcopy(self.config),
                    "assets": manifest, "asset_signature": manifest.get("robot_signature", manifest["robot_urdf_sha256"]),
                    "joint_names": list(self.model.names), "clock": "simulation nanoseconds; no pause compression",
                    "eef_frames": "each shoulder joint-origin base -> flange; XYZW",
                    "wrench": "incoming fixed-joint reaction at force-sensor input (legacy: hand mount); local link frame, N and N.m; includes gravity/inertia and tool masses"}
        metadata["config"]["network"]["token"] = "[redacted]"
        self.recorder = SimRecorder(path, metadata, self.config["cameras"]["width"], self.config["cameras"]["height"], codec=self.codec)
        self.latest_boundary = None
        self.preview.update(recording=str(path), boundary=None)

    def _engage(self):
        if self.state is None or not self.guard.reengage():
            return {"kind": "result", "ok": False, "error": "Need a ready engine, synchronized clock and fresh valid input"}
        self.anchor = {"input": copy.deepcopy(self.guard.latest),
                       "flanges": {side: self.model.flange(side, self.state["q"], world=True) for side in ("left", "right")}}
        self.joint_targets = np.asarray(self.state["q"]).copy()
        self.engaged = True; self._start_recording()
        self.preview.update(state="engaged", reason=None)
        return {"kind": "result", "ok": True, "state": "engaged"}

    def _event(self, message, receive_ns):
        if message.get("token", "") != self.config["network"]["token"]:
            self._reply({"kind": "result", "ok": False, "error": "Unauthorized operator"}); return
        kind = message.get("kind")
        if kind == "clock_ping":
            self._reply({"kind": "clock_pong", "client_send_ns": message["client_send_ns"],
                         "server_receive_ns": receive_ns, "server_send_ns": time.monotonic_ns()}); return
        if kind == "clock_update":
            # Values come from a measured four-timestamp exchange, not arbitrary receiver-time replacement.
            estimate = ClockEstimate.from_exchange(*message["timestamps"])
            self.guard.update_clock(estimate)
            self._reply({"kind": "result", "ok": True}); return
        if kind == "status":
            self._reply({"kind": "result", "ok": True, "engaged": self.engaged, "ready": self.state is not None,
                         "boundary": self.latest_boundary, "input_fault": self.guard.fault,
                         "recording": str(self.recorder.path) if self.recorder else None}); return
        event_id = message.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("Operator events need a unique event_id")
        if event_id in self.event_cache:
            self._reply(self.event_cache[event_id]); return
        if message.get("session") != self.guard.session:
            self._reply({"kind": "result", "ok": False, "error": "Operator event belongs to another input session"}, event_id); return
        action = message.get("action")
        if action == "boundary":
            if not self.engaged or self.recorder is None or self.recorder.failure:
                self._reply({"kind": "result", "ok": False, "error": "Boundary requires an actively recording demonstration"}, event_id)
            else:
                self.pending_boundary = {"event_id": event_id, "client_press_ns": message["client_press_ns"],
                                         "server_receive_ns": receive_ns}
            return
        if action == "toggle":
            if self.engaged:
                self._pause("operator"); result = {"kind": "result", "ok": True, "state": "paused"}
            else:
                result = self._engage()
        elif action == "pause":
            self._pause("operator"); result = {"kind": "result", "ok": True, "state": "paused"}
        elif action in ("save", "discard", "reset", "quit"):
            if self.recorder:
                path = self._finish_recording(discard=action == "discard")
                result = {"kind": "result", "ok": True, "recording": path, "status": "discarding" if action == "discard" else "saving"}
            else:
                result = {"kind": "result", "ok": True, "status": "no_recording"}
            if action == "reset":
                self._pause("reset"); self.task_monitor.reset(); self._send_engine({"kind": "reset"})
            elif action == "quit":
                self._pause("quit"); self.running = False
            elif self.engaged:
                self._start_recording()
        elif action == "home":
            if self.state is None:
                raise ValueError("Simulation engine is not ready")
            self._pause("home")
            q = np.r_[np.deg2rad(self.config["robot"]["ready_deg"]["left"]), np.deg2rad(self.config["robot"]["ready_deg"]["right"]), self.state["q"][14:]]
            self._send_engine({"kind": "target", "q": q.tolist(), "eef_target": [pose_vector(self.model.flange(side, q)).tolist() for side in ("left", "right")],
                               "expires_ns": 0, "command_id": "home"})
            result = {"kind": "result", "ok": True, "state": "paused"}
        else:
            raise ValueError("Unknown simulation operation")
        self._reply(result, event_id)

    def _targets(self):
        q = np.asarray(self.state["q"])
        # Integrate commanded positions, while Jacobians use measured state.
        # Restarting from actual q every tick defeats position holding against
        # gravity. Bound accumulated tracking error during physical obstruction.
        joint_reference = np.clip(self.joint_targets, q-.25, q+.25)
        target = joint_reference.copy()
        eef_goals = []
        for side in ("left", "right"):
            controller = ("right" if side == "left" else "left") if self.config["robot"]["crossed_controllers"] else side
            controller_reference = pose_matrix(self.anchor["input"]["controllers"][controller])
            current = pose_matrix(self.guard.latest["controllers"][controller])
            robot_reference = self.anchor["flanges"][side]
            goal = robot_reference.copy()
            goal[:3, 3] += current[:3, 3]-controller_reference[:3, 3]
            goal[:3, :3] = current[:3, :3] @ controller_reference[:3, :3].T @ robot_reference[:3, :3]
            offset = 0 if side == "left" else 7
            target[offset:offset+7] = self.model.velocity_ik(side, q, goal, 1/self.config["runtime"]["control_hz"], self.config["runtime"], joint_reference)
            eef_goals.append(pose_vector(np.linalg.inv(self.model.base_pose(side)) @ goal).tolist())
            hand_offset = 14 if side == "left" else 34
            selected = slice(hand_offset, hand_offset+20)
            hand = np.clip(self.guard.latest["hands"][side], self.model.lower[selected], self.model.upper[selected])
            maximum = self.config["runtime"]["hand_speed_rad_s"]/self.config["runtime"]["control_hz"]
            target[selected] += np.clip(hand-joint_reference[selected], -maximum, maximum)
        self.joint_targets = target.copy()
        self.command_sequence += 1
        return {"kind": "target", "q": target.tolist(), "eef_target": eef_goals,
                "expires_ns": time.monotonic_ns()+self.config["runtime"]["command_timeout_ms"]*1_000_000,
                "command_id": f"teleop_{self.command_sequence}"}

    def _engine_message(self, identity, message):
        if self.engine_identity and identity != self.engine_identity:
            raise ValueError("Only one simulation engine may connect")
        if message["kind"] == "ready":
            self.engine_identity = identity; self.engine_backend = message.get("backend", "isaac")
            self._send_engine({"kind": "initialize", "ring": self.ring.descriptor()})
            self.preview.update(state="ready"); return
        if message["kind"] == "error":
            self._pause(message["error"]); self.running = False
            raise RuntimeError(message["error"])
        if message["kind"] != "state":
            raise ValueError("Unknown simulation engine message")
        self.state = message; self.last_engine_ns = time.monotonic_ns()
        self.preview.update_pose(message)
        self._inputs()
        message["task_metrics"] = self.task_monitor.update(message)
        if self.engaged and not self.guard.healthy():
            self._pause(self.guard.fault or "operator unavailable")
        if self.engaged and message["tick"] % (self.config["runtime"]["physics_hz"]//self.config["runtime"]["control_hz"]) == 0:
            self._send_engine(self._targets())
        if self.recorder and self.engaged:
            self.recorder.state(message)
        if "image_slot" in message:
            images = self.ring.take(message["image_slot"])
            if self.recorder and self.engaged:
                events = [self.pending_boundary] if self.pending_boundary else []
                points = self.recorder.frame(message, images, events)
                if self.pending_boundary and points:
                    self.latest_boundary = points[0]; self.pending_boundary = None
                    self._reply({"kind": "result", "ok": True, "boundary": self.latest_boundary}, self.latest_boundary["event_id"])
            if self.last_engine_ns-self.last_preview_ns >= 1e9/self.config["runtime"]["preview_hz"]:
                self.preview.update(images, tick=message["tick"], boundary=self.latest_boundary, task=message["task_metrics"])
                self.last_preview_ns = self.last_engine_ns
        if self.recorder and self.recorder.failure:
            failure = self.recorder.failure
            self._finish_recording()
            self.preview.update(recording_error=failure)
            if self.pending_boundary:
                event = self.pending_boundary; self.pending_boundary = None
                self._reply({"kind": "result", "ok": False, "error": failure}, event["event_id"])

    def run(self):
        try:
            while self.running:
                polled = dict(self.poller.poll(10))
                if self.input in polled:
                    self._inputs()
                if self.events in polled and self.pending_boundary is None:
                    try:
                        self._event(unpack(self.events.recv()), time.monotonic_ns())
                    except (ValueError, KeyError, TypeError) as error:
                        self._reply({"kind": "result", "ok": False, "error": str(error)})
                if self.engine in polled:
                    for _ in range(32):
                        try:
                            identity, payload = self.engine.recv_multipart(zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        self._engine_message(identity, unpack(payload))
                # Source expiry remains 100 ms even while the engine renders
                # a frozen state. A missed renderer deadline is separate.
                self._inputs()
                if self.engaged and not self.guard.healthy():
                    self._pause(self.guard.fault or "operator unavailable")
                if self.engaged and self.last_engine_ns and time.monotonic_ns()-self.last_engine_ns > self.config["runtime"]["engine_feedback_timeout_ms"]*1_000_000:
                    self._pause("Simulation engine feedback timeout")
                if self.engine_identity and time.monotonic_ns()-self.last_heartbeat_ns >= 250_000_000:
                    self._send_engine({"kind": "heartbeat"}); self.last_heartbeat_ns = time.monotonic_ns()
        finally:
            self._send_engine({"kind": "stop"})
            if self.recorder:
                self.recorder.close()
            for worker in self.finish_threads:
                worker.join(timeout=30)
            self.preview.close()
            self.input.close(); self.events.close(); self.engine.close(); self.context.term()
            self.ring.close(owner=True)

    def _inputs(self):
        latest = None
        while True:
            try:
                latest = unpack(self.input.recv(zmq.NOBLOCK))
            except zmq.Again:
                break
        if latest is not None and latest.get("token", "") == self.config["network"]["token"]:
            try:
                self.guard.receive(latest)
            except (ValueError, KeyError, TypeError) as error:
                self.guard.fault = str(error)


def main():
    parser = argparse.ArgumentParser(description="Start the simulation-only host bridge")
    parser.add_argument("--config")
    parser.add_argument("--codec", choices=("h264_nvenc", "libx264"), default="h264_nvenc")
    args = parser.parse_args()
    server = SimServer(load_config(args.config), args.codec)
    signal.signal(signal.SIGINT, lambda *_: setattr(server, "running", False))
    signal.signal(signal.SIGTERM, lambda *_: setattr(server, "running", False))
    print(f"Simulation host listening; preview http://{server.config['network']['bind']}:{server.config['network']['preview_port']}")
    server.run()


if __name__ == "__main__":
    main()
