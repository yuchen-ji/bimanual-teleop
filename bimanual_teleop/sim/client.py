"""Local Quest/glove/keyboard-HID client; never connects a robot actuator."""

from __future__ import annotations

import argparse
import copy
import json
import os
import select
import struct
import threading
import time
import uuid

from bimanual_teleop.sim._dependencies import zmq

from .config import load_config
from .protocol import pack, unpack


class OperatorDevices:
    def __init__(self, config):
        from bimanual_teleop.devices.quest.adapter import QuestSource
        from bimanual_teleop.devices.wuji.adapter import WujiGloveSource, WujiSdkSession
        from bimanual_teleop.control.hand.follow import WujiHandRetargeter
        from bimanual_teleop.control.arm.quest import QuestInputMonitor
        self.lock = threading.Lock()
        self.monitor = QuestInputMonitor(require_head=False)
        self.quest = QuestSource(config["operator"]["quest_serial"])
        self.sdk = WujiSdkSession(user_name=config["operator"]["sdk_user_name"])
        self.gloves = {}; self.retargeters = {}; self.last_refs = {}; self.last_hands = {"left": [0.]*20, "right": [0.]*20}
        try:
            self.sdk.open()
            for side in ("left", "right"):
                glove = WujiGloveSource(side, config["operator"]["gloves"][side], manager=self.sdk.manager, sdk=self.sdk.sdk,
                                       timeout_s=config["runtime"]["input_timeout_ms"]/1000)
                self.gloves[side] = glove; glove.start()
                self.retargeters[side] = WujiHandRetargeter(side, sdk=self.sdk.sdk)
            self.quest.start(self.monitor)
        except BaseException:
            self.close(); raise

    def acknowledge(self):
        with self.lock:
            self.monitor.current(time.monotonic_ns(), acknowledge=True)
            for side, glove in self.gloves.items():
                glove.clear_fault(); self.retargeters[side].reset()
                self.last_refs.pop(side, None)

    def frame(self):
        now = time.monotonic_ns()
        with self.lock:
            sample, _deadline, generation = self.monitor.current(now, check_latch=True)
            controllers = {side: list(getattr(sample.payload, side).position_m)+list(getattr(sample.payload, side).orientation_xyzw)
                           for side in ("left", "right")}
            received = [sample.header.received_monotonic_ns]
            references = {"quest": {"session": sample.payload.session, "sequence": sample.payload.sequence,
                                    "query_ns": sample.payload.query_monotonic_ns}}
            for side, glove in self.gloves.items():
                health = glove.health(check_latch=True)
                if not health.ready:
                    raise RuntimeError(f"{side} glove: {health.detail}")
                skeleton = glove.get_latest()
                if skeleton is None:
                    raise RuntimeError(f"Waiting for {side} glove skeleton")
                if self.last_refs.get(side) != skeleton.header.ref:
                    self.last_hands[side] = list(self.retargeters[side].solve(skeleton).position_rad)
                    self.last_refs[side] = skeleton.header.ref
                received.append(skeleton.header.received_monotonic_ns)
                references[side] = {"stream": skeleton.header.ref.stream, "epoch": skeleton.header.ref.epoch,
                                    "sequence": skeleton.header.ref.sequence}
            return {"sample_ns": min(received), "valid": True, "controllers": controllers,
                    "hands": copy.deepcopy(self.last_hands), "origin": f"{sample.payload.session}/{sample.payload.origin}",
                    "fault_generation": generation, "source_refs": references, "input_source": "quest_wuji"}

    def close(self):
        errors = []
        for device in [getattr(self, "quest", None), *getattr(self, "gloves", {}).values(), getattr(self, "sdk", None)]:
            if device is not None:
                try:
                    device.close()
                except Exception as error:
                    errors.append(str(error))
        if errors:
            print("Input device cleanup:", "; ".join(errors))


class SyntheticInput:
    """Software transport test input, marked and excluded from normal export."""
    def frame(self):
        return {"sample_ns": time.monotonic_ns(), "valid": True,
                "controllers": {side: [0, 0, 0, 0, 0, 0, 1] for side in ("left", "right")},
                "hands": {side: [0.]*20 for side in ("left", "right")},
                "origin": "transport-test", "fault_generation": 0, "input_source": "synthetic"}

    def acknowledge(self):
        pass

    def close(self):
        pass


class Pedal:
    # Linux input_event on this supported Linux x86_64 operator host.
    EVENT = struct.Struct("llHHI")
    KEYS = {28: "enter", 57: "space", 48: "b", 31: "s", 45: "x", 35: "h", 19: "r", 16: "q"}

    def __init__(self, path):
        self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        self.buffer = b""

    def read(self):
        if not select.select([self.fd], [], [], 0)[0]:
            return []
        self.buffer += os.read(self.fd, 4096)
        keys = []
        while len(self.buffer) >= self.EVENT.size:
            _seconds, _microseconds, kind, code, value = self.EVENT.unpack(self.buffer[:self.EVENT.size])
            self.buffer = self.buffer[self.EVENT.size:]
            if kind == 1 and value == 1 and code in self.KEYS:
                keys.append(self.KEYS[code])
        return keys

    def close(self):
        os.close(self.fd)


class Client:
    def __init__(self, host, config, devices=None):
        self.config, self.host = config, host
        self.session = uuid.uuid4().hex
        self.context = zmq.Context(); self.stop = threading.Event()
        self.devices = devices if devices is not None else OperatorDevices(config)
        self.last_problem = None
        self.requests = self.context.socket(zmq.REQ)
        self.requests.setsockopt(zmq.LINGER, 0)
        self.requests.connect(f"tcp://{host}:{config['network']['event_port']}")
        self.thread = threading.Thread(target=self._publish, name="sim-operator", daemon=True)
        self.thread.start()

    def request(self, message, timeout_ms=2000):
        payload = pack({**message, "token": self.config["network"]["token"]})
        for _ in range(2):
            self.requests.send(payload)
            if self.requests.poll(timeout_ms):
                return unpack(self.requests.recv())
            self.requests.close()
            self.requests = self.context.socket(zmq.REQ); self.requests.setsockopt(zmq.LINGER, 0)
            self.requests.connect(f"tcp://{self.host}:{self.config['network']['event_port']}")
        raise TimeoutError("Simulation host did not acknowledge the request")

    def synchronize(self):
        exchanges = []
        for _ in range(3):
            sent = time.monotonic_ns()
            reply = self.request({"kind": "clock_ping", "client_send_ns": sent})
            received = time.monotonic_ns()
            times = [sent, reply["server_receive_ns"], reply["server_send_ns"], received]
            delay = (received-sent)-(times[2]-times[1])
            exchanges.append((delay, times))
        result = self.request({"kind": "clock_update", "timestamps": min(exchanges, key=lambda item: item[0])[1]})
        if not result["ok"]:
            raise RuntimeError(result["error"])

    def event(self, action):
        if action == "toggle":
            try:
                self.devices.acknowledge()
                # Let the independently publishing thread expose the acknowledged fresh sample.
                self.stop.wait(.04)
            except RuntimeError as error:
                print(str(error))
        event = {"kind": "event", "event_id": uuid.uuid4().hex, "session": self.session,
                 "action": action, "client_press_ns": time.monotonic_ns()}
        return self.request(event)

    def _publish(self):
        socket = self.context.socket(zmq.PUSH)
        socket.setsockopt(zmq.LINGER, 0); socket.setsockopt(zmq.SNDHWM, 1)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.connect(f"tcp://{self.host}:{self.config['network']['input_port']}")
        sequence = 0
        faults = 0; failing = False
        last = SyntheticInput().frame(); last["valid"] = False
        try:
            while not self.stop.is_set():
                started = time.monotonic()
                try:
                    frame = self.devices.frame()
                    self.last_problem = None
                    last = copy.deepcopy(frame)
                except Exception as error:
                    self.last_problem = str(error)
                    frame = {**last, "valid": False, "problem": str(error)}
                if not frame["valid"] and not failing:
                    faults += 1
                failing = not frame["valid"]
                frame["fault_generation"] = f"{frame.get('fault_generation', 0)}/{faults}"
                sequence += 1
                message = {"kind": "input", "session": self.session, "sequence": sequence,
                           "send_ns": time.monotonic_ns(), "token": self.config["network"]["token"], **frame}
                try:
                    socket.send(pack(message), zmq.NOBLOCK)
                except zmq.Again:
                    pass
                self.stop.wait(max(0, started+1/self.config["runtime"]["control_hz"]-time.monotonic()))
        finally:
            socket.close()

    def close(self):
        self.stop.set(); self.thread.join(timeout=3)
        self.requests.close(); self.context.term(); self.devices.close()


def main():
    from bimanual_teleop.common.terminal import NonblockingTerminal
    parser = argparse.ArgumentParser(description="Local input client for the simulation server")
    parser.add_argument("--host", required=True)
    parser.add_argument("--config")
    parser.add_argument("--user-name")
    parser.add_argument("--serial")
    parser.add_argument("--synthetic", action="store_true", help="Transport tests only; not human demonstration data")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.user_name is not None:
        config["operator"]["sdk_user_name"] = args.user_name
    if args.serial is not None:
        config["operator"]["quest_serial"] = args.serial
    client = Client(args.host, config, SyntheticInput() if args.synthetic else None)
    pedal = Pedal(config["operator"]["pedal_device"]) if config["operator"]["pedal_device"] else None
    print(f"Preview http://{args.host}:{config['network']['preview_port']} | Enter engage/pause, B boundary, S save, X discard, H home, R reset, Q quit")
    last_sync, last_b = 0., 0.
    try:
        client.synchronize()
        with NonblockingTerminal() as keyboard:
            while True:
                if time.monotonic()-last_sync >= 1:
                    client.synchronize(); last_sync = time.monotonic()
                text = keyboard.read(.02)
                keys = [{"\r": "enter", "\n": "enter", " ": "space", "\x03": "q", "\x04": "q"}.get(key, key.lower()) for key in (text or "")]
                if pedal:
                    keys.extend(pedal.read())
                for key in keys:
                    if key not in config["operator"]["keys"]:
                        continue
                    if key == "b":
                        now = time.monotonic()
                        repeated = now-last_b < .75; last_b = now
                        if repeated:
                            continue
                    action = config["operator"]["keys"][key]
                    result = client.event(action)
                    print(json.dumps(result, ensure_ascii=False))
                    if action == "quit":
                        return
    finally:
        if pedal:
            pedal.close()
        client.close()


if __name__ == "__main__":
    main()
