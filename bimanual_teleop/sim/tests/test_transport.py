import copy
import json
import socket
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path

from bimanual_teleop.sim import ROOT
from bimanual_teleop.sim.client import Client, SyntheticInput
from bimanual_teleop.sim.config import load_config
from bimanual_teleop.sim.server import SimServer
from bimanual_teleop.sim.tests.engine_stub import run as stub_run


def free_ports(count):
    sockets = [socket.socket() for _ in range(count)]
    try:
        for item in sockets:
            item.bind(("127.0.0.1", 0))
        return [item.getsockname()[1] for item in sockets]
    finally:
        for item in sockets:
            item.close()


def wait_for(condition, timeout=5):
    end = time.monotonic()+timeout
    while time.monotonic() < end:
        if condition():
            return
        time.sleep(.02)
    raise AssertionError("Timed out waiting for transport test state")


@unittest.skipUnless((ROOT / "assets/generated/robot.urdf").exists(), "Prepare assets first")
class TransportTests(unittest.TestCase):
    def test_marker_retransmission_pause_and_save(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config()
            config["cameras"].update(width=32, height=24)
            for key, port in zip(("input_port", "event_port", "preview_port", "engine_port"), free_ports(4)):
                config["network"][key] = port
            config["paths"]["recordings"] = directory
            server = SimServer(config, codec="libx264")
            errors = []

            def host_run():
                try:
                    server.run()
                except BaseException as error:
                    errors.append(error)

            server_thread = threading.Thread(target=host_run, daemon=True); server_thread.start()
            stop = threading.Event()
            engine_thread = threading.Thread(target=stub_run, args=(config, stop), daemon=True); engine_thread.start()
            client = Client("127.0.0.1", config, SyntheticInput())
            try:
                client.synchronize()
                wait_for(lambda: server.state is not None and server.guard.latest is not None)
                result = client.event("toggle")
                self.assertTrue(result["ok"], result)
                wait_for(lambda: server.recorder is not None and server.recorder.frames >= 4)
                event = {"kind": "event", "event_id": uuid.uuid4().hex, "session": client.session,
                         "action": "boundary", "client_press_ns": time.monotonic_ns()}
                marked = client.request(event)
                self.assertTrue(marked["ok"], marked)
                repeated = client.request(event)
                self.assertEqual(marked["boundary"], repeated["boundary"])
                self.assertEqual(len(server.recorder.boundaries), 1)
                self.assertTrue(client.event("pause")["ok"])
                self.assertFalse(client.event("boundary")["ok"])
                self.assertTrue(client.event("toggle")["ok"])
                wait_for(lambda: server.recorder.frames >= 10)
                response = client.event("save")
                recorded = Path(response["recording"])
                wait_for(lambda: json.loads((recorded / "episode.json").read_text())["status"] == "complete")
                document = json.loads((recorded / "episode.json").read_text())
                self.assertEqual(document["metadata"]["backend"], "transport_test")
                self.assertEqual(document["metadata"]["input_source"], "synthetic")
                self.assertEqual(document["boundaries"], 1)
                self.assertTrue(server.engaged, "Saving must not block and expire control input")
                client.event("quit")
            finally:
                client.close(); stop.set(); server.running = False
                server_thread.join(timeout=10); engine_thread.join(timeout=3)
            self.assertFalse(server_thread.is_alive())
            self.assertFalse(errors, errors)


if __name__ == "__main__":
    unittest.main()
