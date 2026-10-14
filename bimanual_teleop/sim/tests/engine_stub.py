"""Explicitly nonphysical engine used to test transport and recording only."""

import time

from bimanual_teleop.sim._dependencies import np
from bimanual_teleop.sim._dependencies import zmq

from bimanual_teleop.sim.config import sim_path
from bimanual_teleop.sim.kinematics import RobotModel, pose_vector
from bimanual_teleop.sim.protocol import ImageRing, pack, unpack


def run(config, stop):
    context = zmq.Context(); socket = context.socket(zmq.DEALER)
    socket.setsockopt(zmq.IDENTITY, b"transport-test-engine"); socket.setsockopt(zmq.LINGER, 0)
    socket.connect(f"tcp://127.0.0.1:{config['network']['engine_port']}")
    ring = None
    try:
        socket.send(pack({"kind": "ready", "backend": "transport_test"}))
        if not socket.poll(5000):
            raise TimeoutError("Transport-test host did not initialize the ring")
        ring = ImageRing(**unpack(socket.recv())["ring"])
        model = RobotModel(sim_path(config["robot"]["urdf"]))
        q = np.r_[np.deg2rad(config["robot"]["ready_deg"]["left"]), np.deg2rad(config["robot"]["ready_deg"]["right"]), np.zeros(40)]
        goals = [pose_vector(model.flange(side, q)).tolist() for side in ("left", "right")]
        images = np.zeros(ring.shape, dtype=np.uint8)
        tick, command_tick = 0, 0
        start = time.monotonic()
        while not stop.is_set():
            while socket.poll(0):
                message = unpack(socket.recv())
                if message["kind"] == "stop":
                    return
                if message["kind"] == "target":
                    q = np.asarray(message["q"]); goals = message["eef_target"]; command_tick = tick
            tick += 1
            state = {"kind": "state", "tick": tick, "time_ns": round(tick*1e9/240), "q": q.tolist(),
                     "eef": [pose_vector(model.flange(side, q)).tolist() for side in ("left", "right")],
                     "eef_target": goals, "q_target": q.tolist(), "wrench": np.zeros((2, 6)).tolist(),
                     "rail": [0, 0, 0, 0, 0, 0, 1], "command_tick": command_tick, "command_id": "test"}
            if tick % 8 == 0:
                images[:] = (tick//8) % 255
                state["image_slot"] = ring.put(images)
            socket.send(pack(state))
            delay = start+tick/240-time.monotonic()
            if delay > 0:
                stop.wait(delay)
    finally:
        if ring:
            ring.close()
        socket.close(); context.term()
