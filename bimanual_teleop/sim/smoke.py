"""Exercise the real container bridge and recorder with excluded test inputs."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path

from . import ROOT
from ._dependencies import yaml
from .client import Client, SyntheticInput
from .config import load_config
from .recording import atomic_json
from .server import SimServer


def _ports():
    sockets = [socket.socket() for _ in range(4)]
    try:
        for item in sockets:
            item.bind(("127.0.0.1", 0))
        return [item.getsockname()[1] for item in sockets]
    finally:
        for item in sockets:
            item.close()


def run(codec="h264_nvenc"):
    output = ROOT / "data" / ("isaac-smoke-"+uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    config = load_config()
    for name, port in zip(("input_port", "event_port", "preview_port", "engine_port"), _ports()):
        config["network"][name] = port
    config["paths"]["recordings"] = str(output / "recordings")
    config_path = output / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    container_path = "/workspace/bimanual-teleop/bimanual_teleop/sim/"+str(config_path.relative_to(ROOT))
    server = SimServer(config, codec)
    errors = []

    def serve():
        try:
            server.run()
        except BaseException as error:
            errors.append(str(error))

    thread = threading.Thread(target=serve, daemon=True); thread.start()
    client = Client("127.0.0.1", config, SyntheticInput())
    report = {"passed": False, "learning_data": False, "backend": "isaac", "codec": codec}
    engine = None
    try:
        with (output / "engine.log").open("w") as log:
            engine = subprocess.Popen([str(ROOT / "docker/run-apptainer.sh"), "--config", container_path,
                                       "--endpoint", f"tcp://127.0.0.1:{config['network']['engine_port']}"],
                                      stdout=log, stderr=subprocess.STDOUT)

            def wait_for(condition, timeout=30):
                deadline = time.monotonic()+timeout
                while time.monotonic() < deadline:
                    if condition():
                        return
                    if errors or engine.poll() is not None:
                        raise RuntimeError(f"Engine/host stopped: {errors}; see {output / 'engine.log'}")
                    client.synchronize()
                    time.sleep(.05)
                raise TimeoutError("Real engine smoke check timed out")

            client.synchronize()
            wait_for(lambda: server.state is not None and server.guard.latest is not None, 240)
            assert client.event("toggle")["ok"]
            wait_for(lambda: server.recorder is not None and server.recorder.frames >= 30)
            marked = client.event("boundary")
            assert marked["ok"], marked
            wait_for(lambda: server.recorder is not None and server.recorder.frames >= 60)
            assert server.engaged, "Feedback/control deadline failed during capture"
            assert client.event("pause")["ok"]
            assert not client.event("boundary")["ok"]
            saved = client.event("save")
            path = Path(saved["recording"])
            wait_for(lambda: json.loads((path / "episode.json").read_text())["status"] in ("complete", "failed"))
            document = json.loads((path / "episode.json").read_text())
            assert document["status"] == "complete", document
            assert document["frames"] >= 60 and document["boundaries"] == 1
            assert document["metadata"]["backend"] == "isaac"
            assert document["metadata"]["input_source"] == "synthetic"
            report.update(passed=True, episode=str(path), frames=document["frames"], states=document["states"],
                          boundary=marked["boundary"], source_demo_id=document["source_demo_id"])
            client.event("quit")
            from .segments import make_scheme
            from .export import export
            scheme = make_scheme(output / "recordings")
            for episode in scheme["episodes"]:
                for index, part in enumerate(episode["ranges"]):
                    part["task"] = f"transport_part_{index}"
            scheme_path = output / "test-scheme.yaml"
            scheme_path.write_text(yaml.safe_dump(scheme, sort_keys=False))
            report["export_validation"] = export(output / "recordings", output / "export-both", "both",
                                                   scheme_path, allow_test_data=True)
    except BaseException as error:
        report["passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        report["host_errors"] = errors
        report["input_fault"] = server.guard.fault
        report["engaged"] = server.engaged
        report["preview"] = dict(server.preview.state)
        raise
    finally:
        client.close(); server.running = False; thread.join(timeout=30)
        if engine is not None:
            try:
                engine.wait(timeout=20)
            except subprocess.TimeoutExpired:
                engine.terminate()
        atomic_json(output / "report.json", report)
        print(output / "report.json")
    return report


def main():
    parser = argparse.ArgumentParser(description="Validate the real engine, shared memory and recording")
    parser.add_argument("--codec", choices=("h264_nvenc", "libx264"), default="h264_nvenc")
    args = parser.parse_args()
    print(json.dumps(run(args.codec), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
