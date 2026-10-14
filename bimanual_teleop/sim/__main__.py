"""One-command local Isaac desktop; the host only launches its container."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from . import ROOT


def container_path(value):
    path = Path(value).expanduser().resolve()
    project = ROOT.parents[1]
    try:
        return "/workspace/bimanual-teleop/" + path.relative_to(project).as_posix()
    except ValueError as error:
        raise ValueError("Keep simulation configuration and trajectory files inside the project") from error


def main():
    parser = argparse.ArgumentParser(description="Open the Isaac scene directly in your remote desktop")
    parser.add_argument("--headless", action="store_true", help="Run without a desktop window")
    parser.add_argument("--record", action="store_true", help="Enable three RGB cameras and record locally")
    parser.add_argument("--config")
    parser.add_argument("--trajectory", help="Reviewed JSON trajectory with 54 joint targets in radians")
    parser.add_argument("--steps", type=int, default=0, help="Stop after this many physics steps; zero runs until closed")
    parser.add_argument("--codec", choices=("h264_nvenc", "libx264"), default="h264_nvenc")
    parser.add_argument("--display", help="X11 desktop, normally inherited from the Remmina terminal")
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    if args.display:
        os.environ["DISPLAY"] = args.display
    if not args.headless and not os.environ.get("DISPLAY"):
        parser.error("Open a terminal inside the Remmina desktop; DISPLAY is missing (or use --display :10)")
    if not (ROOT / "assets/generated/robot.urdf").is_file():
        parser.error("Prepare models first: python -m bimanual_teleop.sim.assets")
    command = [str(ROOT / "docker/run-apptainer.sh"), "--standalone", "--steps", str(args.steps), "--codec", args.codec]
    if not args.headless:
        os.environ["BIMANUAL_SIM_GUI"] = "1"
        command.append("--gui")
    else:
        os.environ["BIMANUAL_SIM_GUI"] = "0"
    if args.record:
        command.append("--record")
    for flag, value in (("--config", args.config), ("--trajectory", args.trajectory)):
        if value:
            try:
                command += [flag, container_path(value)]
            except ValueError as error:
                parser.error(str(error))
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
