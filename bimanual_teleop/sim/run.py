"""Direct script entry for the simulation launcher and its offline/container tools."""

import argparse
import importlib
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


TOOLS = (
    "run",
    "assets",
    "cad_assets",
    "resources",
    "inspect_assets",
    "doctor",
    "server",
    "client",
    "engine",
    "smoke",
    "segments",
    "export",
    "validate_isaac",
    "validate_head_camera",
    "validate_wrist_camera",
    "validate_appearance",
)


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Omit 'run' for normal launch. Use 'run --help' for launch options.",
    )
    parser.add_argument("command", nargs="?", choices=TOOLS, default="run")
    if arguments in (["--help"], ["-h"]):
        parser.print_help()
        return 0
    command = (
        arguments.pop(0) if arguments and not arguments[0].startswith("-") else "run"
    )
    if command not in TOOLS:
        parser.error(f"Unknown command: {command}")
    module = "__main__" if command == "run" else command
    original = sys.argv
    sys.argv = [f"{__file__} {command}", *arguments]
    try:
        return importlib.import_module(f"bimanual_teleop.sim.{module}").main()
    finally:
        sys.argv = original


if __name__ == "__main__":
    raise SystemExit(main())
