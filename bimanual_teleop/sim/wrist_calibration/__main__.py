"""Compatibility for the previously distributed wrist sampling command."""

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

def main(argv=None):
    from camera_calibration.cli import main as calibrate

    arguments = list(sys.argv[1:] if argv is None else argv)
    return calibrate(arguments if arguments[:1] == ["detect"] else ["wrist", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
