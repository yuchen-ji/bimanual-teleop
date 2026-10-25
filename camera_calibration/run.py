"""Run camera calibration directly: python camera_calibration/run.py --help."""

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera_calibration.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
