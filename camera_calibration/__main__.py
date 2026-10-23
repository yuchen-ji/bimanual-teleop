"""Compatibility entry; the public script is camera_calibration/run.py."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
