"""Compatibility for building the relocated portable sampler."""

import argparse
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

def build(output):
    from camera_calibration.bundle import build as build_sampler

    return build_sampler(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(build(parser.parse_args().output))
