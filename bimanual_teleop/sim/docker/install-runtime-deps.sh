#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/exec-apptainer.sh" \
  /isaac-sim/python.sh -m pip install --no-deps --target /opt/sim-python \
  -r /workspace/bimanual-teleop/bimanual_teleop/sim/docker/requirements.txt
