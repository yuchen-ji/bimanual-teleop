#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/exec-apptainer.sh" \
  /isaac-sim/python.sh /workspace/bimanual-teleop/bimanual_teleop/sim/run.py engine "$@"
