#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/exec-apptainer.sh" \
  /isaac-sim/python.sh -m bimanual_teleop.sim.engine "$@"
