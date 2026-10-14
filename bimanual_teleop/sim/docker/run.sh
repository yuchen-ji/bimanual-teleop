#!/usr/bin/env bash
set -euo pipefail
sim_root=$(cd "$(dirname "$0")/.." && pwd)
project_root=$(cd "$sim_root/../.." && pwd)
mkdir -p "$sim_root/.runtime/docker-cache"
exec docker run --rm --gpus all --network host --ipc host \
  -e ACCEPT_EULA=Y -e PYTHONNOUSERSITE=1 \
  -v "$project_root:/workspace/bimanual-teleop:ro" \
  -v "$sim_root/.runtime/docker-cache:/isaac-sim/.cache:rw" \
  bimanual-sim:5.1.0 "$@"
