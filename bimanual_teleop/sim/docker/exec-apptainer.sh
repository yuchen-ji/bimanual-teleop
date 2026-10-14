#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
sim_root=$(cd "$(dirname "$0")/.." && pwd)
project_root=$(cd "$sim_root/../.." && pwd)
runtime_root="$sim_root/.runtime"
mkdir -p "$runtime_root/container-home" "$runtime_root/apptainer-cache" "$runtime_root/apptainer-tmp" \
  "$runtime_root/tmp" "$runtime_root/pip-cache" "$sim_root/data"
export APPTAINER_CACHEDIR="$runtime_root/apptainer-cache"
export APPTAINER_TMPDIR="$runtime_root/apptainer-tmp"
export APPTAINERENV_ACCEPT_EULA=Y
export APPTAINERENV_PYTHONPATH="/workspace/bimanual-teleop:/opt/sim-python"
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_BIMANUAL_SIM_SHM_DIRECTORY=/sim-shm
export APPTAINERENV_TMPDIR=/sim-runtime/tmp
export APPTAINERENV_PIP_CACHE_DIR=/sim-runtime/pip-cache
if [[ ! -f "$runtime_root/isaac-overlay.img" ]]; then
  "$runtime_root/apptainer/bin/apptainer" overlay create --size 4096 "$runtime_root/isaac-overlay.img"
fi
display_binds=()
if [[ "${BIMANUAL_SIM_GUI:-0}" == 1 ]]; then
  : "${DISPLAY:?Run from the Remmina desktop terminal, or pass --display to the simulation launcher}"
  display_auth=${XAUTHORITY:-$HOME/.Xauthority}
  if [[ ! -r "$display_auth" ]]; then
    echo "Cannot read X11 authorization: $display_auth" >&2
    exit 1
  fi
  export APPTAINERENV_DISPLAY="$DISPLAY"
  export APPTAINERENV_XAUTHORITY=/tmp/bimanual-xauthority
  display_binds=(--bind /tmp/.X11-unix:/tmp/.X11-unix \
    --bind "$display_auth:/tmp/bimanual-xauthority:ro")
fi
exec "$runtime_root/apptainer/bin/apptainer" exec --nv --overlay "$runtime_root/isaac-overlay.img" --contain \
  --pwd /isaac-sim \
  --home "$runtime_root/container-home:/home/sim" \
  --bind "$project_root:/workspace/bimanual-teleop:ro" \
  --bind "$runtime_root:/sim-runtime:rw" \
  --bind "$sim_root/data:/workspace/bimanual-teleop/bimanual_teleop/sim/data:rw" \
  --bind /dev/shm:/sim-shm \
  "${display_binds[@]}" \
  "$runtime_root/isaac-sim-5.1.0.sif" "$@"
