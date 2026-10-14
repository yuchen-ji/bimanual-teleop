#!/usr/bin/env bash
set -euo pipefail
sim_root=$(cd "$(dirname "$0")/.." && pwd)
runtime_root="$sim_root/.runtime"
mkdir -p "$runtime_root/apptainer-cache" "$runtime_root/apptainer-tmp"
if [[ ! -x "$runtime_root/apptainer/bin/apptainer" ]]; then
  export PATH="/opt/miniconda3/bin:$PATH"
  curl -fsSL https://raw.githubusercontent.com/apptainer/apptainer/v1.5.4/tools/install-unprivileged.sh \
    -o "$runtime_root/install-apptainer.sh"
  if command -v rpm2cpio >/dev/null; then
    bash "$runtime_root/install-apptainer.sh" -e -d el8 -v 1.5.4 "$runtime_root/apptainer"
  else
    command -v bsdtar >/dev/null
    sed -e 's/REQUIREDCMDS="curl rpm2cpio cpio"/REQUIREDCMDS="curl bsdtar"/' \
        -e 's/|rpm2cpio -|cpio -idum/|bsdtar -xf -/' \
        -e 's/rpm2cpio "$VERSION"|cpio -idum/bsdtar -xf "$VERSION"/' \
        "$runtime_root/install-apptainer.sh" > "$runtime_root/install-apptainer-local.sh"
    bash "$runtime_root/install-apptainer-local.sh" -e -d el8 -v 1.5.4 "$runtime_root/apptainer"
  fi
fi
export APPTAINER_CACHEDIR="$runtime_root/apptainer-cache"
export APPTAINER_TMPDIR="$runtime_root/apptainer-tmp"
export APPTAINER_MKSQUASHFS_PROCS=4
if [[ ! -f "$runtime_root/isaac-sim-5.1.0.sif" ]]; then
  # NVCR is reachable directly on this server; preserve the user's global proxy.
  HTTPS_PROXY= HTTP_PROXY= ALL_PROXY= https_proxy= http_proxy= all_proxy= \
    "$runtime_root/apptainer/bin/apptainer" pull "$runtime_root/isaac-sim-5.1.0.sif" \
    docker://nvcr.io/nvidia/isaac-sim:5.1.0
fi
"$sim_root/docker/install-runtime-deps.sh"
