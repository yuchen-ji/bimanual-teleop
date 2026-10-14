#!/usr/bin/env bash
# Offline model preparation only; the Isaac runtime does not need this reader.
set -euo pipefail
sim_root=$(cd "$(dirname "$0")/.." && pwd)
python -c 'import pathlib,sys; assert pathlib.Path(sys.prefix).name == "bimanual-teleop", "Activate the bimanual-teleop Conda environment first"'
cad_cache="$sim_root/.runtime/cad-tools"
mkdir -p "$cad_cache/wheels"
shopt -s nullglob
cached=("$cad_cache"/wheels/sldkit-0.1.0-cp310-abi3-linux_x86_64.whl)
if (( ${#cached[@]} )); then
  PIP_USER=false python -m pip install --no-deps "${cached[0]}"
  exit 0
fi
build_root=$(mktemp -d "$cad_cache/build.XXXXXXXX")
trap 'rm -rf "$build_root"' EXIT
# This commit is the last MIT release, before upstream changed its license.
revision=c43b5ef752a0e58535b7f8f744f7d7c68e3c0e90
curl --proto '=https' --tlsv1.2 -fL \
  "https://codeload.github.com/monozukuri-ai/sldkit/tar.gz/$revision" -o "$build_root/source.tar.gz"
mkdir "$build_root/source"
tar -xzf "$build_root/source.tar.gz" --strip-components=1 -C "$build_root/source"
export RUSTUP_HOME="$build_root/rustup"
export CARGO_HOME="$build_root/cargo"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o "$build_root/rustup-init.sh"
sh "$build_root/rustup-init.sh" -y --profile minimal --no-modify-path --default-toolchain stable
export PATH="$CARGO_HOME/bin:$PATH"
PIP_USER=false python -m pip wheel --no-deps "$build_root/source" --wheel-dir "$cad_cache/wheels"
cp "$build_root/source/LICENSE" "$cad_cache/sldkit-MIT-LICENSE"
PIP_USER=false python -m pip install --no-deps "$cad_cache/wheels/sldkit-0.1.0-cp310-abi3-linux_x86_64.whl"
