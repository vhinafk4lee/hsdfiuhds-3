#!/usr/bin/env bash
# Install the worker on a fresh GPU host. Run as root.
set -euo pipefail

TARGET="${HASHBROKER_WORKER_DIR:-/opt/hashbroker}"
BRANCH="${HASHBROKER_BRANCH:-claude/sweet-rubin-w4jyk9}"
REPO="${HASHBROKER_REPO:-https://github.com/vhinafk4lee/hsdfiuhds-3.git}"

if ! command -v nvidia-smi >/dev/null; then
    echo "nvidia-smi not found: install the NVIDIA driver first" >&2
    exit 1
fi
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq || true
apt-get install -y -qq python3 python3-pip python3-venv git || true
command -v git >/dev/null || { echo "git is missing and apt could not install it" >&2; exit 1; }

mkdir -p "$TARGET"
if [ -d "$TARGET/repo/.git" ]; then
    git -C "$TARGET/repo" fetch --depth 1 origin "$BRANCH"
    git -C "$TARGET/repo" checkout -B "$BRANCH" "origin/$BRANCH"
else
    git clone --depth 1 -b "$BRANCH" "$REPO" "$TARGET/repo"
fi

# Some rental images ship without python3-venv; fall back to the system python.
if python3 -m venv "$TARGET/venv" 2>/dev/null; then
    PIP="$TARGET/venv/bin/pip"
    PYTHON="$TARGET/venv/bin/python"
else
    echo "venv unavailable, installing into the system python"
    PIP="python3 -m pip"
    PYTHON="python3"
    ln -sfn "$(command -v python3)" "$TARGET/python3"
fi
$PIP install --quiet --upgrade pip
$PIP install --quiet -r "$TARGET/repo/requirements.txt"

# cupy must match the CUDA the driver supports. Blackwell cards (RTX 50xx,
# sm_120) need a build whose NVRTC can target them: CUDA 13 wheels, or CUDA 12
# wheels from 12.8 onwards.
CUDA_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
CUDA_SUPPORTED="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]*\)\..*/\1/p' | head -1)"
CUDA_SUPPORTED="${CUDA_SUPPORTED:-12}"
if [ "$CUDA_SUPPORTED" -ge 13 ]; then
    CUPY_PACKAGE="${HASHBROKER_CUPY:-cupy-cuda13x}"
else
    CUPY_PACKAGE="${HASHBROKER_CUPY:-cupy-cuda12x}"
fi
echo "driver $CUDA_VERSION supports CUDA $CUDA_SUPPORTED, installing $CUPY_PACKAGE"
$PIP install --quiet "$CUPY_PACKAGE"

$PYTHON "$TARGET/repo/scripts/protocol.py"
echo
echo "worker installed in $TARGET"
echo "benchmark it before renting more:"
echo "  $PYTHON $TARGET/repo/scripts/benchmark.py"
