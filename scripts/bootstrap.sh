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

apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv git

mkdir -p "$TARGET"
if [ -d "$TARGET/repo/.git" ]; then
    git -C "$TARGET/repo" fetch --depth 1 origin "$BRANCH"
    git -C "$TARGET/repo" checkout -B "$BRANCH" "origin/$BRANCH"
else
    git clone --depth 1 -b "$BRANCH" "$REPO" "$TARGET/repo"
fi

python3 -m venv "$TARGET/venv"
"$TARGET/venv/bin/pip" install --quiet --upgrade pip
"$TARGET/venv/bin/pip" install --quiet -r "$TARGET/repo/requirements.txt"

CUDA_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
if [ "$CUDA_MAJOR" -ge 525 ]; then
    "$TARGET/venv/bin/pip" install --quiet cupy-cuda12x
else
    "$TARGET/venv/bin/pip" install --quiet cupy-cuda11x
fi

"$TARGET/venv/bin/python" "$TARGET/repo/scripts/protocol.py"
echo "worker installed in $TARGET"
