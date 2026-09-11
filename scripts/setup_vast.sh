#!/usr/bin/env bash
# One-shot setup on a Vast.ai CUDA instance (tested against the CUDA 13.x base images).
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== GPUs =="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || {
  echo "nvidia-smi not available - this is not a GPU instance"; exit 1; }

if ! command -v nvcc >/dev/null 2>&1; then
  echo "== nvcc missing, installing CUDA toolkit =="
  apt-get update -qq
  apt-get install -y -qq cuda-toolkit || apt-get install -y -qq nvidia-cuda-toolkit
  export PATH="/usr/local/cuda/bin:$PATH"
fi
nvcc --version | tail -2

echo "== python deps =="
pip install --quiet -r requirements.txt

# RTX 5090 (Blackwell) is sm_120; older cards need their own arch.
ARCH="${CUDA_ARCH:-}"
if [ -z "$ARCH" ]; then
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
  ARCH="${CC:-120}"
fi
echo "== building for sm_${ARCH} =="
make -C src/cuda CUDA_ARCH="$ARCH"

echo "== self-tests =="
g++ -O2 -o /tmp/hc-keccak-test tests/test_keccak_host.cpp
python3 tests/test_keccak_vectors.py /tmp/hc-keccak-test

echo
echo "Ready. Next:"
echo "  python3 -m hcminer.cli bench --seconds 15"
echo "  cp config.example.toml config.toml   # then: discover -> solve-schema -> verify"
