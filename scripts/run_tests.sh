#!/usr/bin/env bash
# Everything that can be checked without a GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== keccak core (CPU build of the GPU code) vs pycryptodome =="
g++ -O2 -o /tmp/hc-keccak-test tests/test_keccak_host.cpp
python3 tests/test_keccak_vectors.py /tmp/hc-keccak-test

echo "== schema encoding =="
python3 tests/test_schema.py

echo "== nonce lane splice (the kernel-only arithmetic) =="
python3 tests/test_lane_splice.py

echo "== supervisor <-> miner protocol (CPU stand-in) =="
python3 tests/test_gpu_protocol.py

echo "== schema recovery from a synthetic solved mint =="
python3 tests/test_schema_recovery.py

echo "== automatic contract wiring from chain-shaped data =="
python3 tests/test_autoconfig.py

echo "== real Hashcats wiring (layout, selectors, mint calldata) =="
python3 tests/test_hashcats_wiring.py

if command -v nvcc >/dev/null 2>&1; then
  echo "== CUDA compile check =="
  ARCH="${CUDA_ARCH:-89}"
  nvcc -O3 -std=c++17 --use_fast_math --allow-unsupported-compiler \
       -gencode arch=compute_${ARCH},code=sm_${ARCH} -Xptxas -O3,-v \
       -o /tmp/hcminer-gpu-check src/cuda/miner.cu 2>&1 | grep -E "stack frame|registers|error"
else
  echo "== CUDA compile check skipped (no nvcc) =="
fi

echo "ALL TESTS PASSED"
