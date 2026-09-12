#!/usr/bin/env python3
"""Shared CUDA plumbing for the worker and the benchmark."""
from __future__ import annotations

import cupy as cp
import numpy as np

import pow as powlib
from protocol import PROTOCOL
from sha256_cuda import CUDA_SOURCE

MESSAGE_WORDS = 32


def load_kernels(device: int = 0):
    cp.cuda.Device(device).use()
    module = cp.RawModule(code=CUDA_SOURCE, options=("--std=c++11",))
    return module.get_function("hash_one"), module.get_function("mine_batch")


def device_name(device: int = 0) -> str:
    return cp.cuda.runtime.getDeviceProperties(device)["name"].decode()


def message_buffer(wallet: str, challenge: str) -> tuple[np.ndarray, int]:
    words = powlib.padded_words(wallet, challenge)
    if len(words) > MESSAGE_WORDS:
        raise SystemExit(
            f"preimage of {PROTOCOL.preimage_size} bytes needs more than two SHA-256 blocks"
        )
    blocks = len(words) // 16
    return np.array(words + [0] * (MESSAGE_WORDS - len(words)), dtype=np.uint32), blocks


def target_words(target: int) -> np.ndarray:
    raw = int(target).to_bytes(32, "big")
    return np.array([int.from_bytes(raw[index:index + 4], "big") for index in range(0, 32, 4)],
                    dtype=np.uint32)


def digest_from_words(words) -> bytes:
    return b"".join(int(word).to_bytes(4, "big") for word in words)


def self_test(hash_one, message_gpu, blocks: int, doubled: int, stream_word: int,
              counter_word: int, wallet: str, challenge: str) -> bytes:
    """A GPU that disagrees with hashlib is a hardware fault, not a miner."""
    stream, counter = 0x13579BDF, 0x2468ACE0
    output = cp.zeros(8, dtype=cp.uint32)
    hash_one((1,), (1,), (message_gpu, np.int32(blocks), np.int32(doubled),
                          np.int32(stream_word), np.int32(counter_word),
                          np.uint32(stream), np.uint32(counter), output))
    cp.cuda.runtime.deviceSynchronize()
    actual = digest_from_words(cp.asnumpy(output))
    expected = powlib.digest(wallet, (stream << 32) | counter, challenge)
    if actual != expected:
        raise SystemExit(f"GPU self-test failed: {actual.hex()} != {expected.hex()}")
    return actual
