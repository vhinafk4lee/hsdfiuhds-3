#!/usr/bin/env python3
"""Measures this GPU's proof rate without touching the chain or a wallet.

Run it on a rented host before committing to it: it self-tests the kernel
against hashlib, sweeps a few launch shapes, and reports the expected time per
proof at a given difficulty.
"""
from __future__ import annotations

import argparse
import time

import cupy as cp
import numpy as np

import pow as powlib
from gpu import (device_name, load_kernels, message_buffer, self_test, target_words)
from protocol import PROTOCOL

SAMPLE_WALLET = "0x" + "11" * 20
SAMPLE_CHALLENGE = "0x" + "3a" * 32
SHAPES = ((4096, 256, 32), (8192, 256, 64), (16384, 256, 64), (8192, 512, 128))


def measure(mine_batch, message_gpu, blocks: int, doubled: int, stream_word: int,
            counter_word: int, shape: tuple[int, int, int], seconds: float) -> float:
    grid, threads, iterations = shape
    batch = grid * threads * iterations
    if batch > 2**32:
        raise ValueError("launch shape covers more than 2**32 nonces")
    found = cp.zeros(1, dtype=cp.int32)
    found_counter = cp.zeros(1, dtype=cp.uint32)
    found_hash = cp.zeros(8, dtype=cp.uint32)
    # An unreachable target keeps every thread hashing for the whole batch.
    target = cp.asarray(target_words(1))
    counter = 0
    hashed = 0
    started = time.monotonic()
    while time.monotonic() - started < seconds:
        mine_batch((grid,), (threads,),
                   (message_gpu, np.int32(blocks), np.int32(doubled),
                    np.int32(stream_word), np.int32(counter_word),
                    np.uint32(1), np.uint32(counter), np.uint32(iterations),
                    target, found, found_counter, found_hash))
        cp.cuda.runtime.deviceSynchronize()
        hashed += batch
        counter = (counter + batch) % (2**32)
    return hashed / (time.monotonic() - started)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=5.0, help="per launch shape")
    parser.add_argument("--difficulty", type=int, default=50)
    arguments = parser.parse_args()

    hash_one, mine_batch = load_kernels(arguments.device)
    print(f"GPU {arguments.device} {device_name(arguments.device)}")
    print(f"CUDA runtime {cp.cuda.runtime.runtimeGetVersion()} "
          f"driver {cp.cuda.runtime.driverGetVersion()}")

    message, blocks = message_buffer(SAMPLE_WALLET, SAMPLE_CHALLENGE)
    message_gpu = cp.asarray(message)
    doubled = 1 if PROTOCOL.algorithm == "sha256d" else 0
    stream_word, counter_word = powlib.nonce_word_indices()
    proof = self_test(hash_one, message_gpu, blocks, doubled, stream_word, counter_word,
                      SAMPLE_WALLET, SAMPLE_CHALLENGE)
    print("SELF_TEST_OK", proof.hex())

    best = (0.0, None)
    for shape in SHAPES:
        rate = measure(mine_batch, message_gpu, blocks, doubled, stream_word, counter_word,
                       shape, arguments.seconds)
        grid, threads, iterations = shape
        print(f"blocks={grid:>6} threads={threads:>4} iterations={iterations:>4}"
              f"  {rate / 1e9:7.3f} GH/s")
        if rate > best[0]:
            best = (rate, shape)

    rate, shape = best
    work = 2 ** arguments.difficulty
    print(f"\nbest shape   --blocks {shape[0]} --threads {shape[1]} --iterations {shape[2]}")
    print(f"rate         {rate / 1e9:.3f} GH/s")
    print(f"difficulty   {arguments.difficulty} -> 2^{arguments.difficulty} hashes per proof")
    print(f"expected     {work / rate / 3600:.2f} h per proof on this GPU alone")
    for count in (4, 8, 16, 32):
        print(f"             {work / (rate * count) / 3600:8.2f} h with {count} of these GPUs")


if __name__ == "__main__":
    main()
