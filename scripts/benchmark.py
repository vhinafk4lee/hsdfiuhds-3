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

from gpu import Kernel
from protocol import PROTOCOL

SAMPLE_WALLET = "0x" + "11" * 20
SAMPLE_CHALLENGE = "0x" + "3a" * 32
SAMPLE_BINDINGS = {
    "keccak256": {"prev": "0x" + "5c" * 32, "anchor": "0x" + "a9" * 32, "typeId": 0},
}.get(PROTOCOL.algorithm, {"challenge": SAMPLE_CHALLENGE})
SHAPES = ((4096, 256, 32), (8192, 256, 64), (16384, 256, 64), (8192, 512, 128))


def measure(kernel: Kernel, shape: tuple[int, int, int], seconds: float) -> float:
    grid, threads, iterations = shape
    batch = grid * threads * iterations
    if batch > 2**32:
        raise ValueError("launch shape covers more than 2**32 nonces")
    counter = 0
    hashed = 0
    started = time.monotonic()
    while time.monotonic() - started < seconds:
        # An unreachable target keeps every thread hashing for the whole batch.
        kernel.search(grid, threads, 1, counter, iterations, 1)
        hashed += batch
        counter = (counter + batch) % (2**32)
    return hashed / (time.monotonic() - started)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=5.0, help="per launch shape")
    parser.add_argument("--difficulty", type=int, default=50)
    arguments = parser.parse_args()

    kernel = Kernel(arguments.device)
    print(f"GPU {arguments.device} {kernel.name} algorithm={PROTOCOL.algorithm}")
    print(f"CUDA runtime {cp.cuda.runtime.runtimeGetVersion()} "
          f"driver {cp.cuda.runtime.driverGetVersion()}")

    kernel.bind(SAMPLE_WALLET, SAMPLE_BINDINGS)
    print("SELF_TEST_OK", kernel.self_test(SAMPLE_WALLET).hex())

    best = (0.0, None)
    for shape in SHAPES:
        rate = measure(kernel, shape, arguments.seconds)
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
