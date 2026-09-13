#!/usr/bin/env python3
"""Shared CUDA plumbing for the worker and the benchmark.

Two kernels live behind one object. Hash Broker hashes SHA-256 over a padded
message of 32-bit words; FlyNode hashes Keccak-256 over 17 little-endian lanes,
and pokes the nonce into a lane at a bit offset rather than into a word. The
difference is real but it is all in how a launch is set up, so ``Kernel`` keeps
it here and the worker above stays one loop.
"""
from __future__ import annotations

import cupy as cp
import numpy as np

import pow as powlib
from keccak_cuda import CUDA_SOURCE as KECCAK_SOURCE
from protocol import PROTOCOL, Protocol
from sha256_cuda import CUDA_SOURCE as SHA256_SOURCE

MESSAGE_WORDS = 32


def device_name(device: int = 0) -> str:
    return cp.cuda.runtime.getDeviceProperties(device)["name"].decode()


def target_words(target: int) -> np.ndarray:
    raw = int(target).to_bytes(32, "big")
    return np.array([int.from_bytes(raw[index:index + 4], "big") for index in range(0, 32, 4)],
                    dtype=np.uint32)


def digest_from_words(words) -> bytes:
    return b"".join(int(word).to_bytes(4, "big") for word in words)


def digest_from_lanes(lanes) -> bytes:
    """Keccak's state is little-endian lanes; the digest is those bytes in order."""
    return b"".join(int(lane).to_bytes(8, "little") for lane in lanes)


class Kernel:
    """One compiled kernel, bound to one preimage, ready to be launched at.

    ``bind`` fixes everything about the search but the nonce, so the message
    only crosses to the device when the job actually changes.
    """

    def __init__(self, device: int = 0, protocol: Protocol = PROTOCOL):
        self.protocol = protocol
        self.keccak = protocol.algorithm == "keccak256"
        if not self.keccak and protocol.algorithm not in ("sha256", "sha256d"):
            raise SystemExit(f"no GPU kernel for {protocol.algorithm}")
        cp.cuda.Device(device).use()
        module = cp.RawModule(code=KECCAK_SOURCE if self.keccak else SHA256_SOURCE,
                              options=("--std=c++11",))
        self.device = device
        self._hash_one = module.get_function("hash_one")
        self._mine_batch = module.get_function("mine_batch")
        self._prefix: tuple = ()
        self._bindings: dict = {}

    @property
    def name(self) -> str:
        return device_name(self.device)

    def bind(self, wallet: str, bindings: dict) -> None:
        """Move the fixed part of the preimage onto the device."""
        fields = {"wallet": wallet, **bindings}
        if self.keccak:
            lanes = powlib.keccak_lanes(fields, self.protocol)
            message = cp.asarray(np.array(lanes, dtype=np.uint64))
            positions = powlib.nonce_lane_positions(self.protocol)
            self._prefix = (message, *(np.int32(value) for value in positions))
        else:
            words = powlib.padded_words(fields, self.protocol)
            if len(words) > MESSAGE_WORDS:
                raise SystemExit(f"preimage of {self.protocol.preimage_size} bytes "
                                 "needs more than two SHA-256 blocks")
            padded = np.array(words + [0] * (MESSAGE_WORDS - len(words)), dtype=np.uint32)
            doubled = 1 if self.protocol.algorithm == "sha256d" else 0
            self._prefix = (cp.asarray(padded), np.int32(len(words) // 16), np.int32(doubled))
        self._bindings = dict(bindings)

    def _target(self, target: int):
        if not self.keccak:
            return (cp.asarray(target_words(target)),)
        raw = int(target).to_bytes(32, "big")
        return tuple(np.uint64(int.from_bytes(raw[index:index + 8], "big"))
                     for index in range(0, 32, 8))

    def hash_one(self, stream: int, counter: int) -> bytes:
        """One hash on the device, for checking the device against the CPU."""
        width, reader = (4, digest_from_lanes) if self.keccak else (8, digest_from_words)
        output = cp.zeros(width, dtype=cp.uint64 if self.keccak else cp.uint32)
        self._hash_one((1,), (1,), (*self._prefix, np.uint32(stream), np.uint32(counter), output))
        cp.cuda.runtime.deviceSynchronize()
        return reader(cp.asnumpy(output))

    def search(self, blocks: int, threads: int, stream: int, counter: int,
               iterations: int, target: int) -> tuple[int, bytes] | None:
        """Launch one batch; return the winning nonce tail and its digest, if any."""
        found = cp.zeros(1, dtype=cp.int32)
        found_counter = cp.zeros(1, dtype=cp.uint32)
        found_hash = cp.zeros(4 if self.keccak else 8,
                              dtype=cp.uint64 if self.keccak else cp.uint32)
        self._mine_batch((blocks,), (threads,), (
            *self._prefix, np.uint32(stream), np.uint32(counter), np.uint32(iterations),
            *self._target(target), found, found_counter, found_hash,
        ))
        cp.cuda.runtime.deviceSynchronize()
        if not int(found.get()[0]):
            return None
        reader = digest_from_lanes if self.keccak else digest_from_words
        return int(found_counter.get()[0]), reader(cp.asnumpy(found_hash))

    def self_test(self, wallet: str) -> bytes:
        """A GPU that disagrees with the CPU is a hardware fault, not a miner."""
        stream, counter = 0x13579BDF, 0x2468ACE0
        actual = self.hash_one(stream, counter)
        expected = powlib.digest({"wallet": wallet, "nonce": (stream << 32) | counter,
                                  **self._bindings}, self.protocol)
        if actual != expected:
            raise SystemExit(f"GPU self-test failed: {actual.hex()} != {expected.hex()}")
        return actual
