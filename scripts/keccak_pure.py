#!/usr/bin/env python3
"""Keccak-256 in pure Python, so the collector runs with no dependencies.

Only used to derive 4-byte function selectors. The miner itself uses
pycryptodome when it is installed.
"""
from __future__ import annotations

MASK = (1 << 64) - 1
RATE = 136

RHO = (1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
       27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44)
PI = (10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
      15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1)
ROUND_CONSTANTS = (
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
)


def _rotate(value: int, count: int) -> int:
    return ((value << count) | (value >> (64 - count))) & MASK


def _permute(state: list[int]) -> None:
    for round_constant in ROUND_CONSTANTS:
        parity = [state[i] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20]
                  for i in range(5)]
        for i in range(5):
            spread = parity[(i + 4) % 5] ^ _rotate(parity[(i + 1) % 5], 1)
            for row in range(0, 25, 5):
                state[row + i] ^= spread
        carry = state[1]
        for i in range(24):
            lane = PI[i]
            state[lane], carry = _rotate(carry, RHO[i]), state[lane]
        for row in range(0, 25, 5):
            values = state[row:row + 5]
            for i in range(5):
                state[row + i] = values[i] ^ ((~values[(i + 1) % 5] & MASK) & values[(i + 2) % 5])
        state[0] ^= round_constant


def keccak256(data: bytes) -> bytes:
    padded = bytearray(data)
    padded.append(0x01)
    while len(padded) % RATE:
        padded.append(0x00)
    padded[-1] |= 0x80

    state = [0] * 25
    for offset in range(0, len(padded), RATE):
        block = padded[offset:offset + RATE]
        for lane in range(RATE // 8):
            state[lane] ^= int.from_bytes(block[lane * 8:lane * 8 + 8], "little")
        _permute(state)
    return b"".join(state[lane].to_bytes(8, "little") for lane in range(4))


def selector(signature: str) -> str:
    return keccak256(signature.encode()).hex()[:8]
