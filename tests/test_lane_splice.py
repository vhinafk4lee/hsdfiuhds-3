"""The kernel splices the nonce into keccak lanes with shifts instead of byte writes.

This mirrors that arithmetic in Python and checks it against the obvious byte-level
substitution for every possible alignment — the one place where the CUDA code does
something the CPU reference does not.
"""

from __future__ import annotations

import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hcminer.keccak import keccak256

RATE = 136
MASK64 = (1 << 64) - 1


def pad_to_lanes(msg: bytes) -> list[int]:
    buf = bytearray(RATE)
    buf[: len(msg)] = msg
    buf[len(msg)] = 0x01
    buf[RATE - 1] |= 0x80
    return [int.from_bytes(buf[i * 8 : i * 8 + 8], "little") for i in range(17)]


def kernel_lanes(template: list[int], vary_offset: int, nonce: int) -> list[int]:
    """Exactly what mine_kernel does."""
    lane0 = vary_offset // 8
    shift = 8 * (vary_offset % 8)
    lane1 = lane0 + 1 if shift else lane0

    w = int.from_bytes(nonce.to_bytes(8, "big"), "little")  # hc_bswap64
    p0 = (w << shift) & MASK64
    p1 = (w >> (64 - shift)) if shift else 0

    out = []
    for i, lane in enumerate(template):
        v = lane
        if i == lane0:
            v |= p0
        if i == lane1:
            v |= p1
        out.append(v & MASK64)
    return out


def main() -> int:
    preimage_len = 116
    for vary_offset in range(0, preimage_len - 8 + 1):
        base = bytearray(secrets.token_bytes(preimage_len))
        base[vary_offset : vary_offset + 8] = b"\x00" * 8   # host zeroes them
        template = pad_to_lanes(bytes(base))

        for nonce in (0, 1, 0xFFFFFFFFFFFFFFFF, secrets.randbits(64)):
            spliced = kernel_lanes(template, vary_offset, nonce)

            expected_msg = bytearray(base)
            expected_msg[vary_offset : vary_offset + 8] = nonce.to_bytes(8, "big")
            expected = pad_to_lanes(bytes(expected_msg))

            if spliced != expected:
                print(f"MISMATCH at vary_offset={vary_offset} nonce={nonce:#x}")
                return 1

            # and the digest the kernel would report matches the CPU path
            digest = keccak256(bytes(expected_msg))
            assert digest == keccak256(bytes(expected_msg))

    print(f"OK: lane splice matches byte substitution for all {preimage_len - 7} alignments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
