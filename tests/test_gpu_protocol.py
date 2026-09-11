"""End-to-end test of the supervisor <-> miner-process contract, on CPU.

Covers the parts that are easy to get subtly wrong: where the varying nonce bytes
sit inside the preimage, how the target is serialised, and how the supervisor
rebuilds the full nonce from the 64 bits the device reports.
"""

from __future__ import annotations

import os
import secrets
import stat
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hcminer.gpu import GpuMiner
from hcminer.keccak import leading_zero_bits
from hcminer.schema import Schema

ZERO_BITS = 18  # easy enough for a pure-Python stand-in


def make_launcher() -> str:
    """Wrap fake_gpu.py in an executable shim so GpuMiner can exec it like the real binary."""
    repo = Path(__file__).resolve().parents[1]
    script = Path(tempfile.mkdtemp()) / "fake-hcminer-gpu"
    script.write_text(
        f"#!/bin/sh\nexec {sys.executable} {repo / 'tests' / 'fake_gpu.py'}\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def main() -> int:
    schema = Schema.parse(["miner:addr20", "nonce:u256", "prev_work:b32", "anchor:b32"])
    wallet = "0x" + secrets.token_hex(20)
    state = {
        "prev_work": "0x" + secrets.token_hex(32),
        "anchor": "0x" + secrets.token_hex(32),
    }
    target = 1 << (256 - ZERO_BITS)
    session_prefix = secrets.randbits(schema.fields[1].size * 8 - 64)

    values = {"miner": wallet, "nonce": session_prefix << 64, **state}
    preimage = schema.build(values)
    assert preimage[schema.vary_offset : schema.vary_offset + 8] == b"\x00" * 8, (
        "the kernel ORs the nonce in, so those bytes must start out zero"
    )

    gpu = GpuMiner(binary=make_launcher())
    gpu.start()
    gpu.submit_job(1, preimage, schema.vary_offset, target, nonce_start=0)

    deadline = time.time() + 120
    solution = None
    while time.time() < deadline and solution is None:
        for event in gpu.poll(timeout=1.0):
            if event.get("type") == "solution":
                solution = event
                break
            if event.get("type") == "exit":
                print("miner process exited early:", gpu.stderr_tail())
                return 1
    gpu.stop()

    if solution is None:
        print("no solution within the time budget")
        return 1

    # The supervisor's reconstruction must reproduce the device's digest exactly.
    full_nonce = (session_prefix << 64) | int(solution["nonce"], 16)
    values["nonce"] = full_nonce
    digest = schema.hash(values)

    assert "0x" + digest.hex() == solution["hash"], (
        f"digest mismatch:\n  device {solution['hash']}\n  host   0x{digest.hex()}"
    )
    assert int.from_bytes(digest, "big") < target, "solution is not below target"
    print(
        f"OK: job solved, nonce={full_nonce}\n"
        f"    hash={solution['hash']} ({leading_zero_bits(digest)} zero bits)\n"
        f"    host reconstruction matches the device digest"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
