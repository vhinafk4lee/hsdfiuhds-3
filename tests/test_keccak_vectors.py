"""Cross-check the shared Keccak core (CPU build) against pycryptodome.

Build first:
    g++ -O2 -o /tmp/kt tests/test_keccak_host.cpp
Run:
    python3 tests/test_keccak_vectors.py /tmp/kt
"""
import os
import random
import subprocess
import sys

from Crypto.Hash import keccak


def main() -> int:
    binary = sys.argv[1] if len(sys.argv) > 1 else "/tmp/kt"
    msgs = [b"", b"abc", b"\x00", bytes(range(135))]
    random.seed(1337)
    for _ in range(300):
        msgs.append(os.urandom(random.randint(0, 135)))

    proc = subprocess.run(
        [binary],
        input="\n".join(m.hex() for m in msgs) + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    got = proc.stdout.split()
    assert len(got) == len(msgs), f"{len(got)} outputs for {len(msgs)} inputs"

    for msg, g in zip(msgs, got):
        want = keccak.new(digest_bits=256, data=msg).hexdigest()
        if g != want:
            print(f"MISMATCH len={len(msg)}\n  msg  {msg.hex()}\n  got  {g}\n  want {want}")
            return 1

    print(f"OK: {len(msgs)} vectors match pycryptodome keccak256")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
