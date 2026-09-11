"""The resolver must recover the wiring when nobody tells it which value is which.

Builds two synthetic mints whose hashes were produced by a known layout, hides the
inputs among decoy values, and checks that `resolve` picks out the right layout and
the right labels.
"""

from __future__ import annotations

import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hcminer.autoconfig import MintSample, resolve
from hcminer.keccak import leading_zero_bits
from hcminer.schema import Schema

ZERO_BITS = 18
TRUTH = Schema.parse(["miner:addr20", "nonce:u256", "prev_work:b32", "anchor:b32"])
MINT_ENTRY = {"type": "function", "name": "mine",
              "inputs": [{"name": "nonce", "type": "uint256"}]}


def build_sample(prev_label: str, anchor_label: str, decoys: int = 12) -> MintSample:
    miner = "0x" + secrets.token_hex(20)
    prev = "0x" + secrets.token_hex(32)
    anchor = "0x" + secrets.token_hex(32)

    nonce = 0
    while True:
        digest = TRUTH.hash({"miner": miner, "nonce": nonce,
                             "prev_work": prev, "anchor": anchor})
        if leading_zero_bits(digest) >= ZERO_BITS:
            break
        nonce += 1

    words = {f"decoy{i}": "0x" + secrets.token_hex(32) for i in range(decoys)}
    words[prev_label] = prev
    words[anchor_label] = anchor

    return MintSample(
        tx_hash="0x" + secrets.token_hex(32),
        miner=miner,
        block=1000,
        value_wei=10_080_000_000_000_000,
        nonce_candidates=[nonce],
        words=words,
    )


def main() -> int:
    prev_label, anchor_label = "view:lastWork()", "blockhash-1"
    samples = [build_sample(prev_label, anchor_label) for _ in range(2)]

    found = resolve(samples, MINT_ENTRY, min_zero_bits=ZERO_BITS)
    if not found:
        print("FAIL: nothing resolved")
        return 1

    for res in found:
        print(f"  candidate: {res.schema} prev={res.prev_work_label} "
              f"anchor={res.anchor_label} bits={res.zero_bits}")

    best = found[0]
    assert best.schema.specs() == TRUTH.specs(), f"wrong layout: {best.schema}"
    assert best.prev_work_label == prev_label, f"wrong prev_work: {best.prev_work_label}"
    assert best.anchor_label == anchor_label, f"wrong anchor: {best.anchor_label}"
    assert best.anchor_is_block_hash and best.anchor_block_offset == 1

    print(f"\nOK: wiring recovered from {len(samples)} mints out of "
          f"{len(samples[0].words)} candidate values, with no hints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
