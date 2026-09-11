"""The solver must recover an unknown layout from one solved mint, uniquely."""

from __future__ import annotations

import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hcminer.discover import Observation, candidate_schemas, solve_schema
from hcminer.keccak import leading_zero_bits
from hcminer.schema import Schema

ZERO_BITS = 20


def mine_cpu(schema: Schema, values: dict) -> int:
    nonce = 0
    while True:
        values["nonce"] = nonce
        if leading_zero_bits(schema.hash(values)) >= ZERO_BITS:
            return nonce
        nonce += 1


def check(truth_specs: list) -> None:
    truth = Schema.parse(truth_specs)
    values = {
        "miner": "0x" + secrets.token_hex(20),
        "prev_work": "0x" + secrets.token_hex(32),
        "anchor": "0x" + secrets.token_hex(32),
        "nonce": 0,
    }
    nonce = mine_cpu(truth, dict(values))
    obs = Observation(
        miner=values["miner"], nonce=nonce,
        prev_work=values["prev_work"], anchor=values["anchor"],
    )
    matches = solve_schema([obs], min_zero_bits=ZERO_BITS)
    assert matches, f"no layout recovered for {truth}"
    recovered = [m.schema.specs() for m in matches]
    assert truth.specs() in recovered, f"true layout {truth} missing from {recovered}"
    assert len(matches) == 1, f"ambiguous: {recovered}"
    print(f"  ok  recovered {truth} (nonce={nonce})")


if __name__ == "__main__":
    print(f"  candidate space: {len(candidate_schemas(['miner','nonce','prev_work','anchor']))} layouts")
    check(["miner:addr20", "nonce:u256", "prev_work:b32", "anchor:b32"])
    check(["prev_work:b32", "anchor:b32", "miner:addr32", "nonce:u256"])
    print("schema recovery tests passed")
