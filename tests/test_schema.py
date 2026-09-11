"""Schema encoding: field layout, nonce offsets, error handling."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hcminer.keccak import keccak256
from hcminer.schema import Schema, SchemaError

MINER = "0x" + "11" * 20
PREV = "0x" + "22" * 32
ANCHOR = "0x" + "33" * 32


def test_packed_layout() -> None:
    schema = Schema.parse(["miner:addr20", "nonce:u256", "prev_work:b32", "anchor:b32"])
    values = {"miner": MINER, "nonce": 0xDEAD, "prev_work": PREV, "anchor": ANCHOR}
    blob = schema.build(values)

    assert len(blob) == 116, len(blob)
    assert blob[:20] == bytes.fromhex("11" * 20)
    assert blob[20:52] == (0xDEAD).to_bytes(32, "big")
    assert blob[52:84] == bytes.fromhex("22" * 32)
    assert schema.nonce_offset == 20
    assert schema.vary_offset == 44          # low 8 bytes of a 32-byte nonce
    assert schema.hash(values) == keccak256(blob)


def test_word_layout_matches_abi_encode() -> None:
    """addr32 everywhere reproduces abi.encode(address,uint256,bytes32,bytes32)."""
    schema = Schema.parse(["miner:addr32", "nonce:u256", "prev_work:b32", "anchor:b32"])
    blob = schema.build({"miner": MINER, "nonce": 1, "prev_work": PREV, "anchor": ANCHOR})
    assert len(blob) == 128
    assert blob[:12] == b"\x00" * 12          # address left-padded into a word
    assert blob[12:32] == bytes.fromhex("11" * 20)


def test_short_nonce_has_no_prefix_room() -> None:
    schema = Schema.parse(["miner:addr20", "nonce:u64", "anchor:b32"])
    assert schema.fields[1].size == 8
    assert schema.vary_offset == schema.nonce_offset == 20


def test_rejects_bad_schemas() -> None:
    for specs in (
        ["miner:addr20"],                              # no nonce
        ["nonce:u256", "nonce:u256"],                  # two nonces
        ["miner:addr20", "nonce:uint256"],             # unknown encoding
        ["sender:addr20", "nonce:u256"],               # unknown source
    ):
        try:
            Schema.parse(specs)
        except SchemaError:
            continue
        raise AssertionError(f"should have been rejected: {specs}")


def test_rejects_oversized_preimage() -> None:
    schema = Schema.parse(["nonce:u256"] + ["prev_work:b32"] * 4)
    try:
        schema.build({"nonce": 1, "prev_work": PREV})
    except SchemaError as exc:
        assert "135" in str(exc)
        return
    raise AssertionError("a 160-byte preimage must be rejected: the kernel is single-block")


def test_const_field() -> None:
    schema = Schema.parse(["const:0xcafe", "nonce:u64"])
    assert schema.build({"nonce": 7}) == bytes.fromhex("cafe") + (7).to_bytes(8, "big")
    assert schema.vary_offset == 2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("schema tests passed")
