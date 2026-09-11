"""The shipped Hashcats config must match the format the contract actually accepts.

The layout and selectors are those used by accepted mint transactions on Robinhood
Chain: keccak256(miner || nonce || prevWork || anchor) over 116 bytes with the nonce
big-endian at bytes 20..51, and mine(uint256,uint256) = 0x071e9503 taking the nonce
and the anchor height. This test pins all of it so a refactor cannot silently break
the wiring.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hcminer.abi import function_selector
from hcminer.config import Config
from hcminer.keccak import keccak256
from hcminer.schema import Schema
from hcminer.tx import Submitter

CONFIG = Path(__file__).resolve().parents[1] / "config.hashcats.toml"


def test_selectors_match_the_names() -> None:
    for signature, selector in (
        ("prevWork()", "a4da5da2"),
        ("currentTarget()", "39148c53"),
        ("currentAnchor()", "cd809b11"),
        ("mine(uint256,uint256)", "071e9503"),
    ):
        got = function_selector(signature).hex()
        assert got == selector, f"{signature}: {got} != {selector}"


def test_preimage_layout() -> None:
    cfg = Config.load(CONFIG)
    schema = Schema.parse(cfg.schema_specs())
    values = {
        "miner": "0x" + "ab" * 20,
        "nonce": 0x1234,
        "prev_work": "0x" + "11" * 32,
        "anchor": "0x" + "22" * 32,
    }
    blob = schema.build(values)

    assert len(blob) == 116
    assert blob[0:20] == bytes.fromhex("ab" * 20)
    assert blob[20:52] == (0x1234).to_bytes(32, "big")
    assert blob[52:84] == bytes.fromhex("11" * 32)
    assert blob[84:116] == bytes.fromhex("22" * 32)
    assert schema.vary_offset == 44          # the 8 bytes the GPU increments
    assert schema.hash(values) == keccak256(blob)


def test_mint_calldata() -> None:
    cfg = Config.load(CONFIG)
    submitter = Submitter.__new__(Submitter)   # calldata needs no key or RPC
    submitter.cfg = cfg

    data = submitter.build_calldata({"nonce": 0x1234, "anchor_block": 987654})
    assert data[:4].hex() == "071e9503"
    assert int.from_bytes(data[4:36], "big") == 0x1234
    assert int.from_bytes(data[36:68], "big") == 987654
    assert len(data) == 68


def test_anchor_word_selection() -> None:
    """currentAnchor() returns (height, hash); the hash is word 1, height word 0."""
    from hcminer.chain import Chain

    cfg = Config.load(CONFIG)
    assert Chain._split_word(cfg.get("contract.state.anchor")) == ("currentAnchor()", 1)
    assert Chain._split_word(cfg.get("contract.state.anchor_block")) == ("currentAnchor()", 0)


def test_entry_price_is_sane() -> None:
    cfg = Config.load(CONFIG)
    value = int(cfg.get("contract.mint.value"))
    assert value == 10_080_000_000_000_000, value          # 0.01008 ETH
    assert float(cfg.get("limits.max_spend_eth")) > value / 1e18


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("hashcats wiring pinned")
