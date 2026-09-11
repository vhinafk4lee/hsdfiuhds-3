"""Reads the mining state the contract expects: target, previous work, anchor, price."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from . import abi
from .rpc import Rpc

ARBSYS_ADDRESS = "0x0000000000000000000000000000000000000064"


@dataclass
class MiningState:
    target: int          # 256-bit threshold; a valid hash must be strictly below it
    prev_work: str       # bytes32 hex
    anchor: str          # bytes32 hex
    price_wei: int
    block: int

    @property
    def zero_bits(self) -> int:
        return 256 - max(self.target, 1).bit_length()

    def describe(self) -> str:
        return (
            f"block={self.block} target=0x{self.target:064x} (~{self.zero_bits} zero bits)\n"
            f"prev_work={self.prev_work}\nanchor={self.anchor}\n"
            f"price={self.price_wei / 1e18:.6f} ETH"
        )


class Chain:
    """Thin, config-driven view of the Hashcats contract.

    Every signature comes from config, so a contract rename or a different getter
    shape is a config edit rather than a code change.
    """

    def __init__(self, rpc: Rpc, contract: str, cfg: Any):
        self.rpc = rpc
        self.contract = contract
        self.cfg = cfg

    # --- low level ------------------------------------------------------------

    def _view(self, signature: str, args: Optional[list] = None, block: str = "latest") -> bytes:
        data = abi.calldata(signature, args or [])
        return self.rpc.eth_call(self.contract, data, block)

    def _view_uint(self, signature: str, block: str = "latest") -> int:
        return abi.decode(["uint256"], self._view(signature, block=block))[0]

    def _view_bytes32(self, signature: str, block: str = "latest") -> str:
        raw = self._view(signature, block=block)
        return "0x" + raw[:32].hex()

    # --- state ----------------------------------------------------------------

    def target(self, block: str = "latest") -> int:
        signature = self.cfg.require("contract.state.target")
        value = self._view_uint(signature, block)
        if self.cfg.get("pow.target_is_bits", False):
            if not 0 < value < 256:
                raise RuntimeError(f"{signature} returned {value}, not a zero-bit count")
            return 1 << (256 - value)
        return value

    def prev_work(self, block: str = "latest") -> str:
        return self._view_bytes32(self.cfg.require("contract.state.prev_work"), block)

    def price_wei(self, block: str = "latest") -> int:
        signature = self.cfg.get("contract.state.price")
        if not signature:
            return int(self.cfg.get("contract.mint.value_wei", 0))
        return self._view_uint(signature, block)

    def anchor(self, block: str = "latest") -> str:
        """Anchor value, from the contract itself or from a recent block hash.

        `pow.anchor_source`:
          "contract" - call contract.state.anchor (default)
          "arbsys"   - ArbSys.arbBlockHash(blockNumber - pow.anchor_offset)
          "block"    - eth_getBlockByNumber(latest - pow.anchor_offset).hash
        """
        source = self.cfg.get("pow.anchor_source", "contract")
        if source == "contract":
            return self._view_bytes32(self.cfg.require("contract.state.anchor"), block)

        offset = int(self.cfg.get("pow.anchor_offset", 1))
        height = self.rpc.block_number() - offset
        if source == "arbsys":
            data = abi.calldata("arbBlockHash(uint256)", [height])
            raw = self.rpc.eth_call(ARBSYS_ADDRESS, data)
            return "0x" + raw[:32].hex()
        if source == "block":
            blk = self.rpc.call("eth_getBlockByNumber", [hex(height), False])
            return blk["hash"]
        raise SystemExit(f"unknown pow.anchor_source: {source!r}")

    def read_state(self) -> MiningState:
        block = self.rpc.block_number()
        return MiningState(
            target=self.target(),
            prev_work=self.prev_work(),
            anchor=self.anchor(),
            price_wei=self.price_wei(),
            block=block,
        )
