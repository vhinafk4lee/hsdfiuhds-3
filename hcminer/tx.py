"""Builds, simulates and sends the mint transaction — with spending guards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from . import abi
from .rpc import Rpc, RpcError


class SpendGuardError(RuntimeError):
    """Raised when a transaction would break a configured limit."""


@dataclass
class Limits:
    dry_run: bool = True
    max_mints: int = 1
    max_spend_eth: float = 0.0
    max_gas_price_gwei: float = 0.0
    max_gas: int = 2_000_000

    @staticmethod
    def from_config(cfg: Any) -> "Limits":
        return Limits(
            dry_run=bool(cfg.get("limits.dry_run", True)),
            max_mints=int(cfg.get("limits.max_mints", 1)),
            max_spend_eth=float(cfg.get("limits.max_spend_eth", 0.0)),
            max_gas_price_gwei=float(cfg.get("limits.max_gas_price_gwei", 0.0)),
            max_gas=int(cfg.get("limits.max_gas", 2_000_000)),
        )


@dataclass
class Spend:
    """Running total of what this process has committed on chain."""

    mints: int = 0
    wei: int = 0

    @property
    def eth(self) -> float:
        return self.wei / 1e18


class Submitter:
    def __init__(self, rpc: Rpc, contract: str, cfg: Any, private_key: Optional[str],
                 limits: Limits):
        self.rpc = rpc
        self.contract = contract
        self.cfg = cfg
        self.limits = limits
        self.spend = Spend()
        self._account = None
        if private_key:
            try:
                from eth_account import Account
            except ImportError as exc:
                raise SystemExit(
                    "sending transactions needs eth-account: pip install -r requirements.txt"
                ) from exc
            self._account = Account.from_key(private_key)

    @property
    def address(self) -> Optional[str]:
        return self._account.address if self._account else None

    # --- calldata -------------------------------------------------------------

    def build_calldata(self, values: Dict[str, Any]) -> bytes:
        signature = self.cfg.require("contract.mint.signature")
        arg_sources: List[str] = list(self.cfg.get("contract.mint.args", ["nonce"]))
        types = abi.signature_types(signature)
        if len(types) != len(arg_sources):
            raise SystemExit(
                f"contract.mint.signature takes {len(types)} args but contract.mint.args "
                f"lists {len(arg_sources)}"
            )
        args = []
        for source in arg_sources:
            if source not in values:
                raise SystemExit(f"no value available for mint arg source '{source}'")
            args.append(values[source])
        return abi.calldata(signature, args)

    def mint_value_wei(self, price_wei: int) -> int:
        source = str(self.cfg.get("contract.mint.value", "price"))
        if source == "price":
            return price_wei
        if source in ("0", "none", ""):
            return 0
        return int(source)

    # --- guards ---------------------------------------------------------------

    def check_limits(self, value_wei: int, max_fee: int, gas: int) -> None:
        lim = self.limits
        if lim.max_mints and self.spend.mints >= lim.max_mints:
            raise SpendGuardError(
                f"limits.max_mints={lim.max_mints} already reached; stopping before spending more"
            )
        if lim.max_spend_eth:
            projected = (self.spend.wei + value_wei + max_fee * gas) / 1e18
            if projected > lim.max_spend_eth:
                raise SpendGuardError(
                    f"this mint would bring total spend to {projected:.6f} ETH, over "
                    f"limits.max_spend_eth={lim.max_spend_eth}"
                )
        if lim.max_gas_price_gwei and max_fee > lim.max_gas_price_gwei * 1e9:
            raise SpendGuardError(
                f"maxFeePerGas {max_fee / 1e9:.3f} gwei is over "
                f"limits.max_gas_price_gwei={lim.max_gas_price_gwei}"
            )
        if gas > lim.max_gas:
            raise SpendGuardError(f"gas estimate {gas} is over limits.max_gas={lim.max_gas}")

    # --- send -----------------------------------------------------------------

    def submit(self, values: Dict[str, Any], price_wei: int) -> Dict[str, Any]:
        """Simulate, guard, then (unless dry-run) sign and broadcast."""
        if not self._account:
            raise SystemExit("no private key loaded; set the env var named by miner.private_key_env")

        data = self.build_calldata(values)
        value_wei = self.mint_value_wei(price_wei)
        sender = self._account.address

        call = {
            "from": sender,
            "to": self.contract,
            "data": "0x" + data.hex(),
            "value": hex(value_wei),
        }

        # Simulate first: a revert here means the solution or the state is stale,
        # and costs nothing.
        try:
            self.rpc.call("eth_call", [call, "latest"])
        except RpcError as exc:
            return {"status": "reverted_in_simulation", "error": str(exc)}

        gas = int(self.rpc.estimate_gas(call) * 1.25)
        base = self.rpc.base_fee()
        priority = int(float(self.cfg.get("miner.priority_fee_gwei", 0.01)) * 1e9)
        max_fee = max(base * 2 + priority, self.rpc.gas_price())

        self.check_limits(value_wei, max_fee, gas)

        tx = {
            "type": 2,
            "chainId": int(self.cfg.require("chain.chain_id")),
            "to": self.contract,
            "from": sender,
            "value": value_wei,
            "data": "0x" + data.hex(),
            "gas": gas,
            "maxFeePerGas": max_fee,
            "maxPriorityFeePerGas": priority,
            "nonce": self.rpc.tx_count(sender),
        }

        if self.limits.dry_run:
            return {
                "status": "dry_run",
                "tx": {k: (hex(v) if isinstance(v, int) else v) for k, v in tx.items()},
                "value_eth": value_wei / 1e18,
                "max_cost_eth": (value_wei + max_fee * gas) / 1e18,
            }

        signed = self._account.sign_transaction(tx)
        raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
        tx_hash = self.rpc.send_raw(bytes(raw))

        self.spend.mints += 1
        self.spend.wei += value_wei + max_fee * gas
        return {"status": "sent", "hash": tx_hash, "value_eth": value_wei / 1e18}
