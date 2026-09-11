"""One command that checks everything that can cost money if it is wrong.

Run this after `verify` and before the first real mint. Every check is read-only:
nothing is signed, nothing is broadcast.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional

from . import abi
from .chain import Chain
from .config import Config
from .rpc import Rpc, RpcError
from .schema import Schema, SchemaError

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass
class Check:
    name: str
    status: str
    detail: str


class Preflight:
    def __init__(self, cfg: Config, want_send: bool):
        self.cfg = cfg
        self.want_send = want_send   # True when limits.dry_run is off: stricter checks
        self.results: List[Check] = []
        self.rpc: Optional[Rpc] = None
        self.chain: Optional[Chain] = None
        self.schema: Optional[Schema] = None
        self.price_wei = 0

    # --- plumbing -------------------------------------------------------------

    def _run(self, name: str, fn: Callable[[], tuple]) -> str:
        try:
            status, detail = fn()
        except (RpcError, SchemaError) as exc:
            status, detail = FAIL, str(exc)
        except Exception as exc:  # a preflight must never traceback
            status, detail = FAIL, f"{type(exc).__name__}: {exc}"
        self.results.append(Check(name, status, detail))
        return status

    # --- checks ---------------------------------------------------------------

    def check_schema(self) -> tuple:
        self.schema = Schema.parse(self.cfg.schema_specs())
        if not self.cfg.get("pow.verified", False):
            return FAIL, f"{self.schema} — but pow.verified is not set; run 'verify' first"
        return PASS, f"{self.schema} (verified)"

    def check_rpc(self) -> tuple:
        self.rpc = Rpc(self.cfg.require("chain.rpc_url"), retries=1)
        live = self.rpc.chain_id()
        configured = int(self.cfg.require("chain.chain_id"))
        if live != configured:
            return FAIL, f"RPC reports chain id {live}, config says {configured}"
        return PASS, f"chain id {live}, block {self.rpc.block_number()}"

    def check_contract(self) -> tuple:
        assert self.rpc
        address = self.cfg.require("contract.address")
        code = self.rpc.call("eth_getCode", [address, "latest"])
        if not code or code in ("0x", "0x0"):
            return FAIL, f"no contract code at {address}"
        return PASS, f"{address}, {(len(code) - 2) // 2} bytes of code"

    def check_state(self) -> tuple:
        assert self.rpc
        self.chain = Chain(self.rpc, self.cfg.require("contract.address"), self.cfg)
        state = self.chain.read_state()
        self.price_wei = state.price_wei

        problems = []
        if not 8 <= state.zero_bits <= 96:
            problems.append(f"target implies {state.zero_bits} zero bits, which looks wrong")
        if int(state.prev_work, 16) == 0:
            problems.append("prev_work is zero")
        if int(state.anchor, 16) == 0:
            problems.append("anchor is zero")
        if state.price_wei == 0:
            problems.append("entry price reads as 0 — check contract.state.price")
        if problems:
            return WARN, "; ".join(problems)
        return PASS, (f"{state.zero_bits} zero bits, entry {state.price_wei / 1e18:.6f} ETH, "
                      f"anchor {state.anchor[:12]}...")

    def check_anchor_moves(self) -> tuple:
        """A stale anchor means every solution is stale by the time it lands."""
        assert self.chain
        first = self.chain.anchor()
        source = self.cfg.get("pow.anchor_source", "contract")
        return PASS, f"source={source}, current {first[:12]}... (re-read each poll)"

    def check_mint_signature(self) -> tuple:
        signature = self.cfg.require("contract.mint.signature")
        args = list(self.cfg.get("contract.mint.args", ["nonce"]))
        types = abi.signature_types(signature)
        if len(types) != len(args):
            return FAIL, f"{signature} takes {len(types)} args, contract.mint.args lists {len(args)}"
        selector = abi.function_selector(signature).hex()
        return PASS, f"{signature} -> selector 0x{selector}, args from {args}"

    def check_wallet(self) -> tuple:
        assert self.rpc
        wallet = self.cfg.require("miner.wallet_address")
        if len(wallet) != 42 or not wallet.startswith("0x"):
            return FAIL, f"miner.wallet_address is not an address: {wallet}"

        key_env = self.cfg.get("miner.private_key_env", "HC_PRIVATE_KEY")
        key = os.environ.get(key_env)
        if not key:
            status = FAIL if self.want_send else WARN
            return status, f"${key_env} is not set (needed to send transactions)"

        try:
            from eth_account import Account
        except ImportError:
            return FAIL, "eth-account is not installed: pip install -r requirements.txt"
        derived = Account.from_key(key).address
        if derived.lower() != wallet.lower():
            return FAIL, f"${key_env} holds {derived}, config says {wallet}"
        return PASS, f"{wallet} matches ${key_env}"

    def check_balance(self) -> tuple:
        assert self.rpc
        wallet = self.cfg.require("miner.wallet_address")
        balance = self.rpc.get_balance(wallet)
        gas_budget = int(self.cfg.get("limits.max_gas", 2_000_000)) * max(self.rpc.gas_price(), 1)
        needed = self.price_wei + gas_budget
        if balance == 0:
            status = FAIL if self.want_send else WARN
            return status, f"balance is 0 ETH; need ~{needed / 1e18:.6f} ETH for one mint"
        if balance < needed:
            status = FAIL if self.want_send else WARN
            return status, (f"balance {balance / 1e18:.6f} ETH is below one mint "
                            f"(~{needed / 1e18:.6f} ETH incl. gas)")
        return PASS, f"{balance / 1e18:.6f} ETH, enough for ~{balance // max(needed, 1)} mint(s)"

    def check_limits(self) -> tuple:
        dry = bool(self.cfg.get("limits.dry_run", True))
        max_spend = float(self.cfg.get("limits.max_spend_eth", 0.0))
        max_mints = int(self.cfg.get("limits.max_mints", 1))
        price_eth = self.price_wei / 1e18

        if dry:
            return PASS, f"dry_run=true — nothing will be sent (max_mints={max_mints})"
        if max_spend and max_spend < price_eth:
            return FAIL, (f"limits.max_spend_eth={max_spend} is below the entry price "
                          f"{price_eth:.6f} ETH: every mint would be blocked")
        return WARN, (f"LIVE: dry_run=false, max_mints={max_mints}, "
                      f"max_spend_eth={max_spend} (~{price_eth:.6f} ETH per mint)")

    def check_gpu(self) -> tuple:
        from .gpu import GpuMiner

        binary = self.cfg.get("miner.gpu_binary", "src/cuda/hcminer-gpu")
        if not Path(binary).exists():
            return FAIL, f"{binary} not found — build it: make -C src/cuda CUDA_ARCH=120"

        # Start it with the configured tuning, so a bad value here fails now
        # rather than at the first mining launch.
        gpu = GpuMiner(
            binary=binary,
            devices=str(self.cfg.get("miner.devices", "")),
            threads=int(self.cfg.get("miner.threads", 256)),
            blocks=int(self.cfg.get("miner.blocks", 0)),
            inner=int(self.cfg.get("miner.inner", 256)),
            streams=int(self.cfg.get("miner.streams", 4)),
            blocks_mult=int(self.cfg.get("miner.blocks_mult", 1)),
            max_kernel_ms=float(self.cfg.get("miner.max_kernel_ms", 0)),
        )
        gpu.start()
        try:
            for event in gpu.poll(timeout=10.0):
                if event.get("type") == "ready":
                    return PASS, f"{binary}, {event.get('devices')} device(s) ready"
                if event.get("type") == "error":
                    return FAIL, event.get("message", "unknown GPU error")
            return FAIL, f"{binary} did not report ready: {gpu.stderr_tail()}"
        finally:
            gpu.stop()

    # --- driver ---------------------------------------------------------------

    def run(self) -> int:
        order = [
            ("pow schema", self.check_schema),
            ("rpc", self.check_rpc),
            ("contract", self.check_contract),
            ("mining state", self.check_state),
            ("anchor", self.check_anchor_moves),
            ("mint call", self.check_mint_signature),
            ("wallet", self.check_wallet),
            ("balance", self.check_balance),
            ("limits", self.check_limits),
            ("gpu", self.check_gpu),
        ]

        stop_after_rpc = False
        for name, fn in order:
            if stop_after_rpc and name in ("contract", "mining state", "anchor", "balance"):
                self.results.append(Check(name, FAIL, "skipped: no working RPC"))
                continue
            status = self._run(name, fn)
            if name == "rpc" and status == FAIL:
                stop_after_rpc = True

        width = max(len(c.name) for c in self.results)
        for check in self.results:
            print(f"  {check.status}  {check.name.ljust(width)}  {check.detail}")

        fails = sum(1 for c in self.results if c.status == FAIL)
        warns = sum(1 for c in self.results if c.status == WARN)
        print()
        if fails:
            print(f"{fails} check(s) failed — fix them before mining.")
            return 1
        print(f"All checks passed ({warns} warning(s)). Safe to run 'hcminer mine'.")
        return 0
