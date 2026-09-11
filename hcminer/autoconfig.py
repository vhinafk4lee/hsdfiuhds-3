"""Work out the whole contract wiring automatically, from chain data alone.

`hcminer autoconfig` replaces the manual discover -> observations.json -> solve-schema
-> verify sequence. It takes recent accepted mints and searches for the combination of

  * field layout inside keccak256,
  * which getter supplies the previous work,
  * which value is the anchor,
  * which getter is the target and which is the entry price,
  * which function and argument carry the nonce,

that reproduces those mints' difficulty. A hash landing below a 40+ zero-bit target
cannot happen by chance, so a combination that works for two independent mints is the
contract's real scheme.

Deliberately avoids needing an archive node: candidate values are collected from
current getters, from the logs of the mint transactions themselves, and from the block
hashes around them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import abi
from .discover import (Observation, abi_signature, decode_mint_input, describe_abi,
                       fetch_abi, find_mint_txs)
from .keccak import leading_zero_bits
from .rpc import Rpc, RpcError
from .schema import Schema

MAX_POOL = 64          # candidate bytes32 values kept per mint
BLOCK_LOOKBACK = 24    # block hashes before the mint to try as the anchor


@dataclass
class MintSample:
    """One accepted mint plus everything that might have gone into its hash."""

    tx_hash: str
    miner: str
    block: int
    value_wei: int
    nonce_candidates: List[int] = field(default_factory=list)
    words: Dict[str, str] = field(default_factory=dict)   # label -> 0x.. 32 bytes
    views_uint: Dict[str, int] = field(default_factory=dict)
    views_bytes32: Dict[str, str] = field(default_factory=dict)


@dataclass
class Resolution:
    schema: Schema
    prev_work_label: str
    anchor_label: str
    nonce_arg_index: int
    mint_signature: str
    zero_bits: int
    target_signature: Optional[str] = None
    price_signature: Optional[str] = None
    target_is_bits: bool = False
    anchor_is_block_hash: bool = False
    anchor_block_offset: int = 0

    def describe(self) -> str:
        lines = [
            f"  layout        {self.schema}",
            f"  prev_work     {self.prev_work_label}",
            f"  anchor        {self.anchor_label}",
            f"  mint          {self.mint_signature}, nonce is argument #{self.nonce_arg_index}",
            f"  difficulty    {self.zero_bits} zero bits on the samples",
        ]
        if self.target_signature:
            kind = "zero-bit count" if self.target_is_bits else "256-bit threshold"
            lines.append(f"  target        {self.target_signature} ({kind})")
        if self.price_signature:
            lines.append(f"  price         {self.price_signature}")
        if self.anchor_is_block_hash:
            lines.append(f"  anchor source block hash at height - {self.anchor_block_offset}")
        return "\n".join(lines)


# --------------------------------------------------------------- gathering data

def _words_from_hex(blob: str) -> List[str]:
    raw = bytes.fromhex(blob.removeprefix("0x"))
    return ["0x" + raw[i : i + 32].hex() for i in range(0, len(raw) - 31, 32)]


def collect_sample(rpc: Rpc, contract: str, tx: Dict[str, Any], mint_entry: dict,
                   views: Sequence[dict]) -> MintSample:
    """Everything that could plausibly be an input to this mint's hash."""
    receipt = rpc.get_receipt(tx["hash"]) or {}
    block = int(receipt.get("blockNumber", "0x0"), 16) if receipt else 0

    sample = MintSample(
        tx_hash=tx["hash"],
        miner=(tx.get("from") or "").lower(),
        block=block,
        value_wei=int(tx.get("value") or 0),
    )

    for name, value in (tx.get("args") or {}).items():
        if isinstance(value, int) and value > 0:
            sample.nonce_candidates.append(value)
        elif isinstance(value, str) and value.startswith("0x") and len(value) == 66:
            sample.nonce_candidates.append(int(value, 16))
            sample.words[f"mintarg:{name}"] = value

    # 32-byte words out of this mint's own logs: the new work, and often the
    # previous one and the anchor alongside it.
    for i, log in enumerate(receipt.get("logs", [])):
        for j, topic in enumerate(log.get("topics", [])[1:]):
            if len(topic) == 66:
                sample.words[f"log{i}.topic{j + 1}"] = topic
        for j, word in enumerate(_words_from_hex(log.get("data", "0x"))):
            sample.words[f"log{i}.data{j}"] = word

    # Block hashes around the mint: an anchor taken from a recent block lives here.
    for back in range(1, BLOCK_LOOKBACK + 1):
        height = block - back
        if height <= 0:
            break
        try:
            blk = rpc.call("eth_getBlockByNumber", [hex(height), False])
        except RpcError:
            break
        if blk and blk.get("hash"):
            sample.words[f"blockhash-{back}"] = blk["hash"]

    # Current values of every zero-argument getter.
    for entry in views:
        signature = abi_signature(entry)
        out_type = entry["outputs"][0]["type"]
        try:
            raw = rpc.eth_call(contract, abi.calldata(signature, []))
        except RpcError:
            continue
        if len(raw) < 32:
            continue
        if out_type == "bytes32":
            value = "0x" + raw[:32].hex()
            sample.views_bytes32[signature] = value
            sample.words[f"view:{signature}"] = value
        elif out_type.startswith("uint"):
            sample.views_uint[signature] = int.from_bytes(raw[:32], "big")

    return sample


def state_views(abi_entries: Sequence[dict]) -> List[dict]:
    out = []
    for entry in abi_entries:
        if entry.get("type") != "function":
            continue
        if entry.get("stateMutability") not in ("view", "pure"):
            continue
        if entry.get("inputs"):
            continue
        outputs = entry.get("outputs") or []
        if len(outputs) != 1:
            continue
        if outputs[0]["type"] == "bytes32" or outputs[0]["type"].startswith("uint"):
            out.append(entry)
    return out


# ------------------------------------------------------------------- resolution

def _candidate_layouts() -> List[Schema]:
    """Four-field layouts: miner, nonce and two 32-byte values, in any order."""
    from itertools import permutations

    layouts: List[Schema] = []
    seen = set()
    for order in permutations(("miner", "nonce", "prev_work", "anchor")):
        for miner_enc in ("addr20", "addr32"):
            for nonce_enc in ("u256", "u64"):
                specs = []
                for source in order:
                    if source == "miner":
                        specs.append(f"miner:{miner_enc}")
                    elif source == "nonce":
                        specs.append(f"nonce:{nonce_enc}")
                    else:
                        specs.append(f"{source}:b32")
                key = tuple(specs)
                if key in seen:
                    continue
                seen.add(key)
                layouts.append(Schema.parse(specs))
    return layouts


def resolve(samples: Sequence[MintSample], mint_entry: dict, min_zero_bits: int = 40,
            max_pool: int = MAX_POOL) -> List[Resolution]:
    """Find every wiring that reproduces the difficulty of ALL samples."""
    if not samples:
        return []

    layouts = _candidate_layouts()
    first = samples[0]
    labels = list(first.words)[:max_pool]
    results: List[Resolution] = []
    seen: set = set()   # the same byte layout reached by swapping the two 32-byte roles

    for nonce_index, nonce in enumerate(first.nonce_candidates):
        for schema in layouts:
            for prev_label in labels:
                for anchor_label in labels:
                    if prev_label == anchor_label:
                        continue
                    values = {
                        "miner": first.miner,
                        "nonce": nonce,
                        "prev_work": first.words[prev_label],
                        "anchor": first.words[anchor_label],
                    }
                    try:
                        digest = schema.hash(values)
                    except Exception:
                        continue
                    bits = leading_zero_bits(digest)
                    if bits < min_zero_bits:
                        continue

                    # Confirm on the remaining samples, where the same labels must
                    # keep working with their own values.
                    worst = bits
                    ok = True
                    for other in samples[1:]:
                        if (prev_label not in other.words or anchor_label not in other.words
                                or nonce_index >= len(other.nonce_candidates)):
                            ok = False
                            break
                        other_values = {
                            "miner": other.miner,
                            "nonce": other.nonce_candidates[nonce_index],
                            "prev_work": other.words[prev_label],
                            "anchor": other.words[anchor_label],
                        }
                        other_bits = leading_zero_bits(schema.hash(other_values))
                        worst = min(worst, other_bits)
                        if other_bits < min_zero_bits:
                            ok = False
                            break
                    if not ok:
                        continue

                    canonical = tuple(
                        {"prev_work:b32": prev_label, "anchor:b32": anchor_label}.get(spec, spec)
                        for spec in schema.specs()
                    )
                    if canonical in seen:
                        continue
                    seen.add(canonical)

                    results.append(Resolution(
                        schema=schema,
                        prev_work_label=prev_label,
                        anchor_label=anchor_label,
                        nonce_arg_index=nonce_index,
                        mint_signature=abi_signature(mint_entry),
                        zero_bits=worst,
                        anchor_is_block_hash=anchor_label.startswith("blockhash-"),
                        anchor_block_offset=(int(anchor_label.split("-")[1])
                                             if anchor_label.startswith("blockhash-") else 0),
                    ))
    return results


def identify_target_and_price(sample: MintSample, zero_bits: int,
                              resolution: Resolution) -> None:
    """Pick the getters that behave like the difficulty target and the entry price."""
    for signature, value in sample.views_uint.items():
        if value == sample.value_wei and sample.value_wei > 0 and not resolution.price_signature:
            resolution.price_signature = signature

    best_threshold: Optional[Tuple[str, int]] = None
    for signature, value in sample.views_uint.items():
        if 8 <= value <= 96:                       # a zero-bit count
            if abs(value - zero_bits) <= 2 and not resolution.target_signature:
                resolution.target_signature = signature
                resolution.target_is_bits = True
        elif 0 < value < (1 << 250):               # a 256-bit threshold
            bits = 256 - value.bit_length()
            if abs(bits - zero_bits) <= 2 and (best_threshold is None
                                               or value < best_threshold[1]):
                best_threshold = (signature, value)
    if best_threshold and not resolution.target_signature:
        resolution.target_signature = best_threshold[0]
        resolution.target_is_bits = False


# ---------------------------------------------------------------- config output

def render_config(address: str, rpc_url: str, chain_id: int, explorer: str,
                  resolution: Resolution, wallet: str = "") -> str:
    anchor_source = "block" if resolution.anchor_is_block_hash else "contract"
    anchor_view = ("" if resolution.anchor_is_block_hash
                   else resolution.anchor_label.removeprefix("view:"))
    prev_view = resolution.prev_work_label.removeprefix("view:")

    return f"""# Written by `hcminer autoconfig` from live chain data.
# Every value below was confirmed against real accepted mints.

[chain]
rpc_url  = "{rpc_url}"
chain_id = {chain_id}
explorer = "{explorer}"

[contract]
address = "{address}"

[contract.state]
target    = "{resolution.target_signature or 'REPLACE_ME()'}"
prev_work = "{prev_view}"
anchor    = "{anchor_view or 'unused when anchor_source is not contract'}"
price     = "{resolution.price_signature or 'REPLACE_ME()'}"

[contract.mint]
signature = "{resolution.mint_signature}"
args      = ["nonce"]
value     = "price"

[pow]
schema         = {list(resolution.schema.specs())!r}
verified       = true
target_is_bits = {str(resolution.target_is_bits).lower()}
anchor_source  = "{anchor_source}"
anchor_offset  = {resolution.anchor_block_offset or 1}

[miner]
wallet_address   = "{wallet or '0x0000000000000000000000000000000000000000'}"
private_key_env  = "HC_PRIVATE_KEY"
gpu_binary       = "src/cuda/hcminer-gpu"
devices          = "0"
threads          = 256
blocks           = 0
blocks_mult      = 1
inner            = 256
streams          = 4
max_kernel_ms    = 400
poll_seconds     = 5
priority_fee_gwei = 0.01

[limits]
dry_run            = true
max_mints          = 1
max_spend_eth      = 0.05
max_gas_price_gwei = 5
max_gas            = 2000000
""".replace("'", '"')
