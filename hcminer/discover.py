"""Work out what the contract actually hashes, instead of guessing.

Two halves:

1. `fetch_abi` / `describe_abi` / `find_mint_txs` pull the verified ABI and past
   successful mints from the Blockscout explorer.
2. `solve_schema` replays one past mint through every plausible field order and
   encoding, and keeps the ones whose keccak256 lands below the difficulty target.
   A 42-zero-bit hit cannot happen by accident, so a single surviving candidate is
   the contract's real scheme.

Nothing here signs or sends anything.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from itertools import combinations, permutations
from typing import Any, Dict, List, Optional, Sequence

from .keccak import keccak256, leading_zero_bits
from .schema import Schema

MINT_NAME_HINTS = ("mine", "mint", "solve", "submit", "claim", "work")
STATE_NAME_HINTS = (
    "target", "difficulty", "bits", "price", "cost", "entry", "anchor",
    "work", "last", "tip", "head", "epoch", "nonce", "supply",
)


def _http_json(url: str, timeout: float = 25.0) -> Any:
    req = urllib.request.Request(url, headers={"accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------- explorer I/O

def fetch_abi(explorer: str, address: str) -> List[dict]:
    explorer = explorer.rstrip("/")
    try:
        data = _http_json(f"{explorer}/api/v2/smart-contracts/{address}")
        abi = data.get("abi")
        if abi:
            return abi
    except Exception:
        pass
    data = _http_json(f"{explorer}/api?module=contract&action=getabi&address={address}")
    if data.get("status") != "1":
        raise RuntimeError(f"explorer has no verified ABI for {address}: {data.get('result')}")
    return json.loads(data["result"])


def fetch_source(explorer: str, address: str) -> str:
    data = _http_json(f"{explorer.rstrip('/')}/api/v2/smart-contracts/{address}")
    return data.get("source_code") or ""


def fetch_transactions(explorer: str, address: str, pages: int = 2) -> List[dict]:
    explorer = explorer.rstrip("/")
    url = f"{explorer}/api/v2/addresses/{address}/transactions?filter=to"
    out: List[dict] = []
    for _ in range(pages):
        data = _http_json(url)
        out.extend(data.get("items", []))
        params = data.get("next_page_params")
        if not params:
            break
        query = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
        url = f"{explorer}/api/v2/addresses/{address}/transactions?filter=to&{query}"
    return out


# -------------------------------------------------------------- ABI inspection

def abi_signature(entry: dict) -> str:
    return f"{entry['name']}({','.join(i['type'] for i in entry.get('inputs', []))})"


@dataclass
class AbiReport:
    mint_candidates: List[dict] = field(default_factory=list)
    state_views: List[dict] = field(default_factory=list)
    other_views: List[dict] = field(default_factory=list)
    events: List[dict] = field(default_factory=list)

    def render(self) -> str:
        lines = ["== likely mint / submit functions (payable or nonce-taking) =="]
        for e in self.mint_candidates:
            lines.append(f"  {abi_signature(e)}   [{e.get('stateMutability')}]")
        lines.append("== state views worth wiring into config ==")
        for e in self.state_views:
            outs = ",".join(o["type"] for o in e.get("outputs", []))
            lines.append(f"  {abi_signature(e)} -> ({outs})")
        lines.append("== events ==")
        for e in self.events:
            lines.append(f"  {abi_signature(e)}")
        if self.other_views:
            lines.append(f"== {len(self.other_views)} more view functions (not shown) ==")
        return "\n".join(lines)


def describe_abi(abi: Sequence[dict]) -> AbiReport:
    report = AbiReport()
    for entry in abi:
        kind = entry.get("type")
        if kind == "event":
            report.events.append(entry)
            continue
        if kind != "function":
            continue
        name = entry.get("name", "").lower()
        mutability = entry.get("stateMutability", "")
        inputs = entry.get("inputs", [])
        takes_nonce = any(
            i["type"].startswith(("uint", "bytes32")) for i in inputs
        ) and any(h in name for h in MINT_NAME_HINTS)
        if mutability == "payable" or takes_nonce:
            report.mint_candidates.append(entry)
        elif mutability in ("view", "pure"):
            if not inputs and any(h in name for h in STATE_NAME_HINTS):
                report.state_views.append(entry)
            else:
                report.other_views.append(entry)
    return report


def decode_mint_input(entry: dict, input_hex: str) -> Dict[str, Any]:
    """Decode a transaction's calldata against one ABI function (static args only)."""
    from .abi import decode_arg, function_selector

    data = bytes.fromhex(input_hex.removeprefix("0x"))
    if data[:4] != function_selector(abi_signature(entry)):
        raise ValueError("selector mismatch")
    body = data[4:]
    out: Dict[str, Any] = {}
    for i, inp in enumerate(entry.get("inputs", [])):
        word = body[i * 32 : (i + 1) * 32]
        if len(word) < 32:
            raise ValueError("truncated calldata (dynamic arguments are not supported)")
        out[inp.get("name") or f"arg{i}"] = decode_arg(inp["type"], word)
    return out


def find_mint_txs(explorer: str, address: str, entry: dict, limit: int = 5) -> List[dict]:
    """Recent successful transactions calling `entry`, newest first, with args decoded."""
    from .abi import function_selector

    selector = "0x" + function_selector(abi_signature(entry)).hex()
    found = []
    for tx in fetch_transactions(explorer, address):
        if (tx.get("status") or tx.get("result")) not in ("ok", "success", "1", 1, True):
            continue
        raw = tx.get("raw_input") or tx.get("input") or ""
        if not raw.startswith(selector):
            continue
        try:
            args = decode_mint_input(entry, raw)
        except ValueError:
            continue
        found.append(
            {
                "hash": tx.get("hash"),
                "from": (tx.get("from") or {}).get("hash") if isinstance(tx.get("from"), dict) else tx.get("from"),
                "block": tx.get("block") or tx.get("block_number"),
                "value": tx.get("value"),
                "args": args,
            }
        )
        if len(found) >= limit:
            break
    return found


# ------------------------------------------------------------- schema recovery

@dataclass
class Observation:
    """One known-good mint: the inputs the contract hashed, and how hard it was."""

    miner: str
    nonce: int
    prev_work: Optional[str] = None
    anchor: Optional[str] = None
    epoch: Optional[int] = None
    token_id: Optional[int] = None
    chain_id: Optional[int] = None

    def values(self) -> Dict[str, Any]:
        out = {"miner": self.miner, "nonce": self.nonce}
        for key in ("prev_work", "anchor", "epoch", "token_id", "chain_id"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out


ENCODING_CHOICES = {
    "miner": ["addr20", "addr32"],
    "nonce": ["u256", "u64", "u128"],
    "prev_work": ["b32"],
    "anchor": ["b32"],
    "epoch": ["u256", "u64"],
    "token_id": ["u256", "u64"],
    "chain_id": ["u256", "u64"],
}


def candidate_schemas(sources: Sequence[str], max_fields: Optional[int] = None) -> List[Schema]:
    """Every field order and encoding worth trying, shortest layouts first."""
    sources = [s for s in sources if s in ENCODING_CHOICES]
    if "nonce" not in sources:
        raise ValueError("observations must include a nonce")
    others = [s for s in sources if s != "nonce"]
    max_fields = max_fields or len(sources)

    out: List[Schema] = []
    seen: set = set()
    for extra_count in range(0, min(len(others), max_fields - 1) + 1):
        for chosen in combinations(others, extra_count):
            for order in permutations(("nonce",) + chosen):
                encodings = [ENCODING_CHOICES[s] for s in order]
                for combo in _product(encodings):
                    specs = [f"{s}:{e}" for s, e in zip(order, combo)]
                    key = tuple(specs)
                    if key in seen:
                        continue
                    seen.add(key)
                    try:
                        out.append(Schema.parse(specs))
                    except Exception:
                        continue
    return out


def _product(lists: List[List[str]]) -> List[List[str]]:
    result: List[List[str]] = [[]]
    for options in lists:
        result = [prefix + [o] for prefix in result for o in options]
    return result


@dataclass
class Match:
    schema: Schema
    zero_bits: int
    digest: str


def solve_schema(
    observations: Sequence[Observation],
    min_zero_bits: int = 40,
    max_fields: Optional[int] = None,
) -> List[Match]:
    """Return every layout under which ALL observations satisfy the difficulty."""
    if not observations:
        raise ValueError("need at least one observation")
    sources = sorted({k for obs in observations for k in obs.values()})
    matches: List[Match] = []

    for schema in candidate_schemas(sources, max_fields):
        worst = 256
        digest = b""
        ok = True
        for obs in observations:
            try:
                digest = schema.hash(obs.values())
            except Exception:
                ok = False
                break
            bits = leading_zero_bits(digest)
            worst = min(worst, bits)
            if bits < min_zero_bits:
                ok = False
                break
        if ok:
            matches.append(Match(schema, worst, "0x" + digest.hex()))

    matches.sort(key=lambda m: -m.zero_bits)
    return matches
