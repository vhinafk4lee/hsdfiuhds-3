#!/usr/bin/env python3
"""Turns one accepted mint into a working protocol file.

Point it at an RPC and a mint transaction of a contract the miner has never seen,
and it reads the chain, recovers the contract's selectors, works out what it
hashes, and writes scripts/protocols/<name>.json. After that the miner can mine
it with HASHBROKER_PROTOCOL=<name>.

    python3 scripts/onboard.py --name flynode --rpc "$RPC" --tx 0x...
    python3 scripts/onboard.py --name flynode --report report.json   # already collected
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import collect_protocol
import solve_layout
from protocol import PROTOCOL_DIR

# Which view the miner needs -> the signatures that tend to provide it.
VIEW_ALIASES = {
    "challenge": ("challenge()", "currentChallenge()", "getChallenge()", "lastHash()",
                  "lastWork()", "prevWork()", "seed()"),
    "difficulty": ("currentDifficulty()", "difficulty()", "getDifficulty()"),
    "price": ("mintPrice()", "price()", "currentPrice()", "cost()", "mintCost()"),
    "minted": ("totalSupply()", "totalMinted()", "minted()"),
    "maxSupply": ("MAX_SUPPLY()", "maxSupply()"),
    "lastMintBlock": ("lastMintBlock()", "lastMintTime()", "startBlock()"),
}
TARGET_VIEWS = ("target()", "currentTarget()", "getTarget()")


def pick_views(matched: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    views, missing = {}, []
    for key, candidates in VIEW_ALIASES.items():
        for signature in candidates:
            if signature in matched:
                views[key] = signature
                break
        else:
            missing.append(key)
    return views, missing


def build(report: dict, name: str, proof: str | None) -> tuple[dict, list[str]]:
    pool = solve_layout.components(report)
    targets = solve_layout.proof_candidates(report, proof)
    if not targets:
        raise SystemExit("no proof-shaped value in the mint's logs: pass --hash")
    hits = solve_layout.search(pool, targets)
    if not hits:
        raise SystemExit(
            "could not reproduce the proof from this transaction. The contract may hash a value "
            "it stores rather than one passed in; read that value and pass it with --hash."
        )
    algorithm, labels, _ = hits[0]
    sizes = {label: len(value) for label, value in pool}
    preimage = solve_layout.suggest(labels, sizes)

    mint = report.get("mintCall") or {}
    matched_views = report.get("matchedViews") or {}
    views, missing = pick_views(matched_views)
    # The searched nonce must be an argument of the call, or we cannot submit it.
    mine_args = ["nonce" if field["field"] == "nonce" else "challenge"
                 for field in preimage if field["field"] in ("nonce", "challenge")]

    protocol = {
        "name": name,
        "chainId": int(str(report.get("chainId", "0x0")), 16),
        "contract": report.get("contract", ""),
        "algorithm": algorithm,
        "preimage": preimage,
        "mine": mint.get("signature") or "",
        "mineSelector": mint.get("selector", ""),
        "mineArgs": mine_args,
        "views": views,
        "validate": next((signature for signature in matched_views
                          if signature.startswith("isValidProof")), ""),
        "rpc": [],
        "broadcastRpc": [],
        "verified": False,
    }
    warnings = []
    if missing:
        warnings.append("no view found for: " + ", ".join(missing)
                        + " — fill these in by hand from the selector list")
    if any(signature in matched_views for signature in TARGET_VIEWS):
        warnings.append("this contract exposes a target() view; if it is not a leading-zero-bit "
                        "difficulty, the job reader needs adjusting")
    if not protocol["mine"]:
        warnings.append("the mine() signature is unknown; the raw selector is used instead")
    if not protocol["contract"]:
        warnings.append("no contract address in the report")
    return protocol, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="short name, e.g. flynode")
    parser.add_argument("--rpc", help="RPC endpoint, when collecting now")
    parser.add_argument("--tx", help="an accepted mint transaction")
    parser.add_argument("--wallet", help="a wallet, for per-wallet views")
    parser.add_argument("--report", help="a report from collect_protocol.py instead of --rpc/--tx")
    parser.add_argument("--hash", help="the winning proof, if it is not in the logs")
    parser.add_argument("--out", help="where to write (default scripts/protocols/<name>.json)")
    parser.add_argument("--rpc-url", action="append", default=[],
                        help="RPC endpoints to record in the protocol file; repeatable")
    arguments = parser.parse_args()

    if arguments.report:
        report = json.loads(Path(arguments.report).read_text(encoding="utf-8"))
    elif arguments.rpc and arguments.tx:
        report = collect_protocol.collect(collect_protocol.Rpc(arguments.rpc),
                                          arguments.tx, None, arguments.wallet)
        Path(f"{arguments.name}-report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"collected {arguments.name}-report.json")
    else:
        raise SystemExit("pass --report, or --rpc together with --tx")

    protocol, warnings = build(report, arguments.name, arguments.hash)
    protocol["rpc"] = arguments.rpc_url or ([arguments.rpc] if arguments.rpc else [])
    path = Path(arguments.out) if arguments.out else PROTOCOL_DIR / f"{arguments.name}.json"
    path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    fields = " | ".join(f"{field['field']}[{field['size']}]" for field in protocol["preimage"])
    print(f"\nwrote {path}")
    print(f"  contract   {protocol['contract']}  (chain {protocol['chainId']})")
    print(f"  proof      {protocol['algorithm']}( {fields} )")
    print(f"  mine       {protocol['mine'] or protocol['mineSelector']}")
    print(f"  views      " + ", ".join(f"{key}={value}" for key, value in protocol["views"].items()))
    for warning in warnings:
        print(f"  WARNING    {warning}")
    print(f"\nCheck it, then: HASHBROKER_PROTOCOL={arguments.name} python3 scripts/protocol.py")


if __name__ == "__main__":
    main()
