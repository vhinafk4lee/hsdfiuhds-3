#!/usr/bin/env python3
"""Chain reads for the mining job and the shared job file on worker hosts."""
from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

import pow as powlib
from protocol import PROTOCOL

REQUIRED_JOB_FIELDS = (
    "challenge", "difficulty", "target", "priceWei", "minted", "blockNumber", "fetchedAt",
)


def request_batch(url: str, calls: list[tuple[str, list]], timeout: float = 2.0) -> list:
    payload = [dict(jsonrpc="2.0", id=index, method=method, params=params)
               for index, (method, params) in enumerate(calls, 1)]
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"content-type": "application/json", "user-agent": "hashbroker-miner/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        rows = json.load(response)
    if not isinstance(rows, list):
        raise RuntimeError("RPC response is not a batch")
    by_id = {row["id"]: row for row in rows}
    results = []
    for index in range(1, len(calls) + 1):
        row = by_id[index]
        if "result" not in row:
            raise RuntimeError(str(row.get("error", "RPC error"))[:180])
        results.append(row["result"])
    return results


def call_batch(calls: list[str], preferred: int | None = None,
               failover: bool = True) -> tuple[int, list[str]]:
    contract = PROTOCOL.require_deployed()
    urls = PROTOCOL.rpc
    if preferred is None:
        preferred = int(os.getenv("HASHBROKER_RPC_INDEX", "0"))
    preferred %= len(urls)
    errors = []
    for offset in range(len(urls) if failover else 1):
        url = urls[(preferred + offset) % len(urls)]
        try:
            chain, block = request_batch(url, [("eth_chainId", []), ("eth_blockNumber", [])])
            if int(chain, 16) != PROTOCOL.chain_id:
                raise ValueError(f"wrong chain {int(chain, 16)}")
            block_number = int(block, 16)
            results = request_batch(url, [
                ("eth_call", [{"to": contract, "data": data}, hex(block_number)])
                for data in calls
            ])
            return block_number, results
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {str(exc)[:80]}")
    raise RuntimeError("all RPCs failed: " + "; ".join(errors))


def read_job(wallet: str, preferred: int | None = None, failover: bool = True) -> dict:
    """One snapshot of everything a worker needs, read at a single block."""
    keys = ("challenge", "difficulty", "price", "minted", "maxSupply", "lastMintBlock")
    block_number, values = call_batch([PROTOCOL.view(key) for key in keys], preferred, failover)
    raw = dict(zip(keys, values))
    challenge = "0x" + raw["challenge"][2:].rjust(64, "0")
    difficulty = int(raw["difficulty"], 16)
    minted = int(raw["minted"], 16)
    max_supply = int(raw["maxSupply"], 16)
    if not 0 < difficulty <= 255:
        raise ValueError(f"implausible difficulty {difficulty}")
    if minted > max_supply:
        raise ValueError("minted exceeds max supply")
    return {
        "challenge": challenge,
        "difficulty": difficulty,
        "target": "0x%064x" % powlib.target_for_difficulty(difficulty),
        "priceWei": int(raw["price"], 16),
        "minted": minted,
        "maxSupply": max_supply,
        "lastMintBlock": int(raw["lastMintBlock"], 16),
        "blockNumber": block_number,
        "fetchedAt": time.time(),
    }


def accept_job(previous: dict | None, incoming: dict) -> bool:
    """Reject a snapshot that moves backwards or disagrees at the same block."""
    if previous is None:
        return True
    if incoming["blockNumber"] < previous["blockNumber"]:
        return False
    if incoming["minted"] < previous["minted"]:
        return False
    if incoming["blockNumber"] == previous["blockNumber"]:
        return all(incoming[key] == previous[key] for key in
                   ("challenge", "minted", "difficulty", "priceWei"))
    return True


def write_shared_job(path: Path, wallet: str, job: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"wallet": wallet, **job}, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def read_shared_job(path: Path, wallet: str, max_age: float = 5.0) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("shared job is not an object")
    if str(payload.get("wallet", "")).lower() != wallet.lower():
        raise ValueError("shared job wallet mismatch")
    fetched_at = payload.get("fetchedAt")
    if not isinstance(fetched_at, (int, float)):
        raise ValueError("shared job has no fetch timestamp")
    now = time.time()
    age = max(now - path.stat().st_mtime, now - fetched_at)
    if age > max_age or now - fetched_at < -30:
        raise ValueError(f"shared job is stale: {age:.1f}s")
    missing = [field for field in REQUIRED_JOB_FIELDS if field not in payload]
    if missing:
        raise ValueError("shared job missing fields: " + ",".join(missing))
    job = {field: payload[field] for field in REQUIRED_JOB_FIELDS}
    for optional in ("maxSupply", "lastMintBlock"):
        if optional in payload:
            job[optional] = payload[optional]
    return job
