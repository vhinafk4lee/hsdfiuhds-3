#!/usr/bin/env python3
"""Samples the contract over time: difficulty, challenge, price, supply.

Hash Broker exposes lastMintBlock(), which suggests the difficulty is retargeted
between mints. How fast it moves decides how much hashrate is worth renting, so
measure it before renting anything. Writes one JSON row per sample.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from job_state import read_job

ZERO_WALLET = "0x0000000000000000000000000000000000000000"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wallet", default=ZERO_WALLET)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--duration", type=float, default=0.0, help="0 runs until stopped")
    parser.add_argument("--out", default="state-log.jsonl")
    arguments = parser.parse_args()

    output = Path(arguments.out)
    deadline = time.monotonic() + arguments.duration if arguments.duration else None
    previous: dict | None = None
    print(f"{'block':>10} {'diff':>5} {'minted':>6} {'priceETH':>10}  {'sinceMint':>9}  challenge")
    while deadline is None or time.monotonic() < deadline:
        began = time.monotonic()
        try:
            job = read_job(arguments.wallet)
        except Exception as exc:
            print(f"RPC_RETRY {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            time.sleep(arguments.interval)
            continue
        since_mint = job["blockNumber"] - job.get("lastMintBlock", job["blockNumber"])
        marker = ""
        if previous is not None:
            if job["challenge"] != previous["challenge"]:
                marker += "  <- new challenge"
            if job["difficulty"] != previous["difficulty"]:
                marker += f"  <- difficulty {previous['difficulty']} -> {job['difficulty']}"
        print(f"{job['blockNumber']:>10} {job['difficulty']:>5} {job['minted']:>6} "
              f"{job['priceWei'] / 1e18:>10.6f}  {since_mint:>9}  "
              f"{job['challenge'][:18]}...{marker}", flush=True)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(job, separators=(",", ":")) + "\n")
        previous = job
        time.sleep(max(0.0, arguments.interval - (time.monotonic() - began)))


if __name__ == "__main__":
    main()
