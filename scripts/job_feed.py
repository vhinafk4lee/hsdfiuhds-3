#!/usr/bin/env python3
"""Publishes the current mining job into the shared job file."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from job_state import accept_job, read_job, write_shared_job
from protocol import PROTOCOL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wallet", required=True)
    parser.add_argument("--output", default="/opt/hashbroker/job.json")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--once", action="store_true", help="print one job and exit")
    args = parser.parse_args()

    wallet = args.wallet.strip()
    if not wallet.startswith("0x") or len(wallet) != 42:
        raise SystemExit("invalid wallet address")
    if args.interval < 0.1:
        raise SystemExit("interval must be at least 0.1 seconds")

    output = Path(args.output)
    rpc_index = 0
    last_identity: tuple[str, int] | None = None
    last_job: dict | None = None
    while True:
        began = time.monotonic()
        try:
            job = read_job(wallet, preferred=rpc_index)
            if args.once:
                print(json.dumps(job, indent=2))
                return
            if not accept_job(last_job, job):
                rpc_index = (rpc_index + 1) % len(PROTOCOL.rpc)
                time.sleep(args.interval)
                continue
            write_shared_job(output, wallet, job)
            last_job = job
            identity = (str(job["prev"]), int(job["minted"]))
            if identity != last_identity:
                print("JOB_FEED", json.dumps(job, separators=(",", ":")), flush=True)
                last_identity = identity
        except Exception as exc:
            print(f"RPC_RETRY {type(exc).__name__}: {str(exc)[:240]}", flush=True)
            if args.once:
                raise SystemExit(1)
        rpc_index = (rpc_index + 1) % len(PROTOCOL.rpc)
        time.sleep(max(0.0, args.interval - (time.monotonic() - began)))


if __name__ == "__main__":
    main()
