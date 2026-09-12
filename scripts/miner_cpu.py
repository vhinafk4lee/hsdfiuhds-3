#!/usr/bin/env python3
"""CPU worker: same job and candidate flow as the GPU worker, without CUDA.

It is far too slow to compete for a mint, but it validates a deployment end to
end (job feed -> search -> candidate file -> signer) on any host.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import pow as powlib
from candidate_cache import CandidateCache
from job_state import read_shared_job
from protocol import PROTOCOL

CHUNK = 20_000


def search(wallet: str, job_file: Path, results: mp.Queue, stop: mp.Event) -> None:
    stream = int.from_bytes(os.urandom(4), "big")
    counter = 0
    job: dict | None = None
    refreshed = 0.0
    hashed = 0
    while not stop.is_set():
        now = time.monotonic()
        if job is None or now - refreshed > 1.0:
            try:
                latest = read_shared_job(job_file, wallet)
            except Exception as exc:
                results.put(("error", str(exc)))
                time.sleep(0.5)
                continue
            refreshed = now
            if job is None or latest["challenge"] != job["challenge"]:
                counter = 0
                stream = int.from_bytes(os.urandom(4), "big")
            job = latest
        target = powlib.search_target(job["target"])
        challenge = job["challenge"]
        for _ in range(CHUNK):
            nonce = (stream << 32) | counter
            counter = (counter + 1) & 0xFFFFFFFF
            if counter == 0:
                stream = (stream + 1) & 0xFFFFFFFF
            digest = powlib.digest(wallet, nonce, challenge)
            if int.from_bytes(digest, "big") < target:
                results.put(("candidate", {
                    **job, "wallet": wallet, "nonce": str(nonce),
                    "hash": "0x" + digest.hex(), "challenge": challenge,
                    "difficulty": job["difficulty"], "foundAt": int(time.time()),
                }))
        hashed += CHUNK
        results.put(("rate", hashed))
        hashed = 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wallet", required=True)
    parser.add_argument("--output", default="/opt/hashbroker/solution.json")
    parser.add_argument("--job-file", default="/opt/hashbroker/job.json")
    parser.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--keep-mining", action="store_true",
                        help="keep searching after a solution instead of exiting")
    args = parser.parse_args()

    wallet = args.wallet.strip()
    if not wallet.startswith("0x") or len(wallet) != 42:
        raise SystemExit("invalid wallet address")
    print(f"CPU_MINER processes={args.processes} algorithm={PROTOCOL.algorithm}", flush=True)

    job_file = Path(args.job_file)
    results: mp.Queue = mp.Queue()
    stop = mp.Event()
    workers = [mp.Process(target=search, args=(wallet, job_file, results, stop), daemon=True)
               for _ in range(args.processes)]
    for worker in workers:
        worker.start()

    cache = CandidateCache()
    hashed = 0
    last_log = time.monotonic()
    try:
        while True:
            kind, payload = results.get()
            if kind == "rate":
                hashed += int(payload)
            elif kind == "error":
                print("JOB_FEED_WAIT", str(payload)[:160], flush=True)
                continue
            elif kind == "candidate":
                if cache.remember(payload):
                    print("CANDIDATE_CACHED", payload["hash"], flush=True)
                solution = cache.ready(payload)
                if solution is not None:
                    output = Path(args.output)
                    temporary = output.with_suffix(".tmp")
                    temporary.write_text(json.dumps(solution, indent=2) + "\n")
                    temporary.replace(output)
                    print("SOLUTION", json.dumps(solution, separators=(",", ":")), flush=True)
                    if not args.keep_mining:
                        return
                    cache = CandidateCache()
            now = time.monotonic()
            if now - last_log >= 5.0:
                print(f"RATE {hashed / (now - last_log) / 1e3:.1f} kH/s", flush=True)
                hashed = 0
                last_log = now
    finally:
        stop.set()
        for worker in workers:
            worker.join(timeout=1)
            if worker.is_alive():
                worker.terminate()


if __name__ == "__main__":
    main()
