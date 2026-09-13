#!/usr/bin/env python3
"""GPU worker: searches the protocol's proofs on the device.

The worker never signs or broadcasts anything. It reads the job file published
by the feed and writes an unsigned solution file for the signer to pick up.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from pathlib import Path

import pow as powlib
from candidate_cache import CandidateCache
from gpu import Kernel
from job_state import read_shared_job
from protocol import PROTOCOL


def watch_jobs(wallet: str, job_file: Path, updates: queue.Queue) -> None:
    last_mtime = None
    had_error = False
    while True:
        try:
            mtime = job_file.stat().st_mtime_ns
            if mtime == last_mtime and not had_error:
                time.sleep(0.05)
                continue
            update = (read_shared_job(job_file, wallet), None)
            last_mtime = mtime
            had_error = False
        except Exception as exc:  # the feed may be mid-write or stalled
            update = (None, exc)
            had_error = True
        try:
            updates.put_nowait(update)
        except queue.Full:
            try:
                updates.get_nowait()
            except queue.Empty:
                pass
            updates.put_nowait(update)
        time.sleep(0.05)


def wait_for_shared_job(job_file: Path, wallet: str, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return read_shared_job(job_file, wallet)
        except Exception as exc:
            error = exc
            time.sleep(0.1)
    raise SystemExit(f"shared job unavailable: {error}")


def write_solution(path: Path, solution: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(solution, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wallet", required=True)
    parser.add_argument("--output", default="/opt/hashbroker/solution.json")
    parser.add_argument("--job-file", default="/opt/hashbroker/job.json")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--blocks", type=int,
                        default=int(os.environ.get("HASHBROKER_BLOCKS", "8192")))
    parser.add_argument("--threads", type=int,
                        default=int(os.environ.get("HASHBROKER_THREADS", "256")))
    parser.add_argument("--iterations", type=int,
                        default=int(os.environ.get("HASHBROKER_ITERATIONS", "64")))
    parser.add_argument("--keep-mining", action="store_true",
                        help="keep searching after a solution instead of exiting")
    args = parser.parse_args()

    batch_hashes = args.blocks * args.threads * args.iterations
    if min(args.blocks, args.threads, args.iterations) < 1 or batch_hashes > 2**32:
        raise SystemExit("batch must contain between 1 and 2**32 unique nonces")
    if args.threads < 32:
        raise SystemExit("thread block must cover the 32-word shared message copy")

    wallet = args.wallet.strip()
    if not wallet.startswith("0x") or len(wallet) != 42:
        raise SystemExit("invalid wallet address")

    kernel = Kernel(args.device)
    print(f"GPU {args.device} {kernel.name} algorithm={PROTOCOL.algorithm}", flush=True)

    job_file = Path(args.job_file)
    job = wait_for_shared_job(job_file, wallet)
    kernel.bind(wallet, job["bindings"])
    print("SELF_TEST_OK", kernel.self_test(wallet).hex(), flush=True)

    stream = int.from_bytes(os.urandom(4), "big")
    counter = 0
    total_hashes = 0
    started = time.monotonic()
    last_rate_log = started
    rate_hashes = 0
    cache = CandidateCache()
    awaiting_feed = False
    updates: queue.Queue = queue.Queue(maxsize=1)
    threading.Thread(target=watch_jobs, args=(wallet, job_file, updates), daemon=True).start()

    print("MINING", json.dumps({"wallet": wallet, **job}, separators=(",", ":")), flush=True)
    while True:
        try:
            latest, job_error = updates.get_nowait()
        except queue.Empty:
            pass
        else:
            if job_error is not None:
                if not awaiting_feed:
                    print("JOB_FEED_WAIT", type(job_error).__name__, str(job_error), flush=True)
                awaiting_feed = True
            elif latest is not None:
                if awaiting_feed:
                    print("JOB_FEED_RESUMED", flush=True)
                awaiting_feed = False
                if latest["challenge"] != job["challenge"]:
                    job = latest
                    kernel.bind(wallet, job["bindings"])
                    stream = int.from_bytes(os.urandom(4), "big")
                    counter = 0
                    print("JOB_CHANGED",
                          json.dumps(job, separators=(",", ":")), flush=True)
                else:
                    job = {**job, **latest}
        if awaiting_feed:
            time.sleep(0.05)
            continue

        solution = cache.ready(job)
        if solution is not None:
            write_solution(Path(args.output), solution)
            print("SOLUTION", json.dumps(solution, separators=(",", ":")), flush=True)
            if not args.keep_mining:
                return
            cache = CandidateCache()

        if counter + batch_hashes > 2**32:
            stream = (stream + 1) & 0xFFFFFFFF
            counter = 0

        candidate_target = powlib.search_target(job["target"])
        hit = kernel.search(args.blocks, args.threads, stream, counter,
                            args.iterations, candidate_target)
        total_hashes += batch_hashes
        rate_hashes += batch_hashes
        counter += batch_hashes

        if hit is not None:
            found_counter, reported = hit
            nonce = (stream << 32) | found_counter
            digest = powlib.digest({"wallet": wallet, "nonce": nonce, **job["bindings"]})
            if reported != digest or int.from_bytes(digest, "big") >= candidate_target:
                raise RuntimeError("GPU candidate failed CPU verification")
            if cache.remember({
                "wallet": wallet,
                "nonce": str(nonce),
                "hash": "0x" + digest.hex(),
                "challenge": job["challenge"],
                "difficulty": job["difficulty"],
                "foundAt": int(time.time()),
            }):
                print("CANDIDATE", json.dumps({
                    "hash": "0x" + digest.hex(),
                    "zeroBits": powlib.leading_zero_bits(digest),
                    "difficulty": job["difficulty"],
                }, separators=(",", ":")), flush=True)

        now = time.monotonic()
        if now - last_rate_log >= 1.0:
            rate = rate_hashes / max(now - last_rate_log, 1e-9)
            print(f"RATE {rate / 1e6:.2f} MH/s total={total_hashes} "
                  f"runtime={now - started:.1f}s difficulty={job['difficulty']} "
                  f"minted={job['minted']}/{job.get('maxSupply', '?')}", flush=True)
            last_rate_log = now
            rate_hashes = 0


if __name__ == "__main__":
    main()
