#!/usr/bin/env python3
"""GPU worker: searches SHA-256 proofs for the Hash Broker contract.

The worker never signs or broadcasts anything. It reads a job file published by
the controller and writes unsigned candidate solutions next to it.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from pathlib import Path

import cupy as cp
import numpy as np

import pow as powlib
from candidate_cache import CandidateCache
from job_state import read_shared_job
from protocol import PROTOCOL
from sha256_cuda import CUDA_SOURCE

MESSAGE_WORDS = 32


def message_buffer(wallet: str, prev: str, anchor: str) -> tuple[np.ndarray, int]:
    words = powlib.padded_words(wallet, prev, anchor)
    if len(words) > MESSAGE_WORDS:
        raise SystemExit(
            f"preimage of {PROTOCOL.preimage_size} bytes needs more than two SHA-256 blocks"
        )
    blocks = len(words) // 16
    padded = words + [0] * (MESSAGE_WORDS - len(words))
    return np.array(padded, dtype=np.uint32), blocks


def target_words(target: int) -> np.ndarray:
    raw = int(target).to_bytes(32, "big")
    return np.array(
        [int.from_bytes(raw[index:index + 4], "big") for index in range(0, 32, 4)],
        dtype=np.uint32,
    )


def digest_from_words(words) -> bytes:
    return b"".join(int(word).to_bytes(4, "big") for word in words)


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


def self_test(hash_one, message_gpu, blocks: int, doubled: int,
              stream_word: int, counter_word: int, wallet: str, job: dict) -> None:
    stream, counter = 0x13579BDF, 0x2468ACE0
    output = cp.zeros(8, dtype=cp.uint32)
    hash_one(
        (1,), (1,),
        (message_gpu, np.int32(blocks), np.int32(doubled),
         np.int32(stream_word), np.int32(counter_word),
         np.uint32(stream), np.uint32(counter), output),
    )
    cp.cuda.runtime.deviceSynchronize()
    actual = digest_from_words(cp.asnumpy(output))
    expected = powlib.digest(wallet, (stream << 32) | counter, job["prev"], job["anchor"])
    if actual != expected:
        raise SystemExit(f"GPU self-test failed: {actual.hex()} != {expected.hex()}")
    print("SELF_TEST_OK", actual.hex(), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wallet", required=True)
    parser.add_argument("--output", default="/opt/hashbroker/solution.json")
    parser.add_argument("--job-file", default="/opt/hashbroker/job.json")
    parser.add_argument("--blocks", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=64)
    args = parser.parse_args()

    if PROTOCOL.algorithm not in ("sha256", "sha256d"):
        raise SystemExit(f"GPU kernel supports SHA-256 only, protocol asks for {PROTOCOL.algorithm}")
    batch_hashes = args.blocks * args.threads * args.iterations
    if min(args.blocks, args.threads, args.iterations) < 1 or batch_hashes > 2**32:
        raise SystemExit("batch must contain between 1 and 2**32 unique nonces")
    if args.threads < 32:
        raise SystemExit("thread block must cover the 32-word shared message copy")

    wallet = args.wallet.strip()
    if not wallet.startswith("0x") or len(wallet) != 42:
        raise SystemExit("invalid wallet address")

    properties = cp.cuda.runtime.getDeviceProperties(0)
    print("GPU", properties["name"].decode(), flush=True)
    module = cp.RawModule(code=CUDA_SOURCE, options=("--std=c++11",))
    hash_one = module.get_function("hash_one")
    mine_batch = module.get_function("mine_batch")

    doubled = 1 if PROTOCOL.algorithm == "sha256d" else 0
    stream_word, counter_word = powlib.nonce_word_indices()
    job_file = Path(args.job_file)
    job = wait_for_shared_job(job_file, wallet)
    message, blocks = message_buffer(wallet, job["prev"], job["anchor"])
    message_gpu = cp.asarray(message)
    self_test(hash_one, message_gpu, blocks, doubled, stream_word, counter_word, wallet, job)

    stream = int.from_bytes(os.urandom(4), "big")
    counter = 0
    total_hashes = 0
    started = time.monotonic()
    last_anchor_refresh = started
    last_rate_log = started
    rate_hashes = 0
    cache = CandidateCache()
    awaiting_feed = False
    found = cp.zeros(1, dtype=cp.int32)
    found_counter = cp.zeros(1, dtype=cp.uint32)
    found_hash = cp.zeros(8, dtype=cp.uint32)
    updates: queue.Queue = queue.Queue(maxsize=1)
    threading.Thread(target=watch_jobs, args=(wallet, job_file, updates), daemon=True).start()

    print("MINING", json.dumps({"wallet": wallet, **job}, separators=(",", ":")), flush=True)
    while True:
        now = time.monotonic()
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
                if latest["prev"] != job["prev"]:
                    job = latest
                    message, blocks = message_buffer(wallet, job["prev"], job["anchor"])
                    message_gpu = cp.asarray(message)
                    last_anchor_refresh = now
                    print("JOB_UPDATED", json.dumps(job, separators=(",", ":")), flush=True)
                else:
                    for field in ("target", "priceWei", "minted", "blockNumber",
                                  "fetchedAt", "anchorWindow"):
                        if field in latest:
                            job[field] = latest[field]
                    if now - last_anchor_refresh >= 10:
                        job["anchorBlock"] = latest["anchorBlock"]
                        job["anchor"] = latest["anchor"]
                        message, blocks = message_buffer(wallet, job["prev"], job["anchor"])
                        message_gpu = cp.asarray(message)
                        last_anchor_refresh = now
                        print("ANCHOR_REFRESHED", job["anchorBlock"], flush=True)
        if awaiting_feed:
            time.sleep(0.05)
            continue

        solution = cache.ready(job)
        if solution is not None:
            output = Path(args.output)
            temporary = output.with_suffix(".tmp")
            temporary.write_text(json.dumps(solution, indent=2) + "\n")
            temporary.replace(output)
            print("SOLUTION", json.dumps(solution, separators=(",", ":")), flush=True)
            return

        if counter + batch_hashes > 2**32:
            stream = (stream + 1) & 0xFFFFFFFF
            counter = 0

        candidate_target = powlib.search_target(job["target"])
        found.fill(0)
        mine_batch(
            (args.blocks,), (args.threads,),
            (message_gpu, np.int32(blocks), np.int32(doubled),
             np.int32(stream_word), np.int32(counter_word),
             np.uint32(stream), np.uint32(counter), np.uint32(args.iterations),
             cp.asarray(target_words(candidate_target)),
             found, found_counter, found_hash),
        )
        cp.cuda.runtime.deviceSynchronize()
        total_hashes += batch_hashes
        rate_hashes += batch_hashes
        counter += batch_hashes

        if int(found.get()[0]):
            nonce = (stream << 32) | int(found_counter.get()[0])
            digest = powlib.digest(wallet, nonce, job["prev"], job["anchor"])
            reported = digest_from_words(cp.asnumpy(found_hash))
            if reported != digest or int.from_bytes(digest, "big") >= candidate_target:
                raise RuntimeError("GPU candidate failed CPU verification")
            if cache.remember({
                "wallet": wallet,
                "nonce": str(nonce),
                "hash": "0x" + digest.hex(),
                **job,
                "foundAt": int(time.time()),
            }):
                print("CANDIDATE_CACHED", json.dumps({
                    "hash": "0x" + digest.hex(), "anchorBlock": job["anchorBlock"],
                    "minted": job["minted"], "anchors": len(cache),
                }, separators=(",", ":")), flush=True)

        now = time.monotonic()
        if now - last_rate_log >= 1.0:
            rate = rate_hashes / max(now - last_rate_log, 1e-9)
            print(
                f"RATE {rate / 1e6:.2f} MH/s total={total_hashes} "
                f"runtime={now - started:.1f}s minted={job['minted']}",
                flush=True,
            )
            last_rate_log = now
            rate_hashes = 0


if __name__ == "__main__":
    main()
