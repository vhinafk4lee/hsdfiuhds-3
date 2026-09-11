"""CPU stand-in for hcminer-gpu that speaks the identical JSON protocol.

Lets the supervisor, the job encoding and the nonce reconstruction be tested on a
machine with no CUDA. It is intentionally slow — use it only with easy targets.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hcminer.keccak import keccak256

_job = {"job": None}
_lock = threading.Lock()


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def reader() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        if msg.get("cmd") == "stop":
            with _lock:
                _job["job"] = "stop"
            return
        if msg.get("cmd") == "job":
            with _lock:
                _job["job"] = msg


def main() -> int:
    emit({"type": "ready", "devices": 1})
    threading.Thread(target=reader, daemon=True).start()

    current = None
    nonce = 0
    hashes = 0
    last_report = time.time()

    while True:
        with _lock:
            job = _job["job"]
        if job == "stop":
            return 0
        if job is not current:
            current = job
            nonce = int(current["nonce_start"], 16) if current else 0
        if current is None:
            time.sleep(0.02)
            continue

        preimage = bytearray(bytes.fromhex(current["preimage"]))
        offset = current["vary_offset"]
        target = int(current["target"], 16)

        for _ in range(2000):
            preimage[offset : offset + 8] = nonce.to_bytes(8, "big")
            digest = keccak256(bytes(preimage))
            hashes += 1
            if int.from_bytes(digest, "big") < target:
                emit({"type": "solution", "job": current["id"], "device": 0,
                      "nonce": f"0x{nonce:016x}", "hash": "0x" + digest.hex()})
                with _lock:
                    if _job["job"] is current:
                        _job["job"] = None
                current = None
                break
            nonce += 1

        now = time.time()
        if now - last_report >= 2:
            emit({"type": "status", "job": (current or {}).get("id", 0),
                  "hashrate": hashes / (now - last_report), "total": hashes})
            last_report = now
            hashes = 0


if __name__ == "__main__":
    raise SystemExit(main())
