#!/usr/bin/env python3
"""Works out what a proof-of-work contract actually hashes.

Give it a report from collect_protocol.py for one accepted mint. The winning
hash is almost always in the transaction's logs, and every ingredient — the
miner, the arguments, the token id — is in the transaction itself. This tries
the plausible combinations until one reproduces that hash, which settles the
preimage layout and the hash function at once.

    python3 scripts/collect_protocol.py --rpc "$RPC" --tx 0x... --out report.json
    python3 scripts/solve_layout.py --report report.json
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from keccak_pure import keccak256 as keccak_fallback  # noqa: E402

try:  # pycryptodome is ~100x faster, and the search runs hundreds of thousands of hashes
    from Crypto.Hash import keccak as _keccak

    def keccak256(data: bytes) -> bytes:
        return _keccak.new(digest_bits=256, data=data).digest()
except ImportError:  # pragma: no cover - exercised only without pycryptodome
    keccak256 = keccak_fallback

ALGORITHMS = {
    "sha256": lambda data: hashlib.sha256(data).digest(),
    "sha256d": lambda data: hashlib.sha256(hashlib.sha256(data).digest()).digest(),
    "keccak256": keccak256,
    "sha3_256": lambda data: hashlib.sha3_256(data).digest(),
}
MAX_COMPONENTS = 4
# A winning proof has a run of leading zeros, but a small integer in the same
# log (a token id, a difficulty) has far more; only the middle band is a hash.
PROOF_LIKE_ZERO_BITS = 16
MAX_PROOF_ZERO_BITS = 200


def leading_zero_bits(data: bytes) -> int:
    return len(data) * 8 - int.from_bytes(data, "big").bit_length()


def words_of(hex_data: str) -> list[bytes]:
    body = bytes.fromhex(hex_data.removeprefix("0x"))
    return [body[index:index + 32] for index in range(0, len(body) - 31, 32)]


def components(report: dict) -> list[tuple[str, bytes]]:
    """Every value the contract could plausibly be hashing, labelled."""
    transaction = report.get("transaction") or {}
    pool: list[tuple[str, bytes]] = []
    seen: set[bytes] = set()

    def offer(label: str, value: bytes) -> None:
        if value and value not in seen:
            seen.add(value)
            pool.append((label, value))

    miner = str(transaction.get("from", "")).removeprefix("0x")
    if miner:
        offer("miner20", bytes.fromhex(miner))
        offer("miner32", bytes.fromhex(miner).rjust(32, b"\x00"))
    contract = str(report.get("contract", "")).removeprefix("0x")
    if contract:
        offer("contract20", bytes.fromhex(contract))

    calldata = str(transaction.get("input", ""))
    if len(calldata) > 10:
        for index, word in enumerate(words_of("0x" + calldata[10:])):
            offer(f"arg{index}_32", word)
            offer(f"arg{index}_8", word[-8:])
            offer(f"arg{index}_20", word[-20:])

    for log in (report.get("receipt") or {}).get("logs", []):
        for index, topic in enumerate((log.get("topics") or [])[1:], start=1):
            offer(f"topic{index}_32", bytes.fromhex(str(topic).removeprefix("0x")))
        for index, word in enumerate(words_of(str(log.get("data", "0x")))):
            offer(f"logdata{index}_32", word)

    chain = report.get("chainId")
    if isinstance(chain, str):
        offer("chainId32", int(chain, 16).to_bytes(32, "big"))
    return pool


def proof_candidates(report: dict, explicit: str | None) -> list[bytes]:
    if explicit:
        return [bytes.fromhex(explicit.removeprefix("0x"))]
    found = []
    for log in (report.get("receipt") or {}).get("logs", []):
        for word in words_of(str(log.get("data", "0x"))):
            if PROOF_LIKE_ZERO_BITS <= leading_zero_bits(word) <= MAX_PROOF_ZERO_BITS:
                found.append(word)
        for topic in (log.get("topics") or [])[1:]:
            word = bytes.fromhex(str(topic).removeprefix("0x"))
            if PROOF_LIKE_ZERO_BITS <= leading_zero_bits(word) <= MAX_PROOF_ZERO_BITS:
                found.append(word)
    return found


def search(pool: list[tuple[str, bytes]], targets: list[bytes],
           max_components: int = MAX_COMPONENTS) -> list[tuple[str, tuple[str, ...], bytes]]:
    hits = []
    for size in range(1, max_components + 1):
        for order in itertools.permutations(pool, size):
            material = b"".join(value for _, value in order)
            labels = tuple(label for label, _ in order)
            for name, algorithm in ALGORITHMS.items():
                digest = algorithm(material)
                for target in targets:
                    if digest == target:
                        hits.append((name, labels, digest))
    return hits


FIELD_SIZES = {"wallet": 20, "nonce": 32, "challenge": 32}


def suggest(labels: tuple[str, ...], sizes: dict[str, int]) -> list[dict]:
    """Turn the matched component order into a protocol.json preimage."""
    fields = []
    used_nonce = False
    for label in labels:
        size = sizes[label]
        if label.startswith("miner"):
            fields.append({"field": "wallet", "size": size})
        elif label.startswith("arg") and not used_nonce:
            fields.append({"field": "nonce", "size": size})
            used_nonce = True
        elif label.startswith("arg") or label.startswith("topic") or label.startswith("logdata"):
            fields.append({"field": "challenge", "size": size})
        else:
            fields.append({"field": "const", "size": size, "note": label})
    return fields


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", required=True, help="output of collect_protocol.py")
    parser.add_argument("--hash", help="the winning proof, if it is not in the logs")
    parser.add_argument("--max-components", type=int, default=MAX_COMPONENTS)
    arguments = parser.parse_args()

    report = json.loads(Path(arguments.report).read_text(encoding="utf-8"))
    pool = components(report)
    targets = proof_candidates(report, arguments.hash)
    print(f"ingredients  {len(pool)}: " + ", ".join(label for label, _ in pool))
    if not targets:
        raise SystemExit(
            "no proof-shaped value in the logs: pass the winning hash with --hash"
        )
    print("proof(s)     " + ", ".join("0x" + target.hex() for target in targets))

    hits = search(pool, targets, arguments.max_components)
    if not hits:
        raise SystemExit(
            "no combination reproduced the proof. The contract may hash something not in this "
            "transaction (a stored challenge, a block hash); read it with --hash from a view, or "
            "raise --max-components."
        )
    sizes = {label: len(value) for label, value in pool}
    for algorithm, labels, digest in hits:
        material = " || ".join(labels)
        size = sum(sizes[label] for label in labels)
        print(f"\nMATCH  {algorithm}( {material} )  = {size} bytes")
        print(f"       digest {('0x' + digest.hex())}, "
              f"{leading_zero_bits(digest)} leading zero bits")
        print("       preimage for protocol.json:")
        print("       " + json.dumps(suggest(labels, sizes), separators=(", ", ": ")))


if __name__ == "__main__":
    main()
