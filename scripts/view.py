#!/usr/bin/env python3
"""Calls any read-only function on the protocol's contract.

    python3 scripts/view.py --protocol flynode "neuronsRoot()"
    python3 scripts/view.py --protocol flynode "requiredBits(uint8,address)" 4 0xYourWallet

Arguments are encoded as single words: integers as uint256, 0x-addresses and
0x-bytes32 right-padded into a word. That covers every view these contracts have.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from job_state import request_batch  # noqa: E402
from keccak_pure import selector  # noqa: E402
from protocol import load  # noqa: E402


def encode_argument(value: str) -> str:
    text = value.strip()
    if text.startswith("0x"):
        body = text[2:].lower()
        if len(body) > 64:
            raise SystemExit(f"argument too wide for one word: {text}")
        return body.rjust(64, "0")
    return f"{int(text):064x}"


def decode(raw: str) -> str:
    body = raw.removeprefix("0x")
    lines = []
    for index in range(0, len(body), 64):
        word = body[index:index + 64]
        value = int(word, 16)
        reading = f"uint {value}" if value < 2**64 else ""
        if 2**96 <= value < 2**160:
            reading = f"address 0x{word[-40:]}"
        lines.append(f"  [{index // 64}] 0x{word}  {reading}")
    return "\n".join(lines) or "  (empty)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("signature", help='e.g. "prevWork()"')
    parser.add_argument("arguments", nargs="*", help="one word each")
    parser.add_argument("--protocol", default=os.environ.get("HASHBROKER_PROTOCOL", "hashbroker"))
    parser.add_argument("--rpc", help="override the protocol's endpoints")
    parser.add_argument("--block", default="latest")
    options = parser.parse_args()

    protocol = load() if os.environ.get("HASHBROKER_PROTOCOL_FILE") else None
    if protocol is None:
        os.environ["HASHBROKER_PROTOCOL"] = options.protocol
        protocol = load()
    url = options.rpc or (protocol.rpc[0] if protocol.rpc else "")
    if not url:
        raise SystemExit("no RPC endpoint: pass --rpc")

    data = "0x" + selector(options.signature) + "".join(
        encode_argument(value) for value in options.arguments)
    result = request_batch(url, [("eth_call", [
        {"to": protocol.require_deployed(), "data": data}, options.block])])[0]
    print(f"{options.signature} -> {result}")
    print(decode(result))


if __name__ == "__main__":
    main()
