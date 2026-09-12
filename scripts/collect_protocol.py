#!/usr/bin/env python3
"""Collect everything needed to pin down the Hash Broker contract ABI.

Run this on a host that can reach a Robinhood Chain RPC. It reads a known mint
transaction and the contract's deployed bytecode, extracts the function
selectors out of the dispatcher, matches them against a dictionary of plausible
signatures, calls the read-only ones, and writes a single JSON report.

    python3 scripts/collect_protocol.py \
        --rpc https://your-rpc \
        --tx 0xeeb4cf... \
        --wallet 0xYourWallet \
        --out hashbroker-report.json

No dependencies beyond the standard library.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from keccak_pure import keccak256, selector  # noqa: E402

VIEW_SIGNATURES = [
    f"{name}()" for name in (
        "currentAnchor", "anchor", "anchorBlock", "anchorHash", "getAnchor",
        "prevWork", "lastWork", "lastHash", "lastProof", "challenge",
        "currentChallenge", "getChallenge", "target", "currentTarget", "getTarget",
        "difficulty", "currentDifficulty", "mintPrice", "price", "currentPrice",
        "cost", "totalMinted", "minted", "totalSupply", "maxSupply", "MAX_SUPPLY",
        "ANCHOR_WINDOW", "anchorWindow", "window", "WINDOW", "BLOCK_WINDOW",
        "epoch", "currentEpoch", "nextTokenId", "mintOpen", "paused", "owner",
        "name", "symbol", "MAX_PER_WALLET", "MINT_LIMIT", "startBlock",
    )
] + [
    f"{name}(address)" for name in (
        "targetFor", "targetOf", "getTarget", "difficultyFor", "difficultyOf",
        "workFor", "challengeFor", "balanceOf", "mintedBy", "mintsOf",
        "nonceOf", "lastNonce",
    )
]

MINE_SIGNATURES = [
    "mine(uint256,uint256)", "mine(uint256)", "mine(uint256,bytes32)",
    "mine(bytes32,uint256)", "mine(uint256,uint256,uint256)",
    "mine(address,uint256,uint256)", "mineFor(address,uint256,uint256)",
    "mint(uint256,uint256)", "mint(uint256)", "mint(uint256,bytes32)",
    "mint(bytes32)", "mint(bytes32,uint256)", "mint(address,uint256,uint256)",
    "submit(uint256,uint256)", "submit(uint256)", "solve(uint256,uint256)",
    "claim(uint256,uint256)", "forge(uint256,uint256)", "work(uint256,uint256)",
]

EVENT_SIGNATURES = [
    "Transfer(address,address,uint256)",
    "Mined(address,uint256,bytes32)",
    "Mined(address,uint256,uint256)",
    "Mint(address,uint256)",
    "Minted(address,uint256,bytes32)",
    "Minted(address,uint256,uint256)",
    "Work(address,uint256,bytes32)",
    "ProofAccepted(address,uint256,bytes32)",
]

# EIP-1967 implementation slot: keccak256("eip1967.proxy.implementation") - 1
PROXY_SLOT = "0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc"


class Rpc:
    def __init__(self, url: str, timeout: float = 20.0):
        self.url = url
        self.timeout = timeout
        self._id = 0

    def call(self, method: str, params: list | None = None):
        self._id += 1
        payload = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or []}
        request = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(),
            headers={"content-type": "application/json", "user-agent": "hashbroker-collector/1.0"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            row = json.load(response)
        if "error" in row:
            raise RuntimeError(f"{method}: {str(row['error'])[:200]}")
        return row["result"]

    def try_call(self, method: str, params: list | None = None):
        try:
            return self.call(method, params)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}


def iter_push_values(code: bytes, width: int):
    """Yield the immediates of every PUSH<width> opcode, skipping push data."""
    index = 0
    size = len(code)
    while index < size:
        opcode = code[index]
        index += 1
        if 0x60 <= opcode <= 0x7F:
            length = opcode - 0x5F
            value = code[index:index + length]
            if length == width and len(value) == width:
                yield value.hex()
            index += length


def extract_selectors(code: bytes) -> list[str]:
    return sorted(set(iter_push_values(code, 4)))


def match_signatures(selectors: set[str], signatures: list[str]) -> dict[str, str]:
    return {signature: selector(signature) for signature in signatures
            if selector(signature) in selectors}


def words_of(calldata: str) -> list[str]:
    body = calldata[10:] if calldata.startswith("0x") else calldata[8:]
    return ["0x" + body[index:index + 64] for index in range(0, len(body), 64)]


def describe_word(word: str) -> str:
    value = int(word, 16)
    if value < 2**64:
        return f"uint {value}"
    if 2**96 <= value < 2**160:
        return f"address-shaped 0x{word[-40:]}"
    return f"uint {value}"


def collect(rpc: Rpc, tx_hash: str | None, contract: str | None,
            wallet: str | None) -> dict:
    report: dict = {"rpc": "<redacted>"}
    report["chainId"] = rpc.try_call("eth_chainId")
    report["blockNumber"] = rpc.try_call("eth_blockNumber")

    if tx_hash:
        transaction = rpc.try_call("eth_getTransactionByHash", [tx_hash])
        report["transaction"] = transaction
        report["receipt"] = rpc.try_call("eth_getTransactionReceipt", [tx_hash])
        if isinstance(transaction, dict) and transaction.get("to") and not contract:
            contract = transaction["to"]
    if not contract:
        raise SystemExit("no contract address: pass --contract or a --tx that calls one")

    report["contract"] = contract
    code = rpc.try_call("eth_getCode", [contract, "latest"])
    if not isinstance(code, str) or len(code) <= 2:
        raise SystemExit(f"no bytecode at {contract}: {code}")
    report["codeSize"] = (len(code) - 2) // 2

    implementation = rpc.try_call("eth_getStorageAt", [contract, PROXY_SLOT, "latest"])
    if isinstance(implementation, str) and int(implementation, 16):
        address = "0x" + implementation[-40:]
        report["proxyImplementation"] = address
        implementation_code = rpc.try_call("eth_getCode", [address, "latest"])
        if isinstance(implementation_code, str) and len(implementation_code) > 2:
            code = code + implementation_code[2:]

    selectors = extract_selectors(bytes.fromhex(code[2:]))
    report["selectors"] = selectors
    known = set(selectors)
    report["matchedViews"] = match_signatures(known, VIEW_SIGNATURES)
    report["matchedMine"] = match_signatures(known, MINE_SIGNATURES)
    matched = set(report["matchedViews"].values()) | set(report["matchedMine"].values())
    report["unmatchedSelectors"] = [value for value in selectors if value not in matched]

    if isinstance(report.get("transaction"), dict):
        calldata = report["transaction"].get("input", "")
        if len(calldata) >= 10:
            used = calldata[2:10]
            report["mintCall"] = {
                "selector": "0x" + used,
                "signature": next(
                    (signature for signature in MINE_SIGNATURES if selector(signature) == used),
                    None,
                ),
                "value": report["transaction"].get("value"),
                "args": [{"word": word, "reading": describe_word(word)}
                         for word in words_of(calldata)],
            }

    receipt = report.get("receipt")
    if isinstance(receipt, dict):
        topics = {}
        for log in receipt.get("logs", []):
            topic = (log.get("topics") or [""])[0]
            if not topic:
                continue
            topics[topic] = next(
                (signature for signature in EVENT_SIGNATURES
                 if "0x" + keccak256(signature.encode()).hex() == topic),
                None,
            )
        report["logTopics"] = topics

    values: dict[str, object] = {}
    for signature, sig_selector in report["matchedViews"].items():
        data = "0x" + sig_selector
        if signature.endswith("(address)"):
            if not wallet:
                continue
            data += wallet[2:].lower().rjust(64, "0")
        values[signature] = rpc.try_call("eth_call", [{"to": contract, "data": data}, "latest"])
    report["viewValues"] = values
    return report


def summarize(report: dict) -> str:
    lines = [
        f"chainId      {report.get('chainId')}",
        f"contract     {report.get('contract')}",
        f"code size    {report.get('codeSize')} bytes",
    ]
    if report.get("proxyImplementation"):
        lines.append(f"proxy impl   {report['proxyImplementation']}")
    mint = report.get("mintCall") or {}
    if mint:
        lines.append(f"mint call    {mint.get('selector')} -> {mint.get('signature') or 'UNKNOWN'}")
        lines.append(f"mint value   {mint.get('value')}")
        for index, argument in enumerate(mint.get("args", [])):
            lines.append(f"  arg[{index}]    {argument['word']}  ({argument['reading']})")
    lines.append("views found  " + (", ".join(report.get("matchedViews", {})) or "none"))
    lines.append("mine found   " + (", ".join(report.get("matchedMine", {})) or "none"))
    lines.append(f"unmatched    {len(report.get('unmatchedSelectors', []))} selectors")
    for signature, value in (report.get("viewValues") or {}).items():
        lines.append(f"  {signature} = {value}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rpc", required=True, help="HTTPS RPC endpoint")
    parser.add_argument("--tx", help="a known mint transaction hash")
    parser.add_argument("--contract", help="contract address, if no transaction is given")
    parser.add_argument("--wallet", help="wallet address for per-wallet views")
    parser.add_argument("--out", default="hashbroker-report.json")
    arguments = parser.parse_args()

    report = collect(Rpc(arguments.rpc), arguments.tx, arguments.contract, arguments.wallet)
    Path(arguments.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(summarize(report))
    print(f"\nfull report written to {arguments.out}")


if __name__ == "__main__":
    main()
