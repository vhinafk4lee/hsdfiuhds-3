#!/usr/bin/env python3
"""Controller: verifies a worker's proof, signs mine(), and broadcasts it.

The signer is the only component that holds a key and the only one that can
spend. Every path into a broadcast goes through the same gate: recompute the
proof on the CPU, confirm it still beats the live target for the live
challenge, price the transaction, check it against the configured cap, persist
it, and only then send.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
from pathlib import Path

from eth_account import Account

import pow as powlib
from job_state import call_batch, request_batch
from keyfile import assert_private
from protocol import PROTOCOL

WALLET = os.environ.get("HASHBROKER_WALLET", "").strip()
RUNTIME_DIR = Path(os.environ.get("HASHBROKER_RUNTIME_DIR", "./runtime"))
SUBMIT_CAP_WEI = int(os.environ.get("HASHBROKER_SUBMIT_CAP_WEI", "100000000000000000"))
GAS_LIMIT_CAP = int(os.environ.get("HASHBROKER_GAS_LIMIT_CAP", "400000"))
EVENTS_PATH = RUNTIME_DIR / "events.jsonl"


def log_event(kind: str, **fields) -> None:
    row = {"at": time.time(), "event": kind, **fields}
    print(kind, json.dumps(fields, separators=(",", ":"), default=str), flush=True)
    try:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        with EVENTS_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")
    except OSError as exc:  # logging must never take down the signer
        print(f"EVENT_LOG_FAILED {type(exc).__name__}: {exc}", flush=True)


def load_account() -> Account:
    inline = os.environ.get("HASHBROKER_PRIVATE_KEY", "").strip()
    key_file = os.environ.get("HASHBROKER_PRIVATE_KEY_FILE", "").strip()
    if not inline and not key_file:
        raise SystemExit("set HASHBROKER_PRIVATE_KEY_FILE or HASHBROKER_PRIVATE_KEY")
    if not inline:
        path = Path(key_file)
        assert_private(path)
        inline = path.read_text(encoding="utf-8").strip()
    account = Account.from_key(inline)
    if WALLET and account.address.lower() != WALLET.lower():
        raise SystemExit(f"key is for {account.address}, HASHBROKER_WALLET is {WALLET}")
    return account


def read_state(wallet: str) -> dict:
    """Chain state needed to price and place the transaction, from one endpoint."""
    keys = ("challenge", "difficulty", "price", "minted", "maxSupply")
    block_number, values = call_batch([PROTOCOL.view(key) for key in keys])
    raw = dict(zip(keys, values))
    difficulty = int(raw["difficulty"], 16)
    state = {
        "challenge": "0x" + raw["challenge"][2:].rjust(64, "0"),
        "difficulty": difficulty,
        "target": powlib.target_for_difficulty(difficulty),
        "priceWei": int(raw["price"], 16),
        "minted": int(raw["minted"], 16),
        "maxSupply": int(raw["maxSupply"], 16),
        "blockNumber": block_number,
    }
    account_rows = request_batch(PROTOCOL.rpc[0], [
        ("eth_getTransactionCount", [wallet, "pending"]),
        ("eth_getBalance", [wallet, "latest"]),
        ("eth_maxPriorityFeePerGas", []),
        ("eth_gasPrice", []),
    ])
    nonce, balance, priority, gas_price = account_rows
    state["nonce"] = int(nonce, 16)
    state["balanceWei"] = int(balance, 16)
    state["priorityFeeWei"] = int(priority, 16)
    state["gasPriceWei"] = int(gas_price, 16)
    return state


def verify_solution(solution: dict, wallet: str, state: dict) -> int:
    """Recompute the proof and confirm the chain would still accept it."""
    if str(solution.get("wallet", "")).lower() != wallet.lower():
        raise ValueError("solution belongs to another wallet")
    nonce = int(str(solution["nonce"]))
    if not 0 <= nonce < 2**256:
        raise ValueError("nonce out of range")
    challenge = str(solution["challenge"]).lower()
    if challenge != state["challenge"].lower():
        raise ValueError("solution is for a challenge the chain has moved past")
    digest = powlib.digest(wallet, nonce, challenge)
    if "0x" + digest.hex() != str(solution.get("hash", "")).lower():
        raise ValueError("solution hash does not match its own nonce")
    if int.from_bytes(digest, "big") >= state["target"]:
        raise ValueError(
            f"proof has {powlib.leading_zero_bits(digest)} zero bits, "
            f"difficulty is {state['difficulty']}"
        )
    return nonce


def confirm_on_chain(wallet: str, nonce: int, challenge: str) -> bool | None:
    """Ask the contract itself, when it exposes a validator. None means unknown."""
    if not PROTOCOL.validate_signature:
        return None
    data = ("0x" + PROTOCOL.validate_selector
            + wallet[2:].lower().rjust(64, "0")
            + f"{nonce:064x}"
            + challenge.removeprefix("0x").rjust(64, "0"))
    try:
        result = request_batch(PROTOCOL.rpc[0], [
            ("eth_call", [{"to": PROTOCOL.require_deployed(), "data": data}, "latest"])
        ])[0]
    except Exception as exc:
        log_event("VALIDATOR_UNAVAILABLE", error=f"{type(exc).__name__}: {exc}")
        return None
    return int(result, 16) == 1


def estimate_gas(wallet: str, calldata: str, value: int) -> int:
    call = {"from": wallet, "to": PROTOCOL.require_deployed(),
            "value": hex(value), "data": calldata}
    result = request_batch(PROTOCOL.rpc[0], [("eth_estimateGas", [call])])[0]
    return int(result, 16)


def build_transaction(wallet: str, nonce: int, challenge: str, state: dict,
                      gas_estimate: int) -> dict:
    """Price the mint and refuse anything above the configured cap."""
    price = state["priceWei"]
    if price < 0:
        raise ValueError("negative mint price")
    if gas_estimate <= 0:
        raise ValueError("nonpositive gas estimate")
    gas_limit = (gas_estimate * 125 + 99) // 100
    if gas_limit > GAS_LIMIT_CAP:
        raise ValueError(f"gas estimate {gas_estimate} exceeds the safety limit")
    priority_fee = max(state["priorityFeeWei"], 0)
    max_fee = max(state["gasPriceWei"] * 2, state["gasPriceWei"] + priority_fee)
    total = price + gas_limit * max_fee
    if total > SUBMIT_CAP_WEI:
        raise ValueError(f"mint would cost up to {total} wei, cap is {SUBMIT_CAP_WEI}")
    if state["balanceWei"] < total:
        raise ValueError(f"balance {state['balanceWei']} wei is below the {total} wei ceiling")
    return {
        "chainId": PROTOCOL.chain_id,
        "type": 2,
        "nonce": state["nonce"],
        "to": PROTOCOL.require_deployed(),
        "value": price,
        "data": PROTOCOL.calldata(nonce, challenge),
        "gas": gas_limit,
        "maxFeePerGas": max_fee,
        "maxPriorityFeePerGas": priority_fee,
    }


def broadcast(raw_hex: str, urls: tuple[str, ...], timeout: float = 8.0) -> tuple[str, str]:
    """First endpoint to accept the transaction wins; the rest are redundancy."""
    errors = []

    def send(url: str) -> tuple[str, str]:
        return url, request_batch(url, [("eth_sendRawTransaction", [raw_hex])], timeout)[0]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(urls))) as pool:
        futures = [pool.submit(send, url) for url in urls]
        for future in concurrent.futures.as_completed(futures, timeout=timeout + 2):
            try:
                url, tx_hash = future.result()
                return tx_hash, url
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {str(exc)[:120]}")
    raise RuntimeError("every broadcast endpoint rejected the transaction: " + "; ".join(errors))


def submit(account, solution: dict, dry_run: bool) -> str | None:
    wallet = account.address
    state = read_state(wallet)
    nonce = verify_solution(solution, wallet, state)
    valid = confirm_on_chain(wallet, nonce, state["challenge"])
    if valid is False:
        raise ValueError("contract rejected the proof in isValidProof")

    calldata = PROTOCOL.calldata(nonce, state["challenge"])
    gas_estimate = estimate_gas(wallet, calldata, state["priceWei"])
    transaction = build_transaction(wallet, nonce, state["challenge"], state, gas_estimate)
    signed = account.sign_transaction(transaction)
    raw_hex = "0x" + signed.raw_transaction.hex().removeprefix("0x")
    tx_hash = "0x" + signed.hash.hex().removeprefix("0x")

    # Persist before sending: a crash must not lose track of an in-flight nonce.
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    (RUNTIME_DIR / f"intent-{tx_hash}.json").write_text(
        json.dumps({"transaction": transaction, "solution": solution,
                    "hash": tx_hash, "at": time.time()}, indent=2, default=str) + "\n"
    )
    log_event("SIGNED", txHash=tx_hash, nonce=transaction["nonce"], valueWei=transaction["value"],
              gas=transaction["gas"], maxFeePerGas=transaction["maxFeePerGas"],
              validatedOnChain=valid)
    if dry_run:
        log_event("DRY_RUN", txHash=tx_hash, raw=raw_hex[:24] + "...")
        return None

    urls = PROTOCOL.broadcast_rpc or PROTOCOL.rpc
    sent, url = broadcast(raw_hex, urls)
    log_event("BROADCAST", txHash=sent, endpoint=url)
    return sent


def wait_for_receipt(tx_hash: str, timeout: float = 90.0) -> dict | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            receipt = request_batch(PROTOCOL.rpc[0],
                                    [("eth_getTransactionReceipt", [tx_hash])])[0]
        except Exception:
            receipt = None
        if receipt:
            return receipt
        time.sleep(1.0)
    return None


def check(account) -> None:
    print(f"wallet     {account.address}")
    state = read_state(account.address)
    price = state["priceWei"]
    print(f"balance    {state['balanceWei'] / 1e18:.6f} ETH")
    print(f"challenge  {state['challenge']}")
    print(f"difficulty {state['difficulty']} (target 2^{256 - state['difficulty']})")
    print(f"price      {price / 1e18:.6f} ETH")
    print(f"minted     {state['minted']}/{state['maxSupply']}")
    print(f"nonce      {state['nonce']}  gasPrice {state['gasPriceWei']} wei")
    print(f"cap        {SUBMIT_CAP_WEI / 1e18:.6f} ETH per mint")
    if state["balanceWei"] < price:
        print("WARNING: balance is below the mint price")
    if price > SUBMIT_CAP_WEI:
        print("WARNING: mint price alone is above the configured cap")


def consume_solutions(path: Path) -> list[tuple[Path, dict]]:
    files = sorted(path.glob("solution*.json")) if path.is_dir() else (
        [path] if path.exists() else []
    )
    rows = []
    for file in files:
        try:
            rows.append((file, json.loads(file.read_text(encoding="utf-8"))))
        except (OSError, ValueError):
            continue
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solutions", default="/opt/hashbroker",
                        help="solution file, or a directory of solution*.json")
    parser.add_argument("--check", action="store_true", help="print state and exit")
    parser.add_argument("--dry-run", action="store_true", help="sign but never broadcast")
    parser.add_argument("--once", action="store_true", help="submit one solution and exit")
    parser.add_argument("--interval", type=float, default=0.25)
    arguments = parser.parse_args()

    account = load_account()
    if arguments.check:
        check(account)
        return

    source = Path(arguments.solutions)
    print(f"SIGNER wallet={account.address} contract={PROTOCOL.require_deployed()} "
          f"cap={SUBMIT_CAP_WEI} dryRun={arguments.dry_run}", flush=True)
    seen: set[str] = set()
    while True:
        for file, solution in consume_solutions(source):
            key = f"{solution.get('challenge')}:{solution.get('nonce')}"
            if key in seen:
                continue
            seen.add(key)
            try:
                tx_hash = submit(account, solution, arguments.dry_run)
            except ValueError as exc:
                log_event("SOLUTION_REJECTED", file=str(file), reason=str(exc))
                continue
            except Exception as exc:
                log_event("SUBMIT_FAILED", file=str(file), error=f"{type(exc).__name__}: {exc}")
                continue
            file.replace(file.with_suffix(".submitted.json"))
            if tx_hash:
                receipt = wait_for_receipt(tx_hash)
                status = receipt.get("status") if receipt else None
                log_event("RECEIPT", txHash=tx_hash, status=status,
                          gasUsed=receipt.get("gasUsed") if receipt else None)
            if arguments.once:
                return
        time.sleep(arguments.interval)


if __name__ == "__main__":
    main()
