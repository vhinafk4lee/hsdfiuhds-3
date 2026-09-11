"""Tiny JSON-RPC client for Robinhood Chain (stdlib only, with retries)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, List, Optional


class RpcError(RuntimeError):
    pass


class Rpc:
    def __init__(self, url: str, timeout: float = 20.0, retries: int = 4):
        self.url = url
        self.timeout = timeout
        self.retries = retries
        self._id = 0

    def call(self, method: str, params: Optional[List[Any]] = None) -> Any:
        self._id += 1
        payload = json.dumps(
            {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or []}
        ).encode()
        delay = 2.0
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(
                    self.url, data=payload, headers={"content-type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read())
                if "error" in body:
                    raise RpcError(f"{method}: {body['error']}")
                return body["result"]
            except RpcError:
                raise
            except Exception as exc:  # network hiccup: back off and retry
                last = exc
                if attempt == self.retries:
                    break
                time.sleep(delay)
                delay *= 2
        raise RpcError(f"{method} failed after {self.retries + 1} attempts: {last}")

    # --- convenience wrappers -------------------------------------------------

    def chain_id(self) -> int:
        return int(self.call("eth_chainId"), 16)

    def block_number(self) -> int:
        return int(self.call("eth_blockNumber"), 16)

    def eth_call(self, to: str, data: bytes, block: str = "latest") -> bytes:
        result = self.call("eth_call", [{"to": to, "data": "0x" + data.hex()}, block])
        return bytes.fromhex(result.removeprefix("0x"))

    def get_balance(self, address: str, block: str = "latest") -> int:
        return int(self.call("eth_getBalance", [address, block]), 16)

    def tx_count(self, address: str, block: str = "pending") -> int:
        return int(self.call("eth_getTransactionCount", [address, block]), 16)

    def gas_price(self) -> int:
        return int(self.call("eth_gasPrice"), 16)

    def base_fee(self) -> int:
        block = self.call("eth_getBlockByNumber", ["latest", False])
        return int(block.get("baseFeePerGas", "0x0"), 16)

    def estimate_gas(self, tx: dict) -> int:
        return int(self.call("eth_estimateGas", [tx]), 16)

    def send_raw(self, raw: bytes) -> str:
        return self.call("eth_sendRawTransaction", ["0x" + raw.hex()])

    def get_receipt(self, tx_hash: str) -> Optional[dict]:
        return self.call("eth_getTransactionReceipt", [tx_hash])

    def get_transaction(self, tx_hash: str) -> Optional[dict]:
        return self.call("eth_getTransactionByHash", [tx_hash])
