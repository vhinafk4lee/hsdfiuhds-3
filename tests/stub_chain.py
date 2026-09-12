#!/usr/bin/env python3
"""A minimal in-process JSON-RPC node that answers like the Hash Broker chain."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pow as powlib
from protocol import PROTOCOL


def word(value: int) -> str:
    return f"{value:064x}"


class StubChain:
    def __init__(self, challenge: str, difficulty: int = 8, price: int = 100_000_000_000_000,
                 minted: int = 260, max_supply: int = 4444, balance: int = 10**18,
                 block_number: int = 0x3A85931, chain_id: int | None = None):
        self.challenge = challenge
        self.difficulty = difficulty
        self.price = price
        self.minted = minted
        self.max_supply = max_supply
        self.balance = balance
        self.block_number = block_number
        self.chain_id = PROTOCOL.chain_id if chain_id is None else chain_id
        self.account_nonce = 7
        self.sent: list[str] = []
        self.estimate = 107_089
        self.tx_hash = "0x" + "5c" * 32

    def eth_call(self, call: dict) -> str:
        data = call["data"]
        selector = data[:10]
        views = {key: PROTOCOL.view(key) for key in PROTOCOL.views}
        if selector == views["challenge"]:
            return self.challenge
        if selector == views["difficulty"]:
            return "0x" + word(self.difficulty)
        if selector == views["price"]:
            return "0x" + word(self.price)
        if selector == views["minted"]:
            return "0x" + word(self.minted)
        if selector == views["maxSupply"]:
            return "0x" + word(self.max_supply)
        if selector == views["lastMintBlock"]:
            return "0x" + word(self.block_number - 12)
        if selector == "0x" + PROTOCOL.validate_selector:
            body = data[10:]
            wallet = "0x" + body[24:64]
            nonce = int(body[64:128], 16)
            challenge = "0x" + body[128:192]
            digest = powlib.digest(wallet, nonce, challenge)
            accepted = (challenge.lower() == self.challenge.lower()
                        and int.from_bytes(digest, "big")
                        < powlib.target_for_difficulty(self.difficulty))
            return "0x" + word(1 if accepted else 0)
        raise ValueError(f"stub has no answer for {selector}")

    def handle(self, method: str, params: list):
        if method == "eth_chainId":
            return hex(self.chain_id)
        if method == "eth_blockNumber":
            return hex(self.block_number)
        if method == "eth_call":
            return self.eth_call(params[0])
        if method == "eth_getTransactionCount":
            return hex(self.account_nonce)
        if method == "eth_getBalance":
            return hex(self.balance)
        if method == "eth_maxPriorityFeePerGas":
            return hex(10**9)
        if method == "eth_gasPrice":
            return hex(10**8)
        if method == "eth_estimateGas":
            return hex(self.estimate)
        if method == "eth_sendRawTransaction":
            self.sent.append(params[0])
            return self.tx_hash
        if method == "eth_getTransactionReceipt":
            return {"status": "0x1", "gasUsed": hex(self.estimate),
                    "transactionHash": self.tx_hash}
        raise ValueError(f"stub has no answer for {method}")


class StubServer:
    """Serves one StubChain over HTTP on localhost for the duration of a test."""

    def __init__(self, chain: StubChain):
        self.chain = chain
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler's naming
                length = int(self.headers.get("content-length", "0"))
                payload = json.loads(self.rfile.read(length))
                rows = []
                for item in payload if isinstance(payload, list) else [payload]:
                    try:
                        result = outer.chain.handle(item["method"], item.get("params") or [])
                        rows.append({"jsonrpc": "2.0", "id": item["id"], "result": result})
                    except Exception as exc:
                        rows.append({"jsonrpc": "2.0", "id": item["id"],
                                     "error": {"code": -32000, "message": str(exc)}})
                body = json.dumps(rows if isinstance(payload, list) else rows[0]).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                pass

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def __enter__(self) -> "StubServer":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._server.shutdown()
        self._server.server_close()
