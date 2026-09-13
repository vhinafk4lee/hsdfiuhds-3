#!/usr/bin/env python3
"""A minimal in-process JSON-RPC node that answers like the Hash Broker chain.

It backs the tests and scripts/playground.py: the whole miner can run against it
with no chain, no wallet and no money, at whatever difficulty makes a proof
arrive in seconds instead of days.
"""
from __future__ import annotations

import hashlib
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
                 block_number: int = 0x3A85931, chain_id: int | None = None,
                 advance_on_mint: bool = True):
        self.challenge = challenge
        self.difficulty = difficulty
        self.price = price
        self.minted = minted
        self.max_supply = max_supply
        self.balance = balance
        self.block_number = block_number
        self.chain_id = PROTOCOL.chain_id if chain_id is None else chain_id
        self.advance_on_mint = advance_on_mint
        self.account_nonce = 7
        self.sent: list[str] = []
        self.estimate = 107_089
        self.tx_hash = "0x" + "5c" * 32

    def mint(self, raw_transaction: str) -> None:
        """Accept a mint the way the contract does: new challenge, one more token."""
        seed = self.challenge.removeprefix("0x") + raw_transaction.removeprefix("0x")
        self.challenge = "0x" + hashlib.sha256(bytes.fromhex(seed)).hexdigest()
        self.minted += 1
        self.block_number += 1
        self.tx_hash = "0x" + hashlib.sha256(seed.encode()).hexdigest()

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
            digest = powlib.digest({"wallet": wallet, "nonce": nonce, "challenge": challenge})
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
            if self.advance_on_mint:
                self.mint(params[0])
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


class FlyNodeChain:
    """A stub that answers like FlyNode: two Merkle roots, a lattice, and logs.

    It grows a real graph. Seed it with one claimed cell and it will only accept
    a mint whose parent is already claimed and whose cell is not, which is what
    makes it worth testing the frontier against.
    """

    MINED = "Mined(address,uint32,uint32,uint256)"

    def __init__(self, lattice, protocol, occupied: list[int] | None = None,
                 price: int = 2 * 10**14, retarget_q: int = 8, network_streak: int = 3,
                 address_streak: int = 1, failsafe: int = 0, idle_since: int = 0,
                 deploy_block: int = 62107948, blocks_since_deploy: int = 5000,
                 balance: int = 10**18, miner: str = "0x" + "11" * 20,
                 anchor_at: int | None = None, roots: tuple[str, str] | None = None):
        from keccak_pure import keccak256
        self.lattice = lattice
        self.protocol = protocol
        self.roots = roots or ("0x" + lattice.neurons_root.hex(),
                               "0x" + lattice.edges_root.hex())
        self.price = price
        self.retarget_q = retarget_q
        self.network_streak = network_streak
        self.address_streak = address_streak
        self.failsafe = failsafe
        self.idle_since = idle_since
        self.deploy_block = deploy_block
        self.block_number = deploy_block + blocks_since_deploy
        self.balance = balance
        self.account_nonce = 3
        self.estimate = 240_000
        self.sent: list[str] = []
        self.tx_hash = "0x" + "7e" * 32
        self.prev = "0x" + "5c" * 32
        # An anchor that is some block's hash, when a test wants one to be found.
        self.anchor_at = anchor_at
        self.anchor = (self.block_hash(anchor_at) if anchor_at is not None
                       else "0x" + "a9" * 32)
        self.topic = "0x" + keccak256(self.MINED.encode()).hex()
        self.logs: list[dict] = []
        for index, cell in enumerate(occupied or []):
            parent = next((other for other in lattice.linked(cell)
                           if other in (occupied or [])[:index]), cell)
            self.record(cell, parent, miner, deploy_block + index)

    # --- state ---------------------------------------------------------------

    @staticmethod
    def block_hash(number: int) -> str:
        return "0x" + hashlib.sha256(f"block-{number}".encode()).hexdigest()

    @property
    def occupied(self) -> set[int]:
        return {int(log["topics"][2], 16) for log in self.logs}

    def record(self, cell: int, parent: int, miner: str, block: int) -> None:
        self.logs.append({
            "address": self.protocol.contract,
            "topics": [self.topic,
                       "0x" + miner.removeprefix("0x").rjust(64, "0"),
                       "0x" + word(cell)],
            # parent and the block it landed in: one is a cell, one only looks
            # like one, which is the whole point of the slot search.
            "data": "0x" + word(parent) + word(block),
            "blockNumber": hex(block),
        })

    def required_bits(self, rarity: int) -> int:
        value = (16 + self.retarget_q // 4 + rarity
                 + min(16, self.network_streak) + min(16, self.address_streak)
                 - self.failsafe)
        return max(1, value)

    # --- rpc -----------------------------------------------------------------

    def eth_call(self, call: dict) -> str:
        data = call["data"]
        selector = data[:10]
        views = {key: self.protocol.view(key) for key in self.protocol.views}
        simple = {
            "prev": self.prev, "anchor": self.anchor,
            "price": "0x" + word(self.price),
            "minted": "0x" + word(len(self.occupied)),
            "maxSupply": "0x" + word(self.lattice.size),
            "lastMintBlock": "0x" + word(self.block_number - 3),
            "retargetQ": "0x" + word(self.retarget_q),
            "networkStreak": "0x" + word(self.network_streak),
            "idleSince": "0x" + word(self.idle_since),
            "neuronsRoot": self.roots[0],
            "edgesRoot": self.roots[1],
        }
        for key, answer in simple.items():
            if key in views and selector == views[key]:
                return answer
        if selector == views["addressStreak"]:
            return "0x" + word(self.address_streak)
        if selector == views["requiredBits"]:
            return "0x" + word(self.required_bits(int(data[10:74], 16)))
        if selector == views["mined"]:
            return "0x" + word(1 if int(data[10:74], 16) in self.occupied else 0)
        if selector == "0x" + self.protocol.validate_selector:
            body = data[10:]
            digest = powlib.digest({
                "wallet": "0x" + body[24:64], "nonce": int(body[64:128], 16),
                "prev": "0x" + body[128:192], "anchor": "0x" + body[192:256],
                "typeId": int(body[256:320], 16),
            }, self.protocol)
            return "0x" + digest.hex()
        raise ValueError(f"stub has no answer for {selector}")

    def eth_getLogs(self, query: dict) -> list[dict]:
        low = int(query.get("fromBlock", "0x0"), 16)
        high = int(query.get("toBlock", hex(self.block_number)), 16)
        wanted = (query.get("topics") or [None])[0]
        return [log for log in self.logs
                if low <= int(log["blockNumber"], 16) <= high
                and (wanted is None or log["topics"][0] == wanted)]

    def handle(self, method: str, params: list):
        if method == "eth_chainId":
            return hex(self.protocol.chain_id)
        if method == "eth_blockNumber":
            return hex(self.block_number)
        if method == "eth_call":
            return self.eth_call(params[0])
        if method == "eth_getLogs":
            return self.eth_getLogs(params[0])
        if method == "eth_getBlockByNumber":
            number = int(params[0], 16)
            if number > self.block_number:
                return None
            return {"number": hex(number), "hash": self.block_hash(number)}
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
