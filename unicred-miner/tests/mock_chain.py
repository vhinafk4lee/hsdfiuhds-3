"""Mocked Unichain JSON-RPC (HTTP on 127.0.0.1) with a simplified UNICRED contract.

Enough of the contract to drive the controller end to end: challenge / target /
minted / price views, digest view, mint() simulation and mint() execution with
the real PoW rule. Nothing ever leaves localhost.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

from eth_account import Account
from eth_account.typed_transactions import TypedTransaction
from hexbytes import HexBytes

from unicred import pow as P


def H(n):
    return "0x" + P.keccak256(b"mock-block" + n.to_bytes(8, "big")).hex()


def w(v):
    return "0x" + ("%064x" % v)


class Revert(Exception):
    def __init__(self, sel):
        Exception.__init__(self, sel)
        self.sel = sel


class MockChain(object):
    def __init__(self, target=1 << 242, price=4 * 10 ** 15, minted=650, blocks_per_sec=2.0):
        self.lock = threading.RLock()
        self.t0 = time.time()
        self.base_head = 59_500_000
        self.extra_blocks = 0
        self.bps = blocks_per_sec
        self.challenge = "0x" + P.keccak256(b"genesis").hex()
        self.target = target
        self.global_target = target << 16
        self.price_wei = price
        self.minted = minted
        self.last_mint_block = self.base_head - 5
        self.base_fee = 3_000_000  # 0.003 gwei
        self.balances = {}
        self.nonces = {}
        self.txs = []          # decoded sent transactions
        self.receipts = {}
        self.logs = []
        self.calls = {}        # method -> count
        self.send_count = 0

    # ---------------------------------------------------------------- chain
    def head(self):
        return self.base_head + int((time.time() - self.t0) * self.bps) + self.extra_blocks

    def price(self, n):
        return self.price_wei

    # ------------------------------------------------------------- contract
    def contract_call(self, data, sender=None, value=0, execute=False, block=None):
        sel, args = data[2:10], data[10:]
        words = [int(args[i:i + 64], 16) for i in range(0, len(args), 64)]
        if sel == P.SEL_CHALLENGE:
            return self.challenge
        if sel == P.SEL_MINTED:
            return w(self.minted)
        if sel == P.SEL_TARGET_FOR:
            return w(self.target)
        if sel == P.SEL_GLOBAL_TARGET:
            return w(self.global_target)
        if sel == P.SEL_PRICE:
            return w(self.price(words[0]))
        if sel == P.SEL_LAST_MINT_BLOCK:
            return w(self.last_mint_block)
        if sel == P.SEL_DIGEST:
            return "0x" + P.digest(words[0], words[1], "0x%040x" % words[2], words[3]).hex()
        if sel == P.SEL_MINT:
            return self.mint(words[0], words[1], words[2], sender, value, execute, block)
        raise Revert("00000000")

    def mint(self, anchor, nonce, max_price, sender, value, execute, block):
        number = block if block is not None else self.head() + 1
        if self.minted >= P.MAX_SUPPLY:
            raise Revert("52df9fe5")
        if self.last_mint_block == number:
            raise Revert("440f8b45")
        if not (anchor < number and number - anchor <= 250):
            raise Revert("6e84ebb5")
        d = P.digest(H(anchor), self.challenge, sender, nonce)
        if int.from_bytes(d, "big") >= self.target:
            raise Revert("7ca55c77")
        price = self.price(self.minted)
        if price > max_price:
            raise Revert("a89fb05f")
        if value < price:
            raise Revert("a5cc3e35")
        if execute:
            self.minted += 1
            self.challenge = "0x" + d.hex()
            self.last_mint_block = number
            self.logs.append({
                "address": P.UNICRED.lower(), "blockNumber": hex(number), "blockHash": H(number),
                "topics": [P.MINT_TOPIC, w(self.minted), "0x" + "0" * 24 + sender[2:].lower()],
                "data": "0x" + d.hex() + "%064x%064x" % (anchor, price),
                "transactionHash": None})
            return price
        return "0x"

    # ------------------------------------------------------------------ rpc
    def handle(self, method, params):
        with self.lock:
            self.calls[method] = self.calls.get(method, 0) + 1
            return getattr(self, "rpc_" + method)(*params)

    def rpc_eth_chainId(self):
        return hex(130)

    def rpc_eth_blockNumber(self):
        return hex(self.head())

    def rpc_eth_getBlockByNumber(self, tag, full=False):
        n = self.head() if tag == "latest" else int(tag, 16)
        return {"number": hex(n), "hash": H(n), "parentHash": H(n - 1), "timestamp": hex(1_758_700_000 + n),
                "baseFeePerGas": hex(self.base_fee)}

    def rpc_eth_call(self, obj, block="latest"):
        if obj["to"].lower() != P.UNICRED.lower():
            return "0x"
        try:
            return self.contract_call(obj["data"], obj.get("from", "0x" + "0" * 40),
                                      int(obj.get("value", "0x0"), 16))
        except Revert as r:
            raise RpcFault(3, "execution reverted", "0x" + r.sel)

    def rpc_eth_getBalance(self, addr, block="latest"):
        return hex(self.balances.get(addr.lower(), 10 ** 18))

    def rpc_eth_getTransactionCount(self, addr, block="latest"):
        return hex(self.nonces.get(addr.lower(), 0))

    def rpc_eth_maxPriorityFeePerGas(self):
        return hex(1000)

    def rpc_eth_sendRawTransaction(self, raw):
        self.send_count += 1
        tx = TypedTransaction.from_bytes(HexBytes(raw)).as_dict()
        sender = Account.recover_transaction(raw)
        tx_hash = "0x" + P.keccak256(bytes.fromhex(raw[2:])).hex()
        exp = self.nonces.get(sender.lower(), 0)
        if tx["nonce"] != exp:
            raise RpcFault(-32000, "nonce too low" if tx["nonce"] < exp else "nonce too high")
        self.nonces[sender.lower()] = exp + 1
        self.extra_blocks += 1   # every transaction gets its own block
        number = self.head()
        rec = {"transactionHash": tx_hash, "blockNumber": hex(number), "blockHash": H(number),
               "gasUsed": hex(200_000), "effectiveGasPrice": hex(self.base_fee + tx["maxPriorityFeePerGas"]),
               "l1Fee": hex(1000), "logs": []}
        challenge_before = self.challenge
        try:
            paid = self.contract_call("0x" + bytes(tx["data"]).hex(), sender.lower(), tx["value"],
                                      execute=True, block=number)
            rec["status"] = "0x1"
            rec["logs"] = [dict(self.logs[-1], transactionHash=tx_hash)]
            self.logs[-1]["transactionHash"] = tx_hash
        except Revert as r:
            rec["status"] = "0x0"
            paid = 0
            rec["revert"] = r.sel
        self.txs.append({"raw": raw, "tx": tx, "sender": sender, "hash": tx_hash, "status": rec["status"],
                         "challenge": challenge_before, "paid": paid})
        self.receipts[tx_hash] = rec
        return tx_hash

    def rpc_eth_getTransactionReceipt(self, tx_hash):
        return self.receipts.get(tx_hash)

    def rpc_eth_getLogs(self, flt):
        if "blockHash" in flt:
            return [l for l in self.logs if l["blockHash"] == flt["blockHash"]]
        lo = int(flt.get("fromBlock", "0x0"), 16)
        hi = self.head() if flt.get("toBlock", "latest") == "latest" else int(flt["toBlock"], 16)
        return [l for l in self.logs if lo <= int(l["blockNumber"], 16) <= hi]

    # external "someone else minted"
    def foreign_mint(self):
        with self.lock:
            self.minted += 1
            self.challenge = "0x" + P.keccak256(self.challenge.encode() + b"foreign").hex()
            self.last_mint_block = self.head()


class RpcFault(Exception):
    def __init__(self, code, message, data=None):
        Exception.__init__(self, message)
        self.code, self.message, self.data = code, message, data


class _Server(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(chain, batch=True):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))

            def one(req):
                try:
                    return {"jsonrpc": "2.0", "id": req["id"], "result": chain.handle(req["method"], req["params"])}
                except RpcFault as f:
                    err = {"code": f.code, "message": f.message}
                    if f.data:
                        err["data"] = f.data
                    return {"jsonrpc": "2.0", "id": req["id"], "error": err}

            if isinstance(body, list):
                resp = [one(r) for r in body] if batch else {
                    "jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "batch not supported"}}
            else:
                resp = one(body)
            data = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = _Server(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, "http://127.0.0.1:%d" % srv.server_address[1]
