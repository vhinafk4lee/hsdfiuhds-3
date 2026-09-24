"""Contract / chain access on top of Rpc."""
from . import pow as P
from .rpc import Rpc, RpcError


def hx(v):
    return int(v, 16) if isinstance(v, str) else int(v)


def word(addr_or_int):
    if isinstance(addr_or_int, int):
        return "%064x" % addr_or_int
    a = addr_or_int[2:] if addr_or_int.startswith("0x") else addr_or_int
    return a.lower().rjust(64, "0")


class Chain(object):
    def __init__(self, rpc_url, address, contract=P.UNICRED, send_urls=None, timeout=4.0):
        self.rpc = Rpc(rpc_url, timeout=timeout)
        self.send_rpcs = [Rpc(u, timeout=timeout) for u in (send_urls or []) if u and u != rpc_url]
        self.address = address
        self.contract = contract

    # --- eth_call helpers -------------------------------------------------
    def _call_obj(self, data, sender=None, value=None):
        obj = {"to": self.contract, "data": data}
        if sender:
            obj["from"] = sender
        if value is not None:
            obj["value"] = hex(value)
        return obj

    def call(self, data, block="latest", sender=None, value=None):
        return self.rpc.call("eth_call", [self._call_obj(data, sender, value), block])

    def call_uint(self, data, block="latest"):
        return hx(self.call(data, block))

    def challenge(self):
        return "0x" + self.call("0x" + P.SEL_CHALLENGE)[2:].rjust(64, "0")

    def minted(self):
        return self.call_uint("0x" + P.SEL_MINTED)

    def target_for(self, miner):
        return self.call_uint("0x" + P.SEL_TARGET_FOR + word(miner))

    def global_target(self):
        return self.call_uint("0x" + P.SEL_GLOBAL_TARGET)

    def price(self, n):
        return self.call_uint("0x" + P.SEL_PRICE + word(n))

    def next_price(self, minted):
        """Price to pay for the next mint. The contract indexes price by mint
        number; we don't rely on 0/1-based numbering and take the max of both
        (the excess is refunded by the contract)."""
        res = self.rpc.batch([
            ("eth_call", [self._call_obj("0x" + P.SEL_PRICE + word(minted)), "latest"]),
            ("eth_call", [self._call_obj("0x" + P.SEL_PRICE + word(minted + 1)), "latest"]),
        ])
        vals = [hx(r) for r in res if not isinstance(r, Exception)]
        if not vals:
            raise res[0]
        return max(vals)

    def last_mint_block(self):
        return self.call_uint("0x" + P.SEL_LAST_MINT_BLOCK)

    def contract_digest(self, blockhash, challenge, miner, nonce):
        data = ("0x" + P.SEL_DIGEST + P.b32(blockhash).hex() + P.b32(challenge).hex()
                + word(miner) + word(nonce))
        return "0x" + self.call(data)[2:].rjust(64, "0")

    def simulate_mint(self, anchor_block, nonce, max_price, value, sender):
        """eth_call mint(); returns None on success or (selector, text) on revert."""
        try:
            self.call(P.mint_calldata(anchor_block, nonce, max_price), sender=sender, value=value)
            return None
        except RpcError as exc:
            data = exc.revert_data
            if data:
                return P.decode_error(data)
            return (None, str(exc))

    # --- polling ------------------------------------------------------------
    def poll(self):
        """One batch: latest header, challenge, my target, minted."""
        res = self.rpc.batch([
            ("eth_getBlockByNumber", ["latest", False]),
            ("eth_call", [self._call_obj("0x" + P.SEL_CHALLENGE), "latest"]),
            ("eth_call", [self._call_obj("0x" + P.SEL_TARGET_FOR + word(self.address)), "latest"]),
            ("eth_call", [self._call_obj("0x" + P.SEL_MINTED), "latest"]),
        ])
        for r in res:
            if isinstance(r, Exception):
                raise r
        head = res[0]
        return {
            "head": hx(head["number"]),
            "head_hash": head["hash"],
            "parent_hash": head["parentHash"],
            "timestamp": hx(head["timestamp"]),
            "base_fee": hx(head.get("baseFeePerGas") or "0x0"),
            "challenge": "0x" + res[1][2:].rjust(64, "0"),
            "target": hx(res[2]),
            "minted": hx(res[3]),
        }

    def slow_poll(self):
        res = self.rpc.batch([
            ("eth_call", [self._call_obj("0x" + P.SEL_GLOBAL_TARGET), "latest"]),
            ("eth_call", [self._call_obj("0x" + P.SEL_LAST_MINT_BLOCK), "latest"]),
            ("eth_getBalance", [self.address, "latest"]),
        ])
        out = {}
        for key, r in zip(("global_target", "last_mint_block", "balance"), res):
            if not isinstance(r, Exception):
                out[key] = hx(r)
        return out

    # --- misc ------------------------------------------------------------------
    def chain_id(self):
        return hx(self.rpc.call("eth_chainId"))

    def block_number(self):
        return hx(self.rpc.call("eth_blockNumber"))

    def block(self, n):
        return self.rpc.call("eth_getBlockByNumber", [hex(n) if isinstance(n, int) else n, False])

    def blockhash(self, n):
        return self.block(n)["hash"]

    def balance(self, addr=None):
        return hx(self.rpc.call("eth_getBalance", [addr or self.address, "latest"]))

    def tx_count(self, addr=None, block="pending"):
        return hx(self.rpc.call("eth_getTransactionCount", [addr or self.address, block]))

    def max_priority_fee(self):
        try:
            return hx(self.rpc.call("eth_maxPriorityFeePerGas"))
        except Exception:
            return None

    def send_raw(self, raw_hex):
        """Broadcast to the main RPC and to extra send RPCs; first success wins."""
        errors = []
        result = None
        try:
            result = self.rpc.call("eth_sendRawTransaction", [raw_hex])
        except Exception as exc:
            errors.append(exc)
        for extra in self.send_rpcs:
            try:
                r = extra.call("eth_sendRawTransaction", [raw_hex], timeout=2.0)
                result = result or r
            except Exception as exc:
                errors.append(exc)
        if result is None and errors:
            raise errors[0]
        return result

    def receipt(self, tx_hash):
        return self.rpc.call("eth_getTransactionReceipt", [tx_hash])

    def mint_logs(self, from_block, to_block="latest", block_hash=None):
        flt = {"address": self.contract, "topics": [P.MINT_TOPIC]}
        if block_hash:
            flt["blockHash"] = block_hash
        else:
            flt["fromBlock"] = hex(from_block)
            flt["toBlock"] = to_block if isinstance(to_block, str) else hex(to_block)
        return [parse_mint_log(l) for l in self.rpc.call("eth_getLogs", [flt])]


def parse_mint_log(log):
    data = log["data"][2:]
    return {
        "token_id": hx(log["topics"][1]),
        "miner": "0x" + log["topics"][2][-40:],
        "digest": "0x" + data[0:64],
        "anchor_block": int(data[64:128], 16),
        "price": int(data[128:192], 16),
        "block": hx(log["blockNumber"]),
        "tx": log.get("transactionHash"),
    }
