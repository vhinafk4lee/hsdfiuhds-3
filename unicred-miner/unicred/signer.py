"""Wallet: loads the private key from a local file and signs mint() transactions.

The key never leaves this process: it is not printed, logged or sent to servers.
"""
from pathlib import Path

from eth_account import Account

from . import pow as P
from .config import read_text


class Wallet(object):
    def __init__(self, key_file):
        path = Path(key_file)
        if not path.exists():
            raise SystemExit("нет файла ключа %s (см. README)" % path)
        text = read_text(path).strip()
        if text.startswith("{"):
            raise SystemExit("keystore JSON не поддерживается: положите в %s приватный ключ (hex)" % path)
        key = text if text.startswith("0x") else "0x" + text
        if len(key) != 66:
            raise SystemExit("файл ключа %s: ожидается 32-байтовый hex ключ" % path)
        try:
            self._account = Account.from_key(key)
        except Exception:
            raise SystemExit("файл ключа %s: неверный ключ" % path)
        self.address = self._account.address

    def __repr__(self):
        return "Wallet(%s)" % self.address

    def sign_mint(self, tx_nonce, anchor_block, pow_nonce, price, gas_limit, max_fee, priority_fee,
                  chain_id=P.CHAIN_ID, contract=P.UNICRED):
        tx = {
            "type": 2,
            "chainId": chain_id,
            "nonce": tx_nonce,
            "to": contract,
            "value": price,
            "gas": gas_limit,
            "maxFeePerGas": max_fee,
            "maxPriorityFeePerGas": priority_fee,
            "data": P.mint_calldata(anchor_block, pow_nonce, price),
        }
        signed = self._account.sign_transaction(tx)
        raw = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction")
        tx_hash = signed.hash
        return "0x" + bytes(raw).hex(), "0x" + bytes(tx_hash).hex(), tx
