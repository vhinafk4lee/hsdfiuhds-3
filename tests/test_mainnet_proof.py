#!/usr/bin/env python3
"""Golden test against a real Hash Broker mint on Robinhood Chain.

Transaction 0xeeb4cf12...c75a minted token 0x101. Its calldata carries the nonce
and the challenge; the Mined log carries the resulting hash and the difficulty
it was accepted at. Reproducing that hash from the nonce and challenge is what
proves the miner's proof layout is the contract's.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pow as powlib  # noqa: E402
from protocol import PROTOCOL  # noqa: E402

MINER = "0x7156d3f8dee0659e95a816c08ef5f9a937777777"
NONCE = 0x56792605A02CA634
CHALLENGE = "0x4ebd68b1136b038cc386f91e4166e2e98af3b79db4de2c5691b1d381204a04db"
MINED_HASH = "0x0000000000001662dd2d10f3c56326b00d147e57cbc25a1c2dc3d2f80d3ac016"
DIFFICULTY = 0x32
MINT_VALUE_WEI = 0x5AF3107A4000
CALLDATA = (
    "0xe43e322c"
    "00000000000000000000000000000000000000000000000056792605a02ca634"
    "4ebd68b1136b038cc386f91e4166e2e98af3b79db4de2c5691b1d381204a04db"
)


class MainnetProofTests(unittest.TestCase):
    def test_reproduces_the_mined_hash(self):
        self.assertEqual("0x" + powlib.digest(MINER, NONCE, CHALLENGE).hex(), MINED_HASH)

    def test_preimage_is_84_bytes(self):
        self.assertEqual(len(powlib.preimage(MINER, NONCE, CHALLENGE)), 84)
        self.assertEqual(PROTOCOL.preimage_size, 84)

    def test_proof_beats_the_difficulty_it_was_accepted_at(self):
        proof = powlib.digest(MINER, NONCE, CHALLENGE)
        self.assertGreaterEqual(powlib.leading_zero_bits(proof), DIFFICULTY)
        self.assertTrue(powlib.meets(proof, powlib.target_for_difficulty(DIFFICULTY)))

    def test_a_wrong_wallet_does_not_reproduce_the_proof(self):
        other = "0x7156d3f8dee0659e95a816c08ef5f9a937777778"
        self.assertNotEqual("0x" + powlib.digest(other, NONCE, CHALLENGE).hex(), MINED_HASH)

    def test_rebuilds_the_exact_calldata(self):
        self.assertEqual(PROTOCOL.calldata(NONCE, CHALLENGE), CALLDATA)

    def test_mint_price_is_the_transaction_value(self):
        self.assertEqual(MINT_VALUE_WEI, 100_000_000_000_000)

    def test_gpu_message_agrees_with_the_reference(self):
        words = powlib.padded_words(MINER, CHALLENGE)
        stream_index, counter_index = powlib.nonce_word_indices()
        words[stream_index] = NONCE >> 32
        words[counter_index] = NONCE & 0xFFFFFFFF
        message = b"".join(word.to_bytes(4, "big") for word in words)
        self.assertEqual(message, powlib.sha256_pad(powlib.preimage(MINER, NONCE, CHALLENGE)))


if __name__ == "__main__":
    unittest.main()
