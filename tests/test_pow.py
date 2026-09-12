#!/usr/bin/env python3
"""Checks that the reference PoW, the padding, and the word layout agree."""
import hashlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pow as powlib  # noqa: E402
from protocol import PROTOCOL  # noqa: E402

WALLET = "0x1111111111111111111111111111111111111111"
CHALLENGE = "0x" + "22" * 32


class PreimageTests(unittest.TestCase):
    def test_layout_size(self):
        material = powlib.preimage(WALLET, 0, CHALLENGE)
        self.assertEqual(len(material), PROTOCOL.preimage_size)

    def test_nonce_is_big_endian_at_its_offset(self):
        nonce = 0x1234_5678_9ABC_DEF0
        material = powlib.preimage(WALLET, nonce, CHALLENGE)
        start = PROTOCOL.nonce_offset
        stop = start + PROTOCOL.nonce_size
        self.assertEqual(int.from_bytes(material[start:stop], "big"), nonce)

    def test_digest_matches_hashlib(self):
        material = powlib.preimage(WALLET, 7, CHALLENGE)
        self.assertEqual(powlib.digest(WALLET, 7, CHALLENGE),
                         hashlib.sha256(material).digest())


class MessageLayoutTests(unittest.TestCase):
    def test_padding_is_block_aligned(self):
        padded = powlib.sha256_pad(powlib.preimage(WALLET, 0, CHALLENGE))
        self.assertEqual(len(padded) % 64, 0)
        self.assertEqual(padded[PROTOCOL.preimage_size], 0x80)
        self.assertEqual(int.from_bytes(padded[-8:], "big"), PROTOCOL.preimage_size * 8)

    def test_word_substitution_reproduces_the_nonce(self):
        """Writing stream/counter into the two words equals hashing that nonce."""
        words = powlib.padded_words(WALLET, CHALLENGE)
        stream_index, counter_index = powlib.nonce_word_indices()
        stream, counter = 0x13579BDF, 0x2468ACE0
        words[stream_index] = stream
        words[counter_index] = counter
        message = b"".join(word.to_bytes(4, "big") for word in words)
        nonce = (stream << 32) | counter
        expected = powlib.sha256_pad(powlib.preimage(WALLET, nonce, CHALLENGE))
        self.assertEqual(message, expected)
        self.assertEqual(hashlib.sha256(message[:PROTOCOL.preimage_size]).digest(),
                         powlib.digest(WALLET, nonce, CHALLENGE))

    def test_message_fits_two_blocks(self):
        words = powlib.padded_words(WALLET, CHALLENGE)
        self.assertLessEqual(len(words), 32)


class DifficultyTests(unittest.TestCase):
    def test_target_is_two_to_the_remaining_bits(self):
        self.assertEqual(powlib.target_for_difficulty(50), 1 << 206)
        self.assertEqual(powlib.target_for_difficulty(0), 1 << 256)

    def test_target_rejects_an_impossible_difficulty(self):
        with self.assertRaises(ValueError):
            powlib.target_for_difficulty(256)

    def test_leading_zero_bits(self):
        self.assertEqual(powlib.leading_zero_bits(b"\x00" * 32), 256)
        self.assertEqual(powlib.leading_zero_bits(b"\x00" * 31 + b"\x01"), 255)
        self.assertEqual(powlib.leading_zero_bits(b"\x80" + b"\x00" * 31), 0)


class TargetTests(unittest.TestCase):
    def test_search_target_is_capped(self):
        self.assertEqual(powlib.search_target(hex(powlib.MAX_HASH)), powlib.MAX_HASH)

    def test_search_target_widens(self):
        self.assertEqual(powlib.search_target("0x10", slack=3), 0x80)

    def test_meets(self):
        self.assertTrue(powlib.meets((1).to_bytes(32, "big"), "0x02"))
        self.assertFalse(powlib.meets((2).to_bytes(32, "big"), "0x02"))


if __name__ == "__main__":
    unittest.main()
