#!/usr/bin/env python3
"""The pure-Python keccak must agree with pycryptodome where it is available."""
import os
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from keccak_pure import keccak256, selector  # noqa: E402

try:
    from Crypto.Hash import keccak as _crypto_keccak
except ImportError:  # pragma: no cover - exercised on hosts without pycryptodome
    _crypto_keccak = None

KNOWN = {
    "": "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470",
    "abc": "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45",
}


class KeccakTests(unittest.TestCase):
    def test_known_vectors(self):
        for text, expected in KNOWN.items():
            self.assertEqual(keccak256(text.encode()).hex(), expected)

    def test_known_selectors(self):
        self.assertEqual(selector("transfer(address,uint256)"), "a9059cbb")
        self.assertEqual(selector("totalSupply()"), "18160ddd")
        self.assertEqual(selector("balanceOf(address)"), "70a08231")

    @unittest.skipIf(_crypto_keccak is None, "pycryptodome not installed")
    def test_matches_pycryptodome_across_block_boundaries(self):
        random.seed(4663)
        for size in (0, 1, 135, 136, 137, 271, 272, 500, os.cpu_count() or 1):
            data = bytes(random.randrange(256) for _ in range(size))
            expected = _crypto_keccak.new(digest_bits=256, data=data).digest()
            self.assertEqual(keccak256(data), expected, f"size {size}")


if __name__ == "__main__":
    unittest.main()
