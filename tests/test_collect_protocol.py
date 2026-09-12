#!/usr/bin/env python3
"""Offline checks for the bytecode scan and the calldata reading."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import collect_protocol as collector  # noqa: E402
from keccak_pure import selector  # noqa: E402


def dispatcher(*selectors: str) -> bytes:
    """A dispatcher-shaped stub: PUSH4 <sel> EQ PUSH2 dest JUMPI per entry."""
    code = bytearray(b"\x60\x00\x35\x60\xe0\x1c")  # PUSH1 0 CALLDATALOAD PUSH1 0xe0 SHR
    for index, value in enumerate(selectors):
        code += b"\x80\x63" + bytes.fromhex(value) + b"\x14\x61" + index.to_bytes(2, "big") + b"\x57"
    return bytes(code)


class BytecodeScanTests(unittest.TestCase):
    def test_extracts_dispatcher_selectors(self):
        wanted = [selector("mine(uint256,uint256)"), selector("mintPrice()")]
        found = collector.extract_selectors(dispatcher(*wanted))
        self.assertEqual(found, sorted(wanted))

    def test_skips_bytes_inside_a_push32(self):
        hidden = selector("transfer(address,uint256)")
        constant = b"\x7f" + bytes.fromhex(hidden) + b"\x11" * 28  # PUSH32 with the selector inside
        self.assertEqual(collector.extract_selectors(constant), [])

    def test_ignores_a_truncated_trailing_push(self):
        self.assertEqual(collector.extract_selectors(b"\x63\xaa\xbb"), [])

    def test_matches_only_present_signatures(self):
        present = {selector("prevWork()"), selector("targetFor(address)")}
        matched = collector.match_signatures(present, collector.VIEW_SIGNATURES)
        self.assertIn("prevWork()", matched)
        self.assertIn("targetFor(address)", matched)
        self.assertNotIn("mintPrice()", matched)


class CalldataTests(unittest.TestCase):
    def test_splits_arguments_into_words(self):
        calldata = "0x" + selector("mine(uint256,uint256)") + f"{42:064x}" + f"{7:064x}"
        words = collector.words_of(calldata)
        self.assertEqual([int(word, 16) for word in words], [42, 7])

    def test_reads_an_address_shaped_word(self):
        word = "0x" + "00" * 12 + "ab" * 20
        self.assertIn("address-shaped", collector.describe_word(word))

    def test_reads_a_small_integer(self):
        self.assertEqual(collector.describe_word("0x" + f"{4663:064x}"), "uint 4663")


if __name__ == "__main__":
    unittest.main()
