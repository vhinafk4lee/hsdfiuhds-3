#!/usr/bin/env python3
"""The mining key must land on disk readable by nobody else, and never be reused."""
import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eth_account import Account  # noqa: E402

import newkey  # noqa: E402


class NewKeyTests(unittest.TestCase):
    def setUp(self):
        self.workdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.workdir.cleanup)
        self.path = Path(self.workdir.name) / "wallet.key"

    def test_writes_a_0600_key_matching_the_printed_address(self):
        address = newkey.create(self.path)
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        stored = self.path.read_text(encoding="utf-8").strip()
        self.assertEqual(Account.from_key(stored).address, address)

    def test_refuses_to_overwrite_an_existing_key(self):
        newkey.create(self.path)
        original = self.path.read_text(encoding="utf-8")
        with self.assertRaises(SystemExit):
            newkey.create(self.path)
        self.assertEqual(self.path.read_text(encoding="utf-8"), original)

    def test_keys_are_not_reused(self):
        first = newkey.create(self.path)
        second_path = self.path.with_name("second.key")
        self.assertNotEqual(newkey.create(second_path), first)


if __name__ == "__main__":
    unittest.main()
