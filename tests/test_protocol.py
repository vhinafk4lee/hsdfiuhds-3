#!/usr/bin/env python3
"""Selecting which contract to mine."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import protocol  # noqa: E402


class ProtocolSelectionTests(unittest.TestCase):
    def test_hashbroker_ships_with_the_miner(self):
        self.assertIn("hashbroker", protocol.available())

    def test_resolves_a_name_from_the_environment(self):
        with mock.patch.dict(os.environ, {"HASHBROKER_PROTOCOL": "hashbroker",
                                          "HASHBROKER_PROTOCOL_FILE": ""}):
            self.assertEqual(protocol.resolve().stem, "hashbroker")

    def test_an_unknown_name_lists_what_exists(self):
        with mock.patch.dict(os.environ, {"HASHBROKER_PROTOCOL": "nosuchproject",
                                          "HASHBROKER_PROTOCOL_FILE": ""}):
            with self.assertRaisesRegex(SystemExit, "available: "):
                protocol.resolve()

    def test_an_explicit_file_wins(self):
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "other.json"
            payload = json.loads((protocol.PROTOCOL_DIR / "hashbroker.json").read_text())
            payload["name"] = "other"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.dict(os.environ, {"HASHBROKER_PROTOCOL": "hashbroker"}):
                self.assertEqual(protocol.load(path).name, "other")

    def test_loaded_protocol_builds_the_mine_calldata(self):
        loaded = protocol.load(protocol.PROTOCOL_DIR / "hashbroker.json")
        calldata = loaded.calldata(1, "0x" + "ab" * 32)
        self.assertTrue(calldata.startswith("0x" + loaded.mine_selector))
        self.assertEqual(len(calldata), 2 + 8 + 128)


if __name__ == "__main__":
    unittest.main()
