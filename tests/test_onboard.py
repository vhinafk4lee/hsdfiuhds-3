#!/usr/bin/env python3
"""Onboarding a contract must reproduce the protocol file we hand-wrote.

The fixture is the real Hash Broker report, and the shipped
scripts/protocols/hashbroker.json is the answer key: if the tool cannot derive
that file from that mint, it cannot be trusted to derive a new project's.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import onboard  # noqa: E402
from protocol import PROTOCOL_DIR, load  # noqa: E402
from test_solve_layout import REPORT as BASE_REPORT  # noqa: E402

REPORT = {
    **BASE_REPORT,
    "mintCall": {"selector": "0xe43e322c", "signature": "mine(uint256,bytes32)"},
    "matchedViews": {
        "challenge()": "d2ef7398",
        "currentDifficulty()": "5c062d6c",
        "mintPrice()": "6817c76c",
        "totalSupply()": "18160ddd",
        "MAX_SUPPLY()": "32cb6b0c",
        "lastMintBlock()": "9cf5c3f5",
        "isValidProof(address,uint256,bytes32)": "f8cf640d",
        "owner()": "8da5cb5b",
    },
}


class OnboardTests(unittest.TestCase):
    def setUp(self):
        self.protocol, self.warnings = onboard.build(REPORT, "hashbroker-derived", None)

    def test_recovers_the_proof_layout(self):
        self.assertEqual(self.protocol["algorithm"], "sha256")
        self.assertEqual(self.protocol["preimage"], [
            {"field": "wallet", "size": 20},
            {"field": "nonce", "size": 32},
            {"field": "challenge", "size": 32},
        ])

    def test_recovers_the_call(self):
        self.assertEqual(self.protocol["mine"], "mine(uint256,bytes32)")
        self.assertEqual(self.protocol["mineSelector"], "0xe43e322c")
        self.assertEqual(self.protocol["mineArgs"], ["nonce", "challenge"])

    def test_recovers_the_views_the_miner_needs(self):
        shipped = json.loads((PROTOCOL_DIR / "hashbroker.json").read_text())
        self.assertEqual(self.protocol["views"], shipped["views"])
        self.assertEqual(self.protocol["validate"], shipped["validate"])
        self.assertEqual(self.protocol["chainId"], shipped["chainId"])

    def test_starts_unverified(self):
        self.assertFalse(self.protocol["verified"])
        self.assertEqual(self.warnings, [])

    def test_warns_about_a_view_it_could_not_find(self):
        stripped = {**REPORT, "matchedViews": {"mintPrice()": "6817c76c"}}
        _, warnings = onboard.build(stripped, "partial", None)
        self.assertTrue(any("no view found for" in warning for warning in warnings))
        self.assertTrue(any("challenge" in warning for warning in warnings))

    def test_written_file_loads_as_a_protocol(self):
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "derived.json"
            payload = {**self.protocol, "rpc": ["https://rpc.example"]}
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load(path)
            self.assertEqual(loaded.preimage_size, 84)
            self.assertEqual(loaded.mine_selector, "e43e322c")
            self.assertEqual(
                loaded.calldata(0x56792605A02CA634,
                                "0x4ebd68b1136b038cc386f91e4166e2e98af3b79db4de2c5691b1d381204a04db"),
                REPORT["transaction"]["input"])

    def test_a_report_without_a_proof_says_so(self):
        with self.assertRaisesRegex(SystemExit, "--hash"):
            onboard.build({**REPORT, "receipt": {"logs": []}}, "empty", None)


if __name__ == "__main__":
    unittest.main()
