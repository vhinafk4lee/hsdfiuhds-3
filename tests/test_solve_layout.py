#!/usr/bin/env python3
"""The solver must rediscover the layout we already know is right.

The fixture is the real Hash Broker mint pinned by tests/test_mainnet_proof.py,
so a solver that cannot find sha256(miner || nonce || challenge) here would not
be trusted on a new contract either.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import solve_layout  # noqa: E402

REPORT = {
    "chainId": "0x1237",
    "contract": "0x4272d6f51771839f596082ef48fa84d35239bab3",
    "transaction": {
        "from": "0x7156d3f8dee0659e95a816c08ef5f9a937777777",
        "to": "0x4272d6f51771839f596082ef48fa84d35239bab3",
        "value": "0x5af3107a4000",
        "input": ("0xe43e322c"
                  "00000000000000000000000000000000000000000000000056792605a02ca634"
                  "4ebd68b1136b038cc386f91e4166e2e98af3b79db4de2c5691b1d381204a04db"),
    },
    "receipt": {
        "logs": [
            {"topics": ["0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
                        "0x" + "00" * 32,
                        "0x0000000000000000000000007156d3f8dee0659e95a816c08ef5f9a937777777",
                        "0x0000000000000000000000000000000000000000000000000000000000000101"],
             "data": "0x"},
            {"topics": ["0x78defd2577b2a85477efb3541419a16a34f4db4615553d8c5c02f6b070df79ca",
                        "0x0000000000000000000000007156d3f8dee0659e95a816c08ef5f9a937777777",
                        "0x0000000000000000000000000000000000000000000000000000000000000101"],
             "data": ("0x00000000000000000000000000000000000000000000000056792605a02ca634"
                      "0000000000001662dd2d10f3c56326b00d147e57cbc25a1c2dc3d2f80d3ac016"
                      "0000000000000000000000000000000000000000000000000000000000000032")},
        ]
    },
}
KNOWN_PROOF = "0x0000000000001662dd2d10f3c56326b00d147e57cbc25a1c2dc3d2f80d3ac016"


class SolverTests(unittest.TestCase):
    def test_finds_the_known_layout(self):
        pool = solve_layout.components(REPORT)
        targets = solve_layout.proof_candidates(REPORT, None)
        self.assertIn(bytes.fromhex(KNOWN_PROOF[2:]), targets)
        hits = solve_layout.search(pool, targets)
        self.assertTrue(hits, "the solver failed on a layout we know")
        algorithm, labels, _ = hits[0]
        self.assertEqual(algorithm, "sha256")
        self.assertEqual(labels, ("miner20", "arg0_32", "arg1_32"))

    def test_ignores_small_integers_in_the_logs(self):
        """A token id and a difficulty have more leading zeros than any proof."""
        targets = solve_layout.proof_candidates(REPORT, None)
        self.assertNotIn((0x101).to_bytes(32, "big"), targets)
        self.assertNotIn((0x32).to_bytes(32, "big"), targets)

    def test_suggests_a_preimage_for_the_protocol_file(self):
        pool = solve_layout.components(REPORT)
        sizes = {label: len(value) for label, value in pool}
        suggestion = solve_layout.suggest(("miner20", "arg0_32", "arg1_32"), sizes)
        self.assertEqual(suggestion, [
            {"field": "wallet", "size": 20},
            {"field": "nonce", "size": 32},
            {"field": "challenge", "size": 32},
        ])

    def test_reports_when_nothing_matches(self):
        report = {**REPORT, "receipt": {"logs": []}}
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(SCRIPTS / "solve_layout.py"), "--report", str(path)],
                capture_output=True, text=True, timeout=120)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--hash", result.stdout + result.stderr)

    def test_command_line_prints_the_match(self):
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "report.json"
            path.write_text(json.dumps(REPORT), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(SCRIPTS / "solve_layout.py"), "--report", str(path)],
                capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("MATCH  sha256( miner20 || arg0_32 || arg1_32 )", result.stdout)
        self.assertIn('"field": "challenge"', result.stdout)


if __name__ == "__main__":
    unittest.main()
