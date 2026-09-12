#!/usr/bin/env python3
"""End-to-end signer checks against an in-process stub chain."""
import dataclasses
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eth_account import Account  # noqa: E402

import job_state  # noqa: E402
import pow as powlib  # noqa: E402
import signer  # noqa: E402
from protocol import PROTOCOL  # noqa: E402
from stub_chain import StubChain, StubServer  # noqa: E402

CHALLENGE = "0x4ebd68b1136b038cc386f91e4166e2e98af3b79db4de2c5691b1d381204a04db"
ACCOUNT = Account.from_key("0x" + "11" * 32)


def find_proof(wallet: str, challenge: str, difficulty: int) -> tuple[int, bytes]:
    target = powlib.target_for_difficulty(difficulty)
    for nonce in range(1 << 22):
        digest = powlib.digest(wallet, nonce, challenge)
        if int.from_bytes(digest, "big") < target:
            return nonce, digest
    raise AssertionError("no proof found for the test difficulty")


def solution_for(wallet: str, challenge: str, difficulty: int) -> dict:
    nonce, digest = find_proof(wallet, challenge, difficulty)
    return {"wallet": wallet, "nonce": str(nonce), "hash": "0x" + digest.hex(),
            "challenge": challenge, "difficulty": difficulty, "foundAt": 0}


class SignerTestCase(unittest.TestCase):
    """Points every module at a stub chain and a scratch runtime directory."""

    difficulty = 10

    def setUp(self):
        self.chain = StubChain(CHALLENGE, difficulty=self.difficulty)
        self.server = StubServer(self.chain)
        self.server.__enter__()
        self.addCleanup(self.server.__exit__)
        protocol = dataclasses.replace(PROTOCOL, rpc=(self.server.url,), broadcast_rpc=())
        for module in (job_state, signer):
            patcher = mock.patch.object(module, "PROTOCOL", protocol)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.runtime = tempfile.TemporaryDirectory()
        self.addCleanup(self.runtime.cleanup)
        patcher = mock.patch.object(signer, "RUNTIME_DIR", Path(self.runtime.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.solution = solution_for(ACCOUNT.address, CHALLENGE, self.difficulty)


class JobReadTests(SignerTestCase):
    def test_read_job_reports_the_live_state(self):
        job = job_state.read_job(ACCOUNT.address)
        self.assertEqual(job["challenge"], CHALLENGE)
        self.assertEqual(job["difficulty"], self.difficulty)
        self.assertEqual(int(job["target"], 16), powlib.target_for_difficulty(self.difficulty))
        self.assertEqual(job["minted"], 260)
        self.assertEqual(job["maxSupply"], 4444)
        self.assertEqual(job["blockNumber"], self.chain.block_number)

    def test_read_job_refuses_the_wrong_chain(self):
        self.chain.chain_id = 1
        with self.assertRaises(RuntimeError):
            job_state.read_job(ACCOUNT.address)

    def test_read_job_refuses_an_implausible_difficulty(self):
        """A lagging or lying endpoint fails the read; the feed then rotates RPCs."""
        self.chain.difficulty = 0
        with self.assertRaisesRegex(ValueError, "implausible difficulty"):
            job_state.read_job(ACCOUNT.address)


class VerifyTests(SignerTestCase):
    def test_accepts_a_genuine_proof(self):
        state = signer.read_state(ACCOUNT.address)
        self.assertEqual(signer.verify_solution(self.solution, ACCOUNT.address, state),
                         int(self.solution["nonce"]))

    def test_rejects_a_stale_challenge(self):
        state = signer.read_state(ACCOUNT.address)
        stale = {**self.solution, "challenge": "0x" + "99" * 32}
        with self.assertRaisesRegex(ValueError, "moved past"):
            signer.verify_solution(stale, ACCOUNT.address, state)

    def test_rejects_a_hash_that_does_not_match_its_nonce(self):
        state = signer.read_state(ACCOUNT.address)
        lying = {**self.solution, "hash": "0x" + "00" * 32}
        with self.assertRaisesRegex(ValueError, "does not match"):
            signer.verify_solution(lying, ACCOUNT.address, state)

    def test_rejects_a_proof_that_misses_the_live_difficulty(self):
        state = signer.read_state(ACCOUNT.address)
        state["difficulty"] = 200
        state["target"] = powlib.target_for_difficulty(200)
        with self.assertRaisesRegex(ValueError, "zero bits"):
            signer.verify_solution(self.solution, ACCOUNT.address, state)

    def test_rejects_another_wallets_solution(self):
        state = signer.read_state(ACCOUNT.address)
        with self.assertRaisesRegex(ValueError, "another wallet"):
            signer.verify_solution({**self.solution, "wallet": "0x" + "ab" * 20},
                                   ACCOUNT.address, state)

    def test_asks_the_contract_to_confirm(self):
        nonce = int(self.solution["nonce"])
        self.assertTrue(signer.confirm_on_chain(ACCOUNT.address, nonce, CHALLENGE))
        self.assertFalse(signer.confirm_on_chain(ACCOUNT.address, nonce + 1, CHALLENGE))


class TransactionTests(SignerTestCase):
    def state(self) -> dict:
        return signer.read_state(ACCOUNT.address)

    def test_builds_the_expected_call(self):
        state = self.state()
        nonce = int(self.solution["nonce"])
        transaction = signer.build_transaction(ACCOUNT.address, nonce, CHALLENGE, state, 107_089)
        self.assertEqual(transaction["to"], PROTOCOL.contract)
        self.assertEqual(transaction["chainId"], PROTOCOL.chain_id)
        self.assertEqual(transaction["value"], self.chain.price)
        self.assertEqual(transaction["nonce"], self.chain.account_nonce)
        self.assertEqual(transaction["data"], PROTOCOL.calldata(nonce, CHALLENGE))
        self.assertGreater(transaction["gas"], 107_089)

    def test_refuses_to_exceed_the_cap(self):
        state = self.state()
        with mock.patch.object(signer, "SUBMIT_CAP_WEI", 1):
            with self.assertRaisesRegex(ValueError, "cap is"):
                signer.build_transaction(ACCOUNT.address, 1, CHALLENGE, state, 107_089)

    def test_refuses_an_oversized_gas_estimate(self):
        state = self.state()
        with self.assertRaisesRegex(ValueError, "safety limit"):
            signer.build_transaction(ACCOUNT.address, 1, CHALLENGE, state, 10_000_000)

    def test_refuses_when_the_balance_cannot_cover_the_ceiling(self):
        state = self.state()
        state["balanceWei"] = 1
        with self.assertRaisesRegex(ValueError, "below"):
            signer.build_transaction(ACCOUNT.address, 1, CHALLENGE, state, 107_089)


class SubmitTests(SignerTestCase):
    def test_signs_broadcasts_and_records_the_intent(self):
        tx_hash = signer.submit(ACCOUNT, self.solution, dry_run=False)
        self.assertEqual(tx_hash, self.chain.tx_hash)
        self.assertEqual(len(self.chain.sent), 1)
        intents = list(Path(self.runtime.name).glob("intent-*.json"))
        self.assertEqual(len(intents), 1)
        recorded = json.loads(intents[0].read_text())
        self.assertEqual(recorded["transaction"]["data"],
                         PROTOCOL.calldata(int(self.solution["nonce"]), CHALLENGE))

    def test_dry_run_signs_without_sending(self):
        self.assertIsNone(signer.submit(ACCOUNT, self.solution, dry_run=True))
        self.assertEqual(self.chain.sent, [])

    def test_a_stale_solution_never_reaches_the_chain(self):
        stale = {**self.solution, "challenge": "0x" + "77" * 32}
        with self.assertRaises(ValueError):
            signer.submit(ACCOUNT, stale, dry_run=False)
        self.assertEqual(self.chain.sent, [])


if __name__ == "__main__":
    unittest.main()
