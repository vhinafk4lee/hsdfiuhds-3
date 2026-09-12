#!/usr/bin/env python3
"""Runs the real processes end to end against the stub chain.

feed -> job.json -> CPU worker -> solution.json -> signer (dry run). The pieces
that cannot be exercised on the developer's machine are the GPU kernel (covered
by tests/test_kernel_sha256.py) and the live chain; everything between them is
covered here.
"""
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eth_account import Account  # noqa: E402
from stub_chain import StubChain, StubServer  # noqa: E402

CHALLENGE = "0x" + "3a" * 32
KEY = "0x" + "22" * 32
ACCOUNT = Account.from_key(KEY)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.chain = StubChain(CHALLENGE, difficulty=14)
        self.server = StubServer(self.chain)
        self.server.__enter__()
        self.addCleanup(self.server.__exit__)
        self.workdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.workdir.cleanup)
        self.root = Path(self.workdir.name)
        self.environment = {
            **os.environ,
            "HASHBROKER_RPC_URLS": self.server.url,
            "HASHBROKER_WALLET": ACCOUNT.address,
            "HASHBROKER_PRIVATE_KEY": KEY,
            "HASHBROKER_RUNTIME_DIR": str(self.root / "runtime"),
            "PYTHONPATH": str(SCRIPTS),
        }

    def run_script(self, name: str, *arguments: str, timeout: float = 60.0):
        return subprocess.run(
            [sys.executable, str(SCRIPTS / name), *arguments],
            env=self.environment, capture_output=True, text=True, timeout=timeout,
        )

    def test_feed_publishes_a_job_the_worker_can_read(self):
        result = self.run_script("job_feed.py", "--wallet", ACCOUNT.address, "--once")
        self.assertEqual(result.returncode, 0, result.stderr)
        job = json.loads(result.stdout)
        self.assertEqual(job["challenge"], CHALLENGE)
        self.assertEqual(job["difficulty"], 14)

    def test_full_pipeline_from_feed_to_signed_transaction(self):
        job_file = self.root / "job.json"
        solution_file = self.root / "solution.json"
        feed = subprocess.Popen(
            [sys.executable, str(SCRIPTS / "job_feed.py"), "--wallet", ACCOUNT.address,
             "--output", str(job_file), "--interval", "0.2"],
            env=self.environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        def stop_feed():
            feed.kill()
            feed.wait(timeout=10)
            for stream in (feed.stdout, feed.stderr):
                if stream:
                    stream.close()

        self.addCleanup(stop_feed)

        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and not job_file.exists():
            time.sleep(0.1)
        self.assertTrue(job_file.exists(), "feed never wrote a job file")

        mined = self.run_script(
            "miner_cpu.py", "--wallet", ACCOUNT.address, "--job-file", str(job_file),
            "--output", str(solution_file), "--processes", "2", timeout=120,
        )
        self.assertEqual(mined.returncode, 0, mined.stderr)
        self.assertIn("SOLUTION", mined.stdout)

        solution = json.loads(solution_file.read_text())
        self.assertEqual(solution["challenge"], CHALLENGE)
        self.assertEqual(solution["wallet"], ACCOUNT.address)

        signed = self.run_script("signer.py", "--solutions", str(solution_file),
                                 "--once", "--dry-run", timeout=60)
        self.assertEqual(signed.returncode, 0, signed.stderr)
        self.assertIn("SIGNED", signed.stdout)
        self.assertIn("DRY_RUN", signed.stdout)
        self.assertEqual(self.chain.sent, [], "a dry run must never broadcast")

        intents = list((self.root / "runtime").glob("intent-*.json"))
        self.assertEqual(len(intents), 1)

    def test_signer_check_reports_state_without_spending(self):
        result = self.run_script("signer.py", "--check")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(ACCOUNT.address, result.stdout)
        self.assertIn("difficulty 14", result.stdout)
        self.assertEqual(self.chain.sent, [])


class SupervisorTests(unittest.TestCase):
    """run_all.py must bring the whole miner up, and take it down again, alone."""

    def setUp(self):
        self.chain = StubChain(CHALLENGE, difficulty=14)
        self.server = StubServer(self.chain)
        self.server.__enter__()
        self.addCleanup(self.server.__exit__)
        self.workdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.workdir.cleanup)
        self.root = Path(self.workdir.name)
        self.environment = {
            **os.environ,
            "HASHBROKER_RPC_URLS": self.server.url,
            "HASHBROKER_WALLET": ACCOUNT.address,
            "HASHBROKER_PRIVATE_KEY": KEY,
            "HASHBROKER_RUNTIME_DIR": str(self.root / "runtime"),
            "PYTHONPATH": str(SCRIPTS),
        }

    def test_cpu_mode_mines_and_signs_under_one_command(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "run_all.py"), "--wallet", ACCOUNT.address,
             "--dir", str(self.root), "--mode", "cpu", "--dry-run", "--run-for", "25"],
            env=self.environment, capture_output=True, text=True, timeout=180,
        )
        self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr[-2000:])
        self.assertIn("[feed]", result.stdout)
        self.assertIn("[cpu]", result.stdout)
        self.assertIn("[signer]", result.stdout)
        self.assertIn("SIGNED", result.stdout)
        self.assertIn("STOPPING", result.stdout)
        self.assertNotIn("RPC_RETRY", result.stdout, "the feed must not error on a valid job")
        self.assertNotIn("Traceback", result.stdout)
        self.assertEqual(self.chain.sent, [], "a dry run must never broadcast")

    def test_status_renders_what_the_feed_published(self):
        subprocess.run(
            [sys.executable, str(SCRIPTS / "job_feed.py"), "--wallet", ACCOUNT.address,
             "--output", str(self.root / "job.json"), "--interval", "0.2", "--once"],
            env=self.environment, capture_output=True, text=True, timeout=60, check=True,
        )
        # --once prints the job instead of writing it, so write one the same way.
        job_file = self.root / "job.json"
        if not job_file.exists():
            sys.path.insert(0, str(SCRIPTS))
            import importlib

            job_state = importlib.import_module("job_state")
            protocol = importlib.import_module("protocol")
            original = job_state.PROTOCOL
            job_state.PROTOCOL = dataclasses.replace(original, rpc=(self.server.url,))
            try:
                job = job_state.read_job(ACCOUNT.address)
                job_state.write_shared_job(job_file, ACCOUNT.address, job)
            finally:
                job_state.PROTOCOL = original
                del protocol

        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "status.py"), "--dir", str(self.root), "--once"],
            env=self.environment, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(CHALLENGE, result.stdout)
        self.assertIn("difficulty 14", result.stdout)


if __name__ == "__main__":
    unittest.main()
