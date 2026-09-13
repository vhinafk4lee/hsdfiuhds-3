#!/usr/bin/env python3
"""Fleet controller checks, using a fake ssh that runs commands locally."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
FAKE_SSH = Path(__file__).resolve().parent / "fake_ssh.py"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eth_account import Account  # noqa: E402

import fleet  # noqa: E402
import pow as powlib  # noqa: E402
from stub_chain import StubChain, StubServer  # noqa: E402

CHALLENGE = "0x" + "5b" * 32
KEY = "0x" + "33" * 32
ACCOUNT = Account.from_key(KEY)


def rentals_file(directory: Path, name: str = "box") -> Path:
    path = directory / "rentals.json"
    path.write_text(json.dumps([{"name": name, "host": "192.0.2.10", "port": 40123}]),
                    encoding="utf-8")
    return path


def proof_for(wallet: str, challenge: str, difficulty: int) -> dict:
    target = powlib.target_for_difficulty(difficulty)
    for nonce in range(1 << 22):
        digest = powlib.digest(wallet, nonce, challenge)
        if int.from_bytes(digest, "big") < target:
            return {"wallet": wallet, "nonce": str(nonce), "hash": "0x" + digest.hex(),
                    "challenge": challenge, "difficulty": difficulty, "foundAt": int(time.time())}
    raise AssertionError("no proof found")


class RentalsTests(unittest.TestCase):
    def setUp(self):
        self.workdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.workdir.cleanup)
        self.root = Path(self.workdir.name)

    def test_reads_hosts(self):
        rentals = fleet.load_rentals(rentals_file(self.root))
        self.assertEqual(len(rentals), 1)
        self.assertEqual(rentals[0].target, "root@192.0.2.10")
        self.assertEqual(rentals[0].port, 40123)

    def test_rejects_a_missing_host(self):
        path = self.root / "rentals.json"
        path.write_text(json.dumps([{"name": "box", "port": 22}]), encoding="utf-8")
        with self.assertRaises(SystemExit):
            fleet.load_rentals(path)

    def test_rejects_a_bad_port(self):
        path = self.root / "rentals.json"
        path.write_text(json.dumps([{"host": "h", "port": "ssh"}]), encoding="utf-8")
        with self.assertRaises(SystemExit):
            fleet.load_rentals(path)

    def test_missing_file_explains_itself(self):
        with self.assertRaisesRegex(SystemExit, "rentals.example.json"):
            fleet.load_rentals(self.root / "nope.json")


class ConnectStringTests(unittest.TestCase):
    def test_reads_a_vast_connect_string(self):
        self.assertEqual(
            fleet.parse_ssh_target("ssh -p 41095 root@137.175.76.24 -L 8080:localhost:8080"),
            ("root", "137.175.76.24", 41095))

    def test_reads_a_bare_target(self):
        self.assertEqual(fleet.parse_ssh_target("user@example.net"), ("user", "example.net", 22))

    def test_reads_host_colon_port(self):
        self.assertEqual(fleet.parse_ssh_target("example.net:2222"), ("root", "example.net", 2222))

    def test_refuses_a_string_without_a_host(self):
        with self.assertRaises(SystemExit):
            fleet.parse_ssh_target("ssh -p 22")

    def test_refuses_a_port_that_is_not_a_port(self):
        with self.assertRaises(SystemExit):
            fleet.parse_ssh_target("ssh -p http root@example.net")


class AddTests(unittest.TestCase):
    def setUp(self):
        self.workdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.workdir.cleanup)
        self.path = Path(self.workdir.name) / "rentals.json"

    def add(self, *arguments: str):
        return subprocess.run(
            [sys.executable, str(SCRIPTS / "fleet.py"), "add", "--rentals", str(self.path),
             *arguments],
            capture_output=True, text=True, timeout=60,
        )

    def test_creates_the_file_and_appends(self):
        first = self.add("--target", "ssh -p 41095 root@137.175.76.24 -L 8080:localhost:8080",
                         "--name", "vast-1", "--gpus", "1")
        self.assertEqual(first.returncode, 0, first.stderr)
        second = self.add("--host", "192.0.2.9", "--port", "2200")
        self.assertEqual(second.returncode, 0, second.stderr)

        rentals = fleet.load_rentals(self.path)
        self.assertEqual([rental.name for rental in rentals], ["vast-1", "box2"])
        self.assertEqual(rentals[0].host, "137.175.76.24")
        self.assertEqual(rentals[0].port, 41095)
        self.assertEqual(rentals[0].gpus, 1)
        self.assertEqual(rentals[1].port, 2200)

    def test_refuses_a_duplicate_host(self):
        self.add("--target", "ssh -p 41095 root@137.175.76.24")
        again = self.add("--target", "ssh -p 41095 root@137.175.76.24")
        self.assertNotEqual(again.returncode, 0)
        self.assertIn("already in", again.stdout + again.stderr)
        self.assertEqual(len(fleet.load_rentals(self.path)), 1)


class SshArgvTests(unittest.TestCase):
    def test_carries_port_key_and_command(self):
        rental = fleet.Rental("box", "192.0.2.10", 40123, "root", 1)
        argv = fleet.ssh_argv(rental, "echo hi", "ssh", "/home/me/.hashbroker/id_ed25519")
        self.assertEqual(argv[0], "ssh")
        self.assertIn("40123", argv)
        self.assertIn("BatchMode=yes", argv)
        self.assertIn("/home/me/.hashbroker/id_ed25519", argv)
        self.assertEqual(argv[-2:], ["root@192.0.2.10", "echo hi"])

    def test_key_is_optional(self):
        rental = fleet.Rental("box", "h", 22, "root", None)
        self.assertNotIn("-i", fleet.ssh_argv(rental, "true", "ssh", None))


@unittest.skipUnless(shutil.which("sh"), "the fake ssh runs POSIX shell commands")
class FleetCommandTests(unittest.TestCase):
    """Runs fleet.py as a process, with fake_ssh.py standing in for ssh."""

    def setUp(self):
        self.workdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.workdir.cleanup)
        self.root = Path(self.workdir.name)
        self.remote = self.root / "remote"      # what the worker directory would be
        self.remote.mkdir()
        self.solutions = self.root / "solutions"
        self.rentals = rentals_file(self.root)
        self.environment = {**os.environ, "PYTHONPATH": str(SCRIPTS), "PYTHONUNBUFFERED": "1"}

    def fleet(self, *arguments: str, timeout: float = 120.0):
        return subprocess.run(
            [sys.executable, str(SCRIPTS / "fleet.py"), *arguments,
             "--rentals", str(self.rentals), "--ssh", str(FAKE_SSH), "--dir", str(self.remote),
             "--solutions", str(self.solutions)],
            env=self.environment, capture_output=True, text=True, timeout=timeout,
        )

    def test_check_reports_a_reachable_host(self):
        result = self.fleet("check")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)
        self.assertIn("gpus=", result.stdout)

    def test_collect_claims_a_solution_and_removes_it_remotely(self):
        solution = {"wallet": ACCOUNT.address, "nonce": "1", "hash": "0x" + "0a" * 32,
                    "challenge": CHALLENGE, "difficulty": 12, "foundAt": 0}
        (self.remote / "solution-gpu0.json").write_text(json.dumps(solution), encoding="utf-8")

        result = self.fleet("collect", "--run-for", "4")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SOLUTION from box", result.stdout)

        collected = sorted(self.solutions.glob("solution-box-*.json"))
        self.assertEqual(len(collected), 1)
        self.assertEqual(json.loads(collected[0].read_text())["hash"], solution["hash"])
        self.assertFalse((self.remote / "solution-gpu0.json").exists(),
                         "the worker's file must be claimed, not left to be re-sent")
        self.assertEqual(list(self.remote.glob("*.claim")), [])

    def test_run_collects_from_the_box_and_signs_locally(self):
        chain = StubChain(CHALLENGE, difficulty=12)
        with StubServer(chain) as server:
            self.environment.update({
                "HASHBROKER_RPC_URLS": server.url,
                "HASHBROKER_WALLET": ACCOUNT.address,
                "HASHBROKER_PRIVATE_KEY": KEY,
                "HASHBROKER_RUNTIME_DIR": str(self.root / "runtime"),
            })
            solution = proof_for(ACCOUNT.address, CHALLENGE, 12)
            (self.remote / "solution-gpu0.json").write_text(json.dumps(solution), encoding="utf-8")

            result = self.fleet("run", "--dry-run", "--run-for", "12", timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout[-1500:] + result.stderr[-1500:])
            self.assertIn("SOLUTION from box", result.stdout)
            self.assertIn("[signer] SIGNED", result.stdout)
            self.assertEqual(chain.sent, [], "a dry run must never broadcast")

    def test_stop_reaches_every_host(self):
        result = self.fleet("stop")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("box", result.stdout)


@unittest.skipUnless(shutil.which("sh"), "the collector's remote loop is POSIX shell")
class StreamProtocolTests(unittest.TestCase):
    def test_command_claims_prints_and_deletes(self):
        with tempfile.TemporaryDirectory() as workdir:
            root = Path(workdir)
            (root / "solution-gpu0.json").write_text('{"hash":"0x01"}\n', encoding="utf-8")
            script = textwrap.dedent(fleet.stream_command(str(root)))
            process = subprocess.Popen(["sh", "-c", script], stdout=subprocess.PIPE, text=True)

            def stop():
                process.kill()
                process.wait(timeout=10)
                if process.stdout:
                    process.stdout.close()

            self.addCleanup(stop)
            line = process.stdout.readline().strip()
            self.assertEqual(json.loads(line)["hash"], "0x01")
            self.assertFalse((root / "solution-gpu0.json").exists())


if __name__ == "__main__":
    unittest.main()
