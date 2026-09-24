"""Test 4: end to end with a fake CPU worker and a mocked RPC (127.0.0.1 only).

job -> FOUND -> signed EIP-1559 mint() transaction with the right calldata.
The "worker" is the real worker/worker.py with kernel.cu built for the CPU;
the "chain" is tests/mock_chain.py. No request ever leaves localhost.
"""
import json
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from eth_account import Account

from tests.helpers import ROOT, build_cpu_kernel
from tests.mock_chain import H, MockChain, serve
from unicred import pow as P
from unicred.chain import Chain
from unicred.config import load_config
from unicred.miner import Miner
from unicred.servers import load_servers
from unicred.signer import Wallet

GWEI = 10 ** 9


def wait_for(cond, timeout, step=0.05):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(step)
    return False


class WorkerProc(object):
    """worker.py over pipes (same protocol the SSH channel carries)."""

    def __init__(self, lib=None, devices=2, env=None):
        args = ["--cpu-lib", lib, "--cpu-devices", str(devices)] if lib else []
        self.p = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "worker" / "worker.py"), "--idle-timeout", "20"] + args,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, env=env)
        self.q = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in iter(self.p.stdout.readline, b""):
            self.q.put(line.decode().strip())
        self.q.put(None)

    def send(self, line):
        self.p.stdin.write((line + "\n").encode())
        self.p.stdin.flush()

    def expect(self, prefix, timeout=30):
        end = time.time() + timeout
        while time.time() < end:
            try:
                line = self.q.get(timeout=max(0.01, end - time.time()))
            except queue.Empty:
                break
            if line is None:
                raise AssertionError("worker exited while waiting for %s" % prefix)
            if line.startswith(prefix):
                return line
        raise AssertionError("timeout waiting for %s" % prefix)


class WorkerProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = build_cpu_kernel()

    def test_selftest_cli(self):
        res = subprocess.run([sys.executable, str(ROOT / "worker" / "worker.py"), "--selftest",
                              "--cpu-lib", self.lib], capture_output=True, text=True, timeout=120)
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        self.assertIn("SELFTEST PASS", res.stdout)

    def test_job_found_ping_idle_eof(self):
        w = WorkerProc(self.lib)
        ready = w.expect("READY")
        self.assertTrue(ready.startswith("READY 2 "))
        bh, ch, miner, prefix16 = os.urandom(32), os.urandom(32), os.urandom(20), os.urandom(16)
        target = 1 << 246
        w.send("JOB j1 %s %s %064x %s" % (P.job_prefix(bh, ch).hex(), miner.hex(), target, prefix16.hex()))
        found = w.expect("FOUND j1 ").split()
        nonce, digest = bytes.fromhex(found[2]), bytes.fromhex(found[3])
        self.assertEqual(nonce[:16], prefix16)                  # controller prefix
        self.assertIn(nonce[16], (0, 1))                        # GPU index byte
        self.assertEqual(P.digest(bh, ch, miner, nonce), digest)  # exact re-check
        self.assertLess(int.from_bytes(digest, "big"), target)
        hr = w.expect("HR ").split()
        self.assertGreater(float(hr[1]), 0)
        self.assertEqual(len(hr[2].split(",")), 2)
        w.send("PING tok42")
        self.assertEqual(w.expect("PONG"), "PONG tok42")
        w.send("IDLE")
        w.send("JOB bad zz")
        w.expect("ERR")
        w.p.stdin.close()                                        # controller gone -> EOF
        self.assertEqual(w.p.wait(timeout=10), 0)
        w.p.stdout.close()
        w.p.stderr.close()


class ControllerTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = build_cpu_kernel()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.chain = MockChain(target=1 << 242)
        self.srv, self.url = serve(self.chain)
        self.acct = Account.create()
        (self.dir / "wallet.key").write_text(self.acct.key.hex() if self.acct.key.hex().startswith("0x")
                                             else "0x" + self.acct.key.hex())
        (self.dir / "servers.txt").write_text(
            "# test\nlocal --cpu-lib %s --cpu-devices 2\n" % self.lib)
        self.miner = None

    def tearDown(self):
        if self.miner:
            self.miner.stop()
        self.srv.shutdown()
        time.sleep(0.2)
        self.tmp.cleanup()

    def config(self, **over):
        cfg = {"rpc_url": self.url, "private_key_file": "wallet.key", "servers_file": "servers.txt",
               "runtime_dir": "runtime", "poll_interval": 0.1, "max_mints": 3, "max_total_spend_eth": 1.0,
               "max_price_eth": 0.01, "min_balance_eth": 0.0, "priority_fee_gwei": 0.05,
               "max_fee_gwei": 2.0, "gas_limit": 350000}
        cfg.update(over)
        (self.dir / "config.json").write_text(json.dumps(cfg))
        return load_config(str(self.dir / "config.json"))

    def make_miner(self, cfg, dry_run=False, with_servers=True):
        wallet = Wallet(cfg.path("private_key_file"))
        chain = Chain(cfg.rpc_url, wallet.address)
        specs = load_servers(cfg.path("servers_file")) if with_servers else []
        self.miner = Miner(cfg, chain, wallet.address, wallet=wallet, specs=specs, dry_run=dry_run)
        return self.miner


class ControllerE2ETest(ControllerTestBase):
    def test_mints_with_signed_transactions(self):
        cfg = self.config()
        self.assertTrue(cfg.rpc_url.startswith("http://127.0.0.1:"))  # never the real network
        m = self.make_miner(cfg)
        m.start()
        ok = wait_for(lambda: self.chain.send_count >= 3 and not m.state["pending"], 90)
        self.assertTrue(ok, "no 3 mints; events: %s" % list(m.rt.events)[-10:])
        time.sleep(1.0)  # would-be extra sends show up here
        self.assertEqual(self.chain.send_count, 3, "max_mints must stop sending")
        self.assertIn("max_mints", m.halt or "")

        challenges = set()
        for i, t in enumerate(self.chain.txs):
            tx = t["tx"]
            self.assertEqual(t["sender"], self.acct.address)
            self.assertEqual(tx["type"], 2)
            self.assertEqual(tx["chainId"], 130)
            self.assertEqual(tx["nonce"], i)                     # local nonce counter
            self.assertEqual(tx["to"].hex().lower().replace("0x", ""), P.UNICRED[2:].lower())
            self.assertEqual(tx["value"], 4 * 10 ** 15)
            self.assertEqual(tx["gas"], 350000)
            self.assertEqual(tx["maxPriorityFeePerGas"], int(0.05 * GWEI))
            self.assertEqual(tx["maxFeePerGas"], 2 * 3_000_000 + int(0.05 * GWEI))
            call = P.decode_mint_calldata("0x" + bytes(tx["data"]).hex())
            self.assertEqual(call["max_price"], 4 * 10 ** 15)
            d = P.digest(H(call["anchor_block"]), t["challenge"], self.acct.address, call["nonce"])
            self.assertLess(int.from_bytes(d, "big"), 1 << 242, "PoW of the sent nonce")
            self.assertEqual(t["status"], "0x1")
            challenges.add(t["challenge"])
        self.assertEqual(len(challenges), 3, "one send per challenge")

        wait_for(lambda: m.state["mints_ok"] == 3, 5)
        st = json.loads((cfg.runtime / "state.json").read_text())
        self.assertEqual(st["mints_ok"], 3)
        fee = 200_000 * (3_000_000 + int(0.05 * GWEI)) + 1000
        self.assertEqual(st["spent_wei"], 3 * (4 * 10 ** 15 + fee))
        rows = (cfg.runtime / "txs.csv").read_text().strip().splitlines()
        self.assertEqual(len(rows), 4)  # header + 3
        self.assertTrue(all(",ok," in r for r in rows[1:]))
        log = (cfg.runtime / "log.txt").read_text()
        self.assertNotIn(self.acct.key.hex().replace("0x", ""), log)  # key never logged

    def test_dry_run_never_sends(self):
        cfg = self.config(max_mints=1)
        m = self.make_miner(cfg, dry_run=True)
        m.start()
        self.assertTrue(wait_for(lambda: m.stats["dry"] >= 1, 60))
        self.chain.foreign_mint()  # someone else mints -> new challenge -> new job
        self.assertTrue(wait_for(lambda: m.stats["dry"] >= 2, 60))
        self.assertEqual(self.chain.calls.get("eth_sendRawTransaction", 0), 0)
        self.assertIsNone(m.halt)  # limits do not stop a dry run


class CandidateLogicTest(ControllerTestBase):
    def _prepared(self, **over):
        cfg = self.config(**over)
        m = self.make_miner(cfg, with_servers=False)
        m.tx_nonce = 0
        m.poll_once()
        return m

    def _solve(self, m, job):
        n = 0
        while True:
            d = P.digest(job.anchor_hash, job.challenge, m.address, n)
            if int.from_bytes(d, "big") < job.target:
                return n, d.hex()
            n += 1

    def test_stale_candidate_is_dropped(self):
        m = self._prepared()
        job = m.job
        nonce, dig = self._solve(m, job)
        self.chain.foreign_mint()
        m.poll_once()
        self.assertNotEqual(m.job.challenge, job.challenge)
        m.handle_candidate(job, nonce, dig, "test")
        self.assertEqual(m.stats["stale"], 1)
        self.assertEqual(self.chain.send_count, 0)

    def test_bad_digest_is_rejected(self):
        m = self._prepared()
        nonce, dig = self._solve(m, m.job)
        m.handle_candidate(m.job, nonce + 1, dig, "test")
        self.assertEqual(m.stats["bad"], 1)
        self.assertEqual(self.chain.send_count, 0)

    def test_one_send_per_challenge(self):
        m = self._prepared()
        job = m.job
        n1, d1 = self._solve(m, job)
        m.handle_candidate(job, n1, d1, "a")
        m.handle_candidate(job, n1, d1, "b")
        self.assertEqual(self.chain.send_count, 1)
        self.assertEqual(m.tx_nonce, 1)

    def test_spend_cap_blocks(self):
        m = self._prepared(max_total_spend_eth=0.001)
        self.assertIn("max_total_spend_eth", m.halt)
        self.assertIsNone(m.job)

    def test_price_cap_blocks(self):
        m = self._prepared(max_price_eth=0.001)
        self.assertIn("цена", m.halt)

    def test_new_job_on_target_growth_and_anchor_age(self):
        m = self._prepared(anchor_refresh_blocks=3)
        j1 = m.job
        self.chain.target <<= 1
        m.poll_once()
        self.assertNotEqual(m.job.id, j1.id)
        self.assertEqual(m.job.anchor_block, j1.anchor_block)   # same anchor, bigger target
        j2 = m.job
        wait_for(lambda: self.chain.head() - j2.anchor_block > 3, 5)
        m.poll_once()
        self.assertGreater(m.job.anchor_block, j2.anchor_block)
        self.assertEqual(m.job.anchor_hash, H(m.job.anchor_block))  # blockhash(anchor) via parentHash

    def test_cached_candidate_submitted_when_target_grows(self):
        m = self._prepared(candidate_cache_shift=4)
        job = m.job
        n = 0
        while True:  # a digest in [target, 2*target)
            d = P.digest(job.anchor_hash, job.challenge, m.address, n)
            v = int.from_bytes(d, "big")
            if job.target <= v < 2 * job.target:
                break
            n += 1
        m.handle_candidate(job, n, d.hex(), "t")
        self.assertEqual(self.chain.send_count, 0)
        self.assertEqual(m.stats["cached"], 1)
        self.chain.target <<= 1
        m.running = True
        threading.Thread(target=m._submit_loop, daemon=True).start()
        m.poll_once()
        self.assertTrue(wait_for(lambda: self.chain.send_count == 1, 5))
        self.assertEqual(self.chain.txs[0]["status"], "0x1")
        m.running = False

    def test_servers_txt_hot_reload(self):
        cfg = self.config()
        m = self.make_miner(cfg)
        self.assertEqual([c.spec.name for c in m.conns], ["local"])
        started = []
        import unicred.servers as S
        orig = S.WorkerConn.start
        S.WorkerConn.start = lambda c: started.append(c.spec.name)
        try:
            (self.dir / "servers.txt").write_text(
                "ssh -p 31618 root@151.237.25.16 -L 8080:localhost:8080\n"
                "ssh -p 40017 root@14.234.172.15 -L 8080:localhost:8080\n")
            added, removed = m.reload_servers()
        finally:
            S.WorkerConn.start = orig
        self.assertEqual(sorted(started), ["14.234.172.15:40017", "151.237.25.16:31618"])
        self.assertEqual([c.spec.name for c in removed], ["local"])
        self.assertEqual(sorted(c.spec.name for c in m.conns), sorted(started))
        m.conns = []  # nothing really started

    def test_traits_command(self):
        import random as _r
        for i in range(40):  # 40 mints with random digests
            self.chain.minted += 1
            d = "%064x" % _r.getrandbits(250)
            n = self.chain.head() - 40 + i
            self.chain.logs.append({"address": P.UNICRED.lower(), "blockNumber": hex(n), "blockHash": H(n),
                                    "topics": [P.MINT_TOPIC, "0x%064x" % self.chain.minted, "0x" + "0" * 64],
                                    "data": "0x" + d + "%064x%064x" % (n - 1, 4 * 10 ** 15),
                                    "transactionHash": "0x" + "%064x" % i})
        from unicred import traits as T
        chain = Chain(self.url, self.acct.address)
        rows = T.collect(chain, count=30, progress=lambda *a: None)
        self.assertEqual(len(rows), 30)
        for r in rows:
            want = "Legendary" if int(r["digest"], 16) % 10 == 0 else "Common"
            self.assertEqual(r["trait:Rarity"], want)
        text = "\n".join(T.summary(rows))
        self.assertIn("Common", text)
        out = self.dir / "traits.csv"
        T.write_csv(rows, out)
        self.assertIn("trait:Rarity", out.read_text(encoding="utf-8-sig").splitlines()[0])

    def test_rpc_without_batch(self):
        srv, url = serve(self.chain, batch=False)
        try:
            chain = Chain(url, self.acct.address)
            s = chain.poll()
            self.assertFalse(chain.rpc.batch_ok)
            self.assertEqual(s["challenge"], self.chain.challenge)
            self.assertEqual(s["target"], self.chain.target)
        finally:
            srv.shutdown()


@unittest.skipIf(os.name == "nt", "POSIX signals")
class CliTest(ControllerTestBase):
    def run_cli(self, *args, timeout=120):
        return subprocess.run([sys.executable, str(ROOT / "unicred.py"), "--config",
                               str(self.dir / "config.json")] + list(args),
                              capture_output=True, text=True, timeout=timeout, cwd=str(self.dir))

    def test_check_command(self):
        self.config()
        res = self.run_cli("check")
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        self.assertIn("revert 0x7ca55c77", res.stdout)
        self.assertIn("все проверки пройдены", res.stdout)
        self.assertNotIn(self.acct.key.hex().replace("0x", ""), res.stdout)

    def test_run_dry_run_cli_and_ctrl_c(self):
        self.config()
        p = subprocess.Popen([sys.executable, "-u", str(ROOT / "unicred.py"), "--config",
                              str(self.dir / "config.json"), "run", "--dry-run", "--yes", "--no-dashboard"],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(self.dir))
        lines = []

        def reader():
            for l in iter(p.stdout.readline, b""):
                lines.append(l.decode(errors="replace"))

        threading.Thread(target=reader, daemon=True).start()
        ok = wait_for(lambda: any("DRY-RUN НАЙДЕН" in l for l in lines), 90)
        p.send_signal(signal.SIGINT)
        p.wait(timeout=30)
        out = "".join(lines)
        self.assertTrue(ok, out)
        self.assertIn("остановка", out)
        self.assertEqual(self.chain.calls.get("eth_sendRawTransaction", 0), 0)
        # worker got EOF and exited: no orphan worker process
        time.sleep(1.0)
        ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
        self.assertNotIn(str(self.dir / "runtime" / "local_worker"), ps)


if __name__ == "__main__":
    unittest.main()
