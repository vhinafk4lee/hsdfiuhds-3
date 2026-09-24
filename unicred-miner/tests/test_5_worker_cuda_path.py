"""Worker CUDA path (ctypes Driver API calls) against a fake libcuda that runs kernel.cu on the CPU.

Exercises exactly the code that runs on a vast.ai server: context creation,
PTX module loading (+ fallback to a lower PTX ISA for old drivers), kernel
parameter marshalling, result buffers, selftest, bench and the JOB/FOUND protocol.
"""
import os
import subprocess
import sys
import unittest

from tests.helpers import ROOT, build_fake_libcuda
from tests.test_4_e2e import WorkerProc
from unicred import pow as P

WORKER = str(ROOT / "worker" / "worker.py")


class FakeCudaWorkerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = dict(os.environ, UNICRED_LIBCUDA=build_fake_libcuda())

    def run_worker(self, args, **env):
        e = dict(self.env, **env)
        return subprocess.run([sys.executable, WORKER] + args, capture_output=True, text=True, env=e, timeout=300)

    def test_selftest_via_driver_api(self):
        res = self.run_worker(["--selftest"])
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        self.assertEqual(res.stdout.count("SELFTEST OK"), 2)
        self.assertIn("Fake GPU 1 (sm_89, 2 SM, 78 regs, grid 2x256, embedded PTX)", res.stdout)

    def test_old_driver_ptx_fallback(self):
        res = self.run_worker(["--selftest", "--gpus", "0"], FAKE_CUDA_MAX_PTX="7.0")
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        self.assertIn("embedded PTX as ISA 7.0", res.stdout)

    def test_too_old_driver_fails_cleanly(self):
        res = self.run_worker(["--selftest"], FAKE_CUDA_MAX_PTX="6.0")
        self.assertEqual(res.returncode, 1)
        self.assertIn("SELFTEST FAIL", res.stdout)

    def test_bench(self):
        res = self.run_worker(["--bench", "2"])
        self.assertEqual(res.returncode, 0, res.stderr)
        line = [l for l in res.stdout.splitlines() if l.startswith("BENCH ")][0].split()
        self.assertGreater(int(line[1]), 0)
        self.assertEqual(len(line[2].split(",")), 2)

    def test_protocol_via_driver_api(self):
        w = WorkerProc(env=self.env)
        self.assertTrue(w.expect("READY").startswith('READY 2 ["Fake GPU 0","Fake GPU 1"]'))
        bh, ch, miner, prefix16 = os.urandom(32), os.urandom(32), os.urandom(20), os.urandom(16)
        target = 1 << 245
        w.send("JOB 7 %s %s %064x %s" % (P.job_prefix(bh, ch).hex(), miner.hex(), target, prefix16.hex()))
        f = w.expect("FOUND 7 ").split()
        nonce, digest = bytes.fromhex(f[2]), bytes.fromhex(f[3])
        self.assertEqual(P.digest(bh, ch, miner, nonce), digest)
        self.assertLess(int.from_bytes(digest, "big"), target)
        self.assertEqual(nonce[:16], prefix16)
        w.p.stdin.close()
        self.assertEqual(w.p.wait(timeout=10), 0)
        w.p.stdout.close()
        w.p.stderr.close()


if __name__ == "__main__":
    unittest.main()
