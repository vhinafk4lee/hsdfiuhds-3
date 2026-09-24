"""Test 2: kernel.cu compiled as plain C++ and run on the CPU.

* finds the test-vector nonce with the search kernel;
* the full digest (hash kernel) and the 128-bit prefix (search path) match
  pycryptodome on 1000 random inputs;
* the candidate filter reports exactly the right counters.
"""
import ctypes
import os
import random
import unittest

from tests.helpers import build_cpu_kernel, load_worker
from unicred import pow as P

U64x25 = ctypes.c_uint64 * 25


class KernelCpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = ctypes.CDLL(build_cpu_kernel())
        cls.w = load_worker()

    def cpu_hash(self, cx, ctr):
        out = (ctypes.c_uint64 * 4)()
        self.lib.cpu_hash(U64x25(*cx), ctypes.c_uint64(ctr), out)
        return b"".join(int(v).to_bytes(8, "little") for v in out)

    def cpu_prefix(self, cx, ctr):
        out = (ctypes.c_uint64 * 2)()
        self.lib.cpu_prefix(U64x25(*cx), ctypes.c_uint64(ctr), out)
        return b"".join(int(v).to_bytes(8, "little") for v in out)

    def cpu_search(self, cx, target, base, iters, grid=4, block=256):
        out = (ctypes.c_uint64 * 16)()
        t0, t1 = (target >> 192) & (2 ** 64 - 1), (target >> 128) & (2 ** 64 - 1)
        self.lib.cpu_search(U64x25(*cx), ctypes.c_uint64(t0), ctypes.c_uint64(t1), ctypes.c_uint64(base),
                            ctypes.c_uint32(iters), ctypes.c_uint32(grid), ctypes.c_uint32(block), out)
        return [int(out[1 + i]) for i in range(min(int(out[0]), 15))], int(out[0])

    def test_testvector_hash(self):
        job, gpu8, ctr = self.w.testvector_job()
        self.assertEqual(self.cpu_hash(job.cx_for(gpu8), ctr).hex(), P.TV["digest"][2:])

    def test_testvector_search_finds_nonce(self):
        job, gpu8, ctr = self.w.testvector_job()
        cx = job.cx_for(gpu8)
        per_launch = 4 * 256 * 8
        offset = 5555
        hits, n = self.cpu_search(cx, job.target, ctr - offset, 8)
        self.assertEqual(hits, [ctr])
        nonce = job.nonce(gpu8, hits[0])
        self.assertEqual(nonce, P.TV["nonce"].to_bytes(32, "big"))
        # range that does not contain the nonce: nothing
        hits, n = self.cpu_search(cx, job.target, ctr + 1, 8)
        self.assertEqual(n, 0)
        self.assertLess(offset, per_launch)

    def test_1000_random_inputs_vs_pycryptodome(self):
        rnd = random.Random(12345)
        for i in range(1000):
            bh = rnd.getrandbits(256).to_bytes(32, "big")
            ch = rnd.getrandbits(256).to_bytes(32, "big")
            miner = rnd.getrandbits(160).to_bytes(20, "big")
            nonce = rnd.getrandbits(256).to_bytes(32, "big")
            job = self.w.Job("t", P.job_prefix(bh, ch), miner, 1, nonce[:16])
            gpu8, ctr = nonce[16:24], int.from_bytes(nonce[24:], "big")
            cx = job.cx_for(gpu8)
            expect = P.digest(bh, ch, miner, nonce)
            self.assertEqual(self.cpu_hash(cx, ctr), expect, "input %d" % i)
            self.assertEqual(self.cpu_prefix(cx, ctr), expect[:16], "input %d" % i)

    def test_random_search_and_filter(self):
        """Target = (min digest in the launch range) + 1 -> exactly that counter is reported."""
        rnd = random.Random(7)
        for i in range(25):
            bh, ch, miner = os.urandom(32), os.urandom(32), os.urandom(20)
            prefix16, gpu8 = os.urandom(16), os.urandom(8)
            base = rnd.getrandbits(63)
            job = self.w.Job("t", P.job_prefix(bh, ch), miner, 1, prefix16)
            cx = job.cx_for(gpu8)
            digests = {c: int.from_bytes(P.digest(bh, ch, miner, prefix16 + gpu8 + c.to_bytes(8, "big")), "big")
                       for c in range(base, base + 1024)}
            best = min(digests, key=digests.get)
            hits, n = self.cpu_search(cx, digests[best] + 1, base, 1)
            self.assertEqual((hits, n), ([best], 1))
            # top 128 bits of the target below the best digest -> nothing reported
            hits, n = self.cpu_search(cx, ((digests[best] >> 128) - 1) << 128, base, 1)
            self.assertEqual(n, 0)

    def test_easy_target_counts(self):
        """With target = 2^256/64 about 1/64 of the counters are reported; all are real."""
        bh, ch, miner, prefix16, gpu8 = os.urandom(32), os.urandom(32), os.urandom(20), os.urandom(16), os.urandom(8)
        job = self.w.Job("t", P.job_prefix(bh, ch), miner, 1, prefix16)
        cx = job.cx_for(gpu8)
        target = 1 << 250
        hits, n = self.cpu_search(cx, target, 1000, 1, grid=1, block=256)
        brute = [c for c in range(1000, 1256)
                 if int.from_bytes(P.digest(bh, ch, miner, prefix16 + gpu8 + c.to_bytes(8, "big")), "big") < target]
        self.assertEqual(n, len(brute))
        self.assertEqual(sorted(hits), brute[:15])


if __name__ == "__main__":
    unittest.main()
