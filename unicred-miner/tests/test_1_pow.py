"""Test 1: Python digest == test vector (pycryptodome and the worker's pure-Python Keccak)."""
import os
import random
import unittest

from tests.helpers import load_worker
from unicred import pow as P


class PowTest(unittest.TestCase):
    def test_testvector_pycryptodome(self):
        d = P.digest(P.TV["blockhash"], P.TV["challenge"], P.TV["miner"], P.TV["nonce"])
        self.assertEqual("0x" + d.hex(), P.TV["digest"])
        self.assertLess(int.from_bytes(d, "big"), 1 << 212)

    def test_testvector_worker_pure_python(self):
        w = load_worker()
        job, gpu8, ctr = w.testvector_job()
        self.assertEqual(w.digest_from_cx(job.cx_for(gpu8), ctr).hex(), P.TV["digest"][2:])
        self.assertEqual(job.nonce(gpu8, ctr), P.TV["nonce"].to_bytes(32, "big"))

    def test_prefix_and_message_layout(self):
        w = load_worker()
        bh, ch = bytes.fromhex(P.TV["blockhash"][2:]), bytes.fromhex(P.TV["challenge"][2:])
        self.assertEqual(P.job_prefix(bh, ch), w.job_prefix(bh, ch))
        msg = P.pow_message(bh, ch, P.TV["miner"], P.TV["nonce"])
        self.assertEqual(len(msg), 192)
        self.assertEqual(msg[:32], P.TYPEHASH)
        self.assertEqual(int.from_bytes(msg[32:64], "big"), 130)

    def test_pure_python_keccak_random(self):
        w = load_worker()
        for n in (0, 1, 55, 135, 136, 137, 192, 300):
            data = os.urandom(n)
            self.assertEqual(w.keccak256(data), P.keccak256(data))

    def test_worker_midstate_random(self):
        w = load_worker()
        rnd = random.Random(1)
        for _ in range(50):
            bh, ch, miner = os.urandom(32), os.urandom(32), os.urandom(20)
            nonce = rnd.getrandbits(256).to_bytes(32, "big")
            job = w.Job("t", w.job_prefix(bh, ch), miner, 1, nonce[:16])
            got = w.digest_from_cx(job.cx_for(nonce[16:24]), int.from_bytes(nonce[24:], "big"))
            self.assertEqual(got, P.digest(bh, ch, miner, nonce))

    def test_mint_calldata(self):
        data = P.mint_calldata(59492322, P.TV["nonce"], 4 * 10 ** 15)
        self.assertTrue(data.startswith("0x106c9da1"))
        self.assertEqual(len(data), 2 + 8 + 192)
        dec = P.decode_mint_calldata(data)
        self.assertEqual(dec, {"anchor_block": 59492322, "nonce": P.TV["nonce"], "max_price": 4 * 10 ** 15})

    def test_error_decoding(self):
        self.assertEqual(P.decode_error("0x7ca55c77")[0], "7ca55c77")
        self.assertIn("PoW", P.decode_error("0x7ca55c77")[1])


if __name__ == "__main__":
    unittest.main()
