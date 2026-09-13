#!/usr/bin/env python3
"""Compiles the Keccak kernel's device functions as plain C and checks them.

Same approach as tests/test_kernel_sha256.py: the exact source that runs on the
GPU is verified on a machine without one, against pycryptodome.
"""
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from Crypto.Hash import keccak  # noqa: E402

import pow as powlib  # noqa: E402
from keccak_cuda import CUDA_SOURCE  # noqa: E402
from protocol import PROTOCOL_DIR, load  # noqa: E402

FLYNODE = load(PROTOCOL_DIR / "flynode.json")

SHIM = """
#include <stdio.h>
#include <stdlib.h>
#define __device__ static
#define __forceinline__
"""

HARNESS = """
int main(int argc, char **argv) {
    u64 lanes[17];
    int stream_lane = atoi(argv[1]);
    int stream_shift = atoi(argv[2]);
    int counter_lane = atoi(argv[3]);
    int counter_shift = atoi(argv[4]);
    u32 stream = (u32)strtoul(argv[5], NULL, 16);
    u32 counter = (u32)strtoul(argv[6], NULL, 16);
    for (int i = 0; i < 17; ++i) lanes[i] = strtoull(argv[7 + i], NULL, 16);
    u64 st[25];
    proof_hash(lanes, stream_lane, stream_shift, counter_lane, counter_shift,
               stream, counter, st);
    for (int i = 0; i < 4; ++i)
        for (int b = 0; b < 8; ++b) printf("%02x", (unsigned)((st[i] >> (8 * b)) & 0xff));
    printf("\\n");
    return 0;
}
"""


def device_source() -> str:
    cut = CUDA_SOURCE.index('extern "C" __global__')
    return SHIM + CUDA_SOURCE[:cut] + HARNESS


@unittest.skipUnless(shutil.which("cc") or shutil.which("gcc"), "no C compiler available")
class KernelKeccakTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workdir = tempfile.TemporaryDirectory()
        source = Path(cls.workdir.name) / "kernel.c"
        source.write_text(device_source(), encoding="utf-8")
        cls.binary = Path(cls.workdir.name) / "kernel"
        compiler = shutil.which("cc") or shutil.which("gcc")
        subprocess.run([compiler, "-O2", "-std=c99", "-o", str(cls.binary), str(source)],
                       check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.workdir.cleanup()

    def device_digest(self, lanes, positions, stream, counter) -> bytes:
        arguments = [str(value) for value in positions] + [f"{stream:08x}", f"{counter:08x}"]
        arguments += [f"{lane:016x}" for lane in lanes]
        result = subprocess.run([str(self.binary), *arguments],
                                check=True, capture_output=True, text=True)
        return bytes.fromhex(result.stdout.strip())

    def test_matches_pycryptodome_for_random_messages(self):
        random.seed(4663)
        positions = (0, 0, 0, 32)  # unused halves: overwriting lane 0 is fine when we rebuild it
        for size in (1, 64, 100, 118, 135):
            material = bytes(random.randrange(256) for _ in range(size))
            block = powlib.keccak_pad(material)
            lanes = [int.from_bytes(block[i:i + 8], "little") for i in range(0, 136, 8)]
            stream = int.from_bytes(lanes[0].to_bytes(8, "little")[0:4], "big")
            counter = int.from_bytes(lanes[0].to_bytes(8, "little")[4:8], "big")
            actual = self.device_digest(lanes, positions, stream, counter)
            expected = keccak.new(digest_bits=256, data=material).digest()
            self.assertEqual(actual, expected, f"size {size}")

    def test_reproduces_a_flynode_proof(self):
        wallet = "0x" + "ab" * 20
        fields = {"wallet": wallet, "prev": "0x" + "cd" * 32, "anchor": "0x" + "ef" * 32,
                  "typeId": 7}
        lanes = powlib.keccak_lanes(fields, FLYNODE)
        positions = powlib.nonce_lane_positions(FLYNODE)
        for stream, counter in ((0, 0), (1, 2), (0xDEADBEEF, 0x0BADF00D), (0xFFFFFFFF, 0xFFFFFFFF)):
            actual = self.device_digest(lanes, positions, stream, counter)
            expected = powlib.digest({**fields, "nonce": (stream << 32) | counter}, FLYNODE)
            self.assertEqual(actual, expected, f"nonce {stream:08x}{counter:08x}")

    def test_a_different_type_id_changes_the_proof(self):
        wallet = "0x" + "11" * 20
        base = {"wallet": wallet, "prev": "0x" + "22" * 32, "anchor": "0x" + "33" * 32}
        positions = powlib.nonce_lane_positions(FLYNODE)
        first = self.device_digest(powlib.keccak_lanes({**base, "typeId": 1}, FLYNODE),
                                   positions, 5, 6)
        second = self.device_digest(powlib.keccak_lanes({**base, "typeId": 2}, FLYNODE),
                                    positions, 5, 6)
        self.assertNotEqual(first, second)
        self.assertEqual(first, powlib.digest({**base, "typeId": 1, "nonce": (5 << 32) | 6},
                                              FLYNODE))


if __name__ == "__main__":
    unittest.main()
