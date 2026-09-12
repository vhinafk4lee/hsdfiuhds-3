#!/usr/bin/env python3
"""Compiles the CUDA kernel's SHA-256 core as plain C and checks it against hashlib.

The device functions are ordinary C once the CUDA qualifiers are macro'd away,
so the exact source that runs on the GPU can be verified on a machine without
one. Only the ``__global__`` entry points are stripped.
"""
import hashlib
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pow as powlib  # noqa: E402
from sha256_cuda import CUDA_SOURCE  # noqa: E402

SHIM = """
#include <stdio.h>
#include <stdlib.h>
#define __device__ static
#define __constant__ static const
#define __forceinline__
"""

HARNESS = """
int main(int argc, char **argv) {
    u32 message[32];
    int blocks = atoi(argv[1]);
    int doubled = atoi(argv[2]);
    int stream_word = atoi(argv[3]);
    int counter_word = atoi(argv[4]);
    u32 stream = (u32)strtoul(argv[5], NULL, 16);
    u32 counter = (u32)strtoul(argv[6], NULL, 16);
    for (int i = 0; i < 32; ++i) message[i] = (u32)strtoul(argv[7 + i], NULL, 16);
    u32 digest[8];
    proof_hash(message, blocks, doubled, stream_word, counter_word, stream, counter, digest);
    for (int i = 0; i < 8; ++i) printf("%08x", digest[i]);
    printf("\\n");
    return 0;
}
"""


def device_source() -> str:
    cut = CUDA_SOURCE.index('extern "C" __global__')
    return SHIM + CUDA_SOURCE[:cut] + HARNESS


@unittest.skipUnless(shutil.which("cc") or shutil.which("gcc"), "no C compiler available")
class KernelSha256Tests(unittest.TestCase):
    binary: Path
    workdir: tempfile.TemporaryDirectory

    @classmethod
    def setUpClass(cls):
        cls.workdir = tempfile.TemporaryDirectory()
        root = Path(cls.workdir.name)
        source = root / "kernel.c"
        source.write_text(device_source(), encoding="utf-8")
        cls.binary = root / "kernel"
        compiler = shutil.which("cc") or shutil.which("gcc")
        subprocess.run([compiler, "-O2", "-std=c99", "-o", str(cls.binary), str(source)],
                       check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.workdir.cleanup()

    def device_digest(self, words, blocks, doubled, stream_word, counter_word, stream, counter):
        padded = list(words) + [0] * (32 - len(words))
        arguments = [str(blocks), str(doubled), str(stream_word), str(counter_word),
                     f"{stream:08x}", f"{counter:08x}"] + [f"{word:08x}" for word in padded]
        result = subprocess.run([str(self.binary), *arguments],
                                check=True, capture_output=True, text=True)
        return bytes.fromhex(result.stdout.strip())

    def test_matches_hashlib_for_random_messages(self):
        random.seed(20260912)
        for size in (1, 32, 55, 56, 64, 100, 116, 119):
            material = bytes(random.randrange(256) for _ in range(size))
            padded = powlib.sha256_pad(material)
            words = [int.from_bytes(padded[i:i + 4], "big") for i in range(0, len(padded), 4)]
            actual = self.device_digest(words, len(padded) // 64, 0, -1, -1, 0, 0)
            self.assertEqual(actual, hashlib.sha256(material).digest(), f"size {size}")

    def test_double_sha256(self):
        material = b"hash broker double sha"
        padded = powlib.sha256_pad(material)
        words = [int.from_bytes(padded[i:i + 4], "big") for i in range(0, len(padded), 4)]
        actual = self.device_digest(words, len(padded) // 64, 1, -1, -1, 0, 0)
        expected = hashlib.sha256(hashlib.sha256(material).digest()).digest()
        self.assertEqual(actual, expected)

    def test_nonce_substitution_matches_the_reference_proof(self):
        wallet = "0x" + "ab" * 20
        challenge = "0x" + "cd" * 32
        words = powlib.padded_words(wallet, challenge)
        stream_word, counter_word = powlib.nonce_word_indices()
        for stream, counter in ((0, 0), (1, 2), (0xDEADBEEF, 0x0BADF00D), (0xFFFFFFFF, 0xFFFFFFFF)):
            actual = self.device_digest(words, len(words) // 16, 0,
                                        stream_word, counter_word, stream, counter)
            expected = powlib.digest(wallet, (stream << 32) | counter, challenge)
            self.assertEqual(actual, expected, f"nonce {stream:08x}{counter:08x}")


if __name__ == "__main__":
    unittest.main()
