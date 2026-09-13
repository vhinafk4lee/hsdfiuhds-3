#!/usr/bin/env python3
"""FlyNode's proof, checked against the contract source.

    workHash = keccak256(abi.encodePacked(miner, nonce, prev, anchor, typeId))

with typeId a uint16, so the preimage is 20 + 32 + 32 + 32 + 2 = 118 bytes — one
Keccak block. A proof wins when its leading zero bits reach requiredBits().
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from Crypto.Hash import keccak  # noqa: E402

import pow as powlib  # noqa: E402
from keccak_pure import selector  # noqa: E402
from protocol import load, PROTOCOL_DIR  # noqa: E402

FLYNODE = load(PROTOCOL_DIR / "flynode.json")
MINER = "0xd3754bcbe60f07a14fdbb96399ce96926080c0c6"
NONCE = 0x862081DB4AD0AA16E17F962668EF253E58975C76560D37470000052007EFA320
ANCHOR = "0x6a9248e010007505bbca43db9896be1b5a3de244077c5dbacb8e52fa4f3b3826"
PREV = "0x" + "5c" * 32
TYPE_ID = 0
FIELDS = {"wallet": MINER, "nonce": NONCE, "prev": PREV, "anchor": ANCHOR, "typeId": TYPE_ID}


def solidity_work_hash(miner: str, nonce: int, prev: str, anchor: str, type_id: int) -> bytes:
    """abi.encodePacked(address,uint256,bytes32,bytes32,uint16), hashed."""
    packed = (bytes.fromhex(miner.removeprefix("0x"))
              + nonce.to_bytes(32, "big")
              + bytes.fromhex(prev.removeprefix("0x"))
              + bytes.fromhex(anchor.removeprefix("0x"))
              + type_id.to_bytes(2, "big"))
    return keccak.new(digest_bits=256, data=packed).digest()


class FlyNodeProofTests(unittest.TestCase):
    def test_selector_matches_the_deployed_mint(self):
        """The trace of a real mint called 0x9013cdee."""
        self.assertEqual("0x" + FLYNODE.mine_selector, "0x9013cdee")
        self.assertEqual("0x" + selector(FLYNODE.mine_signature), "0x9013cdee")

    def test_preimage_is_one_keccak_block(self):
        self.assertEqual(FLYNODE.preimage_size, 118)
        self.assertEqual(len(powlib.preimage(FIELDS, FLYNODE)), 118)
        self.assertLess(FLYNODE.preimage_size, powlib.KECCAK_RATE)

    def test_matches_the_contracts_work_hash(self):
        self.assertEqual(powlib.digest(FIELDS, FLYNODE),
                         solidity_work_hash(MINER, NONCE, PREV, ANCHOR, TYPE_ID))

    def test_type_id_is_two_bytes_at_the_end(self):
        material = powlib.preimage({**FIELDS, "typeId": 0x1234}, FLYNODE)
        self.assertEqual(material[-2:], b"\x12\x34")
        self.assertNotEqual(powlib.digest({**FIELDS, "typeId": 1}, FLYNODE),
                            powlib.digest(FIELDS, FLYNODE))

    def test_difficulty_is_leading_zero_bits(self):
        """requiredBits is a count of zero bits, which is the same as a target."""
        proof = bytes.fromhex("00" * 3 + "ff" * 29)
        self.assertEqual(powlib.leading_zero_bits(proof), 24)
        self.assertTrue(powlib.meets(proof, powlib.target_for_difficulty(24)))
        self.assertFalse(powlib.meets(proof, powlib.target_for_difficulty(25)))

    def test_keccak_padding_is_not_sha3_padding(self):
        block = powlib.keccak_pad(powlib.preimage(FIELDS, FLYNODE))
        self.assertEqual(len(block), 136)
        self.assertEqual(block[118], 0x01)
        self.assertEqual(block[135], 0x80)

    def test_lanes_place_the_nonce_where_the_kernel_looks(self):
        """The kernel searches the low 64 bits, so the lanes must reproduce that nonce."""
        searched = 0x13579BDF2468ACE0
        fields = {**FIELDS, "nonce": searched}
        lanes = powlib.keccak_lanes(fields, FLYNODE)
        stream_lane, stream_shift, counter_lane, counter_shift = powlib.nonce_lane_positions(FLYNODE)
        self.assertEqual(len(lanes), 17)
        stream, counter = searched >> 32, searched & 0xFFFFFFFF
        placed = list(lanes)
        for lane, shift, half in ((stream_lane, stream_shift, stream),
                                  (counter_lane, counter_shift, counter)):
            swapped = int.from_bytes(half.to_bytes(4, "big"), "little")
            placed[lane] = (placed[lane] & ~(0xFFFFFFFF << shift)) | (swapped << shift)
        message = b"".join(lane.to_bytes(8, "little") for lane in placed)
        self.assertEqual(message, powlib.keccak_pad(powlib.preimage(fields, FLYNODE)))


if __name__ == "__main__":
    unittest.main()
