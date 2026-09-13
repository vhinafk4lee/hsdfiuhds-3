#!/usr/bin/env python3
"""Reference proof-of-work implementation and GPU message preparation.

Hash Broker proofs are

    sha256( miner[20] || nonce[uint256, big endian] || challenge[bytes32] )

and a proof wins when the digest has at least ``currentDifficulty()`` leading
zero bits. The GPU kernel and the CPU verifier must agree bit for bit, so both
go through this module.
"""
from __future__ import annotations

import hashlib

from Crypto.Hash import keccak

from protocol import PROTOCOL, Protocol

MAX_HASH = (1 << 256) - 1


def _bytes_for(field_name: str, values: dict[str, object], size: int) -> bytes:
    raw = values[field_name]
    if isinstance(raw, int):
        return int(raw).to_bytes(size, "big")
    text = str(raw)
    data = bytes.fromhex(text[2:] if text.startswith("0x") else text)
    if len(data) != size:
        raise ValueError(f"{field_name} must be {size} bytes, got {len(data)}")
    return data


def required_fields(protocol: Protocol = PROTOCOL) -> tuple[str, ...]:
    return tuple(dict.fromkeys(field.name for field in protocol.preimage
                               if field.name != "const"))


def preimage(values: dict, protocol: Protocol = PROTOCOL) -> bytes:
    missing = [name for name in required_fields(protocol) if name not in values]
    if missing:
        raise ValueError("proof needs " + ", ".join(missing))
    chunks = []
    for field in protocol.preimage:
        if field.name == "const":
            chunks.append(field.value or b"")
        else:
            chunks.append(_bytes_for(field.name, values, field.size))
    return b"".join(chunks)


def hash_bytes(material: bytes, protocol: Protocol = PROTOCOL) -> bytes:
    if protocol.algorithm == "sha256":
        return hashlib.sha256(material).digest()
    if protocol.algorithm == "sha256d":
        return hashlib.sha256(hashlib.sha256(material).digest()).digest()
    if protocol.algorithm == "keccak256":
        return keccak.new(digest_bits=256, data=material).digest()
    raise ValueError(f"unsupported algorithm {protocol.algorithm!r}")


def digest(values: dict, protocol: Protocol = PROTOCOL) -> bytes:
    return hash_bytes(preimage(values, protocol), protocol)


def sha256_pad(material: bytes) -> bytes:
    """Merkle-Damgard padding: 0x80, zeros, then a 64-bit big-endian bit length."""
    padded = bytearray(material)
    padded.append(0x80)
    while (len(padded) + 8) % 64:
        padded.append(0x00)
    padded.extend((len(material) * 8).to_bytes(8, "big"))
    return bytes(padded)


def padded_words(values: dict, protocol: Protocol = PROTOCOL) -> list[int]:
    """The padded SHA-256 message with a zero nonce, as big-endian 32-bit words."""
    if protocol.algorithm not in ("sha256", "sha256d"):
        raise ValueError("padded_words only describes SHA-256 messages")
    block = sha256_pad(preimage({**values, "nonce": 0}, protocol))
    return [int.from_bytes(block[index:index + 4], "big") for index in range(0, len(block), 4)]


KECCAK_RATE = 136


def keccak_pad(material: bytes, rate: int = KECCAK_RATE) -> bytes:
    """Keccak's own padding, which is not SHA-3's: 0x01, zeros, 0x80 in the last byte."""
    if len(material) >= rate:
        raise ValueError(f"preimage of {len(material)} bytes needs more than one Keccak block")
    block = bytearray(material) + bytearray(rate - len(material))
    block[len(material)] = 0x01
    block[rate - 1] |= 0x80
    return bytes(block)


def keccak_lanes(values: dict, protocol: Protocol = PROTOCOL) -> list[int]:
    """The padded message with a zero nonce, as the 17 little-endian lanes Keccak absorbs."""
    if protocol.algorithm != "keccak256":
        raise ValueError("keccak_lanes only describes Keccak messages")
    block = keccak_pad(preimage({**values, "nonce": 0}, protocol))
    return [int.from_bytes(block[index:index + 8], "little") for index in range(0, len(block), 8)]


def nonce_lane_positions(protocol: Protocol = PROTOCOL) -> tuple[int, int, int, int]:
    """Which lane and bit offset hold each 32-bit half of the searched nonce tail.

    Lanes are little-endian, so a big-endian nonce byte at offset o lands in lane
    o // 8 at bit (o % 8) * 8 — the kernel byte-swaps each half before placing it.
    """
    offset = protocol.nonce_offset + protocol.nonce_size - 8
    if offset % 4:
        raise ValueError("nonce tail is not 32-bit aligned in the preimage")
    stream_lane, stream_shift = offset // 8, (offset % 8) * 8
    counter_lane, counter_shift = (offset + 4) // 8, ((offset + 4) % 8) * 8
    return stream_lane, stream_shift, counter_lane, counter_shift


def nonce_word_indices(protocol: Protocol = PROTOCOL) -> tuple[int, int]:
    """Word indices of the two 32-bit halves of the searched nonce tail."""
    offset = protocol.nonce_offset + protocol.nonce_size - 8
    if offset % 4:
        raise ValueError("nonce tail is not 32-bit aligned in the preimage")
    return offset // 4, offset // 4 + 1


def target_for_difficulty(difficulty: int) -> int:
    """A proof wins with ``difficulty`` leading zero bits, i.e. hash < 2**(256-d)."""
    value = int(difficulty)
    if not 0 <= value <= 255:
        raise ValueError(f"difficulty {value} out of range")
    return 1 << (256 - value)


def leading_zero_bits(digest_bytes: bytes) -> int:
    return 256 - int.from_bytes(digest_bytes, "big").bit_length()


def search_target(target: str | int, slack: int = 3) -> int:
    """Mine slightly above the live target so proofs survive a small difficulty rise."""
    value = int(target, 16) if isinstance(target, str) else int(target)
    return min(MAX_HASH, value << slack)


def meets(digest_bytes: bytes, target: str | int) -> bool:
    value = int(target, 16) if isinstance(target, str) else int(target)
    return int.from_bytes(digest_bytes, "big") < value
