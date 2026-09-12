#!/usr/bin/env python3
"""Reference proof-of-work implementation and GPU message preparation.

The GPU kernel and the CPU verifier must agree bit for bit, so both go through
this module: ``digest`` is the authoritative CPU implementation, and
``padded_words``/``nonce_word_indices`` describe the same message to the kernel.
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


def preimage(wallet: str, nonce: int, prev: str, anchor: str,
             protocol: Protocol = PROTOCOL) -> bytes:
    values = {"wallet": wallet, "nonce": nonce, "prev": prev, "anchor": anchor}
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


def digest(wallet: str, nonce: int, prev: str, anchor: str,
           protocol: Protocol = PROTOCOL) -> bytes:
    return hash_bytes(preimage(wallet, nonce, prev, anchor, protocol), protocol)


def sha256_pad(material: bytes) -> bytes:
    """SHA-256 / Keccak-free padding: 0x80, zeros, 64-bit big-endian bit length."""
    padded = bytearray(material)
    padded.append(0x80)
    while (len(padded) + 8) % 64:
        padded.append(0x00)
    padded.extend((len(material) * 8).to_bytes(8, "big"))
    return bytes(padded)


def padded_words(wallet: str, prev: str, anchor: str,
                 protocol: Protocol = PROTOCOL) -> list[int]:
    """The padded message with a zero nonce, as big-endian 32-bit words."""
    if protocol.algorithm not in ("sha256", "sha256d"):
        raise ValueError("padded_words only describes SHA-256 messages")
    block = sha256_pad(preimage(wallet, 0, prev, anchor, protocol))
    return [int.from_bytes(block[index:index + 4], "big") for index in range(0, len(block), 4)]


def nonce_word_indices(protocol: Protocol = PROTOCOL) -> tuple[int, int]:
    """Word indices of the two 32-bit halves of the searched nonce tail."""
    offset = protocol.nonce_offset + protocol.nonce_size - 8
    if offset % 4:
        raise ValueError("nonce tail is not 32-bit aligned in the preimage")
    return offset // 4, offset // 4 + 1


def search_target(target: str | int, slack: int = 3) -> int:
    """Mine slightly above the live target so candidates survive small moves."""
    value = int(target, 16) if isinstance(target, str) else int(target)
    return min(MAX_HASH, value << slack)


def meets(digest_bytes: bytes, target: str | int) -> bool:
    value = int(target, 16) if isinstance(target, str) else int(target)
    return int.from_bytes(digest_bytes, "big") < value
