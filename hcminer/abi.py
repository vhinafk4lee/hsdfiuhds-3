"""Minimal ABI helpers for static 32-byte types (all this miner needs).

Avoids a hard dependency on eth-abi for the offline commands; `hcminer.tx`
still prefers eth-abi when it is installed.
"""

from __future__ import annotations

from typing import Any, List, Sequence

from .keccak import keccak256


def is_raw_selector(signature: str) -> bool:
    """True for '0x39148c53' - a selector whose function name is unknown."""
    text = signature.strip().lower()
    return (text.startswith("0x") and len(text) == 10
            and all(c in "0123456789abcdef" for c in text[2:]))


def function_selector(signature: str) -> bytes:
    """'mine(uint256)' -> 4-byte selector. A raw '0x...' selector passes through."""
    if is_raw_selector(signature):
        return bytes.fromhex(signature.strip()[2:])
    return keccak256(signature.replace(" ", "").encode())[:4]


def signature_types(signature: str) -> List[str]:
    if is_raw_selector(signature):
        return []
    inner = signature[signature.index("(") + 1 : signature.rindex(")")].strip()
    return [t.strip() for t in inner.split(",") if t.strip()]


def encode_arg(type_: str, value: Any) -> bytes:
    if type_ == "address":
        if isinstance(value, str):
            value = int(value, 16)
        return int(value).to_bytes(32, "big")
    if type_ == "bool":
        return (1 if value else 0).to_bytes(32, "big")
    if type_.startswith("bytes") and len(type_) > 5:
        data = value if isinstance(value, bytes) else bytes.fromhex(str(value).removeprefix("0x"))
        return data.rjust(32, b"\x00") if len(data) < 32 else data[:32]
    if type_.startswith(("uint", "int")):
        if isinstance(value, bytes):
            value = int.from_bytes(value, "big")
        elif isinstance(value, str):
            value = int(value, 16) if value.startswith("0x") else int(value)
        return int(value).to_bytes(32, "big")
    raise ValueError(f"unsupported static ABI type: {type_}")


def encode(types: Sequence[str], values: Sequence[Any]) -> bytes:
    if len(types) != len(values):
        raise ValueError(f"{len(types)} types but {len(values)} values")
    return b"".join(encode_arg(t, v) for t, v in zip(types, values))


def decode_arg(type_: str, word: bytes) -> Any:
    if type_ == "address":
        return "0x" + word[12:].hex()
    if type_ == "bool":
        return word[-1] != 0
    if type_.startswith("bytes") and len(type_) > 5:
        return "0x" + word[: int(type_[5:])].hex()
    if type_.startswith("uint"):
        return int.from_bytes(word, "big")
    if type_.startswith("int"):
        v = int.from_bytes(word, "big")
        return v - (1 << 256) if v >= 1 << 255 else v
    raise ValueError(f"unsupported static ABI type: {type_}")


def decode(types: Sequence[str], data: bytes) -> List[Any]:
    if len(data) < 32 * len(types):
        raise ValueError(f"need {32 * len(types)} bytes of return data, got {len(data)}")
    return [decode_arg(t, data[i * 32 : (i + 1) * 32]) for i, t in enumerate(types)]


def calldata(signature: str, values: Sequence[Any]) -> bytes:
    return function_selector(signature) + encode(signature_types(signature), values)
