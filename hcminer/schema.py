"""Preimage schema: how the contract packs its fields before keccak256.

A schema is an ordered list of "source:encoding" fields, e.g.

    ["miner:addr20", "nonce:u256", "prev_work:b32", "anchor:b32"]

which reproduces `keccak256(abi.encodePacked(msg.sender, nonce, prevWork, anchor))`.
Use `addr32`/`u256` everywhere for the `abi.encode` (32-byte word) flavour.

The GPU varies the low 8 bytes of the nonce field, so the schema also tells the
supervisor which byte offset to hand the kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

ENCODING_SIZES = {
    "addr20": 20,
    "addr32": 32,
    "u256": 32,
    "u128": 16,
    "u64": 8,
    "u32": 4,
    "b32": 32,
}

SOURCES = ("miner", "nonce", "prev_work", "anchor", "epoch", "token_id", "chain_id", "const")


class SchemaError(ValueError):
    pass


@dataclass(frozen=True)
class Field:
    source: str
    encoding: str
    const_hex: str = ""

    @property
    def size(self) -> int:
        if self.source == "const":
            return len(bytes.fromhex(self.const_hex.removeprefix("0x")))
        return ENCODING_SIZES[self.encoding]

    def encode(self, values: Dict[str, object]) -> bytes:
        if self.source == "const":
            return bytes.fromhex(self.const_hex.removeprefix("0x"))
        if self.source not in values:
            raise SchemaError(f"missing value for field '{self.source}'")
        raw = values[self.source]
        size = ENCODING_SIZES[self.encoding]

        if isinstance(raw, str):
            raw = raw.strip()
            if raw.startswith(("0x", "0X")):
                data = bytes.fromhex(raw[2:])
                if self.encoding in ("addr20", "b32") and len(data) == size:
                    return data
                raw = int.from_bytes(data, "big")
            else:
                raw = int(raw)
        if isinstance(raw, bytes):
            if len(raw) == size:
                return raw
            raw = int.from_bytes(raw, "big")
        if not isinstance(raw, int):
            raise SchemaError(f"cannot encode {self.source}={raw!r}")
        if raw < 0 or raw >= 1 << (8 * size):
            raise SchemaError(f"{self.source} does not fit in {size} bytes")
        return raw.to_bytes(size, "big")


@dataclass(frozen=True)
class Schema:
    fields: List[Field]

    @staticmethod
    def parse(specs: List[str]) -> "Schema":
        fields = []
        for spec in specs:
            parts = spec.split(":")
            if parts[0] == "const":
                if len(parts) != 2:
                    raise SchemaError(f"const field needs hex data: {spec!r}")
                fields.append(Field("const", "raw", parts[1]))
                continue
            if len(parts) != 2:
                raise SchemaError(f"bad field spec {spec!r}, expected 'source:encoding'")
            source, encoding = parts
            if source not in SOURCES:
                raise SchemaError(f"unknown source {source!r}, expected one of {SOURCES}")
            if encoding not in ENCODING_SIZES:
                raise SchemaError(f"unknown encoding {encoding!r}")
            fields.append(Field(source, encoding))
        if sum(1 for f in fields if f.source == "nonce") != 1:
            raise SchemaError("schema must contain exactly one 'nonce' field")
        return Schema(fields)

    def specs(self) -> List[str]:
        return [
            f"const:{f.const_hex}" if f.source == "const" else f"{f.source}:{f.encoding}"
            for f in self.fields
        ]

    def build(self, values: Dict[str, object]) -> bytes:
        out = b"".join(f.encode(values) for f in self.fields)
        if len(out) > 135:
            raise SchemaError(
                f"preimage is {len(out)} bytes; the single-block GPU kernel supports 135"
            )
        return out

    @property
    def nonce_offset(self) -> int:
        """Byte offset of the nonce field inside the preimage."""
        off = 0
        for f in self.fields:
            if f.source == "nonce":
                return off
            off += f.size
        raise SchemaError("no nonce field")

    @property
    def vary_offset(self) -> int:
        """Offset of the 8 bytes the GPU increments (low bytes of the nonce)."""
        field = next(f for f in self.fields if f.source == "nonce")
        return self.nonce_offset + field.size - 8

    def hash(self, values: Dict[str, object]) -> bytes:
        from .keccak import keccak256

        return keccak256(self.build(values))

    def __str__(self) -> str:
        return " | ".join(self.specs())
