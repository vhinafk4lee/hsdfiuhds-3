#!/usr/bin/env python3
"""Everything specific to the Hash Broker contract lives in this module.

The rest of the miner is protocol agnostic: it asks this module for the
proof-of-work preimage layout, the read-only view selectors, and the mint
calldata. Point ``HASHBROKER_PROTOCOL_FILE`` at a JSON file to override the
bundled ``protocol.json`` without touching code.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from Crypto.Hash import keccak

DEFAULT_PROTOCOL_FILE = Path(__file__).resolve().parent / "protocol.json"
VARIABLE_FIELDS = ("wallet", "nonce", "challenge")
SUPPORTED_ALGORITHMS = ("sha256", "sha256d", "keccak256")


def selector(signature: str) -> str:
    return keccak.new(digest_bits=256, data=signature.encode()).hexdigest()[:8]


def checksum(address: str) -> str:
    """EIP-55 casing. eth-account refuses a transaction whose `to` is not checksummed."""
    body = address.removeprefix("0x").lower()
    digest = keccak.new(digest_bits=256, data=body.encode()).hexdigest()
    return "0x" + "".join(
        character.upper() if int(digest[index], 16) > 7 else character
        for index, character in enumerate(body)
    )


@dataclass(frozen=True)
class Field:
    name: str
    size: int
    value: bytes | None = None


@dataclass(frozen=True)
class Protocol:
    name: str
    chain_id: int
    contract: str
    algorithm: str
    preimage: tuple[Field, ...]
    mine_signature: str
    mine_args: tuple[str, ...]
    views: dict[str, str]
    validate_signature: str
    rpc: tuple[str, ...]
    broadcast_rpc: tuple[str, ...]
    verified: bool

    @property
    def preimage_size(self) -> int:
        return sum(field.size for field in self.preimage)

    @property
    def nonce_offset(self) -> int:
        offset = 0
        for field in self.preimage:
            if field.name == "nonce":
                return offset
            offset += field.size
        raise ValueError("preimage layout has no nonce field")

    @property
    def nonce_size(self) -> int:
        for field in self.preimage:
            if field.name == "nonce":
                return field.size
        raise ValueError("preimage layout has no nonce field")

    def view(self, key: str) -> str:
        signature = self.views.get(key)
        if not signature:
            raise KeyError(f"protocol has no view named {key!r}")
        return "0x" + selector(signature)

    @property
    def mine_selector(self) -> str:
        return selector(self.mine_signature)

    @property
    def validate_selector(self) -> str:
        if not self.validate_signature:
            raise KeyError("protocol has no on-chain proof validator")
        return selector(self.validate_signature)

    def calldata(self, nonce: int, challenge: str) -> str:
        """ABI-encoded mine() call. Both arguments are single 32-byte words."""
        words = {"nonce": f"{int(nonce):064x}",
                 "challenge": str(challenge).removeprefix("0x").rjust(64, "0")}
        try:
            body = "".join(words[name] for name in self.mine_args)
        except KeyError as exc:
            raise ValueError(f"unsupported mine argument {exc.args[0]!r}") from exc
        return "0x" + self.mine_selector + body

    def require_deployed(self) -> str:
        if not self.contract:
            raise SystemExit(
                "contract address is unknown: set HASHBROKER_CONTRACT or fill "
                f"\"contract\" in {DEFAULT_PROTOCOL_FILE.name}"
            )
        return self.contract


def _parse(payload: dict) -> Protocol:
    algorithm = str(payload.get("algorithm", "sha256")).lower()
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"unsupported algorithm {algorithm!r}")

    fields = []
    for item in payload.get("preimage", ()):
        name = str(item.get("field", ""))
        size = int(item.get("size", 0))
        if size < 1:
            raise ValueError(f"preimage field {name!r} needs a positive size")
        if name == "const":
            value = bytes.fromhex(str(item.get("value", "")).removeprefix("0x"))
            if len(value) != size:
                raise ValueError("constant preimage field size mismatch")
            fields.append(Field(name, size, value))
            continue
        if name not in VARIABLE_FIELDS:
            raise ValueError(f"unknown preimage field {name!r}")
        fields.append(Field(name, size))
    if not fields:
        raise ValueError("preimage layout is empty")

    contract = str(os.environ.get("HASHBROKER_CONTRACT") or payload.get("contract", "")).strip()
    if contract:
        if not contract.startswith("0x") or len(contract) != 42:
            raise ValueError("contract must be a 0x-prefixed 20-byte address")
        contract = checksum(contract)

    rpc_override = os.environ.get("HASHBROKER_RPC_URLS", "").strip()
    rpc = tuple(url.strip() for url in rpc_override.split(",") if url.strip()) or tuple(
        payload.get("rpc", ())
    )
    broadcast_override = os.environ.get("HASHBROKER_BROADCAST_RPC_URLS", "").strip()
    broadcast = tuple(
        url.strip() for url in broadcast_override.split(",") if url.strip()
    ) or tuple(payload.get("broadcastRpc", ())) or rpc
    if not rpc:
        raise ValueError("protocol needs at least one RPC endpoint")

    mine_args = tuple(str(arg) for arg in payload.get("mineArgs", ()))
    return Protocol(
        name=str(payload.get("name", "hashbroker")),
        chain_id=int(payload.get("chainId", 0)),
        contract=contract,
        algorithm=algorithm,
        preimage=tuple(fields),
        mine_signature=str(payload["mine"]),
        mine_args=mine_args,
        views=dict(payload.get("views", {})),
        validate_signature=str(payload.get("validate", "")),
        rpc=rpc,
        broadcast_rpc=broadcast,
        verified=bool(payload.get("verified", False)),
    )


def load(path: str | os.PathLike[str] | None = None) -> Protocol:
    source = Path(path or os.environ.get("HASHBROKER_PROTOCOL_FILE", DEFAULT_PROTOCOL_FILE))
    return _parse(json.loads(source.read_text(encoding="utf-8")))


PROTOCOL = load()


def describe() -> str:
    fields = " | ".join(f"{field.name}[{field.size}]" for field in PROTOCOL.preimage)
    lines = [
        f"protocol   {PROTOCOL.name} (chainId {PROTOCOL.chain_id})",
        f"contract   {PROTOCOL.contract or '<unset>'}",
        f"algorithm  {PROTOCOL.algorithm}",
        f"preimage   {fields} = {PROTOCOL.preimage_size} bytes",
        f"mine       {PROTOCOL.mine_signature} -> 0x{PROTOCOL.mine_selector}",
        "views      " + ", ".join(
            f"{key}={signature}" for key, signature in sorted(PROTOCOL.views.items())
        ),
        f"validate   {PROTOCOL.validate_signature or '<none>'}",
        f"verified   {PROTOCOL.verified}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
