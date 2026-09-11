"""keccak256 with whichever backend is installed."""

from __future__ import annotations


def _load():
    try:
        from Crypto.Hash import keccak as _k  # pycryptodome

        def f(data: bytes) -> bytes:
            return _k.new(digest_bits=256, data=data).digest()

        return f
    except ImportError:
        pass
    try:
        from eth_hash.auto import keccak as _k  # eth-hash

        return lambda data: _k(data)
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "install pycryptodome or eth-hash: pip install -r requirements.txt"
        ) from exc


keccak256 = _load()


def leading_zero_bits(digest: bytes) -> int:
    """Number of leading zero bits in a 32-byte digest."""
    n = 0
    for byte in digest:
        if byte == 0:
            n += 8
            continue
        while byte & 0x80 == 0:
            n += 1
            byte <<= 1
        break
    return n
