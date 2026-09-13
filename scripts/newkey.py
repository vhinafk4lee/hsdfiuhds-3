#!/usr/bin/env python3
"""Creates a dedicated mining wallet on this machine.

The key is written to a 0600 file and never printed. Fund that address with
only what the mining should be allowed to spend: the signer runs unattended, so
its wallet is a hot wallet by definition — keep it separate from anything else.

    python3 scripts/newkey.py --out wallet.key
"""
from __future__ import annotations

import argparse
from pathlib import Path

from eth_account import Account

from keyfile import describe, write_private


def create(path: Path) -> str:
    account = Account.create()
    write_private(path, "0x" + account.key.hex().removeprefix("0x") + "\n")
    return account.address


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="wallet.key")
    arguments = parser.parse_args()
    path = Path(arguments.out)
    address = create(path)
    print(f"address    {address}")
    print(f"key file   {describe(path.resolve())}")
    print()
    print("Next:")
    print(f"  1. send it the ETH the miner may spend")
    print(f"  2. export HASHBROKER_WALLET={address}")
    print(f"  3. export HASHBROKER_PRIVATE_KEY_FILE={path.resolve()}")
    print("  4. python3 scripts/signer.py --check")


if __name__ == "__main__":
    main()
