#!/usr/bin/env python3
"""Runs the real miner against a fake chain, on any machine, for free.

Nothing here touches Robinhood Chain, a wallet, or a GPU: a stub node serves the
contract's views at a difficulty low enough to solve in seconds, and the actual
feed, worker and signer processes mine against it. Watch the log to see the loop
the live miner runs: job -> proof -> signed mine() -> new challenge -> repeat.

    python3 scripts/playground.py --difficulty 20
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eth_account import Account  # noqa: E402

from stub_chain import StubChain, StubServer  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
THROWAWAY_KEY = "0x" + "42" * 32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", type=int, default=20,
                        help="fake difficulty; 20 solves in seconds on a CPU")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--mode", choices=("cpu", "gpu"), default="cpu")
    arguments = parser.parse_args()

    account = Account.from_key(THROWAWAY_KEY)
    chain = StubChain("0x" + "3a" * 32, difficulty=arguments.difficulty)
    with StubServer(chain) as server, tempfile.TemporaryDirectory() as workdir:
        print(f"fake chain  {server.url}  difficulty {arguments.difficulty}")
        print(f"fake wallet {account.address}  (throwaway key, no real funds)")
        print(f"workdir     {workdir}\n")
        environment = {
            **os.environ,
            "HASHBROKER_RPC_URLS": server.url,
            "HASHBROKER_WALLET": account.address,
            "HASHBROKER_PRIVATE_KEY": THROWAWAY_KEY,
            "HASHBROKER_RUNTIME_DIR": str(Path(workdir) / "runtime"),
            "PYTHONPATH": str(SCRIPTS),
            "PYTHONUNBUFFERED": "1",
        }
        started = time.time()
        subprocess.run(
            [sys.executable, str(SCRIPTS / "run_all.py"),
             "--wallet", account.address, "--dir", workdir, "--mode", arguments.mode,
             "--run-for", str(arguments.seconds)],
            env=environment, check=False,
        )
        print(f"\nmints accepted by the fake chain: {chain.minted - 260} "
              f"in {time.time() - started:.0f}s")
        print(f"final challenge: {chain.challenge}")


if __name__ == "__main__":
    main()
