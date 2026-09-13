#!/usr/bin/env python3
"""Shows where the lattice has grown and what the next cell would cost.

    python3 scripts/frontier.py --protocol flynode --wallet 0xYourWallet

Reads the contract's logs from the deploy block, rebuilds the dataset, and
prints the free cells that touch a claimed one — cheapest first, because rarity
is difficulty. With a wallet it also reads the live job, so the bits it quotes
are requiredBits() rather than an estimate.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import flynode  # noqa: E402
import lattice as lattice_module  # noqa: E402
from job_state import request_batch  # noqa: E402
from protocol import PROTOCOL_DIR, load  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=os.environ.get("HASHBROKER_PROTOCOL", "flynode"))
    parser.add_argument("--wallet", default=os.environ.get("HASHBROKER_WALLET", ""))
    parser.add_argument("--rpc", default="", help="one endpoint, instead of the protocol's")
    parser.add_argument("--limit", type=int, default=15, help="cells to list")
    parser.add_argument("--from-block", type=int, default=flynode.DEPLOY_BLOCK)
    parser.add_argument("--rarest", action="store_true", help="list the rarest cells instead")
    arguments = parser.parse_args()

    protocol = load(PROTOCOL_DIR / f"{arguments.protocol}.json")
    url = arguments.rpc or protocol.rpc[0]
    lattice = lattice_module.Lattice.load()
    lattice.check_roots(*_roots(protocol, url))

    head = int(request_batch(url, [("eth_blockNumber", [])], 20.0)[0], 16)
    frontier = flynode.scan_mined(lattice, url, arguments.from_block, head,
                                  protocol=protocol)
    cells = frontier.open_cells(lattice)
    print(f"contract   {protocol.contract} on chain {protocol.chain_id}")
    print(f"scanned    blocks {arguments.from_block}..{head}")
    print(f"claimed    {len(frontier)} of {lattice.size} cells")
    print(f"frontier   {len(cells)} free cells touching a claimed one")
    if not cells:
        print("\nnothing to mine: no claimed cell has a free neighbour")
        return

    order = sorted(cells, key=lambda cell: ((-1 if arguments.rarest else 1)
                                            * lattice.neuron(cell).rarity_bits, cell))
    print(f"\n{'cell':>9}  {'type':<22} {'rarity':>6}  {'region':>6}  parents")
    for cell in order[:arguments.limit]:
        neuron = lattice.neuron(cell)
        name = lattice.types[neuron.type_id]["name"]
        print(f"{cell:>9}  {name:<22} {neuron.rarity_bits:>6}  {neuron.region:>6}  "
              f"{', '.join(str(parent) for parent in sorted(cells[cell])[:3])}")

    if arguments.wallet:
        best = lattice.neuron(order[0])
        job = flynode.read_job(arguments.wallet, lattice, best.rarity_bits, url, protocol)
        print(f"\nnext mint  cell {order[0]} at {job['difficulty']} bits, "
              f"{job['priceWei'] / 1e18:.6f} ETH")
        print(f"           rule says {job['predictedBits']} "
              f"(retargetQ {job['retargetQ']}, streaks {job['networkStreak']}/"
              f"{job['addressStreak']}), failsafe eases it by {job['failsafeBits']}")
        print(f"           idle since block {job['idleSince']}, "
              f"anchor {job['anchor'][:18]}...")


def _roots(protocol, url: str) -> tuple[str, str]:
    """The contract's roots when it will say, the protocol file's when it will not."""
    try:
        return tuple(request_batch(url, [
            ("eth_call", [{"to": protocol.require_deployed(),
                           "data": flynode.view_data(protocol, key)}, "latest"])
            for key in ("neuronsRoot", "edgesRoot")], 20.0))
    except Exception:
        import json
        payload = json.loads(
            (PROTOCOL_DIR / f"{protocol.name}.json").read_text(encoding="utf-8"))
        return payload["neuronsRoot"], payload["edgesRoot"]


if __name__ == "__main__":
    main()
