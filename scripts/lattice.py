#!/usr/bin/env python3
"""The neuron lattice FlyNode mints from, and the proofs that mint a cell.

The contract holds two Merkle roots and nothing else about the dataset. Every
mint has to hand it a leaf plus a proof for each: which neuron is being claimed
(neuronsRoot) and which already-claimed neuron it grows from (edgesRoot). This
module rebuilds both trees from the bundle the site ships, so those proofs can
be produced offline.

A neuron leaf is ``(id, typeId, rarityBits, region)``. Only ``id`` and
``typeId`` are stored; the other two are properties of the type:

    region     the type's stage, 0..3 — the eye, visual projection,
               central brain, descending
    rarityBits round(log2(LATTICE / how many neurons share this type)), so a
               type with 892 of its kind is worth 4 bits and one with 2 is
               worth 13

Edges are undirected as far as the contract cares, but the tree holds both
directions as separate leaves, in the order the dataset walks them: for each
neuron in order, for each neighbour in order, ``(a, b)`` then ``(b, a)``, with
anything already seen skipped.

Building both trees costs a few seconds, so a process that needs them should
build once and keep the object.
"""
from __future__ import annotations

import base64
import json
import math
from dataclasses import dataclass
from pathlib import Path

import merkle

DATA_FILE = Path(__file__).resolve().parent / "data" / "pathway.json"


@dataclass(frozen=True)
class Neuron:
    id: int
    type_id: int
    rarity_bits: int
    region: int

    @property
    def leaf(self) -> tuple[int, int, int, int]:
        """The tuple the contract hashes, in its struct order."""
        return (self.id, self.type_id, self.rarity_bits, self.region)


def read_varints(data: bytes) -> list[int]:
    values: list[int] = []
    value = shift = 0
    for byte in data:
        value |= (byte & 0x7F) << shift
        if byte & 0x80:
            shift += 7
        else:
            values.append(value)
            value = shift = 0
    if shift:
        raise ValueError("varint stream ends mid-value")
    return values


def rarity_bits(population: int, lattice: int) -> int:
    """How many bits of difficulty a type's scarcity is worth, rounded half up."""
    if not 0 < population <= lattice:
        raise ValueError(f"a type of {population} in a lattice of {lattice}")
    return math.floor(math.log2(lattice / population) + 0.5)


class Lattice:
    """The whole dataset: neurons, their edges, and a tree over each."""

    def __init__(self, payload: dict):
        self.source = str(payload.get("source", "unknown"))
        self.types = list(payload["types"])
        self.regions = list(payload.get("regions", ()))
        count = int(payload["neurons"])

        raw_ids = base64.b64decode(payload["ids"])
        type_ids = base64.b64decode(payload["typeIds"])
        if len(raw_ids) != 4 * count or len(type_ids) != count:
            raise ValueError("dataset columns disagree on how many neurons there are")
        ids = [int.from_bytes(raw_ids[i:i + 4], "big") for i in range(0, len(raw_ids), 4)]

        self.size = count
        self.rarity_of_type = [rarity_bits(int(entry["real"]), count) for entry in self.types]
        self.neurons = [
            Neuron(ids[index], type_ids[index],
                   self.rarity_of_type[type_ids[index]],
                   int(self.types[type_ids[index]]["stage"]))
            for index in range(count)
        ]
        self.by_id = {neuron.id: index for index, neuron in enumerate(self.neurons)}
        if len(self.by_id) != count:
            raise ValueError("two neurons share an id")

        self.adjacency = self._read_adjacency(base64.b64decode(payload["adjacency"]), count)
        self.edges = self._walk_edges()
        # The dataset stores each connection once, in the direction the signal
        # travels. The tree holds both ways, so either end can be the parent of
        # the other — which is the adjacency that matters when picking a cell.
        self.links: list[list[int]] = [[] for _ in range(count)]
        for parent, child in self.edges:
            self.links[self.by_id[parent]].append(child)

        self.neuron_tree = merkle.build_tree(
            [merkle.neuron_leaf(*neuron.leaf) for neuron in self.neurons])
        self.edge_tree = merkle.build_tree(
            [merkle.edge_leaf(parent, child) for parent, child in self.edges])
        self._edge_index = {pair: index for index, pair in enumerate(self.edges)}

    @staticmethod
    def _read_adjacency(blob: bytes, count: int) -> list[list[int]]:
        """Degrees, then each neuron's neighbours as ascending deltas."""
        values = read_varints(blob)
        degrees, targets = values[:count], values[count:]
        if sum(degrees) != len(targets):
            raise ValueError("adjacency degrees do not account for its neighbours")
        rows, cursor = [], 0
        for degree in degrees:
            row, running = [], 0
            for delta in targets[cursor:cursor + degree]:
                running += delta
                row.append(running)
            cursor += degree
            rows.append(row)
        return rows

    def _walk_edges(self) -> list[tuple[int, int]]:
        """Both directions of every edge, deduplicated, in the order first seen."""
        seen: set[tuple[int, int]] = set()
        order: list[tuple[int, int]] = []
        for index, row in enumerate(self.adjacency):
            source = self.neurons[index].id
            for target in row:
                other = self.neurons[target].id
                for pair in ((source, other), (other, source)):
                    if pair not in seen:
                        seen.add(pair)
                        order.append(pair)
        return order

    # --- lookups -------------------------------------------------------------

    def neuron(self, neuron_id: int) -> Neuron:
        try:
            return self.neurons[self.by_id[neuron_id]]
        except KeyError:
            raise KeyError(f"no neuron {neuron_id} in this lattice") from None

    def neighbours(self, neuron_id: int) -> list[int]:
        """Where this neuron sends signal — the dataset's own direction."""
        index = self.by_id.get(neuron_id)
        if index is None:
            raise KeyError(f"no neuron {neuron_id} in this lattice")
        return [self.neurons[target].id for target in self.adjacency[index]]

    def linked(self, neuron_id: int) -> list[int]:
        """Everything this neuron connects to either way: its possible parents."""
        index = self.by_id.get(neuron_id)
        if index is None:
            raise KeyError(f"no neuron {neuron_id} in this lattice")
        return self.links[index]

    # --- proofs --------------------------------------------------------------

    @property
    def neurons_root(self) -> bytes:
        return self.neuron_tree.root

    @property
    def edges_root(self) -> bytes:
        return self.edge_tree.root

    def neuron_proof(self, neuron_id: int) -> list[bytes]:
        return self.neuron_tree.proof_at(self.by_id[neuron_id])

    def edge_proof(self, parent: int, child: int) -> list[bytes]:
        try:
            return self.edge_tree.proof_at(self._edge_index[(parent, child)])
        except KeyError:
            raise KeyError(f"no edge {parent} -> {child} in this lattice") from None

    def check_roots(self, neurons_root: str, edges_root: str) -> None:
        """Refuse to mine against a dataset the contract does not hold."""
        for label, built, expected in (
            ("neuronsRoot", self.neurons_root, neurons_root),
            ("edgesRoot", self.edges_root, edges_root),
        ):
            wanted = bytes.fromhex(str(expected).removeprefix("0x"))
            if built != wanted:
                raise SystemExit(
                    f"{label} mismatch: this dataset builds 0x{built.hex()}, "
                    f"the contract holds 0x{wanted.hex()}"
                )

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Lattice":
        source = Path(path) if path else DATA_FILE
        if not source.exists():
            raise SystemExit(
                f"no lattice dataset at {source}: run "
                "python3 scripts/extract_pathway.py path/to/pathway-*.js"
            )
        return cls(json.loads(source.read_text(encoding="utf-8")))


_CACHED: Lattice | None = None


def shared() -> Lattice:
    """One lattice per process — building it twice is pure waste."""
    global _CACHED
    if _CACHED is None:
        _CACHED = Lattice.load()
    return _CACHED


def main() -> None:
    import argparse
    import time

    parser = argparse.ArgumentParser(description="rebuild the lattice and print its roots")
    parser.add_argument("--protocol", default="flynode")
    arguments = parser.parse_args()

    began = time.monotonic()
    lattice = Lattice.load()
    elapsed = time.monotonic() - began
    print(f"source      {lattice.source}")
    print(f"neurons     {lattice.size} in {len(lattice.types)} types")
    print(f"edges       {len(lattice.edges)} leaves "
          f"({sum(len(row) for row in lattice.adjacency)} directed, both ways, deduplicated)")
    print(f"neuronsRoot 0x{lattice.neurons_root.hex()}")
    print(f"edgesRoot   0x{lattice.edges_root.hex()}")
    print(f"built in    {elapsed:.1f}s")

    from protocol import PROTOCOL_DIR, load as load_protocol
    payload = json.loads((PROTOCOL_DIR / f"{arguments.protocol}.json").read_text())
    expected_neurons = payload.get("neuronsRoot")
    expected_edges = payload.get("edgesRoot")
    if expected_neurons and expected_edges:
        lattice.check_roots(expected_neurons, expected_edges)
        print(f"both roots match {load_protocol(PROTOCOL_DIR / f'{arguments.protocol}.json').name}")


if __name__ == "__main__":
    main()
