#!/usr/bin/env python3
"""Turns the pathway bundle the site ships into the dataset the miner mines.

The bundle is a data-only ES module: a table of the 45 neuron types, then ten
base64 blobs of packed columns. Nothing in it is labelled — the names are
whatever the minifier chose that build — so this reads the blobs by *shape*
instead, and every choice it makes is checked against something independent:

  typeIds    the 20100-byte column whose histogram is exactly the ``real``
             count of each type in the table
  adjacency  the varint stream that splits into 20100 degrees followed by
             exactly that many neighbours
  ids        the varint stream of 20100 zigzag deltas whose running sum gives
             20100 distinct ids

Run it against a new bundle and it either finds all three or says which one it
could not place. The roots are the final word: scripts/lattice.py rebuilds both
trees from what this writes and compares them with the contract's.

    python3 scripts/extract_pathway.py path/to/pathway-*.js
"""
from __future__ import annotations

import argparse
import base64
import collections
import json
import re
from pathlib import Path

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "pathway.json"


def read_varints(data: bytes) -> list[int]:
    """LEB128, as the bundle's own reader consumes it."""
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


def zigzag(value: int) -> int:
    return (value >> 1) ^ -(value & 1)


def running_sum(deltas: list[int]) -> list[int]:
    total = 0
    out = []
    for delta in deltas:
        total += delta
        out.append(total)
    return out


def parse_bundle(source: str) -> tuple[list[dict], list[str], dict[str, bytes]]:
    table = re.search(r"const [A-Za-z_$][\w$]*=(\[\{.*?\}\]),", source)
    if not table:
        raise SystemExit("no neuron type table in this bundle")
    literal = re.sub(r"([{,])(\w+):", r'\1"\2":', table.group(1))
    literal = re.sub(r":(\.\d)", r":0\1", literal)      # JS writes .5, JSON needs 0.5
    types = json.loads(literal)
    after = source[table.end():]
    regions = next((json.loads(found) for found in
                    re.findall(r'(\["[^"\]]+(?:","[^"\]]+)*"\])', after[:400])), [])
    blobs = {name: base64.b64decode(payload) for name, payload in
             re.findall(r'(?:const |,)([A-Za-z_$][\w$]*)="([A-Za-z0-9+/=]{200,})"', source)}
    return types, regions, blobs


def find_type_ids(blobs: dict[str, bytes], real: list[int]) -> bytes:
    """The one byte-per-neuron column that reproduces the table's own counts."""
    wanted = list(real)
    for name, blob in blobs.items():
        if len(blob) != sum(wanted):
            continue
        counts = collections.Counter(blob)
        if [counts.get(index, 0) for index in range(len(wanted))] == wanted:
            return blob
    raise SystemExit("no column in this bundle has one type per neuron")


def find_adjacency(blobs: dict[str, bytes], count: int) -> tuple[bytes, list[int]]:
    """Degrees first, then that many neighbours: the split has to land exactly."""
    for blob in blobs.values():
        try:
            values = read_varints(blob)
        except ValueError:
            continue
        if len(values) > count and sum(values[:count]) == len(values) - count:
            return blob, values
    raise SystemExit("no varint stream in this bundle splits into degrees and neighbours")


def find_ids(blobs: dict[str, bytes], count: int) -> list[int]:
    """One zigzag delta per neuron, and the running sum must not repeat an id."""
    for blob in blobs.values():
        try:
            values = read_varints(blob)
        except ValueError:
            continue
        if len(values) != count:
            continue
        ids = running_sum([zigzag(value) for value in values])
        if len(set(ids)) == count and all(0 <= i < 2**32 for i in ids):
            return ids
    raise SystemExit("no varint stream in this bundle holds one distinct id per neuron")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", help="the pathway-*.js the site loads")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    arguments = parser.parse_args()

    source = Path(arguments.bundle).read_text(encoding="utf-8")
    types, regions, blobs = parse_bundle(source)
    real = [int(entry["real"]) for entry in types]
    count = sum(real)

    type_ids = find_type_ids(blobs, real)
    adjacency_blob, adjacency = find_adjacency(blobs, count)
    ids = find_ids(blobs, count)
    degrees = adjacency[:count]

    payload = {
        "source": Path(arguments.bundle).name,
        "neurons": count,
        "edges": sum(degrees),
        "regions": regions,
        "types": [{"name": entry["name"], "nt": entry["nt"],
                   "real": int(entry["real"]), "stage": int(entry["stage"])}
                  for entry in types],
        "ids": base64.b64encode(b"".join(i.to_bytes(4, "big") for i in ids)).decode(),
        "typeIds": base64.b64encode(type_ids).decode(),
        "adjacency": base64.b64encode(adjacency_blob).decode(),
    }
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{count} neurons, {sum(degrees)} directed edges, {len(types)} types "
          f"-> {output}")


if __name__ == "__main__":
    main()
