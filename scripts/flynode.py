#!/usr/bin/env python3
"""FlyNode's own chain reads: where the lattice has grown, and what a mint costs.

FlyNode does not mint a free-floating token. It grows a graph: every mint claims
one neuron of the lattice and has to name an already-claimed neighbour to grow
from, proving both against the two roots the contract holds. So a job here is
more than a challenge and a target — it is a *cell*, with the proofs that place
it, and a difficulty that depends on how rare that cell's type is.

    frontier   which cells are free and touch a claimed one
    read_job   prevWork, currentAnchor, entryPrice, requiredBits, and the
               streaks and idle counter that move requiredBits around
    calldata   the mine() call, two dynamic arrays and all

The contract is not verified, so the Mined event's shape is not something we can
look up. ``scan_mined`` does not assume it: it reads the contract's logs and
keeps whatever decodes to a neuron the lattice knows, whichever topic or data
word it turns up in. A wrong guess cannot quietly poison the frontier, because a
value that is not a real neuron id is not a value it will accept.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import pow as powlib
from job_state import request_batch
from protocol import PROTOCOL, Protocol

DEPLOY_BLOCK = 62107948
LOG_CHUNK = 8192
MIN_LOG_CHUNK = 64
BASE_BITS = 16
STREAK_CAP = 16


def _word(value: int) -> str:
    return f"{int(value) & (2**256 - 1):064x}"


def _call(protocol: Protocol, url: str, data: str, block: str = "latest",
          timeout: float = 8.0) -> str:
    return request_batch(url, [("eth_call", [
        {"to": protocol.require_deployed(), "data": data}, block])], timeout)[0]


def view_data(protocol: Protocol, key: str, *arguments: int | str) -> str:
    """A view selector with its arguments, each padded into one 32-byte word."""
    body = ""
    for argument in arguments:
        if isinstance(argument, str):
            body += argument.removeprefix("0x").lower().rjust(64, "0")
        else:
            body += _word(argument)
    return protocol.view(key) + body


# --- the frontier ------------------------------------------------------------


@dataclass
class Frontier:
    """Which cells are taken, and which free ones may be claimed next."""

    occupied: dict[int, str] = field(default_factory=dict)   # neuron id -> miner
    scanned_to: int = DEPLOY_BLOCK
    topic: str | None = None        # the Mined event, once its logs have shown it
    slot: int | None = None         # which word of that event holds the neuron id

    def __len__(self) -> int:
        return len(self.occupied)

    def open_cells(self, lattice) -> dict[int, list[int]]:
        """Free neurons that touch a claimed one, each with the parents it may use.

        An empty lattice has no frontier to speak of: nothing is claimed, so
        nothing is adjacent to anything claimed, and the first mint is the
        contract's business rather than ours.
        """
        cells: dict[int, list[int]] = {}
        for taken in self.occupied:
            for neighbour in lattice.linked(taken):
                if neighbour not in self.occupied:
                    cells.setdefault(neighbour, []).append(taken)
        return cells

    def choose(self, lattice, prefer: str = "cheapest",
               rank: int = 0) -> tuple[int, int] | None:
        """Pick a cell and a parent for it: the rarer the type, the more bits it costs.

        ``rank`` is how far down the order to go. Two boxes that both take the
        best cell spend twice the power on one mint, so a fleet gives each box
        its own rank and they mine a different cell each.
        """
        cells = self.open_cells(lattice)
        if not cells:
            return None
        sign = 1 if prefer == "cheapest" else -1
        order = sorted(cells, key=lambda cell: (sign * lattice.neuron(cell).rarity_bits, cell))
        cell = order[rank % len(order)]
        return cell, min(cells[cell])


def log_slots(log: dict) -> list[int | None]:
    """A log flattened into words: the indexed topics first, then the data."""
    values: list[int | None] = []
    data = str(log.get("data", "0x")).removeprefix("0x")
    for word in list(log.get("topics", ())[1:]) + [
            data[index:index + 64] for index in range(0, len(data) - 63, 64)]:
        try:
            values.append(int(str(word), 16))
        except ValueError:
            values.append(None)
    return values


def _grows_from(minted: list[int | None], parent: list[int | None]) -> bool:
    """Could ``parent`` be the parent column if ``minted`` is the claimed column?

    Only if it never reaches forwards. A parent that some log claims must have
    been claimed by an *earlier* log; a parent that no log ever claims is a
    genesis cell the contract seeded, and says nothing either way.
    """
    claimed: set[int | None] = set()
    everything = set(minted)
    for new_cell, from_cell in zip(minted, parent):
        if from_cell in everything and from_cell not in claimed:
            return False
        claimed.add(new_cell)
    return True


def find_mined_slot(logs: list[dict], known: set[int]) -> tuple[str, int] | None:
    """Work out which event, and which word of it, carries the neuron that was claimed.

    Reading "any word that happens to be a known id" is not good enough. A
    neuron id is a plain number, and so is a block number or a token index, so
    one of those could pass for a cell and invent occupancy that is not there.
    Worse, a mint names two real cells — the one it claims and the one it grows
    from — so being a valid id does not even make a word the right one.

    Three things narrow it down, and all of them have to hold:
      * a real id column holds a known neuron in *every* log of its event,
      * a cell is claimed once, so that column never repeats itself,
      * and a parent is a cell that was already claimed, so the parent column
        can only ever point backwards.
    """
    by_topic: dict[str, list[list[int | None]]] = {}
    for log in logs:
        topics = log.get("topics") or []
        if topics:
            by_topic.setdefault(str(topics[0]), []).append(log_slots(log))

    best: tuple[str, int] | None = None
    best_count = 0
    for topic, rows in by_topic.items():
        width = min((len(row) for row in rows), default=0)
        columns = {slot: [row[slot] for row in rows] for slot in range(width)}
        fits = [slot for slot, column in columns.items()
                if all(value in known for value in column)
                and len(set(column)) == len(column)]
        if len(fits) > 1:
            fits = [slot for slot in fits
                    if all(_grows_from(columns[slot], columns[other])
                           for other in fits if other != slot)]
        if len(fits) == 1 and len(rows) > best_count:
            best, best_count = (topic, fits[0]), len(rows)
    return best


def _miner_of(log: dict) -> str:
    """The first indexed topic that looks like an address, when there is one."""
    for topic in (log.get("topics") or [])[1:]:
        value = str(topic).removeprefix("0x").rjust(64, "0")
        if value[:24] == "0" * 24 and int(value, 16) != 0:
            return "0x" + value[24:]
    return ""


def fetch_logs(url: str, address: str, from_block: int, to_block: int, topic: str | None,
               chunk: int, timeout: float) -> list[dict]:
    """Every log in the range, halving the window whenever an endpoint refuses it."""
    collected: list[dict] = []
    start = from_block
    while start <= to_block:
        span = min(chunk, to_block - start + 1)
        query = {"address": address, "fromBlock": hex(start),
                 "toBlock": hex(start + span - 1)}
        if topic:
            query["topics"] = [topic]
        try:
            rows = request_batch(url, [("eth_getLogs", [query])], timeout)[0]
        except Exception:
            # Endpoints cap a range by blocks or by result size and do not agree
            # on which; halving until it fits is the only portable answer.
            if span <= MIN_LOG_CHUNK:
                raise
            chunk = max(MIN_LOG_CHUNK, span // 2)
            continue
        collected.extend(rows or ())
        start += span
    return collected


def scan_mined(lattice, url: str, from_block: int, to_block: int,
               frontier: Frontier | None = None, chunk: int = LOG_CHUNK,
               protocol: Protocol = PROTOCOL, timeout: float = 20.0) -> Frontier:
    """Walk the contract's logs and record every neuron that has been claimed."""
    # Not `frontier or Frontier()`: an empty frontier is falsy, and replacing the
    # caller's would send every cell found to an object nobody else can see.
    if frontier is None:
        frontier = Frontier()
    if to_block < from_block:
        return frontier
    known = set(lattice.by_id)
    logs = fetch_logs(url, protocol.require_deployed(), from_block, to_block,
                      frontier.topic, chunk, timeout)
    if frontier.slot is None:
        found = find_mined_slot(logs, known)
        if found is None:
            frontier.scanned_to = to_block
            return frontier
        frontier.topic, frontier.slot = found
    for log in logs:
        topics = log.get("topics") or []
        if not topics or str(topics[0]) != frontier.topic:
            continue
        slots = log_slots(log)
        if frontier.slot >= len(slots):
            continue
        claimed = slots[frontier.slot]
        if claimed in known:
            frontier.occupied[claimed] = _miner_of(log)
    frontier.scanned_to = to_block
    return frontier


# --- the job -----------------------------------------------------------------


def predicted_bits(rarity: int, retarget_q: int, network_streak: int,
                   address_streak: int) -> int:
    """The difficulty the contract's own rule implies, before its failsafe eases it.

    16 + retargetQ/4 + rarityBits + min(16, network streak) + min(16, your streak)

    requiredBits() is the number that counts; this exists so the difference
    between the two can be reported, because that difference is the failsafe.
    """
    return (BASE_BITS + int(retarget_q) // 4 + int(rarity)
            + min(STREAK_CAP, int(network_streak))
            + min(STREAK_CAP, int(address_streak)))


def read_job(wallet: str, lattice, rarity: int, url: str | None = None,
             protocol: Protocol = PROTOCOL, timeout: float = 8.0) -> dict:
    """One snapshot of everything a mint needs, read at a single block."""
    endpoint = url or protocol.rpc[0]
    chain, block = request_batch(endpoint, [("eth_chainId", []), ("eth_blockNumber", [])],
                                 timeout)
    if int(chain, 16) != protocol.chain_id:
        raise ValueError(f"wrong chain {int(chain, 16)}, wanted {protocol.chain_id}")
    block_number = int(block, 16)
    at = hex(block_number)

    keys = ("prev", "anchor", "price", "minted", "maxSupply", "retargetQ",
            "networkStreak", "idleSince", "lastMintBlock")
    calls = [("eth_call", [{"to": protocol.require_deployed(), "data": view_data(protocol, key)},
                           at]) for key in keys]
    calls.append(("eth_call", [{"to": protocol.require_deployed(),
                                "data": view_data(protocol, "addressStreak", wallet)}, at]))
    calls.append(("eth_call", [{"to": protocol.require_deployed(),
                                "data": view_data(protocol, "requiredBits", rarity, wallet)}, at]))
    rows = request_batch(endpoint, calls, timeout)
    raw = dict(zip(keys, rows))
    address_streak = int(rows[-2], 16)
    required = int(rows[-1], 16)
    if not 0 < required <= 255:
        raise ValueError(f"implausible requiredBits {required}")

    retarget_q = int(raw["retargetQ"], 16)
    network_streak = int(raw["networkStreak"], 16)
    predicted = predicted_bits(rarity, retarget_q, network_streak, address_streak)
    anchor_block = block_number
    return {
        "prev": "0x" + raw["prev"][2:].rjust(64, "0"),
        "anchor": "0x" + raw["anchor"][2:].rjust(64, "0"),
        "anchorBlock": anchor_block,
        "rarityBits": int(rarity),
        "difficulty": required,
        "target": "0x%064x" % powlib.target_for_difficulty(required),
        "priceWei": int(raw["price"], 16),
        "minted": int(raw["minted"], 16),
        "maxSupply": int(raw["maxSupply"], 16),
        "lastMintBlock": int(raw["lastMintBlock"], 16),
        "retargetQ": retarget_q,
        "networkStreak": network_streak,
        "addressStreak": address_streak,
        "idleSince": int(raw["idleSince"], 16),
        "predictedBits": predicted,
        "failsafeBits": predicted - required,
        "blockNumber": block_number,
        "fetchedAt": time.time(),
    }


def is_mined(neuron_id: int, url: str | None = None, protocol: Protocol = PROTOCOL,
             timeout: float = 8.0) -> bool:
    """Ask the contract directly, to catch a cell taken since the last log scan."""
    endpoint = url or protocol.rpc[0]
    result = _call(protocol, endpoint, view_data(protocol, "mined", neuron_id), "latest", timeout)
    return int(result or "0x0", 16) != 0


# --- the mint ----------------------------------------------------------------


def encode_mine(nonce: int, anchor_block: int, leaf: tuple[int, int, int, int],
                neuron_proof: list[bytes], parent: int, edge_proof: list[bytes],
                protocol: Protocol = PROTOCOL) -> str:
    """ABI-encode mine(uint256,uint256,(uint32,uint16,uint8,uint8),bytes32[],uint32,bytes32[]).

    The leaf is a tuple of four static types, so it is static too and sits in the
    head as four words rather than behind an offset. The two arrays are dynamic
    and do sit behind offsets, measured from the start of the arguments.
    """
    for proof in (neuron_proof, edge_proof):
        if any(len(sibling) != 32 for sibling in proof):
            raise ValueError("every proof element is one 32-byte word")
    head_words = 2 + len(leaf) + 1 + 1 + 1          # nonce, block, leaf, ptr, parent, ptr
    neuron_at = head_words * 32
    edge_at = neuron_at + 32 * (1 + len(neuron_proof))
    head = (_word(nonce) + _word(anchor_block)
            + "".join(_word(part) for part in leaf)
            + _word(neuron_at) + _word(parent) + _word(edge_at))
    tail = (_word(len(neuron_proof)) + "".join(s.hex() for s in neuron_proof)
            + _word(len(edge_proof)) + "".join(s.hex() for s in edge_proof))
    return "0x" + protocol.mine_selector + head + tail


def resolve_anchor_block(url: str, anchor: str, head: int, window: int = 256,
                         timeout: float = 8.0) -> int:
    """Which block the contract's anchor is the hash of.

    mine() is handed an anchorBlock alongside the proof, and the proof commits
    to the anchor itself, so the two have to agree or the contract throws. The
    contract does not say which block it picked, but the chain does: walk back
    over the window and find the block whose hash is that anchor. If none is —
    the anchor is not a block hash at all, or it has aged out — the head is the
    only honest answer, and the contract will say so.
    """
    wanted = str(anchor).lower()
    span = max(1, min(window, head + 1))
    numbers = list(range(head, head - span, -1))
    for start in range(0, len(numbers), 32):
        batch = numbers[start:start + 32]
        try:
            rows = request_batch(url, [("eth_getBlockByNumber", [hex(number), False])
                                       for number in batch], timeout)
        except Exception:
            return head      # best effort: an endpoint that will not say is not fatal
        for number, block in zip(batch, rows):
            if block and str(block.get("hash", "")).lower() == wanted:
                return number
    return head


class FlyNodeSource:
    """Picks a cell to mine and keeps publishing it until it is gone.

    Holding the choice matters. Every time the cell changes the preimage changes
    with it, and a worker has to throw away the nonces it has already tried — so
    this keeps mining the same cell until someone claims it, the chain moves the
    difficulty out from under it, or it stops being on the frontier at all.
    """

    def __init__(self, lattice=None, protocol: Protocol = PROTOCOL,
                 prefer: str | None = None, rank: int | None = None,
                 deploy_block: int = DEPLOY_BLOCK, verify_roots: bool = True):
        import lattice as lattice_module
        self.protocol = protocol
        self.lattice = lattice if lattice is not None else lattice_module.shared()
        self.prefer = prefer or os.environ.get("HASHBROKER_CELL_PREFER", "cheapest")
        self.rank = int(os.environ.get("HASHBROKER_CELL_RANK", "0")) if rank is None else rank
        self.deploy_block = deploy_block
        self.frontier = Frontier(scanned_to=deploy_block - 1)
        self.cell: int | None = None
        self.parent: int | None = None
        self._roots_checked = not verify_roots
        self._anchor: tuple[str, int] | None = None

    # --- setup ---------------------------------------------------------------

    def check_roots(self, url: str, timeout: float = 8.0) -> None:
        """The contract's own roots first; the protocol file only if it has none."""
        if self._roots_checked:
            return
        pair = []
        for key in ("neuronsRoot", "edgesRoot"):
            try:
                pair.append(_call(self.protocol, url, view_data(self.protocol, key),
                                  "latest", timeout))
            except Exception:
                pair = []
                break
        if len(pair) != 2:
            import json
            from protocol import PROTOCOL_DIR
            payload = json.loads(
                (PROTOCOL_DIR / f"{self.protocol.name}.json").read_text(encoding="utf-8"))
            pair = [payload.get("neuronsRoot"), payload.get("edgesRoot")]
            if not all(pair):
                raise SystemExit("no Merkle roots to check this dataset against")
        self.lattice.check_roots(pair[0], pair[1])
        self._roots_checked = True

    # --- the cell ------------------------------------------------------------

    def refresh_frontier(self, url: str, head: int, timeout: float = 20.0) -> None:
        scan_mined(self.lattice, url, self.frontier.scanned_to + 1, head,
                   self.frontier, protocol=self.protocol, timeout=timeout)

    def pick_cell(self, url: str, timeout: float = 8.0) -> tuple[int, int]:
        """Keep the current cell while it is still worth mining, else choose again."""
        if self.cell is not None and self.cell not in self.frontier.occupied:
            parents = [other for other in self.lattice.linked(self.cell)
                       if other in self.frontier.occupied]
            if parents:
                if not is_mined(self.cell, url, self.protocol, timeout):
                    self.parent = min(parents)
                    return self.cell, self.parent
                # The contract knows before our log scan catches up. Write it down,
                # or the next choice walks straight back onto the same dead cell.
                self.frontier.occupied.setdefault(self.cell, "")
        chosen = self.frontier.choose(self.lattice, self.prefer, self.rank)
        if chosen is None:
            raise ValueError(
                f"nothing on the frontier: {len(self.frontier)} cells claimed of "
                f"{self.lattice.size}")
        self.cell, self.parent = chosen
        return self.cell, self.parent

    # --- the job -------------------------------------------------------------

    def snapshot(self, wallet: str, preferred: int | None = None,
                 timeout: float = 8.0) -> dict:
        urls = self.protocol.rpc
        url = urls[(preferred or 0) % len(urls)]
        self.check_roots(url, timeout)

        head = int(request_batch(url, [("eth_blockNumber", [])], timeout)[0], 16)
        self.refresh_frontier(url, head)
        cell, parent = self.pick_cell(url, timeout)
        neuron = self.lattice.neuron(cell)

        job = read_job(wallet, self.lattice, neuron.rarity_bits, url, self.protocol, timeout)
        if self._anchor is None or self._anchor[0] != job["anchor"]:
            self._anchor = (job["anchor"],
                            resolve_anchor_block(url, job["anchor"], job["blockNumber"],
                                                 timeout=timeout))
        job["anchorBlock"] = self._anchor[1]

        neuron_proof = self.lattice.neuron_proof(cell)
        edge_proof = self.lattice.edge_proof(parent, cell)
        job.update({
            "cell": cell,
            "parent": parent,
            "leaf": list(neuron.leaf),
            "typeId": neuron.type_id,
            "region": neuron.region,
            "typeName": self.lattice.types[neuron.type_id]["name"],
            "neuronProof": ["0x" + step.hex() for step in neuron_proof],
            "edgeProof": ["0x" + step.hex() for step in edge_proof],
            "occupied": len(self.frontier),
            "bindings": {"prev": job["prev"], "anchor": job["anchor"],
                         "typeId": neuron.type_id},
        })
        job["challenge"] = job_identity(job["bindings"], cell)
        return job


def job_identity(bindings: dict, cell: int) -> str:
    """One bytes32 that changes whenever the work a worker is doing would change.

    The rest of the miner tracks a single 'challenge' and starts over when it
    moves. FlyNode has no such field — it has a preimage and a cell — so this
    stands in for one.
    """
    from merkle import keccak256
    material = (bytes.fromhex(str(bindings["prev"]).removeprefix("0x"))
                + bytes.fromhex(str(bindings["anchor"]).removeprefix("0x"))
                + int(bindings["typeId"]).to_bytes(2, "big")
                + int(cell).to_bytes(4, "big"))
    return "0x" + keccak256(material).hex()
