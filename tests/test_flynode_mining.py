#!/usr/bin/env python3
"""Mining FlyNode: reading the frontier, choosing a cell, and encoding the mint.

The lattice and the proofs are pinned in tests/test_lattice.py against the real
contract. What is left is everything between a proof and a transaction, and it
runs here against a stub that grows a graph the way the contract does.
"""
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eth_abi import encode as abi_encode  # noqa: E402
from eth_account import Account  # noqa: E402

import flynode  # noqa: E402
import lattice as lattice_module  # noqa: E402
import merkle  # noqa: E402
from protocol import PROTOCOL_DIR, load  # noqa: E402
from stub_chain import FlyNodeChain, StubServer  # noqa: E402

FLYNODE = load(PROTOCOL_DIR / "flynode.json")
KEY = "0x" + "33" * 32
ACCOUNT = Account.from_key(KEY)
MINE_TYPES = ["uint256", "uint256", "(uint32,uint16,uint8,uint8)", "bytes32[]",
              "uint32", "bytes32[]"]


def blob(lattice, seed: int, size: int) -> list[int]:
    """A connected patch of the lattice, the way a mined region actually looks."""
    claimed = [seed]
    edge = [seed]
    while len(claimed) < size and edge:
        current = edge.pop(0)
        for other in lattice.linked(current):
            if other not in claimed:
                claimed.append(other)
                edge.append(other)
                if len(claimed) >= size:
                    break
    return claimed


class EncodingTests(unittest.TestCase):
    def test_the_selector_is_the_one_the_deployed_mint_called(self):
        self.assertEqual("0x" + FLYNODE.mine_selector, "0x9013cdee")

    def test_calldata_matches_the_reference_encoder(self):
        leaf = (35160, 0, 4, 0)
        neuron_proof = [bytes([index]) * 32 for index in range(15)]
        edge_proof = [bytes([255 - index]) * 32 for index in range(18)]
        mine = flynode.encode_mine(2**255 + 7, 62107999, leaf, neuron_proof, 90041,
                                   edge_proof, FLYNODE)
        expected = "0x" + FLYNODE.mine_selector + abi_encode(
            MINE_TYPES, [2**255 + 7, 62107999, leaf, neuron_proof, 90041, edge_proof]).hex()
        self.assertEqual(mine, expected)

    def test_the_leaf_sits_in_the_head_and_the_arrays_do_not(self):
        """A tuple of static types is static: four words inline, no offset."""
        mine = flynode.encode_mine(1, 2, (3, 4, 5, 6), [b"\xaa" * 32], 7, [], FLYNODE)
        words = [mine[10 + index * 64:10 + (index + 1) * 64] for index in range(9)]
        self.assertEqual([int(word, 16) for word in words[:6]], [1, 2, 3, 4, 5, 6])
        self.assertEqual(int(words[6], 16), 288)     # neuronProof offset
        self.assertEqual(int(words[7], 16), 7)       # parent, still in the head
        self.assertEqual(int(words[8], 16), 352)     # edgeProof, after one-element array

    def test_a_proof_element_that_is_not_a_word_is_refused(self):
        with self.assertRaises(ValueError):
            flynode.encode_mine(1, 2, (3, 4, 5, 6), [b"\xaa" * 31], 7, [], FLYNODE)


class SlotTests(unittest.TestCase):
    """Finding the neuron id in an event whose shape nobody told us."""

    KNOWN = {10, 20, 30, 40, 50}

    def log(self, topic: str, indexed: list[int], data: list[int]) -> dict:
        return {"topics": [topic] + [f"0x{value:064x}" for value in indexed],
                "data": "0x" + "".join(f"{value:064x}" for value in data)}

    def test_picks_the_claimed_cell_over_the_parent(self):
        """Both are real ids; only the parent points backwards."""
        logs = [self.log("0xaa", [999, 10], [10, 62107948]),
                self.log("0xaa", [999, 20], [10, 62107949]),
                self.log("0xaa", [999, 30], [20, 62107950])]
        self.assertEqual(flynode.find_mined_slot(logs, self.KNOWN), ("0xaa", 1))

    def test_ignores_a_word_that_only_sometimes_looks_like_an_id(self):
        logs = [self.log("0xaa", [999, 10], [777]),
                self.log("0xaa", [999, 20], [30])]
        self.assertEqual(flynode.find_mined_slot(logs, self.KNOWN), ("0xaa", 1))

    def test_refuses_to_guess_when_nothing_identifies_a_cell(self):
        self.assertIsNone(flynode.find_mined_slot(
            [self.log("0xaa", [999], [777])], self.KNOWN))

    def test_a_repeated_value_is_not_a_claim(self):
        """A cell is claimed once, so a column that repeats cannot be the one."""
        logs = [self.log("0xaa", [10], []), self.log("0xaa", [10], [])]
        self.assertIsNone(flynode.find_mined_slot(logs, self.KNOWN))


@unittest.skipUnless(lattice_module.DATA_FILE.exists(),
                     "no dataset: run scripts/extract_pathway.py")
class FrontierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lattice = lattice_module.Lattice.load()
        cls.claimed = blob(cls.lattice, 90041, 24)

    def setUp(self):
        self.chain = FlyNodeChain(self.lattice, FLYNODE, occupied=self.claimed, failsafe=6)
        self.server = StubServer(self.chain)
        self.server.__enter__()
        self.addCleanup(self.server.__exit__)
        self.protocol = dataclasses.replace(FLYNODE, rpc=(self.server.url,))

    def scan(self) -> flynode.Frontier:
        return flynode.scan_mined(self.lattice, self.server.url, self.chain.deploy_block,
                                  self.chain.block_number, protocol=self.protocol)

    def test_reads_every_claimed_cell_back_out_of_the_logs(self):
        frontier = self.scan()
        self.assertEqual(set(frontier.occupied), set(self.claimed))
        self.assertEqual(frontier.slot, 1)

    def test_the_frontier_is_free_cells_touching_claimed_ones(self):
        cells = self.scan().open_cells(self.lattice)
        self.assertTrue(cells)
        for cell, parents in cells.items():
            self.assertNotIn(cell, self.claimed)
            for parent in parents:
                self.assertIn(parent, self.claimed)
                self.assertIn(parent, self.lattice.linked(cell))

    def test_a_chosen_cell_comes_with_a_parent_that_is_already_claimed(self):
        frontier = self.scan()
        cell, parent = frontier.choose(self.lattice)
        self.assertNotIn(cell, frontier.occupied)
        self.assertIn(parent, frontier.occupied)
        self.assertIn(parent, self.lattice.linked(cell))

    def test_cheapest_and_rarest_bracket_the_frontier(self):
        frontier = self.scan()
        cheap = self.lattice.neuron(frontier.choose(self.lattice, "cheapest")[0]).rarity_bits
        rare = self.lattice.neuron(frontier.choose(self.lattice, "rarest")[0]).rarity_bits
        self.assertLessEqual(cheap, rare)

    def test_each_rank_gets_its_own_cell(self):
        frontier = self.scan()
        picks = [frontier.choose(self.lattice, "cheapest", rank)[0] for rank in range(6)]
        self.assertEqual(len(set(picks)), len(picks))

    def test_an_empty_frontier_has_nothing_to_offer(self):
        self.assertIsNone(flynode.Frontier().choose(self.lattice))

    def test_a_rescan_from_where_it_stopped_finds_the_new_mint(self):
        frontier = self.scan()
        before = dict(frontier.occupied)
        fresh = next(cell for cell in frontier.open_cells(self.lattice))
        self.chain.block_number += 1
        self.chain.record(fresh, self.claimed[0], "0x" + "22" * 20, self.chain.block_number)
        flynode.scan_mined(self.lattice, self.server.url, frontier.scanned_to + 1,
                           self.chain.block_number, frontier, protocol=self.protocol)
        self.assertIn(fresh, frontier.occupied)
        self.assertEqual(set(frontier.occupied), set(before) | {fresh})

    def test_the_anchor_resolves_to_the_block_it_is_the_hash_of(self):
        wanted = self.chain.block_number - 7
        self.chain.anchor = self.chain.block_hash(wanted)
        self.assertEqual(
            flynode.resolve_anchor_block(self.server.url, self.chain.anchor,
                                         self.chain.block_number, window=64),
            wanted)

    def test_an_anchor_that_is_not_a_block_hash_falls_back_to_the_head(self):
        """Nothing to find, and guessing a wrong block would be worse than the head."""
        self.assertEqual(
            flynode.resolve_anchor_block(self.server.url, "0x" + "a9" * 32,
                                         self.chain.block_number, window=8),
            self.chain.block_number)


@unittest.skipUnless(lattice_module.DATA_FILE.exists(),
                     "no dataset: run scripts/extract_pathway.py")
class JobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lattice = lattice_module.Lattice.load()

    def setUp(self):
        self.chain = FlyNodeChain(self.lattice, FLYNODE,
                                  occupied=blob(self.lattice, 90041, 24), failsafe=6)
        self.server = StubServer(self.chain)
        self.server.__enter__()
        self.addCleanup(self.server.__exit__)
        self.protocol = dataclasses.replace(FLYNODE, rpc=(self.server.url,))
        self.source = flynode.FlyNodeSource(self.lattice, self.protocol, rank=0)

    def test_a_snapshot_carries_proofs_that_fold_to_the_contracts_roots(self):
        job = self.source.snapshot(ACCOUNT.address)
        leaf = merkle.neuron_leaf(*job["leaf"])
        neuron_proof = [bytes.fromhex(step[2:]) for step in job["neuronProof"]]
        edge_proof = [bytes.fromhex(step[2:]) for step in job["edgeProof"]]
        self.assertTrue(merkle.verify(neuron_proof, self.lattice.neurons_root, leaf))
        self.assertTrue(merkle.verify(
            edge_proof, self.lattice.edges_root,
            merkle.edge_leaf(job["parent"], job["cell"])))

    def test_the_difficulty_is_the_contracts_and_the_gap_is_its_failsafe(self):
        job = self.source.snapshot(ACCOUNT.address)
        self.assertEqual(job["difficulty"], self.chain.required_bits(job["rarityBits"]))
        self.assertEqual(job["predictedBits"], flynode.predicted_bits(
            job["rarityBits"], job["retargetQ"], job["networkStreak"], job["addressStreak"]))
        self.assertEqual(job["failsafeBits"], 6)

    def test_the_streaks_are_capped_at_sixteen(self):
        self.assertEqual(flynode.predicted_bits(0, 0, 900, 900), 16 + 16 + 16)
        self.assertEqual(flynode.predicted_bits(0, 8, 0, 0), 16 + 2)

    def test_the_same_cell_is_held_while_it_is_still_free(self):
        first = self.source.snapshot(ACCOUNT.address)
        second = self.source.snapshot(ACCOUNT.address)
        self.assertEqual(first["cell"], second["cell"])
        self.assertEqual(first["challenge"], second["challenge"])

    def test_a_cell_someone_else_claims_is_given_up(self):
        first = self.source.snapshot(ACCOUNT.address)
        self.chain.block_number += 1
        self.chain.record(first["cell"], first["parent"], "0x" + "44" * 20,
                          self.chain.block_number)
        second = self.source.snapshot(ACCOUNT.address)
        self.assertNotEqual(second["cell"], first["cell"])
        self.assertNotEqual(second["challenge"], first["challenge"])

    def test_a_cell_the_contract_calls_taken_is_dropped_before_the_logs_say_so(self):
        """mined() knows first; the frontier has to learn it or it loops forever."""
        first = self.source.snapshot(ACCOUNT.address)
        self.chain.record(first["cell"], first["parent"], "0x" + "44" * 20,
                          self.chain.block_number)   # a block already scanned
        second = self.source.snapshot(ACCOUNT.address)
        self.assertNotEqual(second["cell"], first["cell"])
        self.assertIn(first["cell"], self.source.frontier.occupied)

    def test_a_dataset_the_contract_does_not_hold_stops_the_run(self):
        """Mining against the wrong dataset only ever buys rejected transactions."""
        self.chain.roots = ("0x" + "11" * 32, "0x" + "22" * 32)
        source = flynode.FlyNodeSource(self.lattice, self.protocol)
        with self.assertRaisesRegex(SystemExit, "neuronsRoot mismatch"):
            source.snapshot(ACCOUNT.address)

    def test_the_job_identity_moves_when_anything_it_binds_moves(self):
        bindings = {"prev": "0x" + "11" * 32, "anchor": "0x" + "22" * 32, "typeId": 3}
        base = flynode.job_identity(bindings, 100)
        self.assertNotEqual(base, flynode.job_identity(bindings, 101))
        self.assertNotEqual(base, flynode.job_identity({**bindings, "typeId": 4}, 100))
        self.assertNotEqual(base, flynode.job_identity(
            {**bindings, "prev": "0x" + "33" * 32}, 100))


@unittest.skipUnless(lattice_module.DATA_FILE.exists(),
                     "no dataset: run scripts/extract_pathway.py")
class EndToEndTests(unittest.TestCase):
    """The real processes, with HASHBROKER_PROTOCOL pointing them at FlyNode."""

    @classmethod
    def setUpClass(cls):
        cls.lattice = lattice_module.Lattice.load()

    def setUp(self):
        self.chain = FlyNodeChain(self.lattice, FLYNODE,
                                  occupied=blob(self.lattice, 90041, 24), failsafe=8)
        self.server = StubServer(self.chain)
        self.server.__enter__()
        self.addCleanup(self.server.__exit__)
        self.workdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.workdir.cleanup)
        self.root = Path(self.workdir.name)
        self.environment = {
            **os.environ,
            "HASHBROKER_PROTOCOL": "flynode",
            "HASHBROKER_RPC_URLS": self.server.url,
            "HASHBROKER_WALLET": ACCOUNT.address,
            "HASHBROKER_PRIVATE_KEY": KEY,
            "HASHBROKER_RUNTIME_DIR": str(self.root / "runtime"),
            "PYTHONPATH": str(SCRIPTS),
        }

    def run_script(self, name: str, *arguments: str, timeout: float = 240.0):
        return subprocess.run([sys.executable, str(SCRIPTS / name), *arguments],
                              env=self.environment, capture_output=True, text=True,
                              timeout=timeout)

    def test_the_feed_publishes_a_mintable_cell(self):
        result = self.run_script("job_feed.py", "--wallet", ACCOUNT.address, "--once")
        self.assertEqual(result.returncode, 0, result.stderr)
        job = json.loads(result.stdout)
        self.assertIn(job["cell"], self.lattice.by_id)
        self.assertNotIn(job["cell"], self.chain.occupied)
        self.assertIn(job["parent"], self.chain.occupied)
        self.assertEqual(job["bindings"]["prev"], self.chain.prev)
        self.assertEqual(job["bindings"]["anchor"], self.chain.anchor)
        self.assertEqual(len(job["neuronProof"]), 15)
        self.assertEqual(len(job["edgeProof"]), 18)

    def test_a_worker_mines_that_cell_and_the_signer_sends_the_mint(self):
        job_file = self.root / "job.json"
        feed = subprocess.Popen(
            [sys.executable, str(SCRIPTS / "job_feed.py"), "--wallet", ACCOUNT.address,
             "--output", str(job_file), "--interval", "1.0"],
            env=self.environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.addCleanup(self.shut_down, feed)

        solution_file = self.root / "solution-cpu.json"
        miner = self.run_script("miner_cpu.py", "--wallet", ACCOUNT.address,
                                "--job-file", str(job_file), "--output", str(solution_file),
                                "--processes", "2")
        self.assertEqual(miner.returncode, 0, miner.stdout[-3000:])
        solution = json.loads(solution_file.read_text())

        signer = self.run_script("signer.py", "--solutions", str(solution_file), "--once")
        self.assertEqual(signer.returncode, 0, signer.stdout[-3000:])
        self.assertIn("BROADCAST", signer.stdout)
        self.assertEqual(len(self.chain.sent), 1)

        sent = self.decode_sent(self.chain.sent[0])
        self.assertEqual(sent["value"], self.chain.price, "value must be exactly entryPrice()")
        self.assertEqual(sent["data"][:10], "0x" + FLYNODE.mine_selector)
        expected = "0x" + FLYNODE.mine_selector + abi_encode(MINE_TYPES, [
            int(solution["nonce"]), int(solution["anchorBlock"]),
            tuple(solution["leaf"]),
            [bytes.fromhex(step[2:]) for step in solution["neuronProof"]],
            int(solution["parent"]),
            [bytes.fromhex(step[2:]) for step in solution["edgeProof"]],
        ]).hex()
        self.assertEqual(sent["data"], expected)

    @staticmethod
    def shut_down(process: subprocess.Popen) -> None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        if process.stdout:
            process.stdout.close()

    @staticmethod
    def decode_sent(raw: str) -> dict:
        from eth_account.typed_transactions import TypedTransaction
        from hexbytes import HexBytes
        transaction = TypedTransaction.from_bytes(HexBytes(raw))
        payload = transaction.as_dict()
        return {"value": payload["value"],
                "data": "0x" + bytes(payload["data"]).hex()}


if __name__ == "__main__":
    unittest.main()
