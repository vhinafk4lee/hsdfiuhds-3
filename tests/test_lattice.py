#!/usr/bin/env python3
"""The lattice rebuilt from the bundle, pinned by the contract and by a mint.

Two independent things have to line up here. The roots this builds must equal
the roots the contract holds — that alone fixes the dataset, the leaf order and
the tree shape all at once. And the proofs it generates for the one mint we
have the calldata for must come out byte for byte identical, which fixes the
neuron's own fields and the edge ordering too.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lattice as lattice_module  # noqa: E402
import merkle  # noqa: E402
from test_merkle import EDGE_PROOF, EDGES_ROOT, LEAF, NEURON_PROOF, NEURONS_ROOT, PARENT  # noqa: E402

PROTOCOL_FILE = ROOT / "scripts" / "protocols" / "flynode.json"


@unittest.skipUnless(lattice_module.DATA_FILE.exists(),
                     "no dataset: run scripts/extract_pathway.py")
class LatticeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lattice = lattice_module.Lattice.load()

    def test_roots_match_the_contract(self):
        payload = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
        self.assertEqual(self.lattice.neurons_root, NEURONS_ROOT)
        self.assertEqual(self.lattice.edges_root, EDGES_ROOT)
        self.lattice.check_roots(payload["neuronsRoot"], payload["edgesRoot"])

    def test_the_mints_neuron_is_the_one_we_rebuilt(self):
        neuron = self.lattice.neuron(LEAF["id"])
        self.assertEqual(neuron.leaf,
                         (LEAF["id"], LEAF["typeId"], LEAF["rarityBits"], LEAF["region"]))

    def test_the_mints_proofs_come_back_byte_for_byte(self):
        self.assertEqual(self.lattice.neuron_proof(LEAF["id"]), NEURON_PROOF)
        self.assertEqual(self.lattice.edge_proof(PARENT, LEAF["id"]), EDGE_PROOF)

    def test_the_mints_parent_is_linked_to_it(self):
        """The dataset stores the connection one way; either end can be a parent."""
        self.assertIn(PARENT, self.lattice.neighbours(LEAF["id"]))
        self.assertNotIn(LEAF["id"], self.lattice.neighbours(PARENT))
        self.assertIn(PARENT, self.lattice.linked(LEAF["id"]))
        self.assertIn(LEAF["id"], self.lattice.linked(PARENT))

    def test_links_are_the_edge_tree_seen_from_each_end(self):
        self.assertEqual(sum(len(row) for row in self.lattice.links),
                         len(self.lattice.edges))
        for neuron in self.lattice.neurons[:200]:
            for other in self.lattice.linked(neuron.id):
                self.assertIn((neuron.id, other), self.lattice._edge_index)

    def test_the_lattice_is_the_size_the_contract_mints(self):
        self.assertEqual(self.lattice.size, 20100)
        self.assertEqual(sum(int(entry["real"]) for entry in self.lattice.types), 20100)
        self.assertEqual(len(self.lattice.by_id), 20100)

    def test_rarity_is_log2_of_how_common_the_type_is(self):
        for index, entry in enumerate(self.lattice.types):
            self.assertEqual(self.lattice.rarity_of_type[index],
                             lattice_module.rarity_bits(int(entry["real"]), 20100))
        self.assertEqual(lattice_module.rarity_bits(892, 20100), 4)    # L1, the commonest
        self.assertEqual(lattice_module.rarity_bits(2, 20100), 13)     # a type with two
        self.assertEqual(lattice_module.rarity_bits(20100, 20100), 0)  # all of them

    def test_every_edge_is_in_the_tree_both_ways(self):
        pairs = set(self.lattice.edges)
        self.assertEqual(len(pairs), len(self.lattice.edges))  # deduplicated
        for parent, child in list(self.lattice.edges)[:2000]:
            self.assertIn((child, parent), pairs)

    def test_edges_are_in_the_order_the_dataset_walks_them(self):
        """First neuron, first neighbour, that way then back — before anything else."""
        first = self.lattice.neurons[0].id
        neighbour = self.lattice.neighbours(first)[0]
        self.assertEqual(self.lattice.edges[0], (first, neighbour))
        self.assertEqual(self.lattice.edges[1], (neighbour, first))

    def test_a_generated_proof_verifies_for_a_sample_of_the_lattice(self):
        for index in range(0, self.lattice.size, 1013):
            neuron = self.lattice.neurons[index]
            self.assertTrue(merkle.verify(self.lattice.neuron_proof(neuron.id),
                                          self.lattice.neurons_root,
                                          merkle.neuron_leaf(*neuron.leaf)), neuron.id)
            for other in self.lattice.neighbours(neuron.id)[:2]:
                self.assertTrue(merkle.verify(self.lattice.edge_proof(other, neuron.id),
                                              self.lattice.edges_root,
                                              merkle.edge_leaf(other, neuron.id)),
                                (other, neuron.id))

    def test_an_unknown_neuron_or_edge_is_refused(self):
        with self.assertRaises(KeyError):
            self.lattice.neuron(2**31)
        with self.assertRaises(KeyError):
            self.lattice.edge_proof(LEAF["id"], LEAF["id"])

    def test_a_dataset_that_builds_other_roots_is_refused(self):
        with self.assertRaises(SystemExit):
            self.lattice.check_roots("0x" + "11" * 32, "0x" + "22" * 32)


class VarintTests(unittest.TestCase):
    def test_varints_are_little_endian_base_128(self):
        self.assertEqual(lattice_module.read_varints(bytes([0x00])), [0])
        self.assertEqual(lattice_module.read_varints(bytes([0x7F])), [127])
        self.assertEqual(lattice_module.read_varints(bytes([0x80, 0x01])), [128])
        self.assertEqual(lattice_module.read_varints(bytes([0xE5, 0x8E, 0x26])), [624485])

    def test_a_truncated_stream_is_refused(self):
        with self.assertRaises(ValueError):
            lattice_module.read_varints(bytes([0x80]))

    def test_an_impossible_population_is_refused(self):
        with self.assertRaises(ValueError):
            lattice_module.rarity_bits(0, 20100)
        with self.assertRaises(ValueError):
            lattice_module.rarity_bits(20101, 20100)


if __name__ == "__main__":
    unittest.main()
