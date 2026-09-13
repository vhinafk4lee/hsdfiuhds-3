#!/usr/bin/env python3
"""Merkle handling, pinned by a real FlyNode mint.

The proofs in that transaction were accepted by the contract, so folding them
must produce the roots the contract holds. That is what makes the leaf hashing
and the pair ordering here provably the contract's, rather than merely plausible.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import merkle  # noqa: E402

# From the mine() call of transaction 0xab1e1995...f919 on Robinhood Chain.
LEAF = {"id": 35160, "typeId": 0, "rarityBits": 4, "region": 0}
PARENT = 90041
NEURON_PROOF = [bytes.fromhex(value) for value in (
    "8df0fcaca5afa90f4cff92ac5d07fd83cd9f3cc7d08652aa9a69196b16ed4efa",
    "d0fec78b47df7c196e83b1f0663f8a8fc021d1fa0532488b719e55bd1ea8553a",
    "145b97deea7c74798b7df059efa353dd3c2f5fbcdecf3020e8701b6926dff12f",
    "cceb78fc9fd9b6a5f34a922331f4d09209fe5d5335848cba39e88c31e1526b78",
    "2f10b74d8ace7559b424a401867dc1032c23e89778741d1df8a0b00d9b3a5864",
    "04993c30bd352ff27308fa932e1146a72d2bb2368a7f417b640e7e57637a68b1",
    "35b89b59b6d4ed4953c50dd3519a4c13baf2bc93e1a3ae56b2d413f5f5dd6f28",
    "4d3485f52e9a109d3b112cbe315c003c40b71e15f61dc9ad910abfbfcb625840",
    "5a3933b869d103b222e6c0d38657361966b22443c74772dce79c6c7e906e6839",
    "de3d3d112110aa63b47caccb8be78bbfa17a7a33edb7a252b31804c7a4ca0860",
    "0f2690b9a32c49c3eb4e7a564c0c173d01c4ec2894402c868943acb17c3172cd",
    "bc9f2b694726adc399dced70217bb47936f0a4e65a0809f5301a794019e5ab60",
    "2e155b53070242725e11d220aa8154e8a6a059eae60e63d7e57d0689d618f28a",
    "06584aa4af294cd03c31eb7fb20864040a79a42014623705028e594361d5a517",
    "92c09aea5d9617bcfa29e6af2471069b379dd66bc85778a623f5d03996736cf7",
)]
EDGE_PROOF = [bytes.fromhex(value) for value in (
    "62c0afbdcfda5fbd184686f0521104b7e8010663900fd289fb3d98bf909aa352",
    "0057046969fa0c643bd707690bfbdd3e792e07d3e8861c8f3470558a59b83c63",
    "e3613786bac8fd9d55acff90d664f5f5b814e4cd5699201a31581844fe716dd2",
    "e7299ad1d0b9ce6d3e50950c4b1ac2d57f93f452fc7dee70dc085405fa502982",
    "74a30fd0db2a62b74229a979f9b7d79119f75875d61c2a928bfbcc092cabbc1e",
    "b385c57d873b5414bfd74a6509d3b5b05f89b08ed41b5dbb756ae04913c3c157",
    "494b2022d80322880fcb21c5a6c2022a6b846ae1807a5299d01d53f54b479abd",
    "2be3019c14b89a045cd8c18d66848fd268967362524160f34205d8e4ceccdb8e",
    "bc0511a9f9a00d657875d3578b8c2160aa9339bb1e99bda7aed5a498fb582ec6",
    "2cd76a3ba8743f4ed560c912be338dde28ae92ffdf70b6bf53b3c93a499dd7e5",
    "7b8668afe7940c80107abbb88da550ff707de1884cd59190ea97bf6572509124",
    "12c131c25719eb4a64b1363df7b1ab66f6291b50fe72cd22d61554c20901fc44",
    "f30617aef657f959752a2a7b4b61ae17cf355da29265622553b9e7ebcdaab677",
    "996c47166356083b5eb2e08b47b89cfd3fe63fcc34e6be46dd792b1837f82834",
    "75996f065a576abac0db626a86703a7d1d41361b7675ab44ef42b448a76252c6",
    "34495bb3cecaf8b22532d7e2a36cc2f7033380423640df2e51b31b4bc01ae8dc",
    "1f4ddcbd0ec8fdab434201a22b4c8ef9fa515915a6dd8e83a2b80aa62ee96367",
    "1647fc985a3bc8ac91d18e40596bb7f393b303a08f76f9ced3642b24515438da",
)]
NEURONS_ROOT = bytes.fromhex("89c949af2a2e7752f5af8ecc1d4a98d3d72a5d5582332bfb595caf7d8c2a3d2b")
EDGES_ROOT = bytes.fromhex("dd0b52d2dbb96aa8f425e66c9bd83a5af6463f950abcc7541524761774c256cd")


class MainnetProofTests(unittest.TestCase):
    def test_neuron_proof_folds_to_the_root(self):
        leaf = merkle.neuron_leaf(LEAF["id"], LEAF["typeId"], LEAF["rarityBits"], LEAF["region"])
        self.assertEqual(merkle.process_proof(NEURON_PROOF, leaf), NEURONS_ROOT)
        self.assertTrue(merkle.verify(NEURON_PROOF, NEURONS_ROOT, leaf))

    def test_edge_proof_folds_to_the_root(self):
        leaf = merkle.edge_leaf(PARENT, LEAF["id"])
        self.assertEqual(merkle.process_proof(EDGE_PROOF, leaf), EDGES_ROOT)

    def test_a_changed_leaf_breaks_the_proof(self):
        leaf = merkle.neuron_leaf(LEAF["id"], LEAF["typeId"], LEAF["rarityBits"] + 1,
                                  LEAF["region"])
        self.assertFalse(merkle.verify(NEURON_PROOF, NEURONS_ROOT, leaf))

    def test_proof_depth_bounds_the_dataset(self):
        self.assertEqual(len(NEURON_PROOF), 15)   # up to 32768 neurons
        self.assertEqual(len(EDGE_PROOF), 18)     # up to 262144 edges


class TreeTests(unittest.TestCase):
    def leaves(self, count: int) -> list[bytes]:
        return [merkle.neuron_leaf(index, index % 7, index % 5, index % 3)
                for index in range(count)]

    def test_generated_proofs_verify(self):
        for count in (1, 2, 3, 8, 17, 64):
            leaves = self.leaves(count)
            tree = merkle.build_tree(leaves)
            root = merkle.root_of(tree)
            for leaf in leaves:
                proof = merkle.proof_for(tree, leaf)
                self.assertTrue(merkle.verify(proof, root, leaf), f"count {count}")

    def test_pair_hashing_is_commutative(self):
        left, right = merkle.keccak256(b"a"), merkle.keccak256(b"b")
        self.assertEqual(merkle.hash_pair(left, right), merkle.hash_pair(right, left))

    def test_a_leaf_outside_the_tree_is_refused(self):
        tree = merkle.build_tree(self.leaves(4))
        with self.assertRaises(KeyError):
            merkle.proof_for(tree, merkle.neuron_leaf(999, 0, 0, 0))

    def test_proof_length_matches_the_tree_depth(self):
        leaves = self.leaves(64)
        tree = merkle.build_tree(leaves)
        self.assertEqual(len(merkle.proof_for(tree, leaves[0])), 6)


if __name__ == "__main__":
    unittest.main()
