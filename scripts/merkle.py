#!/usr/bin/env python3
"""Merkle trees as FlyNode's contract builds and verifies them.

Two halves, and they come from different places.

Verification is OpenZeppelin's. The contract hashes a leaf twice —

    keccak256(bytes.concat(keccak256(abi.encode(leaf))))

— so a leaf can never collide with an inner node, and it folds a proof with
sorted pairs, so a proof carries no left/right information. The accepted mint
pinned in tests/test_merkle.py proves that much.

Construction is *not* OpenZeppelin's. @openzeppelin/merkle-tree sorts the leaves
before it builds, and a tree built that way has a different root than the one
the contract holds. FlyNode's tree is built in dataset order instead: pair the
level up left to right, and carry a node with no partner up to the next level
unchanged. Sorted-pair folding is what lets the two coexist — the verifier never
learns which side a sibling was on, so only the shape has to agree.
"""
from __future__ import annotations

from Crypto.Hash import keccak


def keccak256(data: bytes) -> bytes:
    return keccak.new(digest_bits=256, data=data).digest()


def encode_words(*values: int) -> bytes:
    """abi.encode of a tuple of fixed-size integers: one 32-byte word each."""
    return b"".join(int(value).to_bytes(32, "big") for value in values)


def leaf_hash(*values: int) -> bytes:
    """The double hash OpenZeppelin uses, so a leaf can never collide with a node."""
    return keccak256(keccak256(encode_words(*values)))


def neuron_leaf(leaf_id: int, type_id: int, rarity_bits: int, region: int) -> bytes:
    """keccak256(bytes.concat(keccak256(abi.encode(Leaf))))"""
    return leaf_hash(leaf_id, type_id, rarity_bits, region)


def edge_leaf(parent: int, child: int) -> bytes:
    """keccak256(bytes.concat(keccak256(abi.encode(parent, leaf.id))))"""
    return leaf_hash(parent, child)


def hash_pair(left: bytes, right: bytes) -> bytes:
    """Commutative: the contract sorts the pair, so proofs carry no direction."""
    return keccak256(left + right if left < right else right + left)


def process_proof(proof: list[bytes], leaf: bytes) -> bytes:
    computed = leaf
    for sibling in proof:
        computed = hash_pair(computed, sibling)
    return computed


def verify(proof: list[bytes], root: bytes, leaf: bytes) -> bool:
    return process_proof(proof, leaf) == root


class MerkleTree:
    """A tree over leaves in dataset order, held level by level.

    ``levels[0]`` is the leaves as given, ``levels[-1]`` is the single root.
    Each level pairs its predecessor left to right; an odd node at the end has
    no partner and moves up unchanged, so it is hashed again only once a
    partner appears for it higher in the tree.
    """

    def __init__(self, leaves: list[bytes]):
        if not leaves:
            raise ValueError("a tree needs at least one leaf")
        levels = [list(leaves)]
        while len(levels[-1]) > 1:
            below = levels[-1]
            levels.append([
                hash_pair(below[index], below[index + 1]) if index + 1 < len(below)
                else below[index]
                for index in range(0, len(below), 2)
            ])
        self.levels = levels
        # Built once: a proof for every leaf otherwise rescans the whole level.
        self._index = {leaf: index for index, leaf in enumerate(leaves)}

    def __len__(self) -> int:
        return len(self.levels[0])

    @property
    def leaves(self) -> list[bytes]:
        return self.levels[0]

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    @property
    def depth(self) -> int:
        return len(self.levels) - 1

    def index_of(self, leaf: bytes) -> int:
        try:
            return self._index[leaf]
        except KeyError:
            raise KeyError("leaf is not in this tree") from None

    def proof_at(self, index: int) -> list[bytes]:
        """The siblings on the path from a leaf index to the root.

        A promoted node contributes nothing, so a proof can be shorter than the
        depth. The verifier folds whatever it is given and cannot tell the
        difference.
        """
        if not 0 <= index < len(self.levels[0]):
            raise IndexError(f"leaf index {index} is outside a tree of {len(self)}")
        proof = []
        for level in self.levels[:-1]:
            sibling = index ^ 1
            if sibling < len(level):
                proof.append(level[sibling])
            index //= 2
        return proof

    def proof_for(self, leaf: bytes) -> list[bytes]:
        return self.proof_at(self.index_of(leaf))


def build_tree(leaves: list[bytes]) -> MerkleTree:
    """FlyNode's tree over the leaves in the order given — never sorted."""
    return MerkleTree(leaves)


def root_of(tree: MerkleTree) -> bytes:
    return tree.root


def proof_for(tree: MerkleTree, leaf: bytes) -> list[bytes]:
    return tree.proof_for(leaf)
