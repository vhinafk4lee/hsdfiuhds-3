#!/usr/bin/env python3
"""OpenZeppelin-compatible Merkle trees, as FlyNode's contract verifies them.

The contract hashes a leaf twice —

    keccak256(bytes.concat(keccak256(abi.encode(leaf))))

— which is what @openzeppelin/merkle-tree does, and folds a proof with sorted
pairs, so a proof carries no left/right information. Building a tree here has to
match that library exactly, or the root will not be the one the contract holds.
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


def build_tree(leaves: list[bytes], sort_leaves: bool = True) -> list[bytes]:
    """The flat tree @openzeppelin/merkle-tree builds: root at 0, leaves at the end."""
    if not leaves:
        raise ValueError("a tree needs at least one leaf")
    ordered = sorted(leaves, reverse=True) if sort_leaves else list(leaves)
    tree: list[bytes] = [b""] * (2 * len(ordered) - 1)
    for index, leaf in enumerate(ordered):
        tree[len(tree) - 1 - index] = leaf
    for index in range(len(tree) - 1 - len(ordered), -1, -1):
        tree[index] = hash_pair(tree[2 * index + 1], tree[2 * index + 2])
    return tree


def root_of(tree: list[bytes]) -> bytes:
    return tree[0]


def proof_for(tree: list[bytes], leaf: bytes) -> list[bytes]:
    """The siblings on the path from a leaf to the root."""
    try:
        index = tree.index(leaf)
    except ValueError:
        raise KeyError("leaf is not in this tree") from None
    proof = []
    while index > 0:
        sibling = index - 1 if index % 2 == 0 else index + 1
        proof.append(tree[sibling])
        index = (index - 1) // 2
    return proof
