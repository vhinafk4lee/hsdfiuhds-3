#!/usr/bin/env python3
"""Keeps the best proof found for the live challenge.

A proof is bound to (miner, nonce, challenge). It dies the moment anyone mints,
because the contract moves to a new challenge, so there is nothing to retain
across challenges. Within one challenge the best proof is worth keeping: the
difficulty can ease while the challenge stands, and a proof that just missed
then becomes a winning one without re-mining.
"""
from __future__ import annotations

CANDIDATE_FIELDS = ("wallet", "nonce", "hash", "challenge", "difficulty", "foundAt")


class CandidateCache:
    def __init__(self) -> None:
        self._best: dict | None = None

    def __len__(self) -> int:
        return 1 if self._best else 0

    @property
    def best(self) -> dict | None:
        return self._best

    def remember(self, candidate: dict) -> bool:
        """Store the candidate if it is the best seen for its challenge."""
        current = self._best
        if current is not None and current["challenge"] == candidate["challenge"] \
                and int(current["hash"], 16) <= int(candidate["hash"], 16):
            return False
        self._best = {field: candidate[field] for field in CANDIDATE_FIELDS}
        return True

    def ready(self, job: dict) -> dict | None:
        """Return a submittable solution, dropping work the chain has moved past."""
        candidate = self._best
        if candidate is None:
            return None
        if candidate["challenge"] != job["challenge"]:
            self._best = None
            return None
        if int(candidate["hash"], 16) >= int(job["target"], 16):
            return None
        return {**job, **candidate}
