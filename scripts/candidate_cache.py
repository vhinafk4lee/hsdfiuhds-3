#!/usr/bin/env python3
"""Retain unsigned work that may become valid when the target eases."""
from __future__ import annotations

DEFAULT_ANCHOR_WINDOW = 250
CANDIDATE_FIELDS = ("wallet", "nonce", "hash", "prev", "anchor", "anchorBlock", "foundAt")


class CandidateCache:
    """Best candidate per (prevWork, anchor) pair, pruned to the freshest anchors."""

    def __init__(self, max_anchors: int = 4, submission_margin: int = 20):
        if max_anchors < 1 or submission_margin < 0:
            raise ValueError("invalid candidate cache bounds")
        self.max_anchors = max_anchors
        self.submission_margin = submission_margin
        self._candidates: dict[tuple[str, str], dict] = {}

    def __len__(self) -> int:
        return len(self._candidates)

    def remember(self, candidate: dict) -> bool:
        key = (candidate["prev"], candidate["anchor"])
        previous = self._candidates.get(key)
        if previous is not None and int(previous["hash"], 16) <= int(candidate["hash"], 16):
            return False
        self._candidates[key] = {field: candidate[field] for field in CANDIDATE_FIELDS}
        if len(self._candidates) > self.max_anchors:
            worst = max(self._candidates, key=lambda item: (
                int(self._candidates[item]["hash"], 16),
                -self._candidates[item]["anchorBlock"],
            ))
            del self._candidates[worst]
        return key in self._candidates

    def ready(self, job: dict) -> dict | None:
        window = int(job.get("anchorWindow", DEFAULT_ANCHOR_WINDOW))
        usable_window = max(0, window - self.submission_margin)
        self._candidates = {
            key: candidate for key, candidate in self._candidates.items()
            if candidate["prev"] == job["prev"]
            and job["blockNumber"] < candidate["anchorBlock"] + usable_window
        }
        eligible = [candidate for candidate in self._candidates.values()
                    if int(candidate["hash"], 16) < int(job["target"], 16)]
        if not eligible:
            return None
        best = min(eligible, key=lambda candidate: int(candidate["hash"], 16))
        return {**job, **best}
