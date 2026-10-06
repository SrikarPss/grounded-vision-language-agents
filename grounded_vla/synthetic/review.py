"""Two-person review workflow for the synthetic dataset.

Each candidate task needs sign-off from two reviewers before it can be
promoted into the final JSONL (Section 3.3 of the proposal). We persist
review state to a single JSON file so reviewers can hop in and out without
coordination.

File format (``review_state.json``)::

    {
      "reviewers": ["alice", "bob"],
      "votes": {
        "syn_abc123": {"alice": "approve", "bob": null},
        "syn_def456": {"alice": "reject",  "bob": "reject"}
      }
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_TIEBREAK = "needs_tiebreak"


Vote = Optional[str]  # "approve" | "reject" | None (pending)


@dataclass
class ReviewQueue:
    state_path: Path
    reviewers: tuple[str, str]

    def __post_init__(self) -> None:
        self.state_path = Path(self.state_path)
        self._state = self._load()

    # -- persistence ------------------------------------------------------

    def _load(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"reviewers": list(self.reviewers), "votes": {}}

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")

    # -- operations -------------------------------------------------------

    def enqueue(self, task_id: str) -> None:
        self._state["votes"].setdefault(task_id, {r: None for r in self.reviewers})
        self._save()

    def vote(self, task_id: str, reviewer: str, vote: str) -> None:
        if reviewer not in self.reviewers:
            raise ValueError(f"unknown reviewer: {reviewer}")
        if vote not in ("approve", "reject"):
            raise ValueError("vote must be 'approve' or 'reject'")
        if task_id not in self._state["votes"]:
            raise KeyError(f"task {task_id} not in queue")
        self._state["votes"][task_id][reviewer] = vote
        self._save()

    def status(self, task_id: str) -> ReviewStatus:
        votes = self._state["votes"].get(task_id, {})
        vs = list(votes.values())
        if any(v is None for v in vs):
            return ReviewStatus.PENDING
        if all(v == "approve" for v in vs):
            return ReviewStatus.APPROVED
        if all(v == "reject" for v in vs):
            return ReviewStatus.REJECTED
        return ReviewStatus.NEEDS_TIEBREAK

    def approved_ids(self) -> list[str]:
        return [tid for tid in self._state["votes"] if self.status(tid) == ReviewStatus.APPROVED]

    def summary(self) -> dict[str, int]:
        counts = {s.value: 0 for s in ReviewStatus}
        for tid in self._state["votes"]:
            counts[self.status(tid).value] += 1
        return counts
