"""Per-benchmark scoring functions.

Each function takes a Task + Trajectory and returns a bool (success) plus
some side info. Keeping these dataset-specific functions separate prevents
one loud change (e.g., Mind2Web adding sub-step grading) from bleeding
into the other benchmarks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from ..env import _fuzzy_match
from ..schemas import ActionType, Task, Trajectory


@dataclass
class Score:
    success: bool
    # Fraction of gold steps completed (for partial credit reporting).
    progress: float = 0.0
    # Optional normalized prediction text (used by QA).
    prediction: str = ""


def mind2web_task_success(task: Task, traj: Trajectory) -> Score:
    """Mind2Web success = every gold action matched, in order.

    Agents are allowed intermediate invalid steps; they just need to have
    produced all the gold actions by the end of the rollout.
    """
    gold = task.gold_actions
    if not gold:
        return Score(success=False, progress=0.0)
    gold_idx = 0
    for step in traj.steps:
        if gold_idx >= len(gold):
            break
        g = gold[gold_idx]
        if (
            step.action.type == g.type
            and _fuzzy_match(step.action.target, g.target)
            and _fuzzy_match(step.action.value, g.value)
        ):
            gold_idx += 1
    progress = gold_idx / len(gold)
    return Score(success=(gold_idx == len(gold)), progress=progress)


def scienceqa_task_success(task: Task, traj: Trajectory) -> Score:
    """ScienceQA success = final answer matches gold (normalized)."""
    pred = (traj.final_answer or "").strip()
    gold = (task.gold_answer or "").strip()
    success = _normalize_answer(pred) == _normalize_answer(gold)
    # Also accept matches against the letter label (A/B/C/D) if choices exist.
    choices = task.meta.get("choices") or []
    if not success and pred and choices:
        letter = pred.strip().strip(".").upper()[:1]
        if letter in "ABCDEFG":
            idx = ord(letter) - ord("A")
            if 0 <= idx < len(choices) and _normalize_answer(choices[idx]) == _normalize_answer(gold):
                success = True
    return Score(success=success, progress=1.0 if success else 0.0, prediction=pred)


def synthetic_task_success(task: Task, traj: Trajectory) -> Score:
    """Synthetic success = first action matches the single gold action.

    These tasks are single-step by construction; we don't give partial
    credit but we do fuzzy-match targets for robustness.
    """
    if not task.gold_actions or not traj.steps:
        return Score(success=False, progress=0.0)
    g = task.gold_actions[0]
    a = traj.steps[0].action
    ok = (
        a.type == g.type
        and _fuzzy_match(a.target, g.target)
        and _fuzzy_match(a.value, g.value)
    )
    return Score(success=ok, progress=1.0 if ok else 0.0)


SCORERS: dict[str, Callable[[Task, Trajectory], Score]] = {
    "mind2web": mind2web_task_success,
    "scienceqa": scienceqa_task_success,
    "synthetic": synthetic_task_success,
}


def score_trajectory(task: Task, traj: Trajectory) -> Score:
    scorer = SCORERS.get(task.source)
    if scorer is None:
        raise KeyError(f"no scorer for source {task.source!r}")
    return scorer(task, traj)


def step_efficiency(traj: Trajectory) -> int:
    """Number of agent steps until termination (or len(steps) if truncated)."""
    return traj.num_steps


_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    # drop leading English articles so "a plant cell" == "the plant cell"
    for article in ("the ", "a ", "an "):
        if s.startswith(article):
            s = s[len(article):]
            break
    return s
