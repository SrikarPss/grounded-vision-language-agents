"""Categorize failures into the three buckets the proposal promises:

- ``visual_misgrounding``: model referenced the wrong element in the image
- ``reasoning_error``: right perception, wrong plan
- ``action_parsing_failure``: couldn't even emit a well-formed action

Heuristics here are approximate and that's okay the purpose is to give
the team a starting point for the qualitative error table in the final
report. Humans spot-check the categorization during analysis.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from ..schemas import ActionType, Task, Trajectory


class ErrorCategory(str, Enum):
    NONE = "none"
    VISUAL_MISGROUNDING = "visual_misgrounding"
    REASONING_ERROR = "reasoning_error"
    ACTION_PARSING_FAILURE = "action_parsing_failure"
    TRUNCATED = "truncated"


def categorize_error(task: Task, traj: Trajectory, success: bool) -> ErrorCategory:
    if success:
        return ErrorCategory.NONE

    # Parsing failures leave a trail: NOOP action + error "parse" substring.
    for step in traj.steps:
        if step.action.type == ActionType.NOOP and step.error and "pars" in step.error.lower():
            return ErrorCategory.ACTION_PARSING_FAILURE

    # Truncated without termination = ran out of budget; usually means the
    # model was flailing. We tag this separately so it doesn't look like
    # a perception/reasoning failure.
    if traj.truncated and not traj.terminated:
        return ErrorCategory.TRUNCATED

    # Heuristic: if the first wrong action has the right TYPE but the wrong
    # target, that's usually a visual misgrounding call it (e.g., correct
    # verb, wrong button). If the TYPE itself is wrong, that's a plan-level
    # reasoning error.
    gold = task.gold_actions[0] if task.gold_actions else None
    if gold is not None and traj.steps:
        a = traj.steps[0].action
        if a.type == gold.type:
            return ErrorCategory.VISUAL_MISGROUNDING
        return ErrorCategory.REASONING_ERROR

    # Fallback.
    return ErrorCategory.REASONING_ERROR
