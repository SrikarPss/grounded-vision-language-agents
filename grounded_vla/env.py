"""A minimal Environment abstraction.

An Environment converts a Task into an initial observation and accepts
actions, returning the next observation plus whether the episode ended.

Concrete environments:

- ``TaskReplayEnv``: for Mind2Web/synthetic trajectories where we grade against
  a gold action sequence. It "executes" an action by advancing the gold-action
  cursor if the agent's action matches (by type + fuzzy target match).
- ``StaticQAEnv``: for ScienceQA and any QA-style task, where the only valid
  terminal action is ANSWER and we don't care about intermediate steps.

This deliberately doesn't try to be a real browser or simulator; Mind2Web
provides canonical trajectories that we replay, and full browser execution
is out of scope for this project.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

from .schemas import Action, ActionType, Observation, Task


@dataclass
class StepResult:
    observation: Observation
    done: bool
    valid: bool = True
    error: Optional[str] = None


def _fuzzy_match(a: Optional[str], b: Optional[str], threshold: float = 0.55) -> bool:
    """Element targets vary a lot (CSS selector vs NL description). We accept
    anything above a low-ish similarity bar; the metric layer rewards exact
    matches separately.
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    a_n, b_n = a.strip().lower(), b.strip().lower()
    if a_n == b_n or a_n in b_n or b_n in a_n:
        return True
    return SequenceMatcher(None, a_n, b_n).ratio() >= threshold


class Environment(ABC):
    @abstractmethod
    def reset(self, task: Task) -> Observation: ...

    @abstractmethod
    def step(self, action: Action) -> StepResult: ...


class StaticQAEnv(Environment):
    """Single-observation environment for QA-style tasks."""

    def __init__(self) -> None:
        self._task: Optional[Task] = None
        self._done = False

    def reset(self, task: Task) -> Observation:
        self._task = task
        self._done = False
        return task.initial_observation

    def step(self, action: Action) -> StepResult:
        assert self._task is not None, "call reset() first"
        if action.is_terminal():
            self._done = True
            return StepResult(observation=self._task.initial_observation, done=True)
        # Non-terminal actions are allowed (e.g., the agent "scrolls" the
        # diagram in its head) but they don't change the observation.
        return StepResult(observation=self._task.initial_observation, done=False)


class TaskReplayEnv(Environment):
    """Replays a gold-action trajectory and reports progress.

    For each step we compare the agent's action to the next gold action. If
    it matches (by ActionType and fuzzy target), the cursor advances and we
    hand back the next observation (if the dataset provides per-step frames)
    or the same one. Otherwise the step is marked invalid but we don't
    terminate immediately we let the agent try to recover up to max_steps.
    """

    def __init__(self) -> None:
        self._task: Optional[Task] = None
        self._cursor = 0
        self._frames: list[Observation] = []

    def reset(self, task: Task) -> Observation:
        self._task = task
        self._cursor = 0
        # If the dataset attached per-step frames under `meta["frames"]`, use
        # them; otherwise reuse the initial observation.
        frames = task.meta.get("frames")
        if frames:
            self._frames = [
                Observation(step=i, image_path=f.get("image_path"), text=f.get("text"))
                for i, f in enumerate(frames)
            ]
        else:
            self._frames = [task.initial_observation]
        return self._frames[0]

    def step(self, action: Action) -> StepResult:
        assert self._task is not None, "call reset() first"
        if action.is_terminal():
            return StepResult(observation=self._current_obs(), done=True)

        gold_idx = self._cursor
        gold = self._task.gold_actions[gold_idx] if gold_idx < len(self._task.gold_actions) else None
        matched = (
            gold is not None
            and action.type == gold.type
            and _fuzzy_match(action.target, gold.target)
            and _fuzzy_match(action.value, gold.value)
        )
        if matched:
            self._cursor += 1
            done = self._cursor >= len(self._task.gold_actions)
            return StepResult(observation=self._current_obs(), done=done, valid=True)
        return StepResult(
            observation=self._current_obs(),
            done=False,
            valid=False,
            error=f"action did not match gold step {gold_idx}",
        )

    def _current_obs(self) -> Observation:
        idx = min(self._cursor, len(self._frames) - 1)
        return self._frames[idx]

    @property
    def progress(self) -> float:
        if self._task is None or not self._task.gold_actions:
            return 0.0
        return self._cursor / len(self._task.gold_actions)
