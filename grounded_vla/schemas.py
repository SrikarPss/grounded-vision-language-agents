"""Core data types used throughout the project.

These schemas define the contract between datasets, agents, backends, and
the evaluation harness. Keeping them small and strict pays off once multiple
people are hacking on different parts of the pipeline.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _dump_if_model(v: Any) -> Any:
    """Convert pydantic model instances to dicts so validation is class-identity-safe.

    Colab (and any editable-install environment) can end up with two copies of
    grounded_vla in sys.modules. Pydantic 2 uses strict class-identity checks,
    so an Observation built by copy-A fails in a TrajectoryStep from copy-B.
    Round-tripping through model_dump() lets pydantic reconstruct from the
    locally-correct class regardless of origin.
    """
    if v is None or isinstance(v, dict):
        return v
    if hasattr(v, "model_dump"):
        return v.model_dump()
    return v


class _CoercingBase(BaseModel):
    """Shared base that coerces all nested model fields before validation."""

    @model_validator(mode="before")
    @classmethod
    def _coerce_nested_models(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        for k, v in list(data.items()):
            if isinstance(v, list):
                data[k] = [_dump_if_model(item) for item in v]
            else:
                data[k] = _dump_if_model(v)
        return data


class ActionType(str, Enum):
    """The set of discrete action primitives an agent can emit.

    We intentionally keep the action space small and cross-domain. Mind2Web
    uses CLICK/TYPE/SELECT; ScienceQA degrades into a single ANSWER; the
    synthetic dataset uses the same primitives so a single parser suffices.
    """

    CLICK = "click"
    TYPE = "type"
    SELECT = "select"
    SCROLL = "scroll"
    ANSWER = "answer"
    STOP = "stop"
    NOOP = "noop"


# Synonym map: LLM-emitted strings that aren't canonical ActionType values.
# Applied by the Action.type validator before pydantic attempts enum coercion,
# so alias tolerance is automatic everywhere (parser, data loaders, tests).
_ACTION_TYPE_ALIASES: dict[str, ActionType] = {
    "hover": ActionType.CLICK,
    "tap": ActionType.CLICK,
    "press": ActionType.CLICK,
    "submit": ActionType.CLICK,
    "input": ActionType.TYPE,
    "fill": ActionType.TYPE,
    "choose": ActionType.SELECT,
    "done": ActionType.STOP,
    "finish": ActionType.STOP,
}


class Action(_CoercingBase):
    """A single action emitted by an agent."""

    model_config = ConfigDict(extra="forbid")

    type: ActionType

    @field_validator("type", mode="before")
    @classmethod
    def _coerce_action_type(cls, v: object) -> ActionType:
        # Already the right type — pass straight through.
        # (In Python 3.10, str(ActionType.CLICK) == "ActionType.CLICK", not
        # "click", so we must guard here before calling str().)
        if isinstance(v, ActionType):
            return v
        raw = str(v).strip().lower()
        # Return canonical enum value if it already matches.
        try:
            return ActionType(raw)
        except ValueError:
            pass
        # Try alias table; fall back to NOOP rather than crashing a rollout.
        return _ACTION_TYPE_ALIASES.get(raw, ActionType.NOOP)
    # Target element (selector, element id, or natural-language description).
    target: Optional[str] = None
    # Free-form payload (typed text for TYPE, answer content for ANSWER, etc).
    value: Optional[str] = None
    # Optional structured coordinates (x, y) for pixel-grounded UIs.
    xy: Optional[tuple[int, int]] = None
    # Model's natural-language rationale preceding the action.
    rationale: Optional[str] = None

    def is_terminal(self) -> bool:
        return self.type in (ActionType.ANSWER, ActionType.STOP)


class Observation(_CoercingBase):
    """A snapshot of the current environment state passed to the agent.

    `image_path` points at the current screenshot / diagram frame on disk.
    `text` optionally contains DOM snippets, OCR output, or a question stem.
    """

    model_config = ConfigDict(extra="forbid")

    step: int = 0
    image_path: Optional[Path] = None
    text: Optional[str] = None
    available_actions: list[str] = Field(default_factory=list)

    @field_validator("image_path")
    @classmethod
    def _coerce_path(cls, v: Optional[str | Path]) -> Optional[Path]:
        if v is None:
            return None
        return Path(v)


class Task(_CoercingBase):
    """A single evaluation task. One benchmark row = one Task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    instruction: str
    # The dataset this task came from (e.g., "mind2web", "scienceqa", "synthetic").
    source: str
    # Initial observation.
    initial_observation: Observation
    # Ground-truth action trajectory (when available) for exact-match grading.
    gold_actions: list[Action] = Field(default_factory=list)
    # Freeform ground-truth answer (used by ScienceQA).
    gold_answer: Optional[str] = None
    # Maximum allowed agent steps before we abort.
    max_steps: int = 15
    # Arbitrary metadata carried through (choices for MCQ, URL for Mind2Web, ...).
    meta: dict[str, Any] = Field(default_factory=dict)


class TrajectoryStep(_CoercingBase):
    model_config = ConfigDict(extra="forbid")

    observation: Observation
    action: Action
    # Whether the action is legal / parseable / actionable in the env.
    valid: bool = True
    # Any error message from the environment (parsing, execution, etc).
    error: Optional[str] = None


class Trajectory(_CoercingBase):
    """An agent's rollout for a single task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    steps: list[TrajectoryStep] = Field(default_factory=list)
    final_answer: Optional[str] = None
    # Did the agent emit a terminal action (ANSWER/STOP) on its own?
    terminated: bool = False
    # Was the rollout truncated at max_steps?
    truncated: bool = False

    @property
    def num_steps(self) -> int:
        return len(self.steps)


class RunResult(_CoercingBase):
    """Aggregate result for one (agent, dataset) configuration."""

    model_config = ConfigDict(extra="forbid")

    agent_name: str
    dataset: str
    n_tasks: int
    # Fraction in [0, 1] of tasks scored correct by the dataset's definition.
    task_completion_rate: float
    # Mean number of ORA/ReAct iterations taken on completed tasks.
    mean_steps: float
    # Per-category error counts (see grounded_vla.eval.error_analysis).
    error_breakdown: dict[str, int] = Field(default_factory=dict)
    trajectories: list[Trajectory] = Field(default_factory=list)
