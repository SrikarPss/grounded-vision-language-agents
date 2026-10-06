"""Grounded Vision-Language Agents for Instruction Following.

Implements the ORA (Observe -> Reason -> Act) loop used in this project, along with a ReAct text-only baseline and a LLaVA single-step
baseline, and an evaluation harness over Mind2Web, ScienceQA, and a
synthetic instruction-image corpus.
"""

from .schemas import (
    Action,
    ActionType,
    Observation,
    Task,
    Trajectory,
    TrajectoryStep,
    RunResult,
)

__version__ = "0.1.0"

__all__ = [
    "Action",
    "ActionType",
    "Observation",
    "Task",
    "Trajectory",
    "TrajectoryStep",
    "RunResult",
    "__version__",
]
