"""Agent base class.

An Agent owns the prompting strategy and the control loop. It does NOT own
model weights the Backend does. Keeping this split makes it trivial to run
the same ORA loop with LLaVA today and with (say) InternVL tomorrow.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..backends.base import Backend
from ..env import Environment
from ..schemas import Task, Trajectory


class Agent(ABC):
    name: str = "agent"

    def __init__(self, backend: Backend) -> None:
        self.backend = backend

    @abstractmethod
    def run(self, task: Task, env: Environment) -> Trajectory:
        """Rollout the agent on a single task. Must return a Trajectory."""
