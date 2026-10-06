"""ReAct text-only baseline (Yao et al., 2022).

Runs a Thought/Action loop with no image input. This is Baseline 1 from the
proposal (Section 3.4). The key distinction from ORA is that no visual
re-encoding happens anywhere if the environment has an image, this agent
never sees it.
"""
from __future__ import annotations

from typing import Optional

from ..action_parser import parse as parse_action
from ..backends.base import GenerationConfig
from ..env import Environment
from ..schemas import Action, ActionType, Task, Trajectory, TrajectoryStep
from ..utils.logging import get_logger
from .base import Agent
from .prompts import format_react_prompt

_log = get_logger(__name__)


class ReActAgent(Agent):
    name = "react-text-only"

    def __init__(self, backend, gen_config: Optional[GenerationConfig] = None) -> None:
        super().__init__(backend)
        self.gen_config = gen_config or GenerationConfig()

    def run(self, task: Task, env: Environment) -> Trajectory:
        obs = env.reset(task)
        traj = Trajectory(task_id=task.task_id)

        for step_i in range(task.max_steps):
            prompt = format_react_prompt(task.instruction, obs.text or "", traj)
            raw = self.backend.generate(prompt, image=None, config=self.gen_config)
            parsed = parse_action(raw)

            if not parsed.ok:
                # Record the failure and break we can't recover from an
                # unparseable action in a text-only setting.
                _log.debug("ReAct parse failure on step %d: %s", step_i, parsed.error)
                traj.steps.append(
                    TrajectoryStep(
                        observation=obs,
                        action=Action(type=ActionType.NOOP, rationale=parsed.rationale),
                        valid=False,
                        error=parsed.error,
                    )
                )
                break

            action = parsed.action  # type: ignore[assignment]
            result = env.step(action)
            traj.steps.append(
                TrajectoryStep(
                    observation=obs, action=action, valid=result.valid, error=result.error
                )
            )

            if action.is_terminal():
                traj.terminated = True
                if action.type == ActionType.ANSWER:
                    traj.final_answer = action.value
                return traj
            if result.done:
                return traj

            obs = result.observation

        traj.truncated = True
        return traj
