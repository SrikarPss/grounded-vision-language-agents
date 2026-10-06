"""LLaVA zero-shot single-step baseline (Baseline 2 in the proposal).

Calls the VLM exactly once with the initial observation and expects it to
emit a single Action. Used to isolate the benefit of the agentic loop (H2)
from the benefit of having vision at all (H1).
"""
from __future__ import annotations

from typing import Optional

from ..action_parser import parse as parse_action
from ..backends.base import GenerationConfig
from ..env import Environment
from ..schemas import Action, ActionType, Task, Trajectory, TrajectoryStep
from ..utils.image import load_image
from .base import Agent
from .prompts import format_vlm_single_shot_prompt


class SingleShotVLMAgent(Agent):
    name = "vlm-single-shot"

    def __init__(self, backend, gen_config: Optional[GenerationConfig] = None) -> None:
        super().__init__(backend)
        if not backend.supports_vision:
            raise ValueError("SingleShotVLMAgent needs a vision-capable backend")
        self.gen_config = gen_config or GenerationConfig()

    def run(self, task: Task, env: Environment) -> Trajectory:
        obs = env.reset(task)
        image = load_image(obs.image_path) if obs.image_path else None
        prompt = format_vlm_single_shot_prompt(task.instruction, obs)

        raw = self.backend.generate(prompt, image=image, config=self.gen_config)
        parsed = parse_action(raw)

        traj = Trajectory(task_id=task.task_id)
        if not parsed.ok:
            traj.steps.append(
                TrajectoryStep(
                    observation=obs,
                    action=Action(type=ActionType.NOOP, rationale=parsed.rationale),
                    valid=False,
                    error=parsed.error,
                )
            )
            return traj

        action = parsed.action  # type: ignore[assignment]
        result = env.step(action)
        traj.steps.append(
            TrajectoryStep(observation=obs, action=action, valid=result.valid, error=result.error)
        )
        traj.terminated = action.is_terminal()
        if action.type == ActionType.ANSWER:
            traj.final_answer = action.value
        return traj
