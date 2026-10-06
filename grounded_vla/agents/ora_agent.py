"""ORA (Observe -> Reason -> Act) agent.

This is the project's novel contribution (Section 3.2 of the proposal). The
critical difference from vanilla ReAct is that **at every step** we re-load
the current visual state from disk and feed it through the VLM's vision
encoder before asking for a new action. Nothing in the agent "summarizes"
the image into text that would defeat the point of H2.

Implementation notes:

- We keep the action history as structured objects, not as a rolling text
  transcript, so the prompt stays tight.
- The backend is required to be vision-capable. We raise at construction
  time if it isn't, so misconfigurations fail fast.
- If the action parser fails, we don't give up we surface a short
  remediation hint to the model and retry once before terminating. This
  matches what real agent frameworks do and lets us log
  ``action_parsing_failure`` separately from true reasoning errors.
"""
from __future__ import annotations

from typing import Optional

from ..action_parser import parse as parse_action
from ..backends.base import GenerationConfig
from ..env import Environment
from ..schemas import Action, ActionType, Task, Trajectory, TrajectoryStep
from ..utils.image import load_image
from ..utils.logging import get_logger
from .base import Agent
from .prompts import format_ora_prompt

_log = get_logger(__name__)


class ORAAgent(Agent):
    name = "ora-vlm"

    def __init__(
        self,
        backend,
        gen_config: Optional[GenerationConfig] = None,
        parser_retries: int = 1,
    ) -> None:
        super().__init__(backend)
        if not backend.supports_vision:
            raise ValueError("ORAAgent requires a vision-capable backend (got text-only)")
        self.gen_config = gen_config or GenerationConfig()
        self.parser_retries = parser_retries

    # -- Core loop --------------------------------------------------------

    def run(self, task: Task, env: Environment) -> Trajectory:
        obs = env.reset(task)
        traj = Trajectory(task_id=task.task_id)

        for step_i in range(task.max_steps):
            # OBSERVE: re-encode the current visual state every single step.
            image = load_image(obs.image_path) if obs.image_path else None
            prompt = format_ora_prompt(task.instruction, obs, traj)

            # REASON: ask the VLM for a Thought + Action.
            action = self._generate_action(prompt, image)
            if action is None:
                _log.debug("ORA step %d: action parse failed after retries", step_i)
                traj.steps.append(
                    TrajectoryStep(
                        observation=obs,
                        action=Action(type=ActionType.NOOP),
                        valid=False,
                        error="action parsing failed after retry",
                    )
                )
                break

            # ACT: execute and loop back to Observe with a potentially
            # refreshed observation (different image_path, different DOM).
            result = env.step(action)
            traj.steps.append(
                TrajectoryStep(
                    observation=obs,
                    action=action,
                    valid=result.valid,
                    error=result.error,
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

    # -- Helpers ----------------------------------------------------------

    def _generate_action(self, prompt: str, image) -> Optional[Action]:
        tries = 1 + max(0, self.parser_retries)
        last_err = None
        current_prompt = prompt
        for attempt in range(tries):
            raw = self.backend.generate(current_prompt, image=image, config=self.gen_config)
            parsed = parse_action(raw)
            if parsed.ok:
                return parsed.action
            last_err = parsed.error
            # Append a short remediation hint and retry once. We deliberately
            # don't echo the whole raw response that tends to make the
            # model double down on its mistake.
            current_prompt = (
                prompt
                + "\n\n(Your previous response could not be parsed. "
                "Emit exactly one Action: line with a valid JSON object.)"
            )
        _log.debug("ORA parser failed after %d attempts: %s", tries, last_err)
        return None
