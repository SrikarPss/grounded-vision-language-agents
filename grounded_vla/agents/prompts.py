"""Prompt templates.

These live in one place so that ablation runs (e.g., ORA vs ReAct with the
same surface prompt) are easy to set up and diff.
"""
from __future__ import annotations

from ..schemas import Observation, Trajectory

# Mind2Web rows embed the full cleaned_html which can be 20k–30k tokens.
# We truncate here so the combined prompt stays well within Mistral/LLaVA's
# 32768-token context window and generation doesn't stall.
_MAX_OBS_TEXT_CHARS = 4_000


def _trunc(text: str, limit: int = _MAX_OBS_TEXT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[... truncated {len(text) - limit} chars ...]"


_ACTION_FORMAT = (
    "Respond in exactly this format:\n"
    "Thought: <one short paragraph reasoning about what to do next>\n"
    "Action: <one JSON object like "
    '{"type": "click|type|select|scroll|answer|stop", '
    '"target": "<selector or element description>", '
    '"value": "<optional text payload>"}>\n'
    "Emit a single Action. When you are finished, emit "
    '{"type": "answer", "value": "<your answer>"}.'
)


def format_react_prompt(task_instruction: str, obs_text: str, history: Trajectory) -> str:
    """Text-only ReAct prompt. Observations are rendered as text only."""
    lines = [
        "You are an agent completing a multi-step instruction-following task.",
        f"Task: {task_instruction}",
        "",
        "You act by emitting Thought/Action pairs in a loop. The environment returns the next observation.",
        _ACTION_FORMAT,
        "",
    ]
    if history.steps:
        lines.append("History:")
        for i, step in enumerate(history.steps):
            lines.append(
                f"  Step {i}: Thought: {step.action.rationale or ''} | "
                f"Action: type={step.action.type.value}, target={step.action.target}, value={step.action.value}"
            )
        lines.append("")
    lines.append(f"Current observation (text):\n{_trunc(obs_text) if obs_text else '[no textual observation]'}")
    lines.append("")
    lines.append("What should you do next?")
    return "\n".join(lines)


def format_vlm_single_shot_prompt(task_instruction: str, obs: Observation) -> str:
    """Zero-shot VLM prompt, single step, no history."""
    lines = [
        "You are a vision-language agent. Look at the attached image and the instruction below.",
        f"Instruction: {task_instruction}",
    ]
    if obs.text:
        lines.append(f"Auxiliary text (optional): {_trunc(obs.text)}")
    lines.append("")
    lines.append(_ACTION_FORMAT)
    return "\n".join(lines)


def format_ora_prompt(task_instruction: str, obs: Observation, history: Trajectory) -> str:
    """ORA loop prompt. The image itself is passed separately to the VLM;
    here we include the instruction + action history (but NOT a text
    description of the image the whole point of H2 is that the visual
    state is re-encoded at every step)."""
    lines = [
        "You are a grounded vision-language agent operating in an Observe -> Reason -> Act loop.",
        "A fresh screenshot of the current environment state is attached for you to OBSERVE.",
        f"Task: {task_instruction}",
        "",
        _ACTION_FORMAT,
        "",
    ]
    if history.steps:
        lines.append("Action history so far (most recent last):")
        for i, step in enumerate(history.steps):
            lines.append(
                f"  [{i}] {step.action.type.value}"
                f" target={step.action.target or '-'}"
                f" value={step.action.value or '-'}"
                + (" [invalid]" if not step.valid else "")
            )
        lines.append("")
    if obs.text:
        lines.append(f"Auxiliary text (if present, secondary to the image): {_trunc(obs.text)}")
    lines.append("OBSERVE the image. REASON about the next action. ACT.")
    return "\n".join(lines)
