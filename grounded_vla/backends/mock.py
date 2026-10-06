"""A deterministic mock backend used for tests and CPU-only smoke runs.

The mock parses simple cues from the prompt (task instruction + ground truth
hints embedded in the observation) and returns a plausibly-formatted
``Thought: ... Action: {json}`` response. Crucially, when an image is
provided, the response varies with a fingerprint of the image so the ORA
loop's per-step re-encoding actually changes behavior a feature the tests
for H2 rely on.

This is NOT a model. It lets us exercise the pipeline end-to-end without
downloading 13GB of weights.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Optional

from PIL import Image

from ..utils.image import image_fingerprint
from .base import Backend, GenerationConfig


class MockBackend(Backend):
    name = "mock"

    def __init__(self, supports_vision: bool = True, policy: str = "oracle") -> None:
        """policy in {"oracle", "random", "greedy-click"}.

        - ``oracle``: reads a ``GT:`` cue embedded in the prompt (injected by
          the test harness / synthetic env) and returns the correct action.
          This makes upper-bound analysis easy.
        - ``greedy-click``: always emits click on the first mentioned element.
        - ``random``: deterministic pseudo-random actions based on prompt hash.
        """
        self.supports_vision = supports_vision
        self.policy = policy

    def generate(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        config = config or GenerationConfig()

        # Oracle cue: `[[GT_ACTION: {...}]]` can be injected by the synthetic
        # env or test fixtures to give the mock the correct answer.
        gt_match = re.search(r"\[\[GT_ACTION:\s*(\{.*?\})\s*\]\]", prompt, re.DOTALL)
        if self.policy == "oracle" and gt_match:
            action_json = gt_match.group(1)
            fp = image_fingerprint(image) if image is not None else "no-image"
            return (
                f"Thought: Observed state {fp}. The instruction maps to the "
                f"highlighted element.\nAction: {action_json}\n"
            )

        # Greedy: first token that looks like a selector/target.
        if self.policy == "greedy-click":
            target = _first_token(prompt) or "body"
            body = json.dumps({"type": "click", "target": target})
            return f"Thought: trying the most prominent element.\nAction: {body}\n"

        # Random policy: deterministic from prompt hash so tests are stable.
        h = hashlib.sha256(prompt.encode()).hexdigest()
        bucket = int(h[:8], 16) % 3
        if bucket == 0:
            body = json.dumps({"type": "click", "target": "#main"})
            thought = "guessing at the main content"
        elif bucket == 1:
            body = json.dumps({"type": "answer", "value": "unknown"})
            thought = "not enough information, answering 'unknown'"
        else:
            body = json.dumps({"type": "scroll", "value": "down"})
            thought = "scrolling to reveal more of the page"
        return f"Thought: {thought}.\nAction: {body}\n"


def _first_token(prompt: str) -> Optional[str]:
    m = re.search(r"#\w+|\.[\w-]+|<button[^>]*>", prompt)
    return m.group(0) if m else None
