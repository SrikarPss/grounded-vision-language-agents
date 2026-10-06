"""Backend interface.

Agents talk to models exclusively through a `Backend`. The interface is
deliberately tiny: one `generate` call that accepts text + optional image
and returns a raw string. Anything richer (streaming, structured decoding,
batching) can be added later without breaking callers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from PIL import Image


class BackendError(RuntimeError):
    """Raised when a backend fails to produce a response."""


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    # When True, the backend is asked to format its response as
    #   Thought: ...
    #   Action: {json}
    # so the action_parser can consume it reliably.
    structured_output: bool = True
    # Random seed used by text-only deterministic backends (mock).
    seed: int = 0


class Backend(ABC):
    """Abstract model backend."""

    #: Human-readable name used in logs and results.
    name: str = "backend"
    #: Whether this backend can consume images. Text-only backends ignore them.
    supports_vision: bool = False

    @abstractmethod
    def generate(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        config: Optional[GenerationConfig] = None,
    ) -> str: ...

    def warmup(self) -> None:
        """Optional hook: preload weights, compile graphs, etc."""

    def close(self) -> None:
        """Optional hook: free GPU memory."""
