"""Model backends.

Backends are thin wrappers around a model that implement a small uniform
interface (`generate(prompt, image=None) -> str`). Keeping this surface
minimal lets us slot in LLaVA, Mistral, a mock stub, or a future
OpenAI/Anthropic API behind the same agents without touching agent logic.
"""
from .base import Backend, BackendError, GenerationConfig
from .mock import MockBackend
from .registry import make_backend

__all__ = [
    "Backend",
    "BackendError",
    "GenerationConfig",
    "MockBackend",
    "make_backend",
]
