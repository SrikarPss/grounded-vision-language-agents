"""Simple string -> Backend factory."""
from __future__ import annotations

from typing import Any

from .base import Backend
from .mock import MockBackend


def make_backend(spec: dict[str, Any]) -> Backend:
    """Build a backend from a config dict.

    Example::

        make_backend({"kind": "mock", "policy": "oracle"})
        make_backend({"kind": "llava", "device": "cuda", "quantize": "4bit"})
        make_backend({"kind": "mistral", "device": "cuda"})
    """
    spec = dict(spec)  # copy so we can pop
    kind = spec.pop("kind", "mock").lower()

    if kind == "mock":
        return MockBackend(**spec)
    if kind == "llava":
        from .llava import LLaVABackend

        return LLaVABackend(**spec)
    if kind == "mistral":
        from .mistral import MistralBackend

        return MistralBackend(**spec)
    raise ValueError(f"Unknown backend kind: {kind!r}")
