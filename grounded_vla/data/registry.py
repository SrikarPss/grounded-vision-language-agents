"""Config-driven dataset factory (mirrors backends.registry.make_backend)."""
from __future__ import annotations

from typing import Any

from .base import Dataset, JsonlDataset
from .mind2web import Mind2WebDataset
from .scienceqa import ScienceQADataset
from .synthetic import SyntheticDataset


def make_dataset(spec: dict[str, Any]) -> Dataset:
    spec = dict(spec)
    kind = spec.pop("kind").lower()
    if kind == "mind2web":
        return Mind2WebDataset(**spec)
    if kind == "scienceqa":
        return ScienceQADataset(**spec)
    if kind == "synthetic":
        return SyntheticDataset(**spec)
    if kind == "jsonl":
        return JsonlDataset(**spec)
    raise ValueError(f"Unknown dataset kind: {kind!r}")
