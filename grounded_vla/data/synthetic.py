"""Synthetic instruction-image dataset loader.

The synthetic dataset is produced by ``grounded_vla.synthetic.builder``. It
ships as a JSONL file alongside a folder of Creative-Commons images.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from ..schemas import Task
from .base import Dataset, JsonlDataset


class SyntheticDataset(Dataset):
    name = "synthetic"

    def __init__(
        self,
        jsonl_path: Path | str,
        images_dir: Optional[Path | str] = None,
        limit: Optional[int] = None,
    ) -> None:
        self._inner = JsonlDataset(
            jsonl_path, source="synthetic", base_dir=images_dir, limit=limit
        )

    def __iter__(self) -> Iterator[Task]:
        yield from self._inner
