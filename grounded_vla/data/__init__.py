"""Dataset loaders.

Each loader returns an iterable of :class:`grounded_vla.schemas.Task`
objects. Loaders should be streaming-friendly so we can evaluate on a slice
without materializing the whole dataset in memory.
"""
from .base import Dataset, JsonlDataset
from .mind2web import Mind2WebDataset
from .scienceqa import ScienceQADataset
from .synthetic import SyntheticDataset
from .registry import make_dataset

__all__ = [
    "Dataset",
    "JsonlDataset",
    "Mind2WebDataset",
    "ScienceQADataset",
    "SyntheticDataset",
    "make_dataset",
]
