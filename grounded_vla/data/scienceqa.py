"""ScienceQA loader.

ScienceQA (Lu et al., 2022) is a multiple-choice science dataset with
diagram images. We treat each row as a single-observation QA task whose
only terminal action is ANSWER.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from ..schemas import Action, ActionType, Observation, Task
from .base import Dataset, JsonlDataset


class ScienceQADataset(Dataset):
    name = "scienceqa"

    def __init__(
        self,
        hf_split: Optional[str] = None,
        jsonl_path: Optional[Path | str] = None,
        images_dir: Optional[Path | str] = None,
        only_with_image: bool = True,
        limit: Optional[int] = None,
    ) -> None:
        if not hf_split and not jsonl_path:
            raise ValueError("pass either hf_split=... or jsonl_path=...")
        self.hf_split = hf_split
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self.images_dir = Path(images_dir) if images_dir else None
        self.only_with_image = only_with_image
        self.limit = limit

    def __iter__(self) -> Iterator[Task]:
        if self.jsonl_path:
            yield from JsonlDataset(
                self.jsonl_path, source="scienceqa", base_dir=self.images_dir, limit=self.limit
            )
            return
        yield from self._from_hf()

    def _from_hf(self) -> Iterator[Task]:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise RuntimeError(
                "Install `datasets` to stream ScienceQA directly, or pre-convert to JSONL."
            ) from e

        ds = load_dataset("derek-thomas/ScienceQA", split=self.hf_split, streaming=True)
        count = 0
        for ex in ds:
            if self.only_with_image and not ex.get("image"):
                continue
            count += 1
            if self.limit is not None and count > self.limit:
                return
            yield _example_to_task(ex, self.images_dir)


def _example_to_task(ex: dict, images_dir: Optional[Path]) -> Task:
    choices = ex.get("choices", [])
    answer_idx = ex.get("answer")
    gold_answer = choices[answer_idx] if isinstance(answer_idx, int) and choices else str(answer_idx)
    stem = ex.get("question", "")
    options_block = "\n".join(f"  ({chr(65 + i)}) {c}" for i, c in enumerate(choices))
    text = f"{stem}\nOptions:\n{options_block}" if choices else stem

    # The HF dataset holds a PIL image in memory; callers who want on-disk
    # paths should dump it during a preprocessing step. Here we just keep
    # the text stem and rely on the image being saved out of band.
    img_path = None
    task_id = str(ex.get("id") or ex.get("question_id") or hash(stem))
    if images_dir is not None:
        candidate = images_dir / f"{task_id}.png"
        img_path = candidate if candidate.exists() else None

    obs = Observation(step=0, image_path=img_path, text=text)
    return Task(
        task_id=task_id,
        instruction="Answer the science question using the diagram and text.",
        source="scienceqa",
        initial_observation=obs,
        gold_actions=[Action(type=ActionType.ANSWER, value=gold_answer)],
        gold_answer=gold_answer,
        max_steps=3,  # ScienceQA shouldn't need more than a few ORA iterations
        meta={
            "choices": choices,
            "subject": ex.get("subject"),
            "topic": ex.get("topic"),
            "grade": ex.get("grade"),
        },
    )
