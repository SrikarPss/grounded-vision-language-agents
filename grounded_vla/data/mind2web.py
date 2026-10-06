"""Mind2Web loader.

Mind2Web (Deng et al., 2023) is a real-world web-agent benchmark. The
official release ships as HuggingFace ``osunlp/Mind2Web`` with per-task
screenshots and DOM snippets. We normalize it into our unified Task schema.

This loader has two modes:

1. ``from_hf``: pulls from HuggingFace datasets (requires the `data` extra).
2. ``from_jsonl``: reads a pre-converted JSONL (what you'll typically use
   after one-time preprocessing).

For reproducibility, run `scripts/prepare_mind2web.py` once to produce a
JSONL on disk and then feed that through :class:`JsonlDataset` in eval.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from ..schemas import Action, ActionType, Observation, Task
from .base import Dataset, JsonlDataset


# Mind2Web's raw operation field -> our ActionType.
_OP_MAP = {
    "CLICK": ActionType.CLICK,
    "TYPE": ActionType.TYPE,
    "SELECT": ActionType.SELECT,
    "HOVER": ActionType.CLICK,  # we collapse hover into click
}


class Mind2WebDataset(Dataset):
    name = "mind2web"

    def __init__(
        self,
        hf_split: Optional[str] = None,
        jsonl_path: Optional[Path | str] = None,
        images_dir: Optional[Path | str] = None,
        limit: Optional[int] = None,
    ) -> None:
        if not hf_split and not jsonl_path:
            raise ValueError("pass either hf_split=... or jsonl_path=...")
        self.hf_split = hf_split
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self.images_dir = Path(images_dir) if images_dir else None
        self.limit = limit

    def __iter__(self) -> Iterator[Task]:
        if self.jsonl_path:
            yield from JsonlDataset(
                self.jsonl_path, source="mind2web", base_dir=self.images_dir, limit=self.limit
            )
            return
        yield from self._from_hf()

    def _from_hf(self) -> Iterator[Task]:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise RuntimeError(
                "Install `datasets` to stream Mind2Web directly; "
                "otherwise pre-convert it to JSONL with scripts/prepare_mind2web.py"
            ) from e

        ds = load_dataset("osunlp/Mind2Web", split=self.hf_split, streaming=True)
        for i, ex in enumerate(ds):
            if self.limit is not None and i >= self.limit:
                return
            yield _example_to_task(ex, self.images_dir)


def _example_to_task(ex: dict, images_dir: Optional[Path]) -> Task:
    instruction = ex.get("confirmed_task") or ex.get("task") or ex.get("instruction") or ""
    task_id = str(ex.get("annotation_id") or ex.get("task_id") or ex.get("id"))
    actions_raw = ex.get("actions") or ex.get("action_reprs") or []
    gold_actions: list[Action] = []
    for a in actions_raw:
        if isinstance(a, str):
            # action_reprs are strings like "[button] Sign in -> CLICK"
            op = a.rsplit("->", 1)[-1].strip().upper() if "->" in a else "CLICK"
            target = a.rsplit("->", 1)[0].strip() if "->" in a else a
            gold_actions.append(
                Action(type=_OP_MAP.get(op, ActionType.CLICK), target=target)
            )
        elif isinstance(a, dict):
            op = str(a.get("operation", {}).get("op", "CLICK")).upper()
            gold_actions.append(
                Action(
                    type=_OP_MAP.get(op, ActionType.CLICK),
                    target=a.get("pos_candidates", [{}])[0].get("attributes", {}).get("id")
                    or a.get("raw_html"),
                    value=a.get("operation", {}).get("value"),
                )
            )

    # Mind2Web ships screenshot as base64 inside the row; a preprocessing
    # script is expected to have dumped it to disk under images_dir/<id>.png.
    img_path: Optional[Path] = None
    if images_dir is not None:
        candidate = images_dir / f"{task_id}.png"
        img_path = candidate if candidate.exists() else None

    obs = Observation(step=0, image_path=img_path, text=ex.get("cleaned_html"))
    return Task(
        task_id=task_id,
        instruction=instruction,
        source="mind2web",
        initial_observation=obs,
        gold_actions=gold_actions,
        max_steps=max(15, len(gold_actions) + 5),
        meta={"domain": ex.get("domain"), "website": ex.get("website")},
    )
