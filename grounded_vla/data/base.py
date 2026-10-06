"""Base dataset interface + a generic JSONL loader."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Iterator, Optional

from ..schemas import Action, ActionType, Observation, Task

# Normalize non-canonical action-type strings to valid ActionType values before
# constructing Action objects. This runs in the data layer so it works even on
# Python environments where the schemas.py field_validator hasn't been deployed
# yet (e.g. a Colab session with partially-updated code).
_ACTION_TYPE_ALIASES: dict[str, str] = {
    "hover": ActionType.CLICK,
    "tap": ActionType.CLICK,
    "press": ActionType.CLICK,
    "submit": ActionType.CLICK,
    "input": ActionType.TYPE,
    "fill": ActionType.TYPE,
    "choose": ActionType.SELECT,
    "done": ActionType.STOP,
    "finish": ActionType.STOP,
}


class Dataset(ABC):
    """Streaming iterable of Tasks."""

    name: str = "dataset"

    @abstractmethod
    def __iter__(self) -> Iterator[Task]: ...

    def take(self, n: int) -> list[Task]:
        out: list[Task] = []
        for i, t in enumerate(self):
            if i >= n:
                break
            out.append(t)
        return out


def _action_from_dict(d: dict) -> Action:
    raw = str(d.get("type", "noop")).strip().lower()
    # Resolve aliases first so pydantic only ever sees canonical enum values.
    atype = _ACTION_TYPE_ALIASES.get(raw, raw)
    try:
        ActionType(atype)  # validate; falls through to noop on unknown values
    except ValueError:
        atype = ActionType.NOOP
    return Action(
        type=atype,
        target=d.get("target"),
        value=d.get("value"),
        xy=tuple(d["xy"]) if d.get("xy") else None,
        rationale=d.get("rationale"),
    )


class JsonlDataset(Dataset):
    """Reads a JSONL file where each line is a Task-shaped dict.

    Expected schema per line::

        {
          "task_id": "...",
          "instruction": "...",
          "image_path": "path/to/frame.png",  // relative to `base_dir` if given
          "text": "...",                      // optional
          "gold_actions": [{"type": "click", "target": "#submit"}],
          "gold_answer": "42",
          "max_steps": 15,
          "meta": {...}
        }
    """

    def __init__(
        self,
        path: Path | str,
        source: str,
        base_dir: Optional[Path | str] = None,
        limit: Optional[int] = None,
    ) -> None:
        self.path = Path(path)
        self.source = source
        self.name = source
        self.base_dir = Path(base_dir) if base_dir else self.path.parent
        self.limit = limit

    def __iter__(self) -> Iterator[Task]:
        if not self.path.exists():
            raise FileNotFoundError(f"dataset file not found: {self.path}")
        count = 0
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                obj = json.loads(line)
                yield self._to_task(obj)
                count += 1
                if self.limit is not None and count >= self.limit:
                    return

    def _to_task(self, obj: dict) -> Task:
        img = obj.get("image_path")
        if img and not Path(img).is_absolute():
            candidate = self.base_dir / img
            if not candidate.exists():
                # Detect double-prefix: image_path starts with a directory
                # whose name equals the last component of base_dir.
                # e.g. base_dir=".../mind2web/images", img="images/<id>.png"
                # → try base_dir / <id>.png instead.
                img_parts = Path(img).parts
                if img_parts and img_parts[0] == self.base_dir.name:
                    alt = self.base_dir.joinpath(*img_parts[1:])
                    if alt.exists():
                        candidate = alt
            img = str(candidate)
        obs = Observation(
            step=0,
            image_path=img,
            text=obj.get("text"),
            available_actions=obj.get("available_actions", []),
        )
        gold_actions = [_action_from_dict(a) for a in obj.get("gold_actions", [])]
        return Task(
            task_id=obj["task_id"],
            instruction=obj["instruction"],
            source=self.source,
            initial_observation=obs,
            gold_actions=gold_actions,
            gold_answer=obj.get("gold_answer"),
            max_steps=obj.get("max_steps", 15),
            meta=obj.get("meta", {}),
        )


def write_jsonl(path: Path | str, rows: Iterable[dict]) -> None:
    """Tiny helper used by the synthetic builder and tests."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
