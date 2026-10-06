"""SyntheticBuilder.

Curates ~200 image + instruction + ground-truth-action triples as described
in Section 3.3 of the proposal. The pipeline is:

  1. Intake: a folder of Creative Commons images with a metadata manifest
     (license + attribution). We don't fetch images over the network
     inside this class keep that in a separate fetch script so the build
     step is offline-reproducible.
  2. Instruction generation: call an LLM backend to produce a natural
     language instruction + expected action for each image. The backend is
     pluggable so the proposal's GPT-4 path AND a local Mistral path both
     work.
  3. Staging: write one ``.json`` entry per triple into a staging directory
     and enqueue it for review via ``ReviewQueue``.
  4. Finalization: once two reviewers sign off, flush approved triples to
     a single ``synthetic.jsonl`` that the loader consumes.

The builder is intentionally simple callable and resumable; state lives on
disk under ``out_dir`` so a crash mid-run doesn't cost hours of work.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from ..action_parser import parse as parse_action
from ..backends.base import Backend, GenerationConfig
from ..utils.image import load_image
from ..utils.logging import get_logger
from .review import ReviewQueue, ReviewStatus

_log = get_logger(__name__)


@dataclass
class ImageRecord:
    """Intake schema for a single CC image."""

    image_path: Path
    license: str          # e.g. "CC-BY-4.0"
    attribution: str      # name + source URL
    hint: Optional[str] = None   # optional caption or topic hint


INSTRUCTION_PROMPT = """\
You are helping build a training and evaluation corpus of visually grounded
instruction-following tasks. Look at the attached image and write:

1. A concise natural-language instruction a user might give an agent that
   requires the agent to look at this image to succeed. Avoid instructions
   that can be answered from text alone.
2. A single ground-truth action the agent should take, in this exact JSON
   format:
     {{"type": "click|type|answer|select|scroll", "target": "<short description>", "value": "<optional payload>"}}

Format your response exactly as:

Instruction: <one sentence>
Action: <json object on one line>

Optional hint about the image: {hint}
"""


class SyntheticBuilder:
    def __init__(
        self,
        backend: Backend,
        out_dir: Path | str,
        reviewers: tuple[str, str] = ("reviewer_a", "reviewer_b"),
    ) -> None:
        self.backend = backend
        self.out_dir = Path(out_dir)
        self.staging_dir = self.out_dir / "staging"
        self.images_dir = self.out_dir / "images"
        self.approved_jsonl = self.out_dir / "synthetic.jsonl"
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.review_queue = ReviewQueue(
            state_path=self.out_dir / "review_state.json", reviewers=reviewers
        )

    # -- Public API --------------------------------------------------------

    def build(self, images: Iterable[ImageRecord], n: Optional[int] = None) -> int:
        """Generate candidate triples. Returns the number staged."""
        staged = 0
        for i, rec in enumerate(images):
            if n is not None and staged >= n:
                break
            candidate = self._generate_candidate(rec, index=i)
            if candidate is None:
                continue
            staging_path = self.staging_dir / f"{candidate['task_id']}.json"
            staging_path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
            self.review_queue.enqueue(candidate["task_id"])
            staged += 1
        _log.info("Staged %d synthetic candidates in %s", staged, self.staging_dir)
        return staged

    def finalize(self) -> Path:
        """Write the approved-subset JSONL. Returns the output path."""
        approved = self.review_queue.approved_ids()
        with self.approved_jsonl.open("w", encoding="utf-8") as f:
            for tid in sorted(approved):
                staging = self.staging_dir / f"{tid}.json"
                if not staging.exists():
                    _log.warning("approved task %s missing staging file", tid)
                    continue
                row = json.loads(staging.read_text(encoding="utf-8"))
                f.write(json.dumps(row) + "\n")
        _log.info("Wrote %d approved tasks -> %s", len(approved), self.approved_jsonl)
        return self.approved_jsonl

    # -- Internals --------------------------------------------------------

    def _generate_candidate(self, rec: ImageRecord, index: int) -> Optional[dict]:
        try:
            image = load_image(rec.image_path)
        except FileNotFoundError:
            _log.warning("skipping missing image: %s", rec.image_path)
            return None
        prompt = INSTRUCTION_PROMPT.format(hint=rec.hint or "(none)")
        raw = self.backend.generate(
            prompt, image=image, config=GenerationConfig(max_new_tokens=200, temperature=0.7)
        )
        parsed = parse_action(raw)
        # Extract the natural-language instruction line.
        instruction = _extract_instruction_line(raw) or (rec.hint or "Describe the image.")
        if not parsed.ok:
            _log.debug("candidate %d: parse failed -> %s", index, parsed.error)
            return None

        # Copy the image into our own images folder so the dataset is self-contained.
        task_id = f"syn_{uuid.uuid5(uuid.NAMESPACE_URL, str(rec.image_path))}".replace("-", "")[:16]
        dest_image = self.images_dir / f"{task_id}.png"
        if not dest_image.exists():
            image.save(dest_image, format="PNG")

        action = parsed.action
        return {
            "task_id": task_id,
            "instruction": instruction,
            "image_path": f"images/{dest_image.name}",
            "gold_actions": [
                {
                    "type": action.type.value,
                    "target": action.target,
                    "value": action.value,
                }
            ],
            "gold_answer": action.value if action.type.value == "answer" else None,
            "max_steps": 3,
            "meta": {
                "license": rec.license,
                "attribution": rec.attribution,
                "source_image": str(rec.image_path),
                "generator_backend": self.backend.name,
            },
        }


def _extract_instruction_line(text: str) -> Optional[str]:
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("instruction:"):
            return line.split(":", 1)[1].strip() or None
    return None


# Re-export for convenience so scripts can do `from grounded_vla.synthetic import ReviewStatus`.
__all__ = ["SyntheticBuilder", "ImageRecord", "ReviewStatus"]
