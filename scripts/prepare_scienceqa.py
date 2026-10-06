"""Preprocess ScienceQA: save images to disk + write a JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only-with-image", action="store_true", default=True)
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install `datasets`: pip install -e .[data]")

    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{args.split}.jsonl"

    ds = load_dataset("derek-thomas/ScienceQA", split=args.split, streaming=True)

    with jsonl_path.open("w", encoding="utf-8") as f:
        count = 0
        for ex in tqdm(ds, desc="scienceqa"):
            if args.only_with_image and not ex.get("image"):
                continue
            if args.limit is not None and count >= args.limit:
                break
            count += 1
            task_id = str(ex.get("id") or ex.get("question_id") or count)
            img = ex["image"]
            img.convert("RGB").save(images_dir / f"{task_id}.png")
            choices = ex.get("choices", [])
            gold = choices[ex["answer"]] if isinstance(ex.get("answer"), int) and choices else str(ex.get("answer"))
            options_block = "\n".join(f"  ({chr(65 + i)}) {c}" for i, c in enumerate(choices))
            text = f"{ex['question']}\nOptions:\n{options_block}"
            row = {
                "task_id": task_id,
                "instruction": "Answer the science question using the diagram and text.",
                "image_path": f"images/{task_id}.png",
                "text": text,
                "gold_actions": [{"type": "answer", "value": gold}],
                "gold_answer": gold,
                "max_steps": 3,
                "meta": {
                    "choices": choices,
                    "subject": ex.get("subject"),
                    "topic": ex.get("topic"),
                },
            }
            f.write(json.dumps(row) + "\n")

    print(f"wrote {jsonl_path} ({count} rows)")


if __name__ == "__main__":
    main()
