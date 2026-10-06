"""One-time preprocessing: pull Mind2Web from HuggingFace and write a JSONL
plus a folder of screenshots. After running once, subsequent eval runs use
the local files (no network, no HF auth).

Defaults to ``osunlp/Multimodal-Mind2Web`` — that's the variant that ships
the screenshots LLaVA needs. The non-multimodal ``osunlp/Mind2Web`` is
DOM-only and won't work with our visual agents.

Streaming-mode test splits aren't always exposed by HF's streaming API. If
the requested split isn't in the streaming view, we probe what *is*
available and either use a sensible fallback (e.g., ``test_task`` →
``test_website`` → ``train``) or, with ``--strict``, fail with a helpful
message.

Usage::

    python scripts/prepare_mind2web.py \
        --split test_task \
        --out-dir data/mind2web \
        --limit 200
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm


# Order matters: when the requested split isn't available, we walk this list.
_FALLBACK_SPLITS = (
    "test_task",
    "test_website",
    "test_domain",
    "test",
    "validation",
    "train",
)


def _resolve_split(dataset_id: str, requested: str, strict: bool) -> str:
    """Return a split that exists, with a logged warning if we had to fall back."""
    try:
        from datasets import get_dataset_split_names

        available = list(get_dataset_split_names(dataset_id))
    except Exception as e:
        # Some HF configurations don't expose split discovery without a download;
        # fall through and let load_dataset fail with its own clear error.
        print(f"WARNING: could not probe splits for {dataset_id} ({e}); proceeding")
        return requested

    print(f"Available splits in {dataset_id}: {available}")

    if requested in available:
        return requested

    if strict:
        raise SystemExit(
            f"Requested split {requested!r} not available in {dataset_id}. "
            f"Available: {available}. Pick one or run without --strict."
        )

    for fb in _FALLBACK_SPLITS:
        if fb in available:
            print(f"WARNING: split {requested!r} not found; using {fb!r} instead.")
            return fb

    # Last resort: just pick whatever's there.
    if available:
        fb = available[0]
        print(f"WARNING: no preferred fallback found; using {fb!r}.")
        return fb
    raise SystemExit(f"{dataset_id} exposes no splits we can use.")


def _save_screenshot(ex: dict, dest: Path) -> bool:
    """Multimodal-Mind2Web stores screenshots in a few different shapes.

    Returns True if a PNG was written. We try, in order:
      - ``ex['screenshot']`` already a PIL Image
      - bytes / bytearray
      - base64-encoded str
      - the multimodal repo's ``ex['action_uid']`` -> per-step images is
        handled by callers separately; this fn only does the per-task one.
    """
    shot = ex.get("screenshot")
    if shot is None:
        return False
    if isinstance(shot, Image.Image):
        shot.convert("RGB").save(dest)
        return True
    if isinstance(shot, (bytes, bytearray)):
        Image.open(io.BytesIO(shot)).convert("RGB").save(dest)
        return True
    if isinstance(shot, str) and shot.startswith(("iVBOR", "/9j/")):
        Image.open(io.BytesIO(base64.b64decode(shot))).convert("RGB").save(dest)
        return True
    if isinstance(shot, dict) and "bytes" in shot and shot["bytes"]:
        Image.open(io.BytesIO(shot["bytes"])).convert("RGB").save(dest)
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-id",
        default="osunlp/Multimodal-Mind2Web",
        help="HF dataset id. Default ships screenshots; pass `osunlp/Mind2Web` for DOM-only.",
    )
    ap.add_argument(
        "--split",
        default="test_task",
        help="Preferred split. Falls back gracefully if unavailable in streaming mode.",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of falling back when --split is unavailable.",
    )
    ap.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use streaming mode (default). --no-streaming downloads the whole archive.",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install `datasets`: pip install -e .[data]")

    split = _resolve_split(args.dataset_id, args.split, args.strict)

    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{split}.jsonl"

    ds = load_dataset(args.dataset_id, split=split, streaming=args.streaming)

    n_with_image = 0
    with jsonl_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(tqdm(ds, desc=f"mind2web/{split}")):
            if args.limit is not None and i >= args.limit:
                break
            task_id = str(ex.get("annotation_id") or ex.get("task_id") or ex.get("id") or i)
            img_dest = images_dir / f"{task_id}.png"
            had_image = _save_screenshot(ex, img_dest)
            n_with_image += int(had_image)

            row = {
                "task_id": task_id,
                "instruction": ex.get("confirmed_task") or ex.get("task") or ex.get("instruction"),
                "image_path": f"images/{task_id}.png" if had_image else None,
                "text": ex.get("cleaned_html") or ex.get("html"),
                "gold_actions": _coerce_actions(
                    ex.get("actions") or ex.get("action_reprs") or []
                ),
                "max_steps": 20,
                "meta": {
                    "domain": ex.get("domain"),
                    "website": ex.get("website"),
                    "split": split,
                    "dataset_id": args.dataset_id,
                },
            }
            f.write(json.dumps(row) + "\n")

    print(
        f"wrote {jsonl_path}\n"
        f"  rows: {i + 1 if 'i' in dir() else 0}\n"
        f"  rows_with_image: {n_with_image}\n"
        f"  images_dir: {images_dir}"
    )
    if n_with_image == 0:
        print(
            "WARNING: 0 rows had a screenshot. You probably loaded the DOM-only "
            "`osunlp/Mind2Web`; pass `--dataset-id osunlp/Multimodal-Mind2Web` "
            "for the visual variant."
        )


def _coerce_actions(actions) -> list[dict]:
    out = []
    for a in actions:
        if isinstance(a, str) and "->" in a:
            left, right = a.rsplit("->", 1)
            out.append({"type": right.strip().lower(), "target": left.strip()})
        elif isinstance(a, dict):
            op = (a.get("operation") or {}).get("op", "click")
            out.append(
                {
                    "type": op.lower(),
                    "target": a.get("raw_html") or a.get("target_node"),
                    "value": (a.get("operation") or {}).get("value"),
                }
            )
    return out


if __name__ == "__main__":
    main()
