"""LoRA fine-tuning entry point (Stretch Goal H3).

This module is a thin wrapper around HuggingFace PEFT that's set up for the
synthetic instruction-image corpus. It is deliberately minimal we don't
try to reimplement a trainer; we rely on `transformers.Trainer` so the
risk surface stays small. If GPU resources don't materialize, this file
stays unused the rest of the project still runs.

Usage::

    from grounded_vla.lora import train_lora
    train_lora(
        jsonl_path="data/synthetic/synthetic.jsonl",
        images_dir="data/synthetic/images",
        output_dir="checkpoints/llava-lora-run-1",
    )
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .utils.logging import get_logger

_log = get_logger(__name__)


@dataclass
class LoRAConfig:
    base_model: str = "llava-hf/llava-v1.6-mistral-7b-hf"
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    learning_rate: float = 2e-4
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    warmup_ratio: float = 0.03
    max_seq_len: int = 1024


def train_lora(
    jsonl_path: Path | str,
    images_dir: Path | str,
    output_dir: Path | str,
    config: Optional[LoRAConfig] = None,
) -> Path:
    """Fine-tune LLaVA-1.6 with LoRA on the synthetic dataset.

    Returns the directory containing the trained adapter.
    """
    try:
        import torch
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            LlavaNextForConditionalGeneration,
            Trainer,
            TrainingArguments,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as e:
        raise RuntimeError(
            "LoRA fine-tuning needs the [gpu] extra: `pip install -e .[gpu]`"
        ) from e

    config = config or LoRAConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalise base_model path.  Two cases:
    #   a) Path exists on disk (snapshot_download --local-dir style):
    #      pass the path directly; no cache_dir needed.
    #   b) Path doesn't exist (HF cache tree):
    #      infer Hub ID + cache_dir from parent/basename.
    _base_model = config.base_model
    _cache_dir: Optional[str] = None
    if _base_model.count("/") > 1:
        _p = Path(_base_model)
        if _p.is_dir():
            _cache_dir = None  # load directly from local path
        else:
            _cache_dir = str(_p.parent)
            _base_model = f"llava-hf/{_p.name}"

    import os as _os
    _offline = _os.environ.pop("TRANSFORMERS_OFFLINE", None)
    try:
        processor = AutoProcessor.from_pretrained(_base_model, cache_dir=_cache_dir)
    finally:
        if _offline is not None:
            _os.environ["TRANSFORMERS_OFFLINE"] = _offline
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = LlavaNextForConditionalGeneration.from_pretrained(
        _base_model, quantization_config=bnb, device_map="auto",
        **({"cache_dir": _cache_dir} if _cache_dir else {}),
    )
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    dataset = _load_synthetic_dataset(jsonl_path, images_dir, processor)

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=10,
        save_strategy="epoch",
        bf16=torch.cuda.is_bf16_supported(),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=_collate_fn,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    _log.info("LoRA adapter saved to %s", output_dir)
    return output_dir


def _collate_fn(features):
    """Dynamic-padding collator for multimodal LLaVA batches.

    pixel_values / image_sizes are stacked as-is (same shape per batch).
    Sequence tensors (input_ids, attention_mask, labels) are padded to the
    longest example in the batch so that image tokens are never truncated.
    """
    import torch

    batch = {}
    seq_keys = {"input_ids", "attention_mask", "labels"}
    stack_keys = {"pixel_values", "image_sizes"}

    max_len = max(f["input_ids"].shape[-1] for f in features)

    for key in features[0]:
        if key in stack_keys:
            batch[key] = torch.stack([f[key] for f in features])
        elif key in seq_keys:
            pad_val = -100 if key == "labels" else 0
            tensors = []
            for f in features:
                t = f[key]
                gap = max_len - t.shape[-1]
                if gap > 0:
                    t = torch.cat(
                        [t, torch.full((*t.shape[:-1], gap), pad_val, dtype=t.dtype)], dim=-1
                    )
                tensors.append(t)
            batch[key] = torch.stack(tensors)
        else:
            # unknown key – best-effort stack
            try:
                batch[key] = torch.stack([f[key] for f in features])
            except Exception:
                pass

    return batch


def _load_synthetic_dataset(jsonl_path, images_dir, processor):
    """Build a torch Dataset from the synthetic JSONL.

    Each example is rendered as an instruction + target-action string so
    the model learns to emit actions in our expected format.
    No truncation is applied here — image placeholder tokens must never be
    cut; the collator pads batches dynamically instead.
    """
    from PIL import Image
    from torch.utils.data import Dataset as TorchDataset

    jsonl_path = Path(jsonl_path)
    images_dir = Path(images_dir)
    rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]

    class _DS(TorchDataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            r = rows[i]
            img_path = Path(r["image_path"])
            if not img_path.is_absolute():
                img_path = images_dir / img_path
            image = Image.open(img_path).convert("RGB")
            gold = r["gold_actions"][0]
            target = (
                "Thought: Looking at the image, I will perform the grounded action.\n"
                "Action: " + json.dumps(gold)
            )
            chat = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": r["instruction"]},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": target}]},
            ]
            prompt_str = processor.apply_chat_template(chat, add_generation_prompt=False)
            # No truncation — image placeholder tokens must match vision features exactly.
            enc = processor(images=image, text=prompt_str, return_tensors="pt")
            item = {k: v.squeeze(0) for k, v in enc.items()}

            # Only supervise the assistant response tokens; mask the prompt with -100.
            # Encode just the user turn (with generation prompt) to measure prompt length.
            prompt_only_str = processor.apply_chat_template(
                chat[:1], add_generation_prompt=True
            )
            prompt_enc = processor(images=image, text=prompt_only_str, return_tensors="pt")
            prompt_len = prompt_enc["input_ids"].shape[-1]

            labels = item["input_ids"].clone()
            labels[:prompt_len] = -100  # mask prompt tokens from loss
            item["labels"] = labels
            return item

    return _DS()
