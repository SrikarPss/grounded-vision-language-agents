"""LLaVA-1.6 (7B) backend.

Loads ``llava-hf/llava-v1.6-mistral-7b-hf`` lazily so that environments
without torch/transformers installed can still import this module. The
heavy dependencies are imported inside the class only when weights are
actually requested.

Usage::

    from grounded_vla.backends.llava import LLaVABackend
    be = LLaVABackend(device="cuda", quantize="4bit")
    be.warmup()
    out = be.generate(prompt, image=pil_img)
"""
from __future__ import annotations

from typing import Optional

from PIL import Image

from .base import Backend, BackendError, GenerationConfig


class LLaVABackend(Backend):
    name = "llava-1.6-7b"
    supports_vision = True

    DEFAULT_MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        quantize: Optional[str] = "4bit",
        hf_cache_dir: Optional[str] = None,
    ) -> None:
        # Notebooks sometimes pass model_id as a full local path like
        # "/cache/llava-v1.6-mistral-7b-hf".  Two cases:
        #   a) The directory EXISTS on disk (snapshot_download --local-dir style):
        #      pass the path directly to from_pretrained; no cache_dir needed.
        #   b) The directory does NOT exist (HF cache-dir style):
        #      reconstruct the Hub ID from the basename and use the parent as
        #      cache_dir so HF can find the models--* snapshot tree.
        if model_id.count("/") > 1:
            import pathlib as _pl
            _p = _pl.Path(model_id)
            if _p.is_dir():
                # Local directory from snapshot_download --local-dir
                hf_cache_dir = None  # from_pretrained loads directly from the path
            else:
                # HF cache tree: parent is cache_dir, name maps to Hub ID
                if hf_cache_dir is None:
                    hf_cache_dir = str(_p.parent)
                model_id = f"llava-hf/{_p.name}"
        self.model_id = model_id
        self.device = device
        self.quantize = quantize
        self.hf_cache_dir = hf_cache_dir
        self._model = None
        self._processor = None

    def warmup(self) -> None:
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import (
                AutoProcessor,
                LlavaNextForConditionalGeneration,
            )
        except ImportError as e:
            raise BackendError(
                "LLaVA backend requires transformers + torch. "
                "Install with `pip install -e .[gpu]`."
            ) from e

        load_kwargs: dict = {"cache_dir": self.hf_cache_dir}
        if self.quantize == "4bit":
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as e:
                raise BackendError("bitsandbytes required for 4-bit LLaVA") from e
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = "auto"
        elif self.quantize == "8bit":
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = self.device

        # Temporarily lift TRANSFORMERS_OFFLINE so that a missing
        # processor_config.json can be fetched once from the Hub.
        # This is a no-op when the network is already unrestricted.
        import os as _os
        _offline = _os.environ.pop("TRANSFORMERS_OFFLINE", None)
        try:
            self._processor = AutoProcessor.from_pretrained(
                self.model_id, cache_dir=self.hf_cache_dir
            )
        finally:
            if _offline is not None:
                _os.environ["TRANSFORMERS_OFFLINE"] = _offline

        self._model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_id, **load_kwargs
        )
        self._model.eval()

    def generate(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        if image is None:
            raise BackendError("LLaVABackend requires an image; use a text-only backend instead")
        self._ensure_loaded()
        config = config or GenerationConfig()

        import torch

        # LLaVA-1.6 chat template expects an <image> token + user text.
        chat = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        prompt_str = self._processor.apply_chat_template(chat, add_generation_prompt=True)
        inputs = self._processor(images=image, text=prompt_str, return_tensors="pt").to(
            self._model.device
        )
        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.temperature > 0,
                temperature=max(config.temperature, 1e-5),
                top_p=config.top_p,
            )
        text = self._processor.batch_decode(
            out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]
        return text

    def close(self) -> None:
        if self._model is not None:
            try:
                import torch

                del self._model
                del self._processor
                self._model = None
                self._processor = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
