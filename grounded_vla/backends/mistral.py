"""Mistral-7B text-only backend used by the ReAct baseline.

Loads ``mistralai/Mistral-7B-Instruct-v0.2`` lazily. Images are ignored with
a logged warning (the whole point of the baseline is to see what the model
does without vision).
"""
from __future__ import annotations

from typing import Optional

from PIL import Image

from ..utils.logging import get_logger
from .base import Backend, BackendError, GenerationConfig

_log = get_logger(__name__)


class MistralBackend(Backend):
    name = "mistral-7b-instruct"
    supports_vision = False

    DEFAULT_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        quantize: Optional[str] = "4bit",
        hf_cache_dir: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.quantize = quantize
        self.hf_cache_dir = hf_cache_dir
        self._model = None
        self._tokenizer = None

    def warmup(self) -> None:
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise BackendError(
                "Mistral backend requires transformers + torch. "
                "Install with `pip install -e .[gpu]`."
            ) from e

        load_kwargs: dict = {"cache_dir": self.hf_cache_dir}
        if self.quantize == "4bit":
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = self.device

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, cache_dir=self.hf_cache_dir
        )
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
        self._model.eval()

    def generate(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        if image is not None:
            _log.debug("MistralBackend ignoring image (text-only baseline).")
        self._ensure_loaded()
        config = config or GenerationConfig()

        import torch

        chat = [{"role": "user", "content": prompt}]
        inputs = self._tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.temperature > 0,
                temperature=max(config.temperature, 1e-5),
                top_p=config.top_p,
            )
        return self._tokenizer.decode(out[0, input_len:], skip_special_tokens=True)

    def close(self) -> None:
        if self._model is not None:
            try:
                import torch

                del self._model
                del self._tokenizer
                self._model = None
                self._tokenizer = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
