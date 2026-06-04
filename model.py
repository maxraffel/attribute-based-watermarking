"""Hugging Face causal LM: hub id, device, sampling overrides, and lazy loading."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_ID = DEFAULT_MODEL_ID
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLING: dict[str, float | int] = {
    "temperature": 0.9,
    "top_p": 1.0,
    "top_k": 0,
}

_model: AutoModelForCausalLM | None = None
_tokenizer: AutoTokenizer | None = None


def configure(
    *,
    model_id: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> None:
    """Set hub id and/or sampling; unloads weights when the hub id changes."""
    global MODEL_ID, _model, _tokenizer
    if temperature is not None:
        SAMPLING["temperature"] = temperature
    if top_p is not None:
        SAMPLING["top_p"] = top_p
    if top_k is not None:
        SAMPLING["top_k"] = top_k
    if model_id is not None:
        mid = model_id.strip()
        if mid and mid != MODEL_ID:
            MODEL_ID = mid
            _model = None
            _tokenizer = None


def unload() -> None:
    global _model, _tokenizer
    _model = None
    _tokenizer = None


def load() -> tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        _model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
        for attr, value in SAMPLING.items():
            setattr(_model.generation_config, attr, value)
        if not _model.config.is_encoder_decoder:
            _tokenizer.padding_side = "left"
    return _model, _tokenizer, DEVICE
