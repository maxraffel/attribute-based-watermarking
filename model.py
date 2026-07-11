"""Hugging Face causal LM: hub id, device, sampling overrides, and lazy loading."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from huggingface_hub import auth_check, get_token, login, model_info
from huggingface_hub.utils import GatedRepoError
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[misc, assignment]

DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_ID = DEFAULT_MODEL_ID
SAMPLING: dict[str, float | int] = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 0,
}

# --- inference precision (edit here or pass to ``configure(inference_dtype=...)``) ---
# ``None`` / ``"fp32"`` — full float32 (default)
# ``"fp16"`` — float16 weights/activations (fast; good on most NVIDIA GPUs)
# ``"bf16"`` — bfloat16 (often best on Ampere+; wider exponent range than fp16)
DEFAULT_INFERENCE_DTYPE: str | None = "fp16"
INFERENCE_DTYPE: str | None = DEFAULT_INFERENCE_DTYPE

_model: AutoModelForCausalLM | None = None
_tokenizer: AutoTokenizer | None = None


def require_cuda_device() -> str:
    """Return ``\"cuda\"`` or raise if no CUDA GPU is available."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required but not available. "
            "Install a CUDA-enabled PyTorch build and ensure an NVIDIA driver is loaded."
        )
    return "cuda"


# Kept for callers that read ``model.DEVICE``; never falls back to CPU.
DEVICE = require_cuda_device()


def _normalize_inference_dtype(name: str | None) -> str | None:
    """Return ``None`` for fp32, else ``\"fp16\"`` or ``\"bf16\"``."""
    if name is None:
        return None
    key = name.strip().lower()
    if key in ("", "none", "fp32", "float32", "32", "full"):
        return None
    if key in ("fp16", "float16", "16"):
        return "fp16"
    if key in ("bf16", "bfloat16"):
        return "bf16"
    raise ValueError(
        f"unsupported inference_dtype {name!r}; use None, 'fp32', 'fp16', or 'bf16'"
    )


def _torch_dtype_for_inference(name: str | None) -> torch.dtype | None:
    norm = _normalize_inference_dtype(name)
    if norm is None:
        return None
    if norm == "fp16":
        return torch.float16
    return torch.bfloat16


def inference_dtype_label(name: str | None = None) -> str:
    """Human-readable label for logs (``'fp32'``, ``'fp16'``, ``'bf16'``)."""
    norm = _normalize_inference_dtype(INFERENCE_DTYPE if name is None else name)
    return "fp32" if norm is None else norm


_UNSET = object()


def _apply_sampling_to_generation_config() -> None:
    """Push ``SAMPLING`` into the loaded model's ``generation_config`` (used by ``generate``)."""
    if _model is None:
        return
    for attr, value in SAMPLING.items():
        setattr(_model.generation_config, attr, value)


def configure(
    *,
    model_id: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    inference_dtype: str | None | object = _UNSET,
) -> None:
    """Set hub id, sampling, and/or inference dtype; unloads weights when those change."""
    global MODEL_ID, INFERENCE_DTYPE, _model, _tokenizer
    if temperature is not None:
        SAMPLING["temperature"] = temperature
    if top_p is not None:
        SAMPLING["top_p"] = top_p
    if top_k is not None:
        SAMPLING["top_k"] = top_k
    if any(x is not None for x in (temperature, top_p, top_k)):
        _apply_sampling_to_generation_config()
    if inference_dtype is not _UNSET:
        new_dtype = _normalize_inference_dtype(
            None if inference_dtype is None else str(inference_dtype)
        )
        if new_dtype != INFERENCE_DTYPE:
            INFERENCE_DTYPE = new_dtype
            _model = None
            _tokenizer = None
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


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if load_dotenv is not None and env_path.is_file():
        load_dotenv(env_path)


def _hf_token_from_env() -> str | None:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return None


def _ensure_hf_auth(model_id: str) -> None:
    """Load ``.env``, log in, and fail fast on gated models without access."""
    _load_dotenv()
    token = _hf_token_from_env()
    if token:
        login(token=token, add_to_git_credential=False)

    info = model_info(model_id, token=token or True)
    gated = getattr(info, "gated", False)
    if not gated:
        return

    if not get_token():
        raise RuntimeError(
            f"Model {model_id!r} is gated on Hugging Face. Accept the license at "
            f"https://huggingface.co/{model_id}, then set HF_TOKEN in "
            f"{Path(__file__).resolve().parent / '.env'} or run `huggingface-cli login`."
        )
    try:
        auth_check(model_id)
    except GatedRepoError as exc:
        raise RuntimeError(
            f"No access to gated model {model_id!r}. Accept the license at "
            f"https://huggingface.co/{model_id} and ensure HF_TOKEN is valid."
        ) from exc


def load() -> tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    global _model, _tokenizer
    device = require_cuda_device()
    if _model is None:
        _ensure_hf_auth(MODEL_ID)
        dtype_label = inference_dtype_label()
        print(f"Loading tokenizer for {MODEL_ID}...", flush=True)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=True)
        load_dtype = _torch_dtype_for_inference(INFERENCE_DTYPE)
        print(
            f"Downloading/loading weights for {MODEL_ID} ({dtype_label})...",
            flush=True,
        )
        if load_dtype is None:
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                token=True,
            ).to(device)
        else:
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                dtype=load_dtype,
                token=True,
            ).to(device)
        print(f"Model ready on {device}.", flush=True)
        _model.eval()
        if not _model.config.is_encoder_decoder:
            _tokenizer.padding_side = "left"
    _apply_sampling_to_generation_config()
    return _model, _tokenizer, device
