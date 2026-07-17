from __future__ import annotations

import hashlib
import inspect
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList

_CALIBRATION_CORPUS_PATH = (
    Path(__file__).resolve().parent / "data" / "calibration_corpus.txt"
)
_WEIGHTS_CACHE_DIR = Path(__file__).resolve().parent / ".cache"

# Normalized unigram weights over the partition vocabulary (CPU float32).
_PARTITION_WEIGHTS: torch.Tensor | None = None
_PARTITION_WEIGHTS_KEY: str | None = None


def _partition_seed(seed_index: int) -> int:
    return int(hashlib.md5(f"{seed_index}".encode()).hexdigest()[:16], 16)


def _tokenizer_weight_key(tokenizer: AutoTokenizer, vocab_size: int) -> str:
    name = getattr(tokenizer, "name_or_path", None) or type(tokenizer).__name__
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))
    return f"{safe}_v{int(vocab_size)}"


def _weights_cache_path(key: str) -> Path:
    return _WEIGHTS_CACHE_DIR / f"partition_weights_{key}.pt"


def configure_partition_weights(weights: torch.Tensor) -> torch.Tensor:
    """Install normalized non-negative partition weights (copied to CPU)."""
    global _PARTITION_WEIGHTS, _PARTITION_WEIGHTS_KEY
    if weights.ndim != 1:
        raise ValueError(f"partition weights must be 1-D, got shape {tuple(weights.shape)}")
    w = weights.detach().to(dtype=torch.float32, device="cpu").clamp(min=0.0)
    total = float(w.sum().item())
    if total <= 0.0:
        raise ValueError("partition weights must sum to a positive value")
    _PARTITION_WEIGHTS = w / total
    _PARTITION_WEIGHTS_KEY = None
    return _PARTITION_WEIGHTS


def clear_partition_weights() -> None:
    global _PARTITION_WEIGHTS, _PARTITION_WEIGHTS_KEY
    _PARTITION_WEIGHTS = None
    _PARTITION_WEIGHTS_KEY = None


def get_partition_weights(vocab_size: int | None = None) -> torch.Tensor | None:
    """Return the active weight vector, optionally checking ``vocab_size``."""
    if _PARTITION_WEIGHTS is None:
        return None
    if vocab_size is not None and int(_PARTITION_WEIGHTS.numel()) != int(vocab_size):
        raise ValueError(
            f"partition weights length {_PARTITION_WEIGHTS.numel()} "
            f"!= vocab_size {vocab_size}"
        )
    return _PARTITION_WEIGHTS


def build_unigram_partition_weights(
    tokenizer: AutoTokenizer,
    vocab_size: int,
    *,
    corpus_path: Path | str | None = None,
) -> torch.Tensor:
    """Laplace-smoothed unigram weights from calibrating ``tokenizer`` on a corpus."""
    path = Path(corpus_path) if corpus_path is not None else _CALIBRATION_CORPUS_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"calibration corpus not found at {path}; pass corpus_path or create the file"
        )
    text = path.read_text(encoding="utf-8")
    # +1 Laplace so every id has mass; corpus hits raise frequent tokens.
    counts = torch.ones(int(vocab_size), dtype=torch.float32)
    # Chunk so tokenizers with short model_max_length do not warn/truncate.
    chunk_chars = 800
    for start in range(0, len(text), chunk_chars):
        chunk = text[start : start + chunk_chars]
        if not chunk.strip():
            continue
        enc = tokenizer(chunk, add_special_tokens=False)
        ids = enc["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = [t for row in ids for t in row]
        for tid in ids:
            t = int(tid)
            if 0 <= t < vocab_size:
                counts[t] += 1.0
    return counts / counts.sum()


def ensure_partition_weights(
    vocab_size: int,
    tokenizer: AutoTokenizer | None = None,
    *,
    corpus_path: Path | str | None = None,
    use_cache: bool = True,
) -> torch.Tensor:
    """
    Ensure module-level weights exist for ``vocab_size``.

    Prefer a disk cache keyed by tokenizer identity; otherwise count unigrams on
    the calibration corpus. Without a tokenizer, fall back to uniform weights
    (token-count balance only).
    """
    global _PARTITION_WEIGHTS, _PARTITION_WEIGHTS_KEY
    v = int(vocab_size)
    key: str | None = None
    if tokenizer is not None:
        key = _tokenizer_weight_key(tokenizer, v)

    if _PARTITION_WEIGHTS is not None and int(_PARTITION_WEIGHTS.numel()) == v:
        if key is None or _PARTITION_WEIGHTS_KEY in (None, key):
            return _PARTITION_WEIGHTS

    if tokenizer is not None:
        assert key is not None
        cache_path = _weights_cache_path(key)
        if use_cache and cache_path.is_file():
            loaded = torch.load(cache_path, map_location="cpu", weights_only=True)
            if not isinstance(loaded, torch.Tensor) or int(loaded.numel()) != v:
                raise ValueError(f"corrupt partition weight cache at {cache_path}")
            _PARTITION_WEIGHTS = loaded.to(dtype=torch.float32)
            _PARTITION_WEIGHTS_KEY = key
            return _PARTITION_WEIGHTS
        weights = build_unigram_partition_weights(
            tokenizer, v, corpus_path=corpus_path
        )
        configure_partition_weights(weights)
        _PARTITION_WEIGHTS_KEY = key
        if use_cache:
            _WEIGHTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(_PARTITION_WEIGHTS, cache_path)
        return _PARTITION_WEIGHTS

    # No tokenizer: uniform weights ⇒ random perm takes ~half the tokens by count.
    configure_partition_weights(torch.ones(v, dtype=torch.float32))
    _PARTITION_WEIGHTS_KEY = f"uniform_v{v}"
    return _PARTITION_WEIGHTS


def get_vectorized_partition(vocab_size: int, device: str, seed_index: int) -> torch.BoolTensor:
    """
    Deterministic boolean mask over the vocabulary for one bit index.

    Membership is a seeded random permutation of token ids; set A is the shortest
    prefix of that order whose static unigram weight is at least half the total.
    That keeps masks looking random while balancing estimated probability mass.
    """
    v = int(vocab_size)
    weights = get_partition_weights(v)
    if weights is None:
        weights = ensure_partition_weights(v)

    dev = torch.device(device) if isinstance(device, str) else device
    # randperm with a CUDA generator is unreliable across devices; permute on CPU.
    g = torch.Generator(device="cpu")
    g.manual_seed(_partition_seed(seed_index))
    order = torch.randperm(v, generator=g, device="cpu")
    w = weights[order]
    cum = torch.cumsum(w, dim=0)
    # Shortest prefix with mass >= half, then prefer the nearer of that prefix or
    # the one before it so a single heavy token cannot overshoot too far.
    half = 0.5 * float(cum[-1].item())
    cutoff = int(torch.searchsorted(cum, torch.tensor(half, dtype=cum.dtype)).item())
    if cutoff >= v:
        cutoff = v - 1
    if cutoff > 0:
        over = float(cum[cutoff].item()) - half
        under = half - float(cum[cutoff - 1].item())
        if over > under:
            cutoff -= 1
    mask = torch.zeros(v, dtype=torch.bool, device="cpu")
    mask[order[: cutoff + 1]] = True
    return mask.to(device=dev)


def get_balanced_partition_from_probs(probs: torch.Tensor) -> torch.BoolTensor:
    """
    Split positive-mass tokens into two sets with near-equal probability mass.

    Deterministic given ``probs`` alone: positive-mass tokens are processed by
    descending probability (ties broken by token id), and each token is assigned
    to the currently lighter set. Zero-mass tokens get a fixed parity assignment
    because they cannot affect the probability balance.
    """
    if probs.ndim != 1:
        raise ValueError(f"probs must be 1-D, got shape {tuple(probs.shape)}")
    p = probs.detach().to(dtype=torch.float32)
    if not torch.isfinite(p).all():
        raise ValueError("probs must be finite")
    total = float(p.sum().item())
    if total <= 0.0:
        raise ValueError("probs must sum to a positive value")
    p = p / total
    v = int(p.numel())
    device = p.device

    ids = torch.arange(v, device=device)
    mask = (ids % 2) == 0

    positive = (p > 0).nonzero(as_tuple=False).squeeze(1)
    if positive.numel() == 0:
        return mask

    p_pos = p[positive]
    # Stable descending-prob order among positive-mass tokens; ties -> smaller id.
    keys = (1.0 - p_pos) * float(v + 1) + positive.to(dtype=p.dtype)
    order = torch.argsort(keys)
    pos_sorted = positive[order]
    p_sorted = p_pos[order]

    mass_a = 0.0
    mass_b = 0.0
    for idx, pi in zip(pos_sorted.tolist(), p_sorted.tolist()):
        if mass_a <= mass_b:
            mask[idx] = True
            mass_a += float(pi)
        else:
            mask[idx] = False
            mass_b += float(pi)
    return mask


def balanced_partition_masses(probs: torch.Tensor, mask_A: torch.BoolTensor) -> Tuple[float, float]:
    """Return ``(mass_A, mass_B)`` for ``probs`` under ``mask_A``."""
    p = probs.detach().to(dtype=torch.float32)
    total = float(p.sum().item())
    if total <= 0.0:
        return 0.0, 0.0
    p = p / total
    mass_a = float(p[mask_A].sum().item())
    return mass_a, 1.0 - mass_a


def resolve_partition_vocab_size_for_recovery(
    tokenizer: AutoTokenizer,
    *,
    model: torch.nn.Module | None = None,
    partition_vocab_size: int | None = None,
) -> int:
    v = partition_vocab_size
    if v is None and model is not None:
        cfg = getattr(model, "config", None)
        if cfg is None:
            raise ValueError("model has no config; pass partition_vocab_size explicitly")
        txt = getattr(cfg, "text_config", None)
        base_vocab = getattr(txt or cfg, "vocab_size", None)
        if base_vocab is None:
            raise ValueError("model.config has no vocab_size; pass partition_vocab_size explicitly")
        v = int(base_vocab)
    if v is None:
        v = int(getattr(tokenizer, "vocab_size", len(tokenizer)))
    return int(v)


def _tokenizer_pad_token_id(tokenizer: AutoTokenizer) -> int:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        raise ValueError("tokenizer has no pad_token_id or eos_token_id for generation")
    return int(pad_id)


def encode_prompt_for_generation(
    tokenizer: AutoTokenizer,
    user_prompt: str,
    device: str | torch.device,
) -> Dict[str, torch.Tensor]:
    dev = torch.device(device) if isinstance(device, str) else device
    if getattr(tokenizer, "chat_template", None) is None:
        batch = tokenizer(user_prompt, return_tensors="pt")
    else:
        messages = [{"role": "user", "content": user_prompt}]
        out = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        if isinstance(out, torch.Tensor):
            input_ids = out
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            batch = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids, dtype=torch.long),
            }
        else:
            batch = dict(out)
            if "attention_mask" not in batch and "input_ids" in batch:
                batch["attention_mask"] = torch.ones_like(batch["input_ids"], dtype=torch.long)
    return {k: v.to(dev) for k, v in batch.items()}


def _prepare_sampling_logits_processor_bundle(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt_inputs: Dict[str, torch.Tensor],
    *,
    decode_horizon: int,
    generation_extra: Dict[str, Any] | None = None,
) -> Tuple[Any, LogitsProcessorList]:
    device = prompt_inputs["input_ids"].device
    pad_id = _tokenizer_pad_token_id(tokenizer)
    prep_kwargs: Dict[str, Any] = {
        **prompt_inputs,
        "max_new_tokens": decode_horizon,
        "do_sample": True,
        "pad_token_id": pad_id,
        "use_cache": True,
    }
    if generation_extra:
        prep_kwargs.update(generation_extra)

    generation_config, model_kwargs = model._prepare_generation_config(None, **prep_kwargs)
    inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(
        None,
        generation_config.bos_token_id,
        model_kwargs,
    )
    kwargs_has_attention_mask = model_kwargs.get("attention_mask") is not None
    model._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

    accepts_attention_mask = "attention_mask" in set(inspect.signature(model.forward).parameters.keys())
    if (
        not kwargs_has_attention_mask
        and accepts_attention_mask
        and not model.config.is_encoder_decoder
    ):
        model_kwargs["attention_mask"] = model._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config, model_kwargs
        )

    if model.config.is_encoder_decoder:
        raise NotImplementedError("encoder-decoder models are not supported for watermarked incremental decode")
    if model_input_name != "input_ids":
        model_kwargs.pop("input_ids")
    input_ids_seq_length = int(inputs_tensor.shape[1])

    has_default_max_length = (
        prep_kwargs.get("max_length") is None
        and model.generation_config.max_length is None
    )
    has_default_min_length = (
        prep_kwargs.get("min_length") is None
        and model.generation_config.min_length is None
    )
    generation_config = model._prepare_generated_length(
        generation_config=generation_config,
        has_default_max_length=has_default_max_length,
        has_default_min_length=has_default_min_length,
        model_input_name=model_input_name,
        input_ids_length=input_ids_seq_length,
        inputs_tensor=inputs_tensor,
    )

    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=inputs_tensor,
        device=inputs_tensor.device,
        model_kwargs=model_kwargs,
    )
    return generation_config, logits_processor


def _scores_to_next_token_probs(
    logits_last: torch.Tensor,
    input_ids_for_proc: torch.Tensor,
    logits_processor: LogitsProcessorList,
) -> torch.Tensor:
    next_logits = logits_last.unsqueeze(0).to(copy=True, dtype=torch.float32, device=input_ids_for_proc.device)
    next_scores = logits_processor(input_ids_for_proc, next_logits)
    return F.softmax(next_scores, dim=-1).squeeze(0)


def _choose_watermark_partition_half(secret_bit: int, p: float, q: float) -> Tuple[bool, bool]:
    """Return ``(choose_set_A, used_enforce_branch)``."""
    if random.random() < (2 * q):
        return (secret_bit == 0), True
    return (p > 0.5), False


def _mask_disallowed_half_probs(
    probs: torch.Tensor,
    mask_A: torch.BoolTensor,
    *,
    choose_set_A: bool,
) -> torch.Tensor:
    if choose_set_A:
        return torch.where(mask_A, probs, torch.zeros_like(probs))
    return torch.where(mask_A, torch.zeros_like(probs), probs)


def _watermark_step_probs(
    logits_last: torch.Tensor,
    input_ids_for_proc: torch.Tensor,
    logits_processor: LogitsProcessorList,
    mask_A: torch.BoolTensor,
    secret_bit: int,
) -> Tuple[torch.Tensor, bool]:
    base_probs = _scores_to_next_token_probs(logits_last, input_ids_for_proc, logits_processor)
    p = float(base_probs[mask_A].sum().item())
    q = min(p, 1.0 - p)
    choose_A, used_enforce = _choose_watermark_partition_half(int(secret_bit), p, q)
    return _mask_disallowed_half_probs(base_probs, mask_A, choose_set_A=choose_A), used_enforce


def _decode_suffix_for_recovery(tokenizer: AutoTokenizer, suffix_ids: List[int]) -> str:
    """Same string ``recover_bitstream_from_text`` will tokenize (suffix only, skip specials)."""
    return tokenizer.decode(suffix_ids, skip_special_tokens=True)


def _recovery_non_special_token_ids(
    tokenizer: AutoTokenizer,
    suffix_ids: List[int],
    special_ids: set[int],
) -> List[int]:
    """Non-special token ids from the detect-time decode→tokenize path."""
    suffix_text = _decode_suffix_for_recovery(tokenizer, suffix_ids)
    retok_ids = tokenizer(suffix_text, return_tensors="pt")["input_ids"][0].tolist()
    return [t for t in retok_ids if t not in special_ids]


def _update_recovery_state_after_token(
    *,
    prev_retok_ids_ns: List[int],
    retok_ids_ns: List[int],
    sacrificed_bits: int,
) -> Tuple[List[int], int, int, int, bool]:
    """
    Advance the recovery token snapshot after one generated token.

    The next partition index is always ``len(prev_retok_ids_ns)`` — the slot
    recovery will assign to the next committed payload token. Never advance that
    index independently of how the recovery stream actually changed.
    """
    if retok_ids_ns == prev_retok_ids_ns:
        return prev_retok_ids_ns, sacrificed_bits, 0, 0, True

    prev_len = len(prev_retok_ids_ns)
    new_len = len(retok_ids_ns)

    if new_len > prev_len:
        delta = new_len - prev_len
        sacrificed_delta = delta - 1
        if sacrificed_delta:
            sacrificed_bits += sacrificed_delta
        return list(retok_ids_ns), sacrificed_bits, 0, sacrificed_delta, False

    if new_len == prev_len:
        return list(retok_ids_ns), sacrificed_bits, 1, 0, False

    sacrificed_delta = prev_len - new_len
    sacrificed_bits += sacrificed_delta
    return list(retok_ids_ns), sacrificed_bits, 0, sacrificed_delta, False


def generate_with_watermark(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    secret_bitstream: List[int],
    device: str = "cpu",
    *,
    generation_extra: Dict[str, Any] | None = None,
) -> Dict:
    """Sample tokens until the recovery stream has one payload token per secret bit.

    Partition index ``k`` is always ``len(prev_retok_ids_ns)`` — the same index
    ``recover_bitstream`` uses for the ``k``-th non-special token at detect time.
    Uses static unigram-balanced vocabulary partitions (independent of step probs).
    """
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    inputs = encode_prompt_for_generation(tokenizer, prompt, device)
    input_ids_wm = inputs["input_ids"].clone()
    prompt_len = int(inputs["input_ids"].shape[1])
    attn_mask = inputs.get("attention_mask", None)

    _, logits_processor = _prepare_sampling_logits_processor_bundle(
        model,
        tokenizer,
        inputs,
        decode_horizon=len(secret_bitstream),
        generation_extra=generation_extra,
    )

    prev_retok_ids_ns: List[int] = []
    sacrificed_bits = 0
    natural_partition_choices = 0
    retok_replacements = 0
    recovery_stalls = 0
    partition_vocab_dim: int | None = None
    model_kwargs: Dict[str, Any] = {"use_cache": True}
    if attn_mask is not None:
        model_kwargs["attention_mask"] = attn_mask
    step_input_ids = input_ids_wm
    max_generation_steps = len(secret_bitstream) * 8
    generation_steps = 0

    while len(prev_retok_ids_ns) < len(secret_bitstream):
        generation_steps += 1
        if generation_steps > max_generation_steps:
            raise RuntimeError(
                f"watermarked generation stalled: {len(prev_retok_ids_ns)} recovery slots filled "
                f"of {len(secret_bitstream)} after {max_generation_steps} steps "
                f"(recovery_stalls={recovery_stalls})"
            )

        recovery_slot = len(prev_retok_ids_ns)
        with torch.no_grad():
            outputs = model(input_ids=step_input_ids, **model_kwargs)
            logits_wm = outputs.logits[0, -1, :]
            d_logits = int(logits_wm.shape[-1])
            if partition_vocab_dim is None:
                partition_vocab_dim = d_logits
                ensure_partition_weights(partition_vocab_dim, tokenizer)
            elif partition_vocab_dim != d_logits:
                raise RuntimeError(
                    f"logits vocab width changed mid-decode ({d_logits} vs {partition_vocab_dim})"
                )
            model_kwargs["past_key_values"] = outputs.past_key_values

            mask_A = get_vectorized_partition(partition_vocab_dim, device, recovery_slot)
            secret_bit = int(secret_bitstream[recovery_slot])
            probs_wm, used_enforce = _watermark_step_probs(
                logits_wm, input_ids_wm, logits_processor, mask_A, secret_bit
            )
            if not used_enforce:
                natural_partition_choices += 1
            if torch.isfinite(probs_wm).all() and float(probs_wm.sum().item()) > 0:
                probs_wm = probs_wm.clamp(min=0)
                probs_wm = probs_wm / probs_wm.sum()
                next_token_id_wm = torch.multinomial(probs_wm, num_samples=1)
            else:
                probs_fb = _scores_to_next_token_probs(logits_wm, input_ids_wm, logits_processor)
                next_token_id_wm = torch.multinomial(probs_fb, num_samples=1)

        input_ids_wm = torch.cat([input_ids_wm, next_token_id_wm.unsqueeze(0)], dim=-1)
        step_input_ids = next_token_id_wm.unsqueeze(0)
        if "attention_mask" in model_kwargs:
            model_kwargs["attention_mask"] = torch.cat(
                [
                    model_kwargs["attention_mask"],
                    torch.ones((1, 1), dtype=model_kwargs["attention_mask"].dtype, device=device),
                ],
                dim=-1,
            )

        try:
            suffix_ids = input_ids_wm[0, prompt_len:].tolist()
            retok_ids_ns = _recovery_non_special_token_ids(tokenizer, suffix_ids, special_ids)
            prev_retok_ids_ns, sacrificed_bits, replacements, _, stalled = _update_recovery_state_after_token(
                prev_retok_ids_ns=prev_retok_ids_ns,
                retok_ids_ns=retok_ids_ns,
                sacrificed_bits=sacrificed_bits,
            )
            retok_replacements += replacements
            if stalled:
                recovery_stalls += 1
        except Exception:
            recovery_stalls += 1

    suffix_ids = input_ids_wm[0, prompt_len:].tolist()
    generated_text_wm = _decode_suffix_for_recovery(tokenizer, suffix_ids)
    recovery_ns = _recovery_non_special_token_ids(tokenizer, suffix_ids, special_ids)
    gen_ns = [t for t in suffix_ids if t not in special_ids]
    recovery_ids_aligned = prev_retok_ids_ns == recovery_ns
    recovery_input_ids = tokenizer(generated_text_wm, return_tensors="pt")["input_ids"]

    return {
        "prompt_text": prompt,
        "generated_text_wm": generated_text_wm,
        "input_ids_wm": recovery_input_ids,
        "model_input_ids": input_ids_wm.detach().cpu(),
        "prompt_len": prompt_len,
        "secret_bitstream": secret_bitstream,
        "sacrificed_bits": sacrificed_bits,
        "natural_partition_choices": natural_partition_choices,
        "recovery_stalls": recovery_stalls,
        "retok_replacements": retok_replacements,
        "recovery_ids_aligned": recovery_ids_aligned,
        "recovery_slots_committed": len(prev_retok_ids_ns),
        "incremental_ids_aligned": gen_ns == recovery_ns,
        "partition_mode": "static",
        "partition_vocab_dim": partition_vocab_dim,
    }


def generate_with_watermark_balanced(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    secret_bitstream: List[int],
    device: str = "cpu",
    *,
    generation_extra: Dict[str, Any] | None = None,
) -> Dict:
    """Watermark by biasing each model step with a softmax-balanced vocab split.

    Unlike ``generate_with_watermark``, one secret bit is bound to each generated
    model token (not a retokenized recovery slot). Partitions are recomputed from
    the current next-token distribution so the two halves have near-equal mass.
    Use ``recover_bitstream_balanced`` with the same prompt and model token ids.
    """
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    inputs = encode_prompt_for_generation(tokenizer, prompt, device)
    input_ids_wm = inputs["input_ids"].clone()
    prompt_len = int(inputs["input_ids"].shape[1])
    attn_mask = inputs.get("attention_mask", None)

    n_bits = len(secret_bitstream)
    _, logits_processor = _prepare_sampling_logits_processor_bundle(
        model,
        tokenizer,
        inputs,
        decode_horizon=n_bits,
        generation_extra=generation_extra,
    )
    natural_partition_choices = 0
    partition_mass_gaps: List[float] = []
    partition_vocab_dim: int | None = None
    model_kwargs: Dict[str, Any] = {"use_cache": True}
    if attn_mask is not None:
        model_kwargs["attention_mask"] = attn_mask
    step_input_ids = input_ids_wm

    for bit_idx in range(n_bits):
        with torch.no_grad():
            outputs = model(input_ids=step_input_ids, **model_kwargs)
            logits_wm = outputs.logits[0, -1, :]
            d_logits = int(logits_wm.shape[-1])
            if partition_vocab_dim is None:
                partition_vocab_dim = d_logits
            elif partition_vocab_dim != d_logits:
                raise RuntimeError(
                    f"logits vocab width changed mid-decode ({d_logits} vs {partition_vocab_dim})"
                )
            model_kwargs["past_key_values"] = outputs.past_key_values

            base_probs = _scores_to_next_token_probs(
                logits_wm, input_ids_wm, logits_processor
            )
            mask_A = get_balanced_partition_from_probs(base_probs)
            mass_a, mass_b = balanced_partition_masses(base_probs, mask_A)
            partition_mass_gaps.append(abs(mass_a - mass_b))

            secret_bit = int(secret_bitstream[bit_idx])
            p = float(base_probs[mask_A].sum().item())
            q = min(p, 1.0 - p)
            choose_A, used_enforce = _choose_watermark_partition_half(
                secret_bit, p, q
            )
            probs_wm = _mask_disallowed_half_probs(
                base_probs, mask_A, choose_set_A=choose_A
            )
            if not used_enforce:
                natural_partition_choices += 1
            if torch.isfinite(probs_wm).all() and float(probs_wm.sum().item()) > 0:
                probs_wm = probs_wm.clamp(min=0)
                probs_wm = probs_wm / probs_wm.sum()
                next_token_id_wm = torch.multinomial(probs_wm, num_samples=1)
            else:
                next_token_id_wm = torch.multinomial(base_probs, num_samples=1)

        input_ids_wm = torch.cat([input_ids_wm, next_token_id_wm.unsqueeze(0)], dim=-1)
        step_input_ids = next_token_id_wm.unsqueeze(0)
        if "attention_mask" in model_kwargs:
            model_kwargs["attention_mask"] = torch.cat(
                [
                    model_kwargs["attention_mask"],
                    torch.ones((1, 1), dtype=model_kwargs["attention_mask"].dtype, device=device),
                ],
                dim=-1,
            )

    suffix_ids = input_ids_wm[0, prompt_len:].tolist()
    generated_text_wm = tokenizer.decode(suffix_ids, skip_special_tokens=True)
    recovery_input_ids = tokenizer(generated_text_wm, return_tensors="pt")["input_ids"]
    gap_mean = (
        sum(partition_mass_gaps) / len(partition_mass_gaps) if partition_mass_gaps else 0.0
    )

    return {
        "prompt_text": prompt,
        "generated_text_wm": generated_text_wm,
        "input_ids_wm": recovery_input_ids,
        "model_input_ids": input_ids_wm.detach().cpu(),
        "prompt_len": prompt_len,
        "model_suffix_ids": suffix_ids,
        "secret_bitstream": secret_bitstream,
        "sacrificed_bits": 0,
        "natural_partition_choices": natural_partition_choices,
        "recovery_stalls": 0,
        "retok_replacements": 0,
        "recovery_ids_aligned": True,
        "recovery_slots_committed": n_bits,
        "incremental_ids_aligned": True,
        "partition_mode": "balanced",
        "partition_vocab_dim": partition_vocab_dim,
        "partition_mass_gap_mean": gap_mean,
        "partition_mass_gap_max": max(partition_mass_gaps) if partition_mass_gaps else 0.0,
        "special_ids_in_suffix": sum(1 for t in suffix_ids if t in special_ids),
    }


def generate_baseline(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: str = "cpu",
) -> str:
    inputs = encode_prompt_for_generation(tokenizer, prompt, device)
    prompt_len = int(inputs["input_ids"].shape[1])
    pad_id = _tokenizer_pad_token_id(tokenizer)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            pad_token_id=pad_id,
        )
    gen_ids = out[0, prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def recover_bitstream(
    full_sequence_ids: List[int],
    vocab_size: int,
    device: str,
    special_ids: set,
    *,
    tokenizer: AutoTokenizer | None = None,
) -> Tuple[List[int], List[int]]:
    ensure_partition_weights(int(vocab_size), tokenizer)
    recovered_bits, recovered_tokens = [], []
    filtered_ids = [tid for tid in full_sequence_ids if tid not in special_ids]

    for bit_idx, actual_token_id in enumerate(filtered_ids):
        mask_A = get_vectorized_partition(vocab_size, device, bit_idx)
        if actual_token_id >= mask_A.shape[0]:
            recovered_bits.append(1)
        else:
            recovered_bits.append(0 if mask_A[actual_token_id].item() else 1)
        recovered_tokens.append(actual_token_id)

    return recovered_bits, recovered_tokens


def recover_bitstream_from_text(
    full_text: str,
    tokenizer: AutoTokenizer,
    device: str,
    *,
    model: torch.nn.Module | None = None,
    partition_vocab_size: int | None = None,
) -> Tuple[List[int], List[int]]:
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    enc = tokenizer(full_text, return_tensors="pt")
    vocab_size = resolve_partition_vocab_size_for_recovery(
        tokenizer, model=model, partition_vocab_size=partition_vocab_size
    )
    return recover_bitstream(
        full_sequence_ids=enc["input_ids"][0].tolist(),
        vocab_size=vocab_size,
        device=device,
        special_ids=special_ids,
        tokenizer=tokenizer,
    )


def recover_bitstream_balanced(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    model_suffix_ids: List[int],
    device: str = "cpu",
    *,
    generation_extra: Dict[str, Any] | None = None,
) -> Tuple[List[int], List[int], List[float]]:
    """
    Recover bits embedded by ``generate_with_watermark_balanced``.

    Replays the prompt + each prefix of ``model_suffix_ids``, rebuilds the
    softmax-balanced partition at every step, and reads which half the actual
    token landed in. Returns ``(bits, tokens, mass_gaps)``.
    """
    inputs = encode_prompt_for_generation(tokenizer, prompt, device)
    prompt_ids = inputs["input_ids"]
    prompt_len = int(prompt_ids.shape[1])
    attn_mask = inputs.get("attention_mask", None)

    n = len(model_suffix_ids)
    _, logits_processor = _prepare_sampling_logits_processor_bundle(
        model,
        tokenizer,
        inputs,
        decode_horizon=max(n, 1),
        generation_extra=generation_extra,
    )

    full_ids = torch.cat(
        [
            prompt_ids,
            torch.tensor([model_suffix_ids], dtype=prompt_ids.dtype, device=prompt_ids.device),
        ],
        dim=-1,
    )
    input_ids = prompt_ids.clone()
    model_kwargs: Dict[str, Any] = {"use_cache": True}
    if attn_mask is not None:
        model_kwargs["attention_mask"] = attn_mask
    step_input_ids = input_ids

    recovered_bits: List[int] = []
    recovered_tokens: List[int] = []
    mass_gaps: List[float] = []

    for bit_idx, token_id in enumerate(model_suffix_ids):
        with torch.no_grad():
            outputs = model(input_ids=step_input_ids, **model_kwargs)
            logits = outputs.logits[0, -1, :]
            model_kwargs["past_key_values"] = outputs.past_key_values
            base_probs = _scores_to_next_token_probs(
                logits, input_ids, logits_processor
            )
            mask_A = get_balanced_partition_from_probs(base_probs)
            mass_a, mass_b = balanced_partition_masses(base_probs, mask_A)
            mass_gaps.append(abs(mass_a - mass_b))

            tid = int(token_id)
            if tid < 0 or tid >= int(mask_A.numel()):
                recovered_bits.append(1)
            else:
                recovered_bits.append(0 if bool(mask_A[tid].item()) else 1)
            recovered_tokens.append(tid)

        next_tok = torch.tensor([[tid]], dtype=input_ids.dtype, device=device)
        input_ids = torch.cat([input_ids, next_tok], dim=-1)
        step_input_ids = next_tok
        if "attention_mask" in model_kwargs:
            model_kwargs["attention_mask"] = torch.cat(
                [
                    model_kwargs["attention_mask"],
                    torch.ones((1, 1), dtype=model_kwargs["attention_mask"].dtype, device=device),
                ],
                dim=-1,
            )

    # Sanity: reconstructed length matches prompt + suffix.
    if int(input_ids.shape[1]) != prompt_len + n:
        raise RuntimeError(
            f"balanced recovery length mismatch: got {int(input_ids.shape[1])}, "
            f"expected {prompt_len + n}"
        )
    if not torch.equal(input_ids, full_ids):
        raise RuntimeError("balanced recovery replay diverged from provided suffix ids")

    return recovered_bits, recovered_tokens, mass_gaps


def recover_bitstream_balanced_from_generation(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    generation_out: Dict[str, Any],
    device: str = "cpu",
    *,
    generation_extra: Dict[str, Any] | None = None,
) -> Tuple[List[int], List[int], List[float]]:
    """Recover using ``prompt_text`` / ``model_suffix_ids`` from a balanced generate call."""
    prompt = str(generation_out["prompt_text"])
    if "model_suffix_ids" in generation_out:
        suffix = [int(t) for t in generation_out["model_suffix_ids"]]
    else:
        model_ids = generation_out["model_input_ids"]
        if isinstance(model_ids, torch.Tensor):
            row = model_ids[0] if model_ids.dim() > 1 else model_ids
            prompt_len = int(generation_out["prompt_len"])
            suffix = [int(t) for t in row[prompt_len:].tolist()]
        else:
            raise KeyError("generation_out missing model_suffix_ids / model_input_ids")
    return recover_bitstream_balanced(
        model,
        tokenizer,
        prompt,
        suffix,
        device,
        generation_extra=generation_extra,
    )


def negative_control_transcript_like(
    reference_text: str,
    tokenizer: AutoTokenizer,
    device: str,
    *,
    n_bits: int,
    model: torch.nn.Module | None = None,
    phrase: str = "Unrelated decoy text used only as a negative control. ",
) -> str:
    """Build a decoy string with enough tokens/chars for a negative-control detect.

    Uses tokenizer length only (no model rewind). ``model`` is accepted for API
    compatibility with callers that already have a loaded LM.
    """
    del model, device
    ref_chars = max(len(reference_text), 1)
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    phrase_ids = tokenizer(phrase, add_special_tokens=False)["input_ids"]
    if phrase_ids and isinstance(phrase_ids[0], list):
        phrase_ids = [t for row in phrase_ids for t in row]
    phrase_ns = sum(1 for t in phrase_ids if t not in special_ids)
    if phrase_ns <= 0:
        raise ValueError("negative-control phrase must yield at least one non-special token")

    # Ceiling of needed phrases for bit count and character length.
    n_for_bits = (int(n_bits) + phrase_ns - 1) // phrase_ns
    n_for_chars = (ref_chars + len(phrase) - 1) // len(phrase)
    return phrase * max(n_for_bits, n_for_chars, 1)


__all__ = [
    "get_vectorized_partition",
    "get_balanced_partition_from_probs",
    "balanced_partition_masses",
    "configure_partition_weights",
    "clear_partition_weights",
    "get_partition_weights",
    "build_unigram_partition_weights",
    "ensure_partition_weights",
    "resolve_partition_vocab_size_for_recovery",
    "encode_prompt_for_generation",
    "generate_with_watermark",
    "generate_with_watermark_balanced",
    "generate_baseline",
    "recover_bitstream",
    "recover_bitstream_from_text",
    "recover_bitstream_balanced",
    "recover_bitstream_balanced_from_generation",
    "negative_control_transcript_like",
]
