from __future__ import annotations

import hashlib
import inspect
import os
import random
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList


def _balanced_orientation_flip(step_index: int) -> bool:
    """Deterministic pseudorandom orientation of the balanced mask at ``step_index``.

    The greedy split always assigns the heaviest token to set A first, so
    without this flip the heavier half is almost always A. On natural
    (non-enforce) steps the sampler takes the heavier half, which would bias
    recovered bits toward 0. Flipping the orientation pseudorandomly per step
    makes natural-step bits ~fair coins, and (unlike strict alternation) stays
    uncorrelated across the depth-layout replicas of one logical bit.
    """
    digest = hashlib.md5(f"balanced-flip:{int(step_index)}".encode()).digest()
    return bool(digest[0] & 1)


def get_balanced_partition_from_probs(
    probs: torch.Tensor, step_index: int = 0
) -> torch.BoolTensor:
    """
    Split positive-mass tokens into two sets with near-equal probability mass.

    Deterministic given ``probs`` and ``step_index``: positive-mass tokens are
    processed by descending probability (ties broken by token id), and each
    token is assigned to the currently lighter set. Zero-mass tokens get a
    fixed parity assignment because they cannot affect the probability balance.
    The A/B orientation is then flipped pseudorandomly per ``step_index`` (see
    ``_balanced_orientation_flip``); generation and recovery must pass the same
    ``step_index`` to reconstruct the identical mask.
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

    flip = _balanced_orientation_flip(step_index)
    ids = torch.arange(v, device=device)
    mask = (ids % 2) == 0

    positive = (p > 0).nonzero(as_tuple=False).squeeze(1)
    if positive.numel() == 0:
        return ~mask if flip else mask

    p_pos = p[positive]
    # ``positive`` is already ascending token id; stable descending sort keeps
    # smaller ids first on exact probability ties.
    order = torch.argsort(-p_pos, stable=True)
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
    return ~mask if flip else mask


def balanced_partition_masses(probs: torch.Tensor, mask_A: torch.BoolTensor) -> Tuple[float, float]:
    """Return ``(mass_A, mass_B)`` for ``probs`` under ``mask_A``."""
    p = probs.detach().to(dtype=torch.float32)
    total = float(p.sum().item())
    if total <= 0.0:
        return 0.0, 0.0
    p = p / total
    mass_a = float(p[mask_A].sum().item())
    return mass_a, 1.0 - mass_a


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


def encode_prompts_for_generation(
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    device: str | torch.device,
) -> Dict[str, torch.Tensor]:
    """Left-pad a batch of prompts for causal LM ``generate`` (same templates as single)."""
    if not prompts:
        raise ValueError("prompts must be non-empty")
    dev = torch.device(device) if isinstance(device, str) else device
    pad_id = _tokenizer_pad_token_id(tokenizer)
    rows: List[torch.Tensor] = []
    for prompt in prompts:
        single = encode_prompt_for_generation(tokenizer, prompt, "cpu")
        rows.append(single["input_ids"].squeeze(0).to(dtype=torch.long))
    max_len = max(int(r.shape[0]) for r in rows)
    input_rows: List[torch.Tensor] = []
    attn_rows: List[torch.Tensor] = []
    for row in rows:
        pad = max_len - int(row.shape[0])
        if pad > 0:
            input_rows.append(
                torch.cat([torch.full((pad,), pad_id, dtype=torch.long), row], dim=0)
            )
            attn_rows.append(
                torch.cat(
                    [torch.zeros(pad, dtype=torch.long), torch.ones(row.shape[0], dtype=torch.long)],
                    dim=0,
                )
            )
        else:
            input_rows.append(row)
            attn_rows.append(torch.ones(row.shape[0], dtype=torch.long))
    return {
        "input_ids": torch.stack(input_rows, dim=0).to(dev),
        "attention_mask": torch.stack(attn_rows, dim=0).to(dev),
    }


def suggest_baseline_batch_size(
    n_prompts: int,
    *,
    max_new_tokens: int,
    max_input_tokens: int,
    reserved_bytes: int = 512 * 1024 * 1024,
) -> int:
    """Heuristic max batch size from free VRAM (overridden by env / OOM backoff)."""
    if n_prompts <= 0:
        return 1
    env = os.environ.get("BENCHMARK_BASELINE_BATCH_SIZE", "").strip()
    if env:
        return max(1, min(n_prompts, int(env)))
    if not torch.cuda.is_available():
        return 1
    free_bytes, _total = torch.cuda.mem_get_info()
    usable = max(int(free_bytes) - int(reserved_bytes), 0)
    seq = max(int(max_input_tokens), 1) + max(int(max_new_tokens), 1)
    # Conservative per-sequence budget for 1B–3B fp16 generate (KV + activations).
    bytes_per_seq = seq * 48 * 1024
    if bytes_per_seq <= 0:
        return min(8, n_prompts)
    return max(1, min(n_prompts, usable // bytes_per_seq))


def _count_new_tokens(gen_ids: torch.Tensor, pad_id: int, *, batched: bool) -> int:
    """Count newly generated token ids (strip trailing pad fill in batched decode)."""
    if gen_ids.numel() == 0:
        return 0
    if not batched:
        return int(gen_ids.numel())
    ids = gen_ids.detach().cpu().tolist()
    end = len(ids)
    while end > 0 and int(ids[end - 1]) == int(pad_id):
        end -= 1
    return int(end)


def _generate_baseline_batch(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    max_new_tokens: int,
    device: str | torch.device,
) -> List[Tuple[str, int]]:
    inputs = encode_prompts_for_generation(tokenizer, prompts, device)
    prompt_width = int(inputs["input_ids"].shape[1])
    pad_id = _tokenizer_pad_token_id(tokenizer)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            pad_token_id=pad_id,
        )
    rows: List[Tuple[str, int]] = []
    for i in range(out.shape[0]):
        gen_ids = out[i, prompt_width:]
        n_tok = _count_new_tokens(gen_ids, pad_id, batched=True)
        rows.append(
            (tokenizer.decode(gen_ids, skip_special_tokens=True), n_tok)
        )
    return rows


def generate_baselines(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    max_new_tokens: int,
    device: str = "cpu",
    *,
    batch_size: int | None = None,
    on_batch_done: Any | None = None,
) -> Tuple[List[str], List[int]]:
    """Generate baseline texts for many prompts with adaptive GPU batching.

    Returns ``(texts, n_new_tokens)`` parallel lists. ``on_batch_done(n)`` is
    invoked after each successful micro-batch (for progress bars). On CUDA OOM
    the batch size is halved and the micro-batch is retried.
    """
    prompt_list = [str(p) for p in prompts]
    n = len(prompt_list)
    if n == 0:
        return [], []
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")

    max_input = 1
    for p in prompt_list:
        enc = encode_prompt_for_generation(tokenizer, p, "cpu")
        max_input = max(max_input, int(enc["input_ids"].shape[-1]))

    if batch_size is None:
        bs = suggest_baseline_batch_size(
            n,
            max_new_tokens=max_new_tokens,
            max_input_tokens=max_input,
        )
    else:
        bs = max(1, min(n, int(batch_size)))

    out_texts: List[str] = [""] * n
    out_n_tokens: List[int] = [0] * n
    i = 0
    while i < n:
        take = min(bs, n - i)
        while True:
            try:
                chunk = _generate_baseline_batch(
                    model,
                    tokenizer,
                    prompt_list[i : i + take],
                    max_new_tokens,
                    device,
                )
                break
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if take == 1:
                    raise
                take = max(1, take // 2)
                bs = take
        for j, (text, n_tok) in enumerate(chunk):
            out_texts[i + j] = text
            out_n_tokens[i + j] = int(n_tok)
        if on_batch_done is not None:
            on_batch_done(take)
        i += take
    return out_texts, out_n_tokens


def generate_baseline(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: str = "cpu",
) -> Tuple[str, int]:
    """Single-prompt baseline. Returns ``(text, n_new_tokens)``."""
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
    n_tok = _count_new_tokens(gen_ids, pad_id, batched=False)
    return tokenizer.decode(gen_ids, skip_special_tokens=True), n_tok


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


def _seed_token_id(tokenizer: AutoTokenizer) -> int:
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None:
        return int(bos)
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        return int(eos)
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is not None:
        return int(pad)
    raise ValueError("tokenizer needs bos_token_id, eos_token_id, or pad_token_id")


def _append_generated_token(
    *,
    input_ids: torch.Tensor,
    step_input_ids: torch.Tensor,
    next_token_id: torch.Tensor,
    model_kwargs: Dict[str, Any],
    device: str | torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)
    step_input_ids = next_token_id.unsqueeze(0)
    if "attention_mask" in model_kwargs:
        model_kwargs["attention_mask"] = torch.cat(
            [
                model_kwargs["attention_mask"],
                torch.ones(
                    (1, 1),
                    dtype=model_kwargs["attention_mask"].dtype,
                    device=device,
                ),
            ],
            dim=-1,
        )
    return input_ids, step_input_ids


def _tokenize_text_fragment(tokenizer: AutoTokenizer, text: str) -> List[int]:
    """Tokenize a text fragment without specials (for cascade-isolated prefix/suffix)."""
    enc = tokenizer(text, add_special_tokens=False)
    ids = enc["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = [t for row in ids for t in row]
    return [int(t) for t in ids]


def _fit_text_to_token_count(
    tokenizer: AutoTokenizer,
    text: str,
    n_tokens: int,
) -> Tuple[str, List[int]]:
    """Return a prefix of ``text`` whose retokenization has exactly ``n_tokens`` ids.

    Uses a char-prefix search on ``text``, then stabilizes decode↔encode drift.
    Raises ``ValueError`` if ``text`` cannot cover ``n_tokens`` tokens.
    """
    if n_tokens < 0:
        raise ValueError(f"n_tokens must be >= 0, got {n_tokens}")
    if n_tokens == 0:
        return "", []

    full_ids = _tokenize_text_fragment(tokenizer, text)
    if len(full_ids) < n_tokens:
        raise ValueError(
            f"text retokenizes to {len(full_ids)} tokens, need at least {n_tokens}"
        )

    # Longest char prefix with <= n_tokens, then grow to exactly n_tokens.
    lo, hi = 0, len(text)
    best_len = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        k = len(_tokenize_text_fragment(tokenizer, text[:mid]))
        if k <= n_tokens:
            best_len = mid
            lo = mid + 1
        else:
            hi = mid - 1

    prefix = text[:best_len]
    while (
        len(prefix) < len(text)
        and len(_tokenize_text_fragment(tokenizer, prefix)) < n_tokens
    ):
        prefix = text[: len(prefix) + 1]

    got = _tokenize_text_fragment(tokenizer, prefix)
    if len(got) != n_tokens:
        # Fallback: decode the first n source tokens and re-stabilize.
        prefix = tokenizer.decode(full_ids[:n_tokens], skip_special_tokens=True)
        got = _tokenize_text_fragment(tokenizer, prefix)

    # Decode/encode can drift; iterate toward a fixed point with exact length.
    for _ in range(16):
        if len(got) == n_tokens:
            roundtrip = tokenizer.decode(got, skip_special_tokens=True)
            got2 = _tokenize_text_fragment(tokenizer, roundtrip)
            if got2 == got:
                return roundtrip, got
            if len(got2) == n_tokens:
                got = got2
                continue
        if len(got) > n_tokens:
            got = got[:n_tokens]
            prefix = tokenizer.decode(got, skip_special_tokens=True)
            got = _tokenize_text_fragment(tokenizer, prefix)
        elif len(got) < n_tokens:
            need = n_tokens - len(got)
            # Extend from full_ids beyond what we have.
            merged = got + full_ids[len(got) : len(got) + need]
            if len(merged) < n_tokens:
                raise ValueError(
                    f"cannot stabilize burn-in to {n_tokens} tokens "
                    f"(stuck at {len(got)})"
                )
            got = merged[:n_tokens]
            prefix = tokenizer.decode(got, skip_special_tokens=True)
            got = _tokenize_text_fragment(tokenizer, prefix)
        else:
            break

    got = _tokenize_text_fragment(tokenizer, prefix)
    if len(got) != n_tokens:
        raise ValueError(
            f"burn-in stabilize failed: retok length {len(got)} != {n_tokens}"
        )
    return prefix, got


def _sample_next_token_id(
    model: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    step_input_ids: torch.Tensor,
    model_kwargs: Dict[str, Any],
    logits_processor: LogitsProcessorList,
    device: str | torch.device,
    partition_vocab_dim: int | None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """One unwatermarked multinomial step. Returns ids, step_ids, token, vocab_dim."""
    with torch.no_grad():
        outputs = model(input_ids=step_input_ids, **model_kwargs)
        logits = outputs.logits[0, -1, :]
        d_logits = int(logits.shape[-1])
        if partition_vocab_dim is None:
            partition_vocab_dim = d_logits
        elif partition_vocab_dim != d_logits:
            raise RuntimeError(
                f"logits vocab width changed mid-decode ({d_logits} vs {partition_vocab_dim})"
            )
        model_kwargs["past_key_values"] = outputs.past_key_values
        sample_probs = _scores_to_next_token_probs(logits, input_ids, logits_processor)
        next_token_id = torch.multinomial(sample_probs, num_samples=1)
    input_ids, step_input_ids = _append_generated_token(
        input_ids=input_ids,
        step_input_ids=step_input_ids,
        next_token_id=next_token_id,
        model_kwargs=model_kwargs,
        device=device,
    )
    return input_ids, step_input_ids, next_token_id, int(partition_vocab_dim)


def _init_prompt_free_decode_state(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: str | torch.device,
    *,
    decode_horizon: int,
    generation_extra: Dict[str, Any] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], LogitsProcessorList]:
    """BOS-only decode state matching ``recover_bitstream`` (no user prompt)."""
    seed = _seed_token_id(tokenizer)
    dev = torch.device(device) if isinstance(device, str) else device
    seed_inputs = {
        "input_ids": torch.tensor([[seed]], dtype=torch.long, device=dev),
        "attention_mask": torch.ones((1, 1), dtype=torch.long, device=dev),
    }
    _, logits_processor = _prepare_sampling_logits_processor_bundle(
        model,
        tokenizer,
        seed_inputs,
        decode_horizon=max(int(decode_horizon), 1),
        generation_extra=generation_extra,
    )
    input_ids = seed_inputs["input_ids"].clone()
    model_kwargs: Dict[str, Any] = {
        "use_cache": True,
        "attention_mask": seed_inputs["attention_mask"].clone(),
    }
    return input_ids, input_ids.clone(), model_kwargs, logits_processor


def _prefill_prompt_free_tokens(
    model: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    model_kwargs: Dict[str, Any],
    token_ids: Sequence[int],
    device: str | torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], torch.Tensor]:
    """Teacher-force known tokens in one forward (burn-in prefill).

    ``input_ids`` is the current prompt-free prefix (usually ``[[BOS]]``) with
    no ``past_key_values`` yet. Returns updated ids / kwargs whose KV cache
    covers ``prefix + token_ids``, plus ``logits`` at the last position
    (``P(next | prefix + token_ids)``).
    """
    if not token_ids:
        raise ValueError("token_ids must be non-empty for prefill")
    if model_kwargs.get("past_key_values") is not None:
        raise ValueError("prefill expects a cache-free prefix (no past_key_values)")

    dev = torch.device(device) if isinstance(device, str) else device
    extra = torch.tensor([list(token_ids)], dtype=input_ids.dtype, device=dev)
    full_ids = torch.cat([input_ids, extra], dim=-1)
    attn = torch.ones(
        (1, full_ids.shape[1]),
        dtype=(
            model_kwargs["attention_mask"].dtype
            if model_kwargs.get("attention_mask") is not None
            else torch.long
        ),
        device=dev,
    )
    with torch.no_grad():
        outputs = model(input_ids=full_ids, attention_mask=attn, use_cache=True)
    new_kwargs: Dict[str, Any] = {
        "use_cache": True,
        "attention_mask": attn,
        "past_key_values": outputs.past_key_values,
    }
    logits_last = outputs.logits[0, -1, :]
    step_input_ids = full_ids[:, -1:]
    return full_ids, step_input_ids, new_kwargs, logits_last


def _warm_prompt_free_with_burn_in(
    model: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    step_input_ids: torch.Tensor,
    model_kwargs: Dict[str, Any],
    burn_in_ids: Sequence[int],
    device: str | torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Advance prompt-free state through known burn-in tokens via prefill.

    Matches the end state of the old token-by-token warmup: KV cache covers
    ``BOS + burn_in[:-1]`` and ``step_input_ids`` is the last burn-in token
    (consumed on the next watermark forward). Empty ``burn_in_ids`` is a no-op.
    """
    if not burn_in_ids:
        return input_ids, step_input_ids, model_kwargs

    prefix = list(burn_in_ids[:-1])
    last = int(burn_in_ids[-1])
    if prefix:
        input_ids, step_input_ids, model_kwargs, _ = _prefill_prompt_free_tokens(
            model,
            input_ids=input_ids,
            model_kwargs=model_kwargs,
            token_ids=prefix,
            device=device,
        )
    else:
        with torch.no_grad():
            outputs = model(input_ids=step_input_ids, **model_kwargs)
            model_kwargs = dict(model_kwargs)
            model_kwargs["past_key_values"] = outputs.past_key_values

    next_tok = torch.tensor([last], dtype=input_ids.dtype, device=input_ids.device)
    input_ids, step_input_ids = _append_generated_token(
        input_ids=input_ids,
        step_input_ids=step_input_ids,
        next_token_id=next_tok,
        model_kwargs=model_kwargs,
        device=device,
    )
    return input_ids, step_input_ids, model_kwargs


def generate_with_watermark(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    secret_bitstream: List[int],
    device: str = "cpu",
    *,
    burn_in_tokens: int = 100,
    generation_extra: Dict[str, Any] | None = None,
) -> Dict:
    """Watermark after a free burn-in prefix using softmax-balanced vocab splits.

    The first ``burn_in_tokens`` **retokenized** prefix tokens are free (no
    watermark). Raw model burn-in may be longer/shorter after decode↔encode; we
    grow free sampling until the decoded prefix covers at least
    ``burn_in_tokens`` retokenized ids, then publish a stabilized prefix whose
    retokenization has **exactly** that length. Prompt-free partitions are warmed
    on those retokenized ids (matching integrity recovery). Channel bit indices
    are offset by ``len(retok_prefix) - burn_in_tokens`` (normally 0) so recovery
    that skips the first ``burn_in_tokens`` tokens of the full retokenized
    transcript lines up with generation.

    During the watermarked phase, two LM contexts run in parallel:

    - **Sampling state** (prompt-conditioned): provides next-token probabilities
      used for soft watermarking and multinomial sampling.
    - **Partition state** (prompt-free, BOS + retokenized burn-in + wm tokens):
      provides the distribution used to build balanced masks — the same context
      recovery uses.
    """
    if burn_in_tokens < 0:
        raise ValueError(f"burn_in_tokens must be >= 0, got {burn_in_tokens}")

    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    inputs = encode_prompt_for_generation(tokenizer, prompt, device)
    input_ids_wm = inputs["input_ids"].clone()
    prompt_len = int(inputs["input_ids"].shape[1])
    attn_mask = inputs.get("attention_mask", None)

    n_bits = len(secret_bitstream)
    # Horizon: raw burn-in may exceed scheme length when retok shrinks.
    decode_horizon = max(int(burn_in_tokens) + n_bits + 64, 1)
    _, logits_processor_sample = _prepare_sampling_logits_processor_bundle(
        model,
        tokenizer,
        inputs,
        decode_horizon=decode_horizon,
        generation_extra=generation_extra,
    )
    natural_partition_choices = 0
    partition_mass_gaps: List[float] = []
    partition_vocab_dim: int | None = None
    model_kwargs_sample: Dict[str, Any] = {"use_cache": True}
    if attn_mask is not None:
        model_kwargs_sample["attention_mask"] = attn_mask
    step_input_ids_sample = input_ids_wm

    # --- Phase 1: free burn-in until retokenized prefix covers scheme length ---
    max_burn_raw = max(int(burn_in_tokens) * 4, int(burn_in_tokens) + 64)
    while True:
        burn_in_ids = input_ids_wm[0, prompt_len:].tolist()
        prefix_raw_text = tokenizer.decode(burn_in_ids, skip_special_tokens=True)
        retok_len = len(_tokenize_text_fragment(tokenizer, prefix_raw_text))
        if retok_len >= int(burn_in_tokens):
            break
        if len(burn_in_ids) >= max_burn_raw:
            raise RuntimeError(
                f"burn-in retokenization stayed at {retok_len} tokens after "
                f"{len(burn_in_ids)} raw tokens (need {burn_in_tokens})"
            )
        (
            input_ids_wm,
            step_input_ids_sample,
            _tid,
            partition_vocab_dim,
        ) = _sample_next_token_id(
            model,
            input_ids=input_ids_wm,
            step_input_ids=step_input_ids_sample,
            model_kwargs=model_kwargs_sample,
            logits_processor=logits_processor_sample,
            device=device,
            partition_vocab_dim=partition_vocab_dim,
        )

    burn_in_ids = input_ids_wm[0, prompt_len:].tolist()
    prefix_raw_text = tokenizer.decode(burn_in_ids, skip_special_tokens=True)
    if int(burn_in_tokens) == 0:
        prefix_text = ""
        retok_burn_ids: List[int] = []
        retok_replacements = 0
    else:
        prefix_text, retok_burn_ids = _fit_text_to_token_count(
            tokenizer, prefix_raw_text, int(burn_in_tokens)
        )
        # Count how far raw retok drifted from the scheme cut.
        raw_retok = _tokenize_text_fragment(tokenizer, prefix_raw_text)
        retok_replacements = abs(len(raw_retok) - int(burn_in_tokens)) + (
            0 if raw_retok[: int(burn_in_tokens)] == retok_burn_ids else 1
        )

    burn_in_char_len = len(prefix_text)
    # Integrity recovery skips the first burn_in_tokens of the full retokenized
    # transcript. If stabilize left a residual length delta, shift bit indices.
    bit_index_offset = len(retok_burn_ids) - int(burn_in_tokens)

    # Warm the prompt-free partition state on the *retokenized* burn-in (recovery).
    (
        input_ids_part,
        step_input_ids_part,
        model_kwargs_part,
        logits_processor_part,
    ) = _init_prompt_free_decode_state(
        model,
        tokenizer,
        device,
        decode_horizon=max(len(retok_burn_ids) + n_bits, 1),
        generation_extra=generation_extra,
    )
    input_ids_part, step_input_ids_part, model_kwargs_part = _warm_prompt_free_with_burn_in(
        model,
        input_ids=input_ids_part,
        step_input_ids=step_input_ids_part,
        model_kwargs=model_kwargs_part,
        burn_in_ids=retok_burn_ids,
        device=device,
    )

    # --- Phase 2: watermarked tokens (dual forward) ---
    for bit_idx in range(n_bits):
        partition_step = int(bit_idx) + int(bit_index_offset)
        with torch.no_grad():
            outputs_sample = model(
                input_ids=step_input_ids_sample, **model_kwargs_sample
            )
            logits_sample = outputs_sample.logits[0, -1, :]
            d_logits = int(logits_sample.shape[-1])
            if partition_vocab_dim is None:
                partition_vocab_dim = d_logits
            elif partition_vocab_dim != d_logits:
                raise RuntimeError(
                    f"logits vocab width changed mid-decode ({d_logits} vs {partition_vocab_dim})"
                )
            model_kwargs_sample["past_key_values"] = outputs_sample.past_key_values
            sample_probs = _scores_to_next_token_probs(
                logits_sample, input_ids_wm, logits_processor_sample
            )

            outputs_part = model(input_ids=step_input_ids_part, **model_kwargs_part)
            logits_part = outputs_part.logits[0, -1, :]
            if int(logits_part.shape[-1]) != partition_vocab_dim:
                raise RuntimeError(
                    f"partition logits vocab width {int(logits_part.shape[-1])} "
                    f"!= sample width {partition_vocab_dim}"
                )
            model_kwargs_part["past_key_values"] = outputs_part.past_key_values
            partition_probs = _scores_to_next_token_probs(
                logits_part, input_ids_part, logits_processor_part
            )

            # Masks must match recovery: built from prompt-free probs only.
            mask_A = get_balanced_partition_from_probs(partition_probs, partition_step)
            mass_a, mass_b = balanced_partition_masses(partition_probs, mask_A)
            partition_mass_gaps.append(abs(mass_a - mass_b))

            # Soft watermark / sampling still uses prompt-conditioned masses.
            secret_bit = int(secret_bitstream[bit_idx])
            p = float(sample_probs[mask_A].sum().item())
            q = min(p, 1.0 - p)
            choose_A, used_enforce = _choose_watermark_partition_half(
                secret_bit, p, q
            )
            probs_wm = _mask_disallowed_half_probs(
                sample_probs, mask_A, choose_set_A=choose_A
            )
            if not used_enforce:
                natural_partition_choices += 1
            if torch.isfinite(probs_wm).all() and float(probs_wm.sum().item()) > 0:
                probs_wm = probs_wm.clamp(min=0)
                probs_wm = probs_wm / probs_wm.sum()
                next_token_id_wm = torch.multinomial(probs_wm, num_samples=1)
            else:
                next_token_id_wm = torch.multinomial(sample_probs, num_samples=1)

        input_ids_wm, step_input_ids_sample = _append_generated_token(
            input_ids=input_ids_wm,
            step_input_ids=step_input_ids_sample,
            next_token_id=next_token_id_wm,
            model_kwargs=model_kwargs_sample,
            device=device,
        )
        input_ids_part, step_input_ids_part = _append_generated_token(
            input_ids=input_ids_part,
            step_input_ids=step_input_ids_part,
            next_token_id=next_token_id_wm,
            model_kwargs=model_kwargs_part,
            device=device,
        )

    wm_ids = input_ids_wm[0, prompt_len + len(burn_in_ids) :].tolist()
    wm_token_texts = [tokenizer.decode([tid], skip_special_tokens=True) for tid in wm_ids]
    wm_token_char_lens = [len(piece) for piece in wm_token_texts]
    wm_text = "".join(wm_token_texts)
    # Publish stabilized burn-in (exact scheme token length) + wm payload text.
    generated_text_wm = prefix_text + wm_text
    recovery_input_ids = tokenizer(generated_text_wm, return_tensors="pt")["input_ids"]
    full_retok = _tokenize_text_fragment(tokenizer, generated_text_wm)
    recovery_ids_aligned = full_retok[: int(burn_in_tokens)] == retok_burn_ids
    gap_mean = (
        sum(partition_mass_gaps) / len(partition_mass_gaps) if partition_mass_gaps else 0.0
    )
    suffix_ids = list(retok_burn_ids) + list(wm_ids)

    return {
        "prompt_text": prompt,
        "generated_text_wm": generated_text_wm,
        "burn_in_text": prefix_text,
        "wm_text": wm_text,
        "burn_in_tokens": int(burn_in_tokens),
        "burn_in_char_len": int(burn_in_char_len),
        "burn_in_ids": list(burn_in_ids),
        "burn_in_retok_ids": list(retok_burn_ids),
        "burn_in_bit_index_offset": int(bit_index_offset),
        "wm_suffix_ids": list(wm_ids),
        "wm_token_char_lens": list(wm_token_char_lens),
        "input_ids_wm": recovery_input_ids,
        "model_input_ids": input_ids_wm.detach().cpu(),
        "prompt_len": prompt_len,
        "model_suffix_ids": suffix_ids,
        "secret_bitstream": secret_bitstream,
        "natural_partition_choices": natural_partition_choices,
        "retok_replacements": int(retok_replacements),
        "recovery_ids_aligned": bool(recovery_ids_aligned),
        "recovery_slots_committed": n_bits,
        "partition_vocab_dim": partition_vocab_dim,
        "partition_mass_gap_mean": gap_mean,
        "partition_mass_gap_max": max(partition_mass_gaps) if partition_mass_gaps else 0.0,
        "special_ids_in_suffix": sum(1 for t in suffix_ids if t in special_ids),
    }


def recover_bitstream(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    token_ids: List[int],
    device: str = "cpu",
    *,
    burn_in_tokens: int = 100,
    generation_extra: Dict[str, Any] | None = None,
) -> Tuple[List[int], List[int], List[float]]:
    """
    Prompt-free recovery of bits embedded by ``generate_with_watermark``.

    Teacher-forces ``token_ids`` from a BOS-only context in **one** forward
    (full prefill), then rebuilds balanced partitions for tokens after
    ``burn_in_tokens`` with ``step_index = 0, 1, …``. Returns
    ``(bits, tokens, mass_gaps)``.
    """
    results = recover_bitstreams_batched(
        model,
        tokenizer,
        [token_ids],
        device,
        burn_in_tokens=burn_in_tokens,
        generation_extra=generation_extra,
    )
    return results[0]


def recover_bitstreams_batched(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    token_id_seqs: Sequence[Sequence[int]],
    device: str = "cpu",
    *,
    burn_in_tokens: int = 100,
    generation_extra: Dict[str, Any] | None = None,
    batch_size: int | None = None,
    on_batch_done: Any | None = None,
) -> List[Tuple[List[int], List[int], List[float]]]:
    """
    Batched prompt-free recovery via full-prefill teacher forcing.

    Each sequence is independent. Sequences may differ in length; the batch is
    left-padded. ``burn_in_tokens`` is the scheme warm-up length (same for all).
    """
    if burn_in_tokens < 0:
        raise ValueError(f"burn_in_tokens must be >= 0, got {burn_in_tokens}")

    seqs = [[int(t) for t in seq] for seq in token_id_seqs]
    n = len(seqs)
    if n == 0:
        return []

    if batch_size is None:
        max_len = max((len(s) for s in seqs), default=1) + 1  # +BOS
        bs = suggest_baseline_batch_size(
            n,
            max_new_tokens=0,
            max_input_tokens=max_len,
        )
    else:
        bs = max(1, min(n, int(batch_size)))

    out: List[Tuple[List[int], List[int], List[float]]] = [
        ([], [], []) for _ in range(n)
    ]
    i = 0
    while i < n:
        take = min(bs, n - i)
        while True:
            try:
                chunk = _recover_bitstreams_batch(
                    model,
                    tokenizer,
                    seqs[i : i + take],
                    device,
                    burn_in_tokens=burn_in_tokens,
                    generation_extra=generation_extra,
                )
                break
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if take == 1:
                    raise
                take = max(1, take // 2)
                bs = take
        out[i : i + take] = chunk
        if on_batch_done is not None:
            on_batch_done(take)
        i += take
    return out


def _recover_bitstreams_batch(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    seqs: List[List[int]],
    device: str | torch.device,
    *,
    burn_in_tokens: int,
    generation_extra: Dict[str, Any] | None,
) -> List[Tuple[List[int], List[int], List[float]]]:
    """One micro-batch full-prefill recover (left-padded)."""
    dev = torch.device(device) if isinstance(device, str) else device
    bos = _seed_token_id(tokenizer)
    pad_id = _tokenizer_pad_token_id(tokenizer)
    burn = int(burn_in_tokens)

    # Empty / burn-only sequences: no payload bits.
    results: List[Tuple[List[int], List[int], List[float]] | None] = [None] * len(seqs)
    active_idx: List[int] = []
    active_seqs: List[List[int]] = []
    for i, ids in enumerate(seqs):
        if min(burn, len(ids)) >= len(ids):
            results[i] = ([], [], [])
        else:
            active_idx.append(i)
            active_seqs.append(ids)

    if not active_seqs:
        return [r if r is not None else ([], [], []) for r in results]

    # full_row = [PAD…] + [BOS] + ids
    full_rows: List[List[int]] = [[bos] + ids for ids in active_seqs]
    max_len = max(len(r) for r in full_rows)
    input_rows: List[torch.Tensor] = []
    attn_rows: List[torch.Tensor] = []
    pad_lens: List[int] = []
    for row in full_rows:
        pad = max_len - len(row)
        pad_lens.append(pad)
        if pad > 0:
            input_rows.append(
                torch.tensor([pad_id] * pad + row, dtype=torch.long, device=dev)
            )
            attn_rows.append(
                torch.tensor([0] * pad + [1] * len(row), dtype=torch.long, device=dev)
            )
        else:
            input_rows.append(torch.tensor(row, dtype=torch.long, device=dev))
            attn_rows.append(torch.ones(len(row), dtype=torch.long, device=dev))

    input_ids = torch.stack(input_rows, dim=0)
    attention_mask = torch.stack(attn_rows, dim=0)

    # Logits processor from a BOS-only template (sampling config only).
    seed_inputs = {
        "input_ids": torch.tensor([[bos]], dtype=torch.long, device=dev),
        "attention_mask": torch.ones((1, 1), dtype=torch.long, device=dev),
    }
    _, logits_processor = _prepare_sampling_logits_processor_bundle(
        model,
        tokenizer,
        seed_inputs,
        decode_horizon=max(max_len, 1),
        generation_extra=generation_extra,
    )

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        all_logits = outputs.logits  # [B, T, V]

    for local_i, (ids, pad) in enumerate(zip(active_seqs, pad_lens)):
        n = len(ids)
        b = min(burn, n)
        recovered_bits: List[int] = []
        recovered_tokens: List[int] = []
        mass_gaps: List[float] = []
        for t in range(b, n):
            # logits index after seeing BOS + ids[:t] in the unpadded full row.
            # full = [BOS]+ids; position t is BOS+ids[:t] (length t+1); index = t.
            # With left pad: index = pad + t.
            logit_idx = pad + t
            logits_t = all_logits[local_i, logit_idx, :]
            prefix = torch.tensor([[bos] + ids[:t]], dtype=torch.long, device=dev)
            base_probs = _scores_to_next_token_probs(logits_t, prefix, logits_processor)
            tid = int(ids[t])
            bit_idx = t - b
            mask_A = get_balanced_partition_from_probs(base_probs, bit_idx)
            mass_a, mass_b = balanced_partition_masses(base_probs, mask_A)
            mass_gaps.append(abs(mass_a - mass_b))
            if tid < 0 or tid >= int(mask_A.numel()):
                recovered_bits.append(1)
            else:
                recovered_bits.append(0 if bool(mask_A[tid].item()) else 1)
            recovered_tokens.append(tid)
        results[active_idx[local_i]] = (recovered_bits, recovered_tokens, mass_gaps)

    return [r if r is not None else ([], [], []) for r in results]


def recover_bitstream_from_watermarked_text(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    text: str,
    device: str = "cpu",
    *,
    burn_in_tokens: int,
    n_channel_bits: int | None = None,
    generation_extra: Dict[str, Any] | None = None,
) -> Tuple[List[int], List[int], List[float]]:
    """Integrity-preserving recovery: watermarked text + scheme params only.

    Tokenizes ``text`` with no specials, treats the first ``burn_in_tokens`` as
    warm-up, and extracts channel bits from the remainder (optionally truncated
    to ``n_channel_bits``). Does **not** use generation metadata (char splits,
    token ids, etc.).
    """
    bits_list = recover_bitstreams_from_watermarked_texts(
        model,
        tokenizer,
        [text],
        device,
        burn_in_tokens=burn_in_tokens,
        n_channel_bits=n_channel_bits,
        generation_extra=generation_extra,
    )
    return bits_list[0]


def recover_bitstreams_from_watermarked_texts(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    texts: Sequence[str],
    device: str = "cpu",
    *,
    burn_in_tokens: int,
    n_channel_bits: int | None = None,
    generation_extra: Dict[str, Any] | None = None,
    batch_size: int | None = None,
    on_batch_done: Any | None = None,
) -> List[Tuple[List[int], List[int], List[float]]]:
    """Batched integrity-preserving recovery from watermarked texts only."""
    if burn_in_tokens < 0:
        raise ValueError(f"burn_in_tokens must be >= 0, got {burn_in_tokens}")
    token_seqs: List[List[int]] = []
    for text in texts:
        ids = _tokenize_text_fragment(tokenizer, text)
        if n_channel_bits is not None:
            need = int(burn_in_tokens) + int(n_channel_bits)
            ids = ids[:need]
        token_seqs.append(ids)
    return recover_bitstreams_batched(
        model,
        tokenizer,
        token_seqs,
        device,
        burn_in_tokens=burn_in_tokens,
        generation_extra=generation_extra,
        batch_size=batch_size,
        on_batch_done=on_batch_done,
    )


def recover_bitstream_from_text(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    text: str,
    device: str = "cpu",
    *,
    burn_in_char_len: int,
    wm_token_char_lens: List[int] | None = None,
    generation_extra: Dict[str, Any] | None = None,
) -> Tuple[List[int], List[int], List[float]]:
    """Prompt-free recovery using the cascade-isolated character split.

    Splits ``text`` at ``burn_in_char_len``, tokenizes prefix and watermarked
    suffix independently, then rewinds. If per-token payload character lengths
    are available, each payload token text is retokenized independently so a
    local mismatch cannot shift later channel bits.
    """
    if burn_in_char_len < 0:
        raise ValueError(f"burn_in_char_len must be >= 0, got {burn_in_char_len}")
    if burn_in_char_len > len(text):
        raise ValueError(
            f"burn_in_char_len {burn_in_char_len} exceeds text length {len(text)}"
        )
    prefix = text[:burn_in_char_len]
    wm_part = text[burn_in_char_len:]
    prefix_ids = _tokenize_text_fragment(tokenizer, prefix)
    if wm_token_char_lens is None:
        wm_ids = _tokenize_text_fragment(tokenizer, wm_part)
    else:
        wm_ids = []
        pos = 0
        for n_chars in wm_token_char_lens:
            n = int(n_chars)
            if n < 0:
                raise ValueError(f"wm_token_char_lens entries must be >= 0, got {n}")
            piece = wm_part[pos : pos + n]
            pos += n
            piece_ids = _tokenize_text_fragment(tokenizer, piece)
            # Preserve one channel slot per generated payload token. If local
            # retokenization expands, use the first id; if it vanishes, use an
            # impossible id that recovers as an error without shifting later bits.
            wm_ids.append(piece_ids[0] if piece_ids else -1)
        if pos != len(wm_part):
            raise ValueError(
                f"wm_token_char_lens cover {pos} chars but payload has {len(wm_part)}"
            )
    return recover_bitstream(
        model,
        tokenizer,
        prefix_ids + wm_ids,
        device,
        burn_in_tokens=len(prefix_ids),
        generation_extra=generation_extra,
    )


def recover_bitstream_from_generation(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    generation_out: Dict[str, Any],
    device: str = "cpu",
    *,
    generation_extra: Dict[str, Any] | None = None,
    prefer_text_split: bool = True,
) -> Tuple[List[int], List[int], List[float]]:
    """Prompt-free recovery from a generate call.

    Default: character-split text path (cascade-isolated). Set
    ``prefer_text_split=False`` to replay exact ``burn_in_ids + wm_suffix_ids``.
    """
    if prefer_text_split and "burn_in_char_len" in generation_out:
        return recover_bitstream_from_text(
            model,
            tokenizer,
            str(generation_out["generated_text_wm"]),
            device,
            burn_in_char_len=int(generation_out["burn_in_char_len"]),
            wm_token_char_lens=(
                [int(n) for n in generation_out["wm_token_char_lens"]]
                if "wm_token_char_lens" in generation_out
                else None
            ),
            generation_extra=generation_extra,
        )

    burn_in_tokens = int(generation_out.get("burn_in_tokens", 0))
    if "burn_in_ids" in generation_out and "wm_suffix_ids" in generation_out:
        token_ids = [int(t) for t in generation_out["burn_in_ids"]] + [
            int(t) for t in generation_out["wm_suffix_ids"]
        ]
        burn_in_tokens = len(generation_out["burn_in_ids"])
    elif "model_suffix_ids" in generation_out:
        token_ids = [int(t) for t in generation_out["model_suffix_ids"]]
    else:
        model_ids = generation_out["model_input_ids"]
        if isinstance(model_ids, torch.Tensor):
            row = model_ids[0] if model_ids.dim() > 1 else model_ids
            prompt_len = int(generation_out["prompt_len"])
            token_ids = [int(t) for t in row[prompt_len:].tolist()]
        else:
            raise KeyError("generation_out missing model_suffix_ids / model_input_ids")

    return recover_bitstream(
        model,
        tokenizer,
        token_ids,
        device,
        burn_in_tokens=burn_in_tokens,
        generation_extra=generation_extra,
    )


def uncorrelated_bits_from_text(
    text: str,
    tokenizer: AutoTokenizer,
    *,
    n_bits: int,
) -> List[int]:
    """Deterministic pseudo-random bits for negative-control / text-only detect.

    Decoys without a recoverable channel need an uncorrelated bit string of
    length ``n_bits`` so PRC detection rejects.
    """
    del tokenizer  # API compat with callers that already have a tokenizer
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little")
    rng = random.Random(seed)
    return [rng.randint(0, 1) for _ in range(int(n_bits))]


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
    "get_balanced_partition_from_probs",
    "balanced_partition_masses",
    "encode_prompt_for_generation",
    "encode_prompts_for_generation",
    "suggest_baseline_batch_size",
    "generate_with_watermark",
    "generate_baseline",
    "generate_baselines",
    "recover_bitstream",
    "recover_bitstreams_batched",
    "recover_bitstream_from_watermarked_text",
    "recover_bitstreams_from_watermarked_texts",
    "recover_bitstream_from_text",
    "recover_bitstream_from_generation",
    "uncorrelated_bits_from_text",
    "negative_control_transcript_like",
]
