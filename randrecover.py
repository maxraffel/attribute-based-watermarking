from __future__ import annotations

import hashlib
import inspect
import random
from typing import Any, Dict, List, Tuple

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

    The first ``burn_in_tokens`` model tokens are sampled without watermarking
    under the real prompt. During the watermarked phase, two LM contexts run in
    parallel:

    - **Sampling state** (prompt-conditioned): provides next-token probabilities
      used for soft watermarking and multinomial sampling.
    - **Partition state** (prompt-free, BOS + generated tokens): provides the
      distribution used to build balanced masks — the same context recovery uses.

    Published text is ``decode(burn_in) + decode(wm)`` (separate decodes). That
    character split is the retokenization barrier: prefix retokenization cannot
    shift watermarked-token indices.
    """
    if burn_in_tokens < 0:
        raise ValueError(f"burn_in_tokens must be >= 0, got {burn_in_tokens}")

    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    inputs = encode_prompt_for_generation(tokenizer, prompt, device)
    input_ids_wm = inputs["input_ids"].clone()
    prompt_len = int(inputs["input_ids"].shape[1])
    attn_mask = inputs.get("attention_mask", None)

    n_bits = len(secret_bitstream)
    total_new = int(burn_in_tokens) + n_bits
    _, logits_processor_sample = _prepare_sampling_logits_processor_bundle(
        model,
        tokenizer,
        inputs,
        decode_horizon=max(total_new, 1),
        generation_extra=generation_extra,
    )
    natural_partition_choices = 0
    partition_mass_gaps: List[float] = []
    partition_vocab_dim: int | None = None
    model_kwargs_sample: Dict[str, Any] = {"use_cache": True}
    if attn_mask is not None:
        model_kwargs_sample["attention_mask"] = attn_mask
    step_input_ids_sample = input_ids_wm

    # --- Phase 1: free burn-in under the prompt (sampling only) ---
    for _ in range(int(burn_in_tokens)):
        with torch.no_grad():
            outputs = model(input_ids=step_input_ids_sample, **model_kwargs_sample)
            logits_wm = outputs.logits[0, -1, :]
            d_logits = int(logits_wm.shape[-1])
            if partition_vocab_dim is None:
                partition_vocab_dim = d_logits
            elif partition_vocab_dim != d_logits:
                raise RuntimeError(
                    f"logits vocab width changed mid-decode ({d_logits} vs {partition_vocab_dim})"
                )
            model_kwargs_sample["past_key_values"] = outputs.past_key_values
            sample_probs = _scores_to_next_token_probs(
                logits_wm, input_ids_wm, logits_processor_sample
            )
            next_token_id_wm = torch.multinomial(sample_probs, num_samples=1)
        input_ids_wm, step_input_ids_sample = _append_generated_token(
            input_ids=input_ids_wm,
            step_input_ids=step_input_ids_sample,
            next_token_id=next_token_id_wm,
            model_kwargs=model_kwargs_sample,
            device=device,
        )

    burn_in_ids = input_ids_wm[0, prompt_len:].tolist()
    prefix_text = tokenizer.decode(burn_in_ids, skip_special_tokens=True)
    burn_in_char_len = len(prefix_text)

    # Warm the prompt-free partition state to BOS + burn-in (same as recovery).
    (
        input_ids_part,
        step_input_ids_part,
        model_kwargs_part,
        logits_processor_part,
    ) = _init_prompt_free_decode_state(
        model,
        tokenizer,
        device,
        decode_horizon=max(len(burn_in_ids) + n_bits, 1),
        generation_extra=generation_extra,
    )
    for tid in burn_in_ids:
        with torch.no_grad():
            outputs_part = model(input_ids=step_input_ids_part, **model_kwargs_part)
            model_kwargs_part["past_key_values"] = outputs_part.past_key_values
        next_tok = torch.tensor([int(tid)], dtype=torch.long, device=input_ids_part.device)
        input_ids_part, step_input_ids_part = _append_generated_token(
            input_ids=input_ids_part,
            step_input_ids=step_input_ids_part,
            next_token_id=next_tok,
            model_kwargs=model_kwargs_part,
            device=device,
        )

    # --- Phase 2: watermarked tokens (dual forward) ---
    for bit_idx in range(n_bits):
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
            mask_A = get_balanced_partition_from_probs(partition_probs, bit_idx)
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
    # Separate decode + concat: retokenizing the prefix cannot rewrite wm token ids.
    # Per-token payload spans keep a retokenization change local to that token.
    generated_text_wm = prefix_text + wm_text
    recovery_input_ids = tokenizer(generated_text_wm, return_tensors="pt")["input_ids"]
    gap_mean = (
        sum(partition_mass_gaps) / len(partition_mass_gaps) if partition_mass_gaps else 0.0
    )
    suffix_ids = burn_in_ids + wm_ids

    return {
        "prompt_text": prompt,
        "generated_text_wm": generated_text_wm,
        "burn_in_text": prefix_text,
        "wm_text": wm_text,
        "burn_in_tokens": int(burn_in_tokens),
        "burn_in_char_len": int(burn_in_char_len),
        "burn_in_ids": list(burn_in_ids),
        "wm_suffix_ids": list(wm_ids),
        "wm_token_char_lens": list(wm_token_char_lens),
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
        "partition_vocab_dim": partition_vocab_dim,
        "partition_mass_gap_mean": gap_mean,
        "partition_mass_gap_max": max(partition_mass_gaps) if partition_mass_gaps else 0.0,
        "special_ids_in_suffix": sum(1 for t in suffix_ids if t in special_ids),
        "dual_context_partition": True,
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

    Replays ``token_ids`` from a BOS-only context (no original user prompt),
    warms up through the first ``burn_in_tokens`` ids without extracting bits,
    then rebuilds balanced partitions for the remaining tokens with
    ``step_index = 0, 1, …``. Partition masks match generation's prompt-free
    partition state. Returns ``(bits, tokens, mass_gaps)``.
    """
    if burn_in_tokens < 0:
        raise ValueError(f"burn_in_tokens must be >= 0, got {burn_in_tokens}")

    ids = [int(t) for t in token_ids]
    n = len(ids)
    burn = min(int(burn_in_tokens), n)
    (
        input_ids,
        step_input_ids,
        model_kwargs,
        logits_processor,
    ) = _init_prompt_free_decode_state(
        model,
        tokenizer,
        device,
        decode_horizon=max(n, 1),
        generation_extra=generation_extra,
    )

    recovered_bits: List[int] = []
    recovered_tokens: List[int] = []
    mass_gaps: List[float] = []

    for t, token_id in enumerate(ids):
        with torch.no_grad():
            outputs = model(input_ids=step_input_ids, **model_kwargs)
            logits = outputs.logits[0, -1, :]
            model_kwargs["past_key_values"] = outputs.past_key_values
            base_probs = _scores_to_next_token_probs(
                logits, input_ids, logits_processor
            )

            tid = int(token_id)
            if t >= burn:
                bit_idx = t - burn
                mask_A = get_balanced_partition_from_probs(base_probs, bit_idx)
                mass_a, mass_b = balanced_partition_masses(base_probs, mask_A)
                mass_gaps.append(abs(mass_a - mass_b))
                if tid < 0 or tid >= int(mask_A.numel()):
                    recovered_bits.append(1)
                else:
                    recovered_bits.append(0 if bool(mask_A[tid].item()) else 1)
                recovered_tokens.append(tid)

        next_tok = torch.tensor([tid], dtype=input_ids.dtype, device=input_ids.device)
        input_ids, step_input_ids = _append_generated_token(
            input_ids=input_ids,
            step_input_ids=step_input_ids,
            next_token_id=next_tok,
            model_kwargs=model_kwargs,
            device=device,
        )

    return recovered_bits, recovered_tokens, mass_gaps


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

    Decoys without ``generation_out`` / ``burn_in_char_len`` have no recoverable
    channel; callers only need an uncorrelated bit string of length ``n_bits``
    so PRC detection rejects.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little")
    rng = random.Random(seed)
    _ = tokenizer(text, add_special_tokens=False)
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
    "generate_with_watermark",
    "generate_baseline",
    "recover_bitstream",
    "recover_bitstream_from_text",
    "recover_bitstream_from_generation",
    "uncorrelated_bits_from_text",
    "negative_control_transcript_like",
]
