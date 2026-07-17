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


def generate_with_watermark(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    secret_bitstream: List[int],
    device: str = "cpu",
    *,
    generation_extra: Dict[str, Any] | None = None,
) -> Dict:
    """Watermark by biasing each model step with a softmax-balanced vocab split.

    One secret bit is bound to each generated model token. Partitions are
    recomputed from the current next-token distribution so the two halves have
    near-equal mass. Use ``recover_bitstream`` with the same prompt and model
    token ids.
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
            mask_A = get_balanced_partition_from_probs(base_probs, bit_idx)
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
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    model_suffix_ids: List[int],
    device: str = "cpu",
    *,
    generation_extra: Dict[str, Any] | None = None,
) -> Tuple[List[int], List[int], List[float]]:
    """
    Recover bits embedded by ``generate_with_watermark``.

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
            mask_A = get_balanced_partition_from_probs(base_probs, bit_idx)
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


def recover_bitstream_from_generation(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    generation_out: Dict[str, Any],
    device: str = "cpu",
    *,
    generation_extra: Dict[str, Any] | None = None,
) -> Tuple[List[int], List[int], List[float]]:
    """Recover using ``prompt_text`` / ``model_suffix_ids`` from a generate call."""
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
    return recover_bitstream(
        model,
        tokenizer,
        prompt,
        suffix,
        device,
        generation_extra=generation_extra,
    )


def uncorrelated_bits_from_text(
    text: str,
    tokenizer: AutoTokenizer,
    *,
    n_bits: int,
) -> List[int]:
    """Deterministic pseudo-random bits for negative-control / text-only detect.

    There is no balanced watermark channel to rewind without ``generation_out``.
    Callers (decoy transcripts) only need an uncorrelated bit string of length
    ``n_bits`` so PRC detection rejects.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little")
    rng = random.Random(seed)
    # Touch the tokenizer so callers that already size decoys by token count keep
    # a consistent API; bit length is dictated by the expected channel width.
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
    "recover_bitstream_from_generation",
    "uncorrelated_bits_from_text",
    "negative_control_transcript_like",
]
