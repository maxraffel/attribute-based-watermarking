from __future__ import annotations

import hashlib
import inspect
import random
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList


def get_vectorized_partition(vocab_size: int, device: str, seed_index: int) -> torch.BoolTensor:
    """Deterministic boolean mask over the vocabulary for one bit index."""
    seed = int(hashlib.md5(f"{seed_index}".encode()).hexdigest()[:16], 16)
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.rand(vocab_size, generator=g, device=device) > 0.5


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
        "secret_bitstream": secret_bitstream,
        "sacrificed_bits": sacrificed_bits,
        "natural_partition_choices": natural_partition_choices,
        "recovery_stalls": recovery_stalls,
        "retok_replacements": retok_replacements,
        "recovery_ids_aligned": recovery_ids_aligned,
        "recovery_slots_committed": len(prev_retok_ids_ns),
        "incremental_ids_aligned": gen_ns == recovery_ns,
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
) -> Tuple[List[int], List[int]]:
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
    ref_chars = max(len(reference_text), 1)
    s = ""
    while True:
        s += phrase
        bits, _ = recover_bitstream_from_text(s, tokenizer, device, model=model)
        if len(bits) >= n_bits and len(s) >= ref_chars:
            return s


__all__ = [
    "get_vectorized_partition",
    "resolve_partition_vocab_size_for_recovery",
    "encode_prompt_for_generation",
    "generate_with_watermark",
    "generate_baseline",
    "recover_bitstream",
    "recover_bitstream_from_text",
    "negative_control_transcript_like",
]
