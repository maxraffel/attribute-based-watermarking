from __future__ import annotations

import hashlib
import inspect
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList

def get_vectorized_partition(vocab_size: int, device: str, seed_index: int) -> torch.BoolTensor:
    """Generates a boolean mask for the entire vocabulary using a deterministic seed."""
    seed_string = f"{seed_index}"
    seed = int(hashlib.md5(seed_string.encode()).hexdigest()[:16], 16)
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.rand(vocab_size, generator=g, device=device) > 0.5


def resolve_partition_vocab_size_for_recovery(
    tokenizer: AutoTokenizer,
    *,
    model: torch.nn.Module | None = None,
    partition_vocab_size: int | None = None,
) -> int:
    """
    Same ``vocab_size`` rule as ``recover_bitstream_from_text`` / ``recover_bitstream`` callers:
    explicit ``partition_vocab_size`` wins, then ``model.config.vocab_size``, else tokenizer vocab.
    """
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
    """
    Build model inputs for causal generation: apply the tokenizer's ``chat_template``
    (user turn + ``add_generation_prompt``) when defined, else plain ``tokenizer(...)``.

    ``prompt_len`` for slicing new tokens is ``input_ids.shape[1]`` of the returned batch
    (includes the full templated prefix).
    """
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


_SAMPLING_LOGITS_CONFIG_KEYS: Tuple[str, ...] = (
    "do_sample",
    "temperature",
    "top_k",
    "top_p",
    "top_h",
    "min_p",
    "typical_p",
    "repetition_penalty",
    "encoder_repetition_penalty",
    "no_repeat_ngram_size",
    "max_length",
    "max_new_tokens",
    "min_length",
    "min_new_tokens",
    "pad_token_id",
    "eos_token_id",
    "renormalize_logits",
    "guidance_scale",
    "watermarking_config",
    "epsilon_cutoff",
    "eta_cutoff",
    "remove_invalid_values",
    "bad_words_ids",
    "suppress_tokens",
    "sequence_bias",
)


def _serialize_generation_config_sampling_fields(generation_config: Any) -> Dict[str, Any]:
    """Stable subset of ``GenerationConfig`` fields that affect logits before multinomial."""
    out: Dict[str, Any] = {}
    for k in _SAMPLING_LOGITS_CONFIG_KEYS:
        if not hasattr(generation_config, k):
            continue
        v = getattr(generation_config, k)
        if v is None:
            continue
        if k == "watermarking_config":
            out[k] = repr(v)[:240]
        elif k == "bad_words_ids" and isinstance(v, list) and len(v) > 8:
            out[k] = f"<list len={len(v)}>"
        elif k == "suppress_tokens" and isinstance(v, list) and len(v) > 16:
            out[k] = f"<list len={len(v)}>"
        else:
            out[k] = v
    return out


def build_sampling_logits_debug_info(
    generation_config: Any,
    logits_processor: LogitsProcessorList,
) -> Dict[str, Any]:
    """
    Snapshot of resolved sampling / logits-processing settings (HF ``GenerationConfig`` subset
    plus ordered ``LogitsProcessorList`` class names).
    """
    return {
        "generation_config_sampling": _serialize_generation_config_sampling_fields(generation_config),
        "logits_processor_pipeline": [type(p).__name__ for p in logits_processor],
        "logits_processor_count": len(logits_processor),
    }


def format_sampling_logits_debug_lines(debug: Dict[str, Any]) -> List[str]:
    """Multi-line strings suitable for ``logging.info`` / print."""
    lines: List[str] = ["--- HF sampling / logits processing (resolved for incremental watermark path) ---"]
    g = debug.get("generation_config_sampling") or {}
    if not g:
        lines.append("  (no generation_config_sampling fields captured)")
    else:
        lines.append("  GenerationConfig (sampling-related):")
        for k in sorted(g.keys()):
            lines.append(f"    {k}={g[k]!r}")
    cr = debug.get("caller_request")
    if cr:
        lines.append("  Caller request (merged into HF prep):")
        for k in sorted(cr.keys()):
            lines.append(f"    {k}={cr[k]!r}")
    lines.append(
        f"  LogitsProcessorList: count={debug.get('logits_processor_count')} "
        f"pipeline={debug.get('logits_processor_pipeline')}"
    )
    return lines


def infer_sampling_logits_debug_for_prompt(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    user_prompt: str,
    decode_horizon: int,
    device: str,
    *,
    generation_extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Same bundle as ``generate_with_watermark`` / ``generate_baseline``-aligned ``_prepare_*``:
    chat-encoded prompt + ``max_new_tokens=decode_horizon``, ``do_sample=True``.
    """
    inputs = encode_prompt_for_generation(tokenizer, user_prompt, device)
    gc, lp = _prepare_sampling_logits_processor_bundle(
        model,
        tokenizer,
        inputs,
        decode_horizon=decode_horizon,
        generation_extra=generation_extra,
    )
    info = build_sampling_logits_debug_info(gc, lp)
    info["caller_request"] = {
        "max_new_tokens": decode_horizon,
        "do_sample": True,
        "generation_extra": dict(generation_extra) if generation_extra else {},
    }
    return info


def _prepare_sampling_logits_processor_bundle(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt_inputs: Dict[str, torch.Tensor],
    *,
    decode_horizon: int,
    generation_extra: Dict[str, Any] | None = None,
) -> Tuple[Any, LogitsProcessorList]:
    """
    Build the resolved ``GenerationConfig`` and ``LogitsProcessorList`` used by
    ``model.generate(do_sample=True, ...)`` so manual sampling uses the same post-processed
    scores as HF. ``prompt_inputs`` should match ``encode_prompt_for_generation``.
    """
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


def _prepare_sampling_logits_processor(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt_inputs: Dict[str, torch.Tensor],
    *,
    decode_horizon: int,
    generation_extra: Dict[str, Any] | None = None,
) -> LogitsProcessorList:
    _, lp = _prepare_sampling_logits_processor_bundle(
        model,
        tokenizer,
        prompt_inputs,
        decode_horizon=decode_horizon,
        generation_extra=generation_extra,
    )
    return lp


def _scores_to_next_token_probs(
    logits_last: torch.Tensor,
    input_ids_for_proc: torch.Tensor,
    logits_processor: LogitsProcessorList,
) -> torch.Tensor:
    """Match ``GenerationMixin._sample`` (last-step logits -> processor -> softmax)."""
    next_logits = logits_last.unsqueeze(0).to(copy=True, dtype=torch.float32, device=input_ids_for_proc.device)
    next_scores = logits_processor(input_ids_for_proc, next_logits)
    return F.softmax(next_scores, dim=-1).squeeze(0)


def _choose_watermark_partition_half(secret_bit: int, p: float, q: float) -> bool:
    """True iff sampling stays on partition set A; same random channel as legacy prob-space masking."""
    return (secret_bit == 0) if random.random() < (2 * q) else (p > 0.5)


def _logits_mask_disallowed_half(
    logits: torch.Tensor,
    mask_A: torch.BoolTensor,
    *,
    choose_set_A: bool,
) -> torch.Tensor:
    """Set disallowed vocabulary logits to -inf before HF logits processors (temperature, top-k/p, …)."""
    lf = logits.to(dtype=torch.float32)
    neg_inf = torch.tensor(float("-inf"), dtype=lf.dtype, device=lf.device)
    if choose_set_A:
        return torch.where(mask_A, lf, neg_inf)
    return torch.where(mask_A, neg_inf, lf)


def _probs_via_partition_masked_logits_then_process(
    logits_last: torch.Tensor,
    input_ids_for_proc: torch.Tensor,
    logits_processor: LogitsProcessorList,
    mask_A: torch.BoolTensor,
    secret_bit: int,
) -> Tuple[torch.Tensor, float]:
    """
    Partition on **raw** model logits (-inf mask), run ``logits_processor``, then softmax.
    ``p`` / ``q`` / half choice use softmax(**raw** logits), matching the legacy control channel.
    """
    logits_f = logits_last.to(dtype=torch.float32)
    raw_probs = F.softmax(logits_f, dim=-1)
    p = float(raw_probs[mask_A].sum().item())
    q = min(p, 1.0 - p)
    choose_A = _choose_watermark_partition_half(int(secret_bit), p, q)
    part_logits = _logits_mask_disallowed_half(logits_f, mask_A, choose_set_A=choose_A)
    proc = logits_processor(input_ids_for_proc, part_logits.unsqueeze(0))
    return F.softmax(proc, dim=-1).squeeze(0), p


def _generate_watermark_incremental(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    secret_bitstream: List[int],
    device: str,
    track_token_alignment: bool,
    *,
    full_vocab_only: bool,
    generation_extra: Dict[str, Any] | None = None,
) -> Dict:
    """
    Shared KV-cache loop: stop once ``len(secret_bitstream)`` non-special tokens have
    advanced ``bit_index``. Full-vocab mode: raw logits → HF logits processors → softmax
    → multinomial. Watermark mode: raw logits → **half-vocab mask (-inf)** (bit + stochastic
    choice as before) → same HF processors → softmax → multinomial.
    """
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    inputs = encode_prompt_for_generation(tokenizer, prompt, device)
    input_ids_wm = inputs["input_ids"].clone()
    prompt_len = int(inputs["input_ids"].shape[1])
    attn_mask = inputs.get("attention_mask", None)

    gen_cfg, logits_processor = _prepare_sampling_logits_processor_bundle(
        model,
        tokenizer,
        inputs,
        decode_horizon=len(secret_bitstream),
        generation_extra=generation_extra,
    )
    sampling_logits_debug = build_sampling_logits_debug_info(gen_cfg, logits_processor)
    sampling_logits_debug["caller_request"] = {
        "max_new_tokens": len(secret_bitstream),
        "do_sample": True,
        "generation_extra": dict(generation_extra) if generation_extra else {},
    }

    bit_index = 0
    bit_index_to_token_id = [-1] * len(secret_bitstream)
    running_mean_p, p_count = 0.0, 0
    partition_vocab_dim: int | None = None

    model_kwargs: Dict[str, Any] = {"use_cache": True}
    if attn_mask is not None:
        model_kwargs["attention_mask"] = attn_mask
    step_input_ids = input_ids_wm

    while bit_index < len(secret_bitstream):
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

            if full_vocab_only:
                probs_wm = _scores_to_next_token_probs(logits_wm, input_ids_wm, logits_processor)
                next_token_id_wm = torch.multinomial(probs_wm, num_samples=1)
            else:
                mask_A = get_vectorized_partition(partition_vocab_dim, device, bit_index)
                probs_wm, p = _probs_via_partition_masked_logits_then_process(
                    logits_wm,
                    input_ids_wm,
                    logits_processor,
                    mask_A,
                    secret_bitstream[bit_index],
                )
                p_count += 1
                running_mean_p += (p - running_mean_p) / p_count
                if torch.isfinite(probs_wm).all() and float(probs_wm.sum().item()) > 0:
                    probs_wm = probs_wm.clamp(min=0)
                    probs_wm = probs_wm / probs_wm.sum()
                    next_token_id_wm = torch.multinomial(probs_wm, num_samples=1)
                else:
                    probs_fb = _scores_to_next_token_probs(
                        logits_wm, input_ids_wm, logits_processor
                    )
                    next_token_id_wm = torch.multinomial(probs_fb, num_samples=1)

        tid = int(next_token_id_wm.item())
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

        if track_token_alignment:
            if "_retok_offset" not in locals():
                _retok_offset = 0
            try:
                gen_text_so_far = tokenizer.decode(input_ids_wm[0], skip_special_tokens=False)
                retok_ids_ns = [t for t in tokenizer(gen_text_so_far)["input_ids"] if t not in special_ids]
                gen_ids_ns = [t for t in input_ids_wm[0].tolist() if t not in special_ids]

                if len(gen_ids_ns) != (len(retok_ids_ns) + _retok_offset):
                    if len(gen_ids_ns) > (len(retok_ids_ns) + _retok_offset):
                        if tid not in special_ids:
                            bit_index_to_token_id[bit_index] = tid
                        _retok_offset += 1
                    else:
                        if tid not in special_ids:
                            for i in range(bit_index, min(bit_index + 2, len(bit_index_to_token_id))):
                                if i < len(bit_index_to_token_id):
                                    bit_index_to_token_id[i] = tid
                        _retok_offset -= 1
                        bit_index += 2
                else:
                    if tid not in special_ids:
                        bit_index_to_token_id[bit_index] = tid
                    bit_index += 1
            except Exception:
                bit_index += 1
        else:
            if tid not in special_ids:
                bit_index_to_token_id[bit_index] = tid
                bit_index += 1

    return {
        "prompt_text": prompt,
        "generated_text_wm": tokenizer.decode(input_ids_wm[0, prompt_len:], skip_special_tokens=True),
        "input_ids_wm": input_ids_wm[:, prompt_len:],
        "secret_bitstream": secret_bitstream,
        "bit_index_to_token_id": bit_index_to_token_id,
        "running_mean_p": running_mean_p,
        "p_count": p_count,
        "partition_vocab_dim": partition_vocab_dim
        if partition_vocab_dim is not None
        else int(getattr(getattr(model, "config", None), "vocab_size", 0)),
        "sampling_logits_debug": sampling_logits_debug,
    }


def generate_with_watermark(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    secret_bitstream: List[int],
    device: str = "cpu",
    track_token_alignment: bool = False,
    *,
    generation_extra: Dict[str, Any] | None = None,
) -> Dict:
    """Partitioned vocabulary: mask raw logits to a half-space, then HF logits warpers + sample."""
    return _generate_watermark_incremental(
        model,
        tokenizer,
        prompt,
        secret_bitstream,
        device,
        track_token_alignment,
        full_vocab_only=False,
        generation_extra=generation_extra,
    )


def generate_with_watermark_full_vocab_sample(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    secret_bitstream: List[int],
    device: str = "cpu",
    track_token_alignment: bool = False,
    *,
    generation_extra: Dict[str, Any] | None = None,
) -> Dict:
    """Same loop and horizon as ``generate_with_watermark``, but draws from full (HF-processed) distribution."""
    return _generate_watermark_incremental(
        model,
        tokenizer,
        prompt,
        secret_bitstream,
        device,
        track_token_alignment,
        full_vocab_only=True,
        generation_extra=generation_extra,
    )

def generate_baseline(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: str = "cpu",
) -> str:
    """
    Sample with ``model.generate`` from a **user** ``prompt``: chat template plus generation
    prefix when ``tokenizer.chat_template`` is set; otherwise encode ``prompt`` as raw text.
    ``max_new_tokens`` counts assistant tokens **after** that encoded prefix (same slicing rule
    as the watermark KV loop).
    """
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
    ground_truth_tokens: List[int] = None,
) -> Tuple[List[int], List[int]]:
    """
    Recover side-of-partition bits for each **non-special** token in order.

    ``vocab_size`` must be the **same** integer used when encoding (the model logits width).
    Do not enlarge it here: ``get_vectorized_partition(vocab_size, …)`` draws
    ``vocab_size`` uniforms per bit index; changing the length desynchronizes masks from
    generation. Out-of-range token ids use the existing OOB rule (treated as set B).
    """
    recovered_bits, recovered_tokens = [], []
    matches, total_checks = 0, 0
    filtered_ids = [tid for tid in full_sequence_ids if tid not in special_ids]

    for bit_idx, actual_token_id in enumerate(filtered_ids):
        if ground_truth_tokens and bit_idx >= len(ground_truth_tokens): break

        mask_A = get_vectorized_partition(vocab_size, device, bit_idx)
        # Guard against any remaining out-of-bounds access just in case.
        if actual_token_id >= mask_A.shape[0]:
            # If token id is outside, treat it as belonging to the B set (bit=1)
            recovered_bits.append(1)
        else:
            recovered_bits.append(0 if mask_A[actual_token_id].item() else 1)
        recovered_tokens.append(actual_token_id)

        if ground_truth_tokens:
            expected_id = ground_truth_tokens[bit_idx]
            if expected_id != -1:
                total_checks += 1
                if actual_token_id == expected_id: matches += 1

    if ground_truth_tokens and total_checks > 0:
        print(f"Token Accuracy: {matches}/{total_checks} ({(matches/total_checks)*100:.2f}%)")

    return recovered_bits, recovered_tokens

def recover_bitstream_from_text(
    full_text: str,
    tokenizer: AutoTokenizer,
    device: str,
    ground_truth_tokens: List[int] | None = None,
    *,
    model: torch.nn.Module | None = None,
    partition_vocab_size: int | None = None,
) -> Tuple[List[int], List[int]]:
    """
    Tokenize **generated transcript** only (not the encoded chat prompt used at generation time).

    Pass ``partition_vocab_size`` from the watermark artifact's ``partition_vocab_dim``, or pass
    ``model`` so we use ``model.config.vocab_size`` — must match the logits width used when
    building partitions during ``generate_with_watermark``.
    """
    enc = tokenizer(full_text, return_tensors="pt")
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    vocab_size = resolve_partition_vocab_size_for_recovery(
        tokenizer, model=model, partition_vocab_size=partition_vocab_size
    )

    return recover_bitstream(
        full_sequence_ids=enc["input_ids"][0].tolist(),
        vocab_size=vocab_size,
        device=device,
        special_ids=special_ids,
        ground_truth_tokens=ground_truth_tokens,
    )


@dataclass
class TokenPartitionAuditRow:
    """Per payload token: generation vs recovery partition tensors and inferred bits."""

    bit_index: int
    token_id: int
    generation_vocab_dim: int
    recovery_vocab_dim: int
    generation_mask_shape: Tuple[int, ...]
    recovery_mask_shape: Tuple[int, ...]
    masks_bitwise_identical: bool
    num_vocab_cells_mismatched: int
    sample_mismatch_vocab_indices: List[int]
    in_set_A_by_generation_mask: Optional[bool]
    in_set_A_by_recovery_mask: Optional[bool]
    bit_from_recovery_pipeline: Optional[int]
    bit_implied_by_generation_mask: Optional[int]
    bit_implied_by_recovery_mask_only: Optional[int]
    mapping_entry_agrees_with_token_id: Optional[bool]


@dataclass
class PartitionReplicationAudit:
    generation_vocab_dim: int
    recovery_vocab_dim: int
    num_suffix_tokens_raw: int
    num_payload_tokens: int
    recovery_bitstream_length: int
    rows: List[TokenPartitionAuditRow] = field(default_factory=list)

    @property
    def all_partitions_recreated(self) -> bool:
        return all(r.masks_bitwise_identical for r in self.rows)

    @property
    def all_memberships_agree_masks(self) -> bool:
        for r in self.rows:
            a = r.in_set_A_by_generation_mask
            b = r.in_set_A_by_recovery_mask
            if a is None or b is None:
                return False
            if bool(a) != bool(b):
                return False
        return True


def audit_partition_replication_tokenwise(
    wm_out: Dict[str, Any],
    tokenizer: AutoTokenizer,
    device: str,
    *,
    model: torch.nn.Module | None = None,
    recovery_partition_vocab_size: int | None = None,
) -> PartitionReplicationAudit:
    """
    For each payload (non-special) token ``k``: build the partition ``get_vectorized_partition(dim, …, k)``
    that generation used (``generation_vocab_dim`` from ``wm_out``) and the partition recovery would use
    (``recovery_vocab_size`` resolver), compare **every** vocabulary cell, membership for ``token_id``,
    and ``recover_bitstream`` bits on the suffix.
    """
    gen_dim = wm_out.get("partition_vocab_dim")
    if gen_dim is None:
        raise ValueError("wm_out missing partition_vocab_dim")
    gen_dim = int(gen_dim)

    if recovery_partition_vocab_size is None:
        if model is None:
            rec_dim = gen_dim
        else:
            rec_dim = resolve_partition_vocab_size_for_recovery(tokenizer, model=model, partition_vocab_size=None)
    else:
        rec_dim = int(recovery_partition_vocab_size)

    suffix = wm_out["input_ids_wm"][0].tolist()
    special = set(getattr(tokenizer, "all_special_ids", []))
    filtered = [tid for tid in suffix if tid not in special]
    mapping: List[int] = wm_out.get("bit_index_to_token_id", [])

    rec_bits, _ = recover_bitstream(suffix, rec_dim, device, special)

    rows: List[TokenPartitionAuditRow] = []
    for k, tid in enumerate(filtered):
        mg = get_vectorized_partition(gen_dim, device, k).cpu().flatten()
        mr = get_vectorized_partition(rec_dim, device, k).cpu().flatten()

        sg, sr = tuple(mg.shape), tuple(mr.shape)
        shape_ok = sg == sr
        if shape_ok:
            cell_diff = mg != mr
            n_mismatch = int(cell_diff.sum().item())
            sample_idx = torch.nonzero(cell_diff, as_tuple=False).flatten().tolist()[:32]
            masks_identical = n_mismatch == 0
        else:
            mlen = min(mg.numel(), mr.numel())
            pref = mg[:mlen] != mr[:mlen]
            n_mismatch = int(pref.sum().item())
            sample_idx = torch.nonzero(pref, as_tuple=False).flatten().tolist()[:32]
            masks_identical = False

        def _side(mask_1d: torch.Tensor, token: int) -> Optional[bool]:
            if token >= mask_1d.numel():
                return None
            return bool(mask_1d[token].item())

        side_g = _side(mg, tid)
        side_r = _side(mr, tid)

        impl_g: Optional[int] = None if side_g is None else (0 if side_g else 1)
        bit_implied_rec_mask: Optional[int]
        if tid >= mr.numel():
            bit_implied_rec_mask = 1
        else:
            bit_implied_rec_mask = 0 if mr[tid].item() else 1

        bit_rec_pipe: Optional[int] = None
        if k < len(rec_bits):
            bit_rec_pipe = int(rec_bits[k])
        map_ok: Optional[bool] = None
        if k < len(mapping):
            if mapping[k] == -1:
                map_ok = None
            else:
                map_ok = mapping[k] == tid

        rows.append(
            TokenPartitionAuditRow(
                bit_index=k,
                token_id=tid,
                generation_vocab_dim=gen_dim,
                recovery_vocab_dim=rec_dim,
                generation_mask_shape=sg,
                recovery_mask_shape=sr,
                masks_bitwise_identical=masks_identical,
                num_vocab_cells_mismatched=n_mismatch,
                sample_mismatch_vocab_indices=sample_idx,
                in_set_A_by_generation_mask=side_g,
                in_set_A_by_recovery_mask=side_r,
                bit_from_recovery_pipeline=bit_rec_pipe,
                bit_implied_by_generation_mask=impl_g,
                bit_implied_by_recovery_mask_only=bit_implied_rec_mask,
                mapping_entry_agrees_with_token_id=map_ok,
            )
        )

    return PartitionReplicationAudit(
        generation_vocab_dim=gen_dim,
        recovery_vocab_dim=rec_dim,
        num_suffix_tokens_raw=len(suffix),
        num_payload_tokens=len(filtered),
        recovery_bitstream_length=len(rec_bits),
        rows=rows,
    )


def format_partition_replication_audit(audit: PartitionReplicationAudit) -> List[str]:
    """Human-readable lines highlighting mask equality and any disagreements per token."""
    lines: List[str] = [
        (
            f"partition replication: generation_dim={audit.generation_vocab_dim} "
            f"recovery_dim={audit.recovery_vocab_dim} "
            f"suffix_tokens={audit.num_suffix_tokens_raw} payload_tokens={audit.num_payload_tokens}"
        ),
        (
            f"summary: all_full_masks_match={audit.all_partitions_recreated} "
            f"A_vs_A_membership_same={audit.all_memberships_agree_masks} "
            f"recover_bits_len={audit.recovery_bitstream_length}"
        ),
    ]
    for r in audit.rows:
        if r.masks_bitwise_identical:
            badge = "MASKS_IDENTICAL"
        else:
            badge = f"MASK_MISMATCH cells={r.num_vocab_cells_mismatched} sample_idxs={r.sample_mismatch_vocab_indices}"

        memb = "?/?"
        if r.in_set_A_by_generation_mask is not None and r.in_set_A_by_recovery_mask is not None:
            memb = f"A_gen={r.in_set_A_by_generation_mask} A_rec={r.in_set_A_by_recovery_mask}"

        map_txt = ""
        if r.mapping_entry_agrees_with_token_id is False:
            map_txt = " MAPPING_ENTRY_MISMATCH"

        bit_txt = ""
        if (
            r.bit_from_recovery_pipeline is not None
            and r.bit_implied_by_generation_mask is not None
            and r.bit_from_recovery_pipeline != r.bit_implied_by_generation_mask
        ):
            bit_txt = (
                f" BIT_MISMATCH rec_pipeline={r.bit_from_recovery_pipeline} "
                f"gen_mask_implies={r.bit_implied_by_generation_mask}"
            )

        if (
            r.bit_from_recovery_pipeline is not None
            and r.bit_implied_by_recovery_mask_only is not None
            and r.bit_from_recovery_pipeline != r.bit_implied_by_recovery_mask_only
        ):
            bit_txt += (
                f" INTERNAL_rec_mask_implies={r.bit_implied_by_recovery_mask_only}"
                f"(≠recover_bitstream)"
            )

        lines.append(
            f"  k={r.bit_index} tid={r.token_id} {badge} gen_shape={r.generation_mask_shape} "
            f"rec_shape={r.recovery_mask_shape} {memb}"
            f"{bit_txt}{map_txt}"
        )
    return lines


def verify_partition_embed_recovery_sync(
    wm_out: Dict[str, Any],
    tokenizer: AutoTokenizer,
    device: str,
    *,
    model: torch.nn.Module | None = None,
    recovery_partition_vocab_size: int | None = None,
    log: Optional[Callable[[str], None]] = None,
) -> PartitionReplicationAudit:
    """
    Token-by-token audit: full boolean partition used at generation vs the one recovery rebuilds must match
    (same ``get_vectorized_partition`` arguments). Optionally ``log`` each line via e.g. ``logging.info``.
    Raises ``AssertionError`` on any bitwise mask mismatch or A/B disagreement for ``token_id``.

    Defaults: recovery uses ``partition_vocab_dim`` from ``wm_out``. Pass ``model`` to audit against the
    resolver used by ``recover_bitstream_from_text`` instead (omit ``recovery_partition_vocab_size``).
    """
    if recovery_partition_vocab_size is None and model is None:
        dim = wm_out.get("partition_vocab_dim")
        if dim is None:
            raise ValueError("wm_out missing partition_vocab_dim")
        recovery_partition_vocab_size = int(dim)

    audit = audit_partition_replication_tokenwise(
        wm_out,
        tokenizer,
        device,
        model=model,
        recovery_partition_vocab_size=recovery_partition_vocab_size,
    )

    text_lines = format_partition_replication_audit(audit)
    if log is not None:
        for line in text_lines:
            log(line)

    errs: List[str] = []
    if not audit.all_partitions_recreated:
        errs.append("one or more full partition masks did not bitwise match generation vs recovery")
    if not audit.all_memberships_agree_masks:
        errs.append("token membership in half-space A differed between generation and recovery partitions")
    for r in audit.rows:
        if r.mapping_entry_agrees_with_token_id is False:
            errs.append(
                f"k={r.bit_index} bit_index_to_token_id disagreed with filtered suffix token "
                f"(tid={r.token_id})"
            )
        if (
            r.bit_from_recovery_pipeline is not None
            and r.bit_implied_by_generation_mask is not None
            and r.bit_from_recovery_pipeline != r.bit_implied_by_generation_mask
        ):
            errs.append(
                f"k={r.bit_index} recover_bitstream bit {r.bit_from_recovery_pipeline} != "
                f"generation-mask-implied bit {r.bit_implied_by_generation_mask} (tid={r.token_id})"
            )
        if (
            r.bit_from_recovery_pipeline is not None
            and r.bit_implied_by_recovery_mask_only is not None
            and r.bit_from_recovery_pipeline != r.bit_implied_by_recovery_mask_only
        ):
            errs.append(
                f"k={r.bit_index} recover_bitstream bit {r.bit_from_recovery_pipeline} != "
                f"recovery-partition-rule bit {r.bit_implied_by_recovery_mask_only} (tid={r.token_id})"
            )

    if audit.recovery_bitstream_length != audit.num_payload_tokens:
        raise AssertionError(
            "\n".join(
                text_lines
                + ["--- ASSERTION FAILURE ---"]
                + [
                    f"recover_bitstream length {audit.recovery_bitstream_length} != payload tokens "
                    f"{audit.num_payload_tokens} (suffix vs special filtering)"
                ]
            )
        )

    if errs:
        raise AssertionError("\n".join(text_lines + ["--- ASSERTION FAILURE ---"] + errs))

    return audit


def negative_control_transcript_like(
    reference_text: str,
    tokenizer: AutoTokenizer,
    device: str,
    *,
    n_bits: int,
    model: torch.nn.Module | None = None,
    phrase: str = "Unrelated decoy text used only as a negative control. ",
) -> str:
    """
    Build semantically unrelated text whose tokenized length supports ``n_bits`` of bit recovery
    (same path as ``master_detect`` / ``detect``) and whose character length is at least that of
    ``reference_text`` (e.g. the watermarked example) so the negative control matches that horizon.
    """
    ref_chars = max(len(reference_text), 1)
    s = ""
    while True:
        s += phrase
        bits, _ = recover_bitstream_from_text(s, tokenizer, device, model=model)
        if len(bits) >= n_bits and len(s) >= ref_chars:
            return s


def log_generation_result(out: Dict) -> None:
    print(f"\n--- Generation Result ---\nPrompt: {out['prompt_text']}\nOutput: {out['generated_text_wm']}")

def log_recovery_evaluation(secret, extracted, label=""):
    if not secret: return
    # Handle length mismatches for BER calculation
    min_len = min(len(secret), len(extracted))
    errs = sum(1 for i in range(min_len) if secret[i] != extracted[i])
    errs += abs(len(secret) - len(extracted)) # Penalize length difference
    print(f"BER [{label}]: {(errs/len(secret))*100:.2f}%")

__all__ = [
    "get_vectorized_partition",
    "resolve_partition_vocab_size_for_recovery",
    "TokenPartitionAuditRow",
    "PartitionReplicationAudit",
    "audit_partition_replication_tokenwise",
    "format_partition_replication_audit",
    "encode_prompt_for_generation",
    "generate_with_watermark",
    "generate_with_watermark_full_vocab_sample",
    "generate_baseline",
    "recover_bitstream",
    "recover_bitstream_from_text",
    "verify_partition_embed_recovery_sync",
    "negative_control_transcript_like",
    "log_generation_result",
    "log_recovery_evaluation",
]