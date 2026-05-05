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
    """Generates a boolean mask for the entire vocabulary using a deterministic seed."""
    seed_string = f"{seed_index}"
    seed = int(hashlib.md5(seed_string.encode()).hexdigest()[:16], 16)
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.rand(vocab_size, generator=g, device=device) > 0.5


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


def _prepare_sampling_logits_processor(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt_inputs: Dict[str, torch.Tensor],
    *,
    decode_horizon: int,
    generation_extra: Dict[str, Any] | None = None,
) -> LogitsProcessorList:
    """
    Build the ``LogitsProcessorList`` used by ``model.generate(do_sample=True, ...)`` so
    manual sampling uses the same post-processed scores as HF (temperature, top-k/p, repetition
    penalty, etc.). ``prompt_inputs`` should match ``encode_prompt_for_generation`` (chat prefix
    when ``tokenizer.chat_template`` is set). Mirrors the ``generate`` preparation path in
    transformers 5.x.
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
    return logits_processor


def _scores_to_next_token_probs(
    logits_last: torch.Tensor,
    input_ids_for_proc: torch.Tensor,
    logits_processor: LogitsProcessorList,
) -> torch.Tensor:
    """Match ``GenerationMixin._sample`` (last-step logits -> processor -> softmax)."""
    next_logits = logits_last.unsqueeze(0).to(copy=True, dtype=torch.float32, device=input_ids_for_proc.device)
    next_scores = logits_processor(input_ids_for_proc, next_logits)
    return F.softmax(next_scores, dim=-1).squeeze(0)


def _sample_modified_token_partition(
    probs: torch.Tensor, mask_A: torch.BoolTensor, p: float, q: float, random_bit: int
) -> torch.Tensor:
    """Pick a half-space from ``mask_A`` (bit + randomness), then multinomial on renormalized ``probs``."""
    choose_set_A = (random_bit == 0) if random.random() < (2 * q) else (p > 0.5)
    if choose_set_A:
        modified = torch.where(mask_A, probs, torch.zeros_like(probs))
    else:
        modified = torch.where(mask_A, torch.zeros_like(probs), probs)
    total = float(modified.sum().item())
    if total <= 0.0 or total != total:  # second: NaN guard
        modified = probs
    else:
        modified = modified / modified.sum()
    return torch.multinomial(modified, num_samples=1)


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
    advanced ``bit_index``. Sampling uses Hugging Face ``generate`` logits processing
    (same as ``do_sample=True``), then either full-vocab multinomial or partition-based
    half-space sampling via ``_sample_modified_token_partition``.
    """
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    inputs = encode_prompt_for_generation(tokenizer, prompt, device)
    input_ids_wm = inputs["input_ids"].clone()
    prompt_len = int(inputs["input_ids"].shape[1])
    attn_mask = inputs.get("attention_mask", None)

    logits_processor = _prepare_sampling_logits_processor(
        model,
        tokenizer,
        inputs,
        decode_horizon=len(secret_bitstream),
        generation_extra=generation_extra,
    )

    bit_index = 0
    bit_index_to_token_id = [-1] * len(secret_bitstream)
    running_mean_p, p_count = 0.0, 0

    model_kwargs: Dict[str, Any] = {"use_cache": True}
    if attn_mask is not None:
        model_kwargs["attention_mask"] = attn_mask
    step_input_ids = input_ids_wm

    while bit_index < len(secret_bitstream):
        with torch.no_grad():
            outputs = model(input_ids=step_input_ids, **model_kwargs)
            logits_wm = outputs.logits[0, -1, :]
            probs_wm = _scores_to_next_token_probs(logits_wm, input_ids_wm, logits_processor)
            model_kwargs["past_key_values"] = outputs.past_key_values

        if full_vocab_only:
            next_token_id_wm = torch.multinomial(probs_wm, num_samples=1)
        else:
            mask_A = get_vectorized_partition(probs_wm.shape[-1], device, bit_index)
            p = probs_wm[mask_A].sum().item()
            p_count += 1
            running_mean_p += (p - running_mean_p) / p_count
            next_token_id_wm = _sample_modified_token_partition(
                probs_wm, mask_A, p, min(p, 1 - p), secret_bitstream[bit_index]
            )

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
    """Partitioned vocabulary + bit-driven half-space multinomial sampling (HF-processed probs)."""
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
    recovered_bits, recovered_tokens = [], []
    matches, total_checks = 0, 0
    filtered_ids = [tid for tid in full_sequence_ids if tid not in special_ids]
    # Ensure the mask covers all token ids present in the sequence. Some tokenizers
    # may produce ids >= `vocab_size` (e.g. added tokens); extend the effective
    # vocab size so mask indexing never goes out of bounds.
    if filtered_ids:
        max_token_id = max(filtered_ids)
        if max_token_id >= vocab_size:
            vocab_size = max_token_id + 1
    
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
) -> Tuple[List[int], List[int]]:
    """Tokenize **generated transcript** only (not the encoded chat prompt used at generation time)."""
    enc = tokenizer(full_text, return_tensors="pt")
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    return recover_bitstream(
        full_sequence_ids=enc["input_ids"][0].tolist(),
        vocab_size=tokenizer.vocab_size,
        device=device,
        special_ids=special_ids,
        ground_truth_tokens=ground_truth_tokens,
    )


def negative_control_transcript_like(
    reference_text: str,
    tokenizer: AutoTokenizer,
    device: str,
    *,
    n_bits: int,
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
        bits, _ = recover_bitstream_from_text(s, tokenizer, device)
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
    "encode_prompt_for_generation",
    "generate_with_watermark",
    "generate_with_watermark_full_vocab_sample",
    "generate_baseline",
    "recover_bitstream",
    "recover_bitstream_from_text",
    "negative_control_transcript_like",
    "log_generation_result",
    "log_recovery_evaluation",
]