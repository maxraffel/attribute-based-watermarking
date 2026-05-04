"""
Replicate the watermarked decoding loop (incremental KV cache, same stopping rule on
non-special tokens) but sample from the full next-token softmax instead of partitioning
the vocabulary and renormalizing to embed bits.

Logs:
  - ``baseline``: greedy continuation from ``randrecover.generate_baseline`` (same as the
    attribute-derivation path in ``watermarking.generate``).
  - ``full_prob_sample``: categorical sampling from the unmodified distribution each step.

Run: ``python test_full_prob_sampling_generation.py`` or ``uv run python ...``
"""

from __future__ import annotations

import argparse
import logging
import sys

import torch
from transformers import AutoTokenizer

import randrecover
import watermarking as wm


def generate_like_watermark_full_probs(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    num_content_tokens: int,
    device: str,
    *,
    seed: int | None = None,
    max_extra_steps: int = 512,
) -> str:
    """
    Same outer structure as ``randrecover.generate_with_watermark`` (KV cache, attention
    mask growth), but each step draws ``next_token ~ Categorical(softmax(logits))`` over
    the full vocabulary instead of using ``get_vectorized_partition`` and a bit-driven
    half-space restriction.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"].clone()
    prompt_len = inputs["input_ids"].shape[1]
    attn_mask = inputs.get("attention_mask", None)

    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    content_count = 0
    model_kwargs: dict = {"use_cache": True}
    if attn_mask is not None:
        model_kwargs["attention_mask"] = attn_mask
    step_input_ids = input_ids

    gen: torch.Generator | None = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))

    max_iters = num_content_tokens + max_extra_steps
    iters = 0
    while content_count < num_content_tokens and iters < max_iters:
        iters += 1
        with torch.no_grad():
            outputs = model(input_ids=step_input_ids, **model_kwargs)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            model_kwargs["past_key_values"] = outputs.past_key_values

        if gen is not None:
            next_token_id = torch.multinomial(probs, num_samples=1, generator=gen)
        else:
            next_token_id = torch.multinomial(probs, num_samples=1)

        tid = int(next_token_id.item())
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)
        step_input_ids = next_token_id.unsqueeze(0)
        if "attention_mask" in model_kwargs and model_kwargs["attention_mask"] is not None:
            am = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [
                    am,
                    torch.ones((1, 1), dtype=am.dtype, device=device),
                ],
                dim=-1,
            )

        if tid not in special_ids:
            content_count += 1

    if content_count < num_content_tokens:
        logging.getLogger(__name__).warning(
            "stopped after %d iterations with %d/%d non-special tokens (max_iters=%d)",
            iters,
            content_count,
            num_content_tokens,
            max_iters,
        )

    return tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=None,
        help="Hugging Face hub id (default: watermarking.MODEL_ID)",
    )
    parser.add_argument(
        "--code-length",
        type=int,
        default=300,
        help="Horizon: same as SECURITY_PARAM / non-special tokens to match watermark gen",
    )
    parser.add_argument(
        "--prompt",
        default="Write one short paragraph about the Roman Empire.",
        help="Prompt shared with baseline and full-probability sampling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=67,
        help="RNG seed for multinomial sampling (omit for non-deterministic sampling)",
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Do not fix a seed; sampling is non-deterministic",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("test_full_prob_sampling")

    wm.set_prc_code_length(args.code_length)
    if args.model:
        wm.set_llm_model_id(args.model.strip())
    else:
        wm.set_llm_model_id(wm.MODEL_ID)

    assert wm.MODEL is not None and wm.TOKENIZER is not None
    n = wm.SECURITY_PARAM
    seed = None if args.no_seed else args.seed

    log.info("model=%s device=%s SECURITY_PARAM=%d", wm.MODEL_ID, wm.DEVICE, n)
    log.info("greedy baseline (%d new tokens)", n)
    baseline = randrecover.generate_baseline(
        wm.MODEL, wm.TOKENIZER, args.prompt, n, wm.DEVICE
    )

    log.info(
        "full-vocab softmax sampling (%d non-special steps, seed=%s)",
        n,
        "None" if seed is None else repr(seed),
    )
    sampled = generate_like_watermark_full_probs(
        wm.MODEL,
        wm.TOKENIZER,
        args.prompt,
        n,
        wm.DEVICE,
        seed=seed,
    )

    log.info("--- BASELINE (greedy, full text) ---\n%s", baseline)
    log.info("--- FULL-PROB SAMPLING (full text) ---\n%s", sampled)
    return 0


if __name__ == "__main__":
    sys.exit(main())
