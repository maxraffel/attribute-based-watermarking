"""
Compare three generations at a fixed horizon ``--length``:

1. **Baseline** — ``randrecover.generate_baseline`` (``model.generate``, ``do_sample=True``,
   ``max_new_tokens=length``), same as ``watermarking._baseline``.

2. **Partition watermark** — ``randrecover.generate_with_watermark``: partition + bit-driven
   half-space, **sampling** from the restricted distribution (same logic as before, but
   multinomial instead of argmax).

3. **Full-vocab control** — ``randrecover.generate_with_watermark_full_vocab_sample``: same
   incremental loop and bit-index stopping rule as (2), but **no** partition; sample from the
   full softmax each step (secret bits are only used for horizon length, not decoding).

Secret bits for (2) and (3) are drawn with ``secrets``.

Run: ``python test_full_prob_sampling_generation.py`` or ``uv run python ...``
"""

from __future__ import annotations

import argparse
import logging
import secrets
import sys

import randrecover
import watermarking as wm

LOG = logging.getLogger("test_full_prob_sampling")


def random_secret_bits(n: int) -> list[int]:
    if n < 1:
        raise ValueError("length must be positive")
    return [secrets.randbelow(2) for _ in range(n)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=None,
        help="Hugging Face hub id (default: watermarking.MODEL_ID)",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=300,
        help="Horizon: max_new_tokens for baseline; bitstream length / decode steps for both watermark paths",
    )
    parser.add_argument(
        "--prompt",
        default="Write one short paragraph about the Roman Empire.",
        help="Prompt shared by all three generators",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    wm.set_prc_code_length(args.length)
    if args.model:
        wm.set_llm_model_id(args.model.strip())
    else:
        wm.set_llm_model_id(wm.MODEL_ID)

    assert wm.MODEL is not None and wm.TOKENIZER is not None
    n = args.length
    bits = random_secret_bits(n)

    baseline_text = randrecover.generate_baseline(
        wm.MODEL, wm.TOKENIZER, args.prompt, n, wm.DEVICE
    )

    wm_partition = randrecover.generate_with_watermark(
        wm.MODEL,
        wm.TOKENIZER,
        args.prompt,
        bits,
        wm.DEVICE,
    )
    partition_text = wm_partition["generated_text_wm"]

    wm_full = randrecover.generate_with_watermark_full_vocab_sample(
        wm.MODEL,
        wm.TOKENIZER,
        args.prompt,
        bits,
        wm.DEVICE,
    )
    full_vocab_text = wm_full["generated_text_wm"]

    LOG.info("=== (1) Baseline (HF generate, do_sample=True, max_new_tokens=%d) ===", n)
    LOG.info("%s", baseline_text)
    LOG.info("")
    LOG.info("=== (2) Partition watermark (sample restricted half; secrets bits) ===")
    LOG.info("%s", partition_text)
    LOG.info("")
    LOG.info("=== (3) Same loop as (2), no partition (full-vocab multinomial each step) ===")
    LOG.info("%s", full_vocab_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
