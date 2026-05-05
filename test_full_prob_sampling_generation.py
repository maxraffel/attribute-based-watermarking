"""
For one user ``--prompt`` and one horizon ``--length``, print three completions. The prompt is
passed through ``tokenizer.chat_template`` (with ``add_generation_prompt``) whenever the tokenizer
defines a template—otherwise it is encoded as plain text.

1. **``model.generate``** — ``randrecover.generate_baseline``: Hugging Face
   ``generate(..., do_sample=True, max_new_tokens=length)``.

2. **Incremental sampling with the HF logits-processor stack** —
   ``randrecover.generate_with_watermark_full_vocab_sample``: same KV-cache loop as the
   watermark path, but each step takes last-step logits through
   ``_prepare_sampling_logits_processor`` (same list as (1)), softmaxes **after** processing,
   and draws one token via full-vocab ``multinomial``. Secret bits are **not** used for
   decoding; they only tie the stepping horizon to ``length``.

3. **Partition before logits processing** —
   ``randrecover.generate_with_watermark``: restricts to one vocabulary half by masking logits
   *before* the same Hugging Face logits-processor stack as (2); then softmax and multinomial.
   Cryptographic bits (`secrets`) drive the scheme together with coin flips tied to marginal mass
   ``p`` computed on the full HF-processed scores (same as before), while the actual token draw
   applies the half-vocab mask **before** that processing stack at each step. After printing (3), the script runs
   ``recover_bitstream_from_text`` on that transcript and logs **BER** against the encoded
   ``bits``.

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


def _ber_vs_secret(secret: list[int], extracted: list[int]) -> tuple[float, int]:
    """
    Same BER rule as ``randrecover.log_recovery_evaluation``: bitwise mismatches over the
    overlap plus ``abs(len(secret) - len(extracted))``, denominator ``len(secret)``.
    """
    if not secret:
        return (0.0, 0)
    min_len = min(len(secret), len(extracted))
    errs = sum(1 for i in range(min_len) if secret[i] != extracted[i])
    errs += abs(len(secret) - len(extracted))
    return ((errs / len(secret)) * 100.0, errs)


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
        help="Horizon n: baseline uses max_new_tokens=n; modes (2)-(3) use n decode steps aligned to bits",
    )
    parser.add_argument(
        "--prompt",
        default="Explain the rise and fall of the Roman Empire.",
        help="User message (decoded with chat template when available; shared by all generators)",
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

    hf_generate_out = randrecover.generate_baseline(
        wm.MODEL,
        wm.TOKENIZER,
        args.prompt,
        n,
        wm.DEVICE,
    )

    logits_processor_full_vocab = randrecover.generate_with_watermark_full_vocab_sample(
        wm.MODEL,
        wm.TOKENIZER,
        args.prompt,
        bits,
        wm.DEVICE,
    )
    processor_only_text = logits_processor_full_vocab["generated_text_wm"]

    partition_wm = randrecover.generate_with_watermark(
        wm.MODEL,
        wm.TOKENIZER,
        args.prompt,
        bits,
        wm.DEVICE,
    )
    partition_text = partition_wm["generated_text_wm"]

    LOG.info("=== (1) Model.generate (do_sample=True, max_new_tokens=%d) ===", n)
    LOG.info("%s", hf_generate_out)
    LOG.info("")
    LOG.info(
        "=== (2) Incremental: _prepare_sampling_logits_processor scores → softmax → "
        "full-vocab multinomial (repeat; bits only fix horizon) ==="
    )
    LOG.info("%s", processor_only_text)
    LOG.info("")
    LOG.info(
        "=== (3) Partition-half logit mask → same HF logits processing as (2) → multinomial ==="
    )
    LOG.info("%s", partition_text)

    recovered_bits, _ = randrecover.recover_bitstream_from_text(
        partition_text, wm.TOKENIZER, wm.DEVICE
    )
    ber_pct, err_count = _ber_vs_secret(bits, recovered_bits)
    LOG.info("")
    LOG.info(
        "--- Entropy recovery: decode partition transcript → recovered bits vs secret "
        "(len secret=%d, len recovered=%d) ---",
        len(bits),
        len(recovered_bits),
    )
    LOG.info(
        "BER [partition-encoded entropy recovery]: %.2f%% (%d errors, denom = len(secret))",
        ber_pct,
        err_count,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
