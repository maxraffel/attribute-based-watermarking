#!/usr/bin/env python3
"""
One-time unigram count for watermark partition weights.

Streams a public corpus, tokenizes with the target HF tokenizer, and writes a
packaged ``.pt`` artifact under ``data/partition_weights/`` for the repo to load
at runtime (no recounting on generate/detect).

Examples:
  uv sync --extra weights
  uv run python scripts/build_partition_weights.py
  uv run python scripts/build_partition_weights.py --max-tokens 5_000_000 --corpus wikipedia
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer

import model
import randrecover as rr


def _iter_wikipedia_texts(*, language: str = "en"):
    from datasets import load_dataset

    # Fixed dump id keeps rebuilds reproducible when the hub still hosts it.
    ds = load_dataset(
        "wikimedia/wikipedia",
        f"20231101.{language}",
        split="train",
        streaming=True,
    )
    for row in ds:
        text = row.get("text") or ""
        if text.strip():
            yield text


def _iter_wikitext_texts():
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    for row in ds:
        text = row.get("text") or ""
        if text.strip():
            yield text


def _iter_corpus(name: str):
    key = name.strip().lower()
    if key in ("wikipedia", "wiki"):
        yield from _iter_wikipedia_texts()
    elif key in ("wikitext", "wikitext-103"):
        yield from _iter_wikitext_texts()
    else:
        raise ValueError(f"unknown corpus {name!r}; use wikipedia or wikitext")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-id",
        default=None,
        help=f"HF tokenizer/model id (default: {model.DEFAULT_MODEL_ID})",
    )
    p.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Partition vocab width (default: len(tokenizer))",
    )
    p.add_argument(
        "--corpus",
        default="wikipedia",
        choices=("wikipedia", "wikitext"),
        help="Public corpus to stream for counts",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=5_000_000,
        help="Stop after this many counted tokens (0 = unlimited)",
    )
    p.add_argument(
        "--laplace",
        type=float,
        default=1.0,
        help="Add-k smoothing before normalization",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .pt path (default: packaged path for this tokenizer/vocab)",
    )
    p.add_argument(
        "--chunk-chars",
        type=int,
        default=2000,
        help="Character chunk size when tokenizing long documents",
    )
    args = p.parse_args()

    model_id = (args.model_id or model.DEFAULT_MODEL_ID).strip()
    print(f"Loading tokenizer {model_id!r}...", flush=True)
    model._ensure_hf_auth(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=True)
    vocab_size = int(args.vocab_size) if args.vocab_size else max(
        int(getattr(tokenizer, "vocab_size", 0) or 0),
        len(tokenizer),
    )
    key = rr.partition_weights_key(tokenizer, vocab_size, tokenizer_id=model_id)
    out = args.output or rr.packaged_partition_weights_path(key)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Counting unigrams: corpus={args.corpus} vocab_size={vocab_size} "
        f"max_tokens={args.max_tokens or '∞'} → {out}",
        flush=True,
    )
    t0 = time.perf_counter()
    counts, n_tokens, n_docs = rr.count_unigram_tokens(
        tokenizer,
        vocab_size,
        _iter_corpus(args.corpus),
        max_tokens=args.max_tokens if args.max_tokens > 0 else None,
        chunk_chars=args.chunk_chars,
        progress_every=250_000,
    )
    elapsed = time.perf_counter() - t0
    path = rr.save_partition_weights_artifact(
        out,
        counts,
        tokenizer_id=model_id,
        vocab_size=vocab_size,
        corpus=args.corpus,
        num_tokens_counted=n_tokens,
        num_documents=n_docs,
        laplace=args.laplace,
        key=key,
    )
    print(
        f"Wrote {path}  tokens={n_tokens} docs={n_docs} "
        f"elapsed={elapsed:.1f}s nonzero≈{int((counts > 0).sum())}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
