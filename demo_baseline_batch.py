"""Quick smoke demo: batched baseline generation via ``generate_baselines``.

Prints each prompt and its completion so you can inspect left-pad / decode
alignment. Does not need CPRF/PRC.

  uv run demo_baseline_batch.py
  uv run demo_baseline_batch.py --batch-size 2 --max-new-tokens 64
"""

from __future__ import annotations

import argparse
import time

import model
import randrecover

DEMO_PROMPTS = (
    "Name three primary colors.",
    "What is 7 + 5? Answer with just the number.",
    "Write one sentence about the ocean.",
    "List two mammals.",
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="tokens to generate per prompt (default: 64, short for inspection)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="micro-batch size (default: VRAM heuristic / BENCHMARK_BASELINE_BATCH_SIZE)",
    )
    args = p.parse_args()

    prompts = list(DEMO_PROMPTS)
    m, tok, device = model.load()

    print(
        f"\nBatched baselines: n={len(prompts)}  max_new_tokens={args.max_new_tokens}  "
        f"batch_size={args.batch_size or 'auto'}  device={device}\n"
    )

    batches: list[int] = []

    def _on_batch(n: int) -> None:
        batches.append(int(n))
        print(f"  micro-batch done: {n} prompt(s)", flush=True)

    t0 = time.perf_counter()
    texts, token_counts = randrecover.generate_baselines(
        m,
        tok,
        prompts,
        args.max_new_tokens,
        device,
        batch_size=args.batch_size,
        on_batch_done=_on_batch,
    )
    elapsed = time.perf_counter() - t0
    total_tok = sum(token_counts)
    tok_s = (total_tok / elapsed) if elapsed > 0 else 0.0

    print(
        f"\nFinished in {elapsed:.2f}s  ({tok_s:.1f} tok/s, "
        f"micro-batches: {batches})\n"
    )
    print("=" * 72)
    for i, (prompt, text, n_tok) in enumerate(
        zip(prompts, texts, token_counts), start=1
    ):
        print(f"[{i}/{len(prompts)}] PROMPT  ({n_tok} new tokens)")
        print(prompt)
        print("--- OUTPUT ---")
        print(text if text.strip() else "(empty)")
        print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
