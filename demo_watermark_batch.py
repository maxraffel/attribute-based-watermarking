"""Quick smoke demo: batched dual-context watermark generation.

  uv run demo_watermark_batch.py
  uv run demo_watermark_batch.py --batch-size 2 --burn-in 8 --bits 16
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
    p.add_argument("--burn-in", type=int, default=8)
    p.add_argument("--bits", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=None)
    args = p.parse_args()

    prompts = list(DEMO_PROMPTS)
    bits = [[(i + j) % 2 for j in range(args.bits)] for i in range(len(prompts))]
    m, tok, device = model.load()

    print(
        f"\nBatched watermark gen: n={len(prompts)}  burn_in={args.burn_in}  "
        f"bits={args.bits}  batch_size={args.batch_size or 'auto'}  device={device}\n"
    )
    batches: list[int] = []

    def _on_batch(n: int) -> None:
        batches.append(int(n))
        print(f"  micro-batch done: {n} prompt(s)", flush=True)

    t0 = time.perf_counter()
    outs = randrecover.generate_with_watermarks(
        m,
        tok,
        prompts,
        bits,
        device,
        burn_in_tokens=args.burn_in,
        batch_size=args.batch_size,
        on_batch_done=_on_batch,
    )
    elapsed = time.perf_counter() - t0
    total_tok = sum(len(o["burn_in_ids"]) + len(o["wm_suffix_ids"]) for o in outs)
    tok_s = (total_tok / elapsed) if elapsed > 0 else 0.0
    print(
        f"\nFinished in {elapsed:.2f}s  ({tok_s:.1f} tok/s, micro-batches: {batches})\n"
    )
    print("=" * 72)
    for i, (prompt, out) in enumerate(zip(prompts, outs), start=1):
        n_tok = len(out["burn_in_ids"]) + len(out["wm_suffix_ids"])
        print(f"[{i}/{len(prompts)}] PROMPT  ({n_tok} tokens, natural={out['natural_partition_choices']})")
        print(prompt)
        print("--- OUTPUT ---")
        text = str(out["generated_text_wm"])
        print(text if text.strip() else "(empty)")
        print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
