"""
Compare channel redundancy *layouts* at fixed ``wm_bit_redundancy``.

Both layouts use the same R and the same strict-majority decode; only replica
placement differs:

  depth — R full passes of the codeword (replicas of bit i at i, i+n, …).
           Short bursts hit at most one replica per logical bit.
  block — each bit repeated R times contiguously.
           A burst of length R can wipe every replica of one bit.

This is the fair interleaving comparison. Contrasting R=1 vs R>1 conflates
"having redundancy" with "layout"; use ``benchmark_watermark.py
--wm-bit-redundancy`` for that.

Examples:
  uv run python benchmark_interleaving.py
  uv run python benchmark_interleaving.py --redundancy 5 --layouts depth,block
  uv run python benchmark_interleaving.py --partition-mode balanced --repeats 2
"""

from __future__ import annotations

import argparse
import logging
import statistics
from pathlib import Path
from typing import Sequence

import benchmark_io
import benchmark_watermark as bw
import model


def _parse_layouts(spec: str) -> list[str]:
    vals: list[str] = []
    seen: set[str] = set()
    for part in spec.split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key not in ("depth", "block"):
            raise ValueError(f"layout must be 'depth' or 'block', got {part!r}")
        if key not in seen:
            seen.add(key)
            vals.append(key)
    if not vals:
        raise ValueError("need at least one layout")
    return vals


def _load_prompts(path: Path | None) -> list[str]:
    return bw._load_prompts(path)


def _row(
    agg: bw.RunAggregate, layout: str, redundancy: int, code_length: int
) -> dict[str, float | int | str]:
    return {
        "layout": layout,
        "R": redundancy,
        "channel_bits": code_length * redundancy,
        "master_detect_%": 100.0 * agg.master_detect_rate,
        "all_ok_%": 100.0 * agg.all_ok_rate,
        "avg_BER_%": agg.avg_master_ber,
        "max_BER_%": agg.max_master_ber,
        "avg_wm_gen_s": agg.avg_watermarked_gen_s,
        "avg_detect_s": agg.avg_detect_s,
        "neg_control_%": 100.0 * agg.neg_control_pass_rate,
        "n_runs": agg.n_runs,
    }


def _print_table(rows: Sequence[dict[str, float | int | str]]) -> None:
    headers = [
        "layout",
        "R",
        "channel_bits",
        "master_detect_%",
        "all_ok_%",
        "avg_BER_%",
        "max_BER_%",
        "avg_wm_gen_s",
        "avg_detect_s",
        "neg_control_%",
        "n_runs",
    ]
    widths = {
        h: max(
            len(h),
            *(
                len(f"{row[h]:.2f}" if isinstance(row[h], float) else str(row[h]))
                for row in rows
            ),
        )
        for h in headers
    }

    def fmt(h: str, v: object) -> str:
        if isinstance(v, float):
            return f"{v:.2f}".rjust(widths[h])
        return str(v).rjust(widths[h])

    print(" ".join(h.ljust(widths[h]) for h in headers))
    print(" ".join("-" * widths[h] for h in headers))
    for row in rows:
        print(" ".join(fmt(h, row[h]) for h in headers))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--redundancy",
        type=int,
        default=5,
        help="Fixed wm_bit_redundancy R for all layouts (default: 5). Must be > 1.",
    )
    p.add_argument(
        "--layouts",
        default="depth,block",
        help="Comma-separated layouts to compare (default: depth,block).",
    )
    p.add_argument("--prompts", type=Path, help="One prompt per line (# comments ok).")
    p.add_argument("--repeats", type=int, default=1, help="Runs per prompt (default: 1).")
    p.add_argument("--modulus", type=int, default=1024)
    p.add_argument("--code-length", type=int, default=100)
    p.add_argument(
        "--partition-mode",
        choices=("static", "balanced"),
        default="static",
    )
    p.add_argument("--model-id", default=None)
    p.add_argument(
        "--no-reuse-baseline",
        action="store_true",
        help="Re-sample baseline text on every repeat.",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional JSON dump of per-layout aggregates.",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    for name in ("httpx", "httpcore", "huggingface_hub", "urllib3", "text_attributes"):
        logging.getLogger(name).setLevel(logging.WARNING)

    args = _parse_args(argv)
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.redundancy < 2:
        raise SystemExit(
            "--redundancy must be >= 2 (layouts are identical at R=1; "
            "use benchmark_watermark.py --wm-bit-redundancy 1 to measure no-redundancy)"
        )
    layouts = _parse_layouts(args.layouts)
    prompts = _load_prompts(args.prompts)

    print("=== Redundancy layout comparison (interleaving) ===")
    print(
        f"prompts={len(prompts)}  repeats/prompt={args.repeats}  "
        f"code_length={args.code_length}  R={args.redundancy}  "
        f"partition_mode={args.partition_mode}  layouts={layouts}"
    )
    print(f"LLM={model.MODEL_ID if args.model_id is None else args.model_id}")
    print(
        "Note: same R and majority decode; only replica placement differs "
        "(depth = interleaved passes, block = contiguous repeats)."
    )
    print()

    rows: list[dict[str, float | int | str]] = []
    payloads: list[dict] = []
    by_layout: dict[str, dict[str, float | int | str]] = {}
    for layout in layouts:
        cfg = bw.BenchmarkConfig(
            modulus=args.modulus,
            code_length=args.code_length,
            wm_bit_redundancy=args.redundancy,
            redundancy_layout=layout,
            partition_mode=args.partition_mode,
            model_id=args.model_id,
            repeats_per_prompt=args.repeats,
            reuse_baseline=not args.no_reuse_baseline,
        )
        print(
            f"-- running layout={layout} R={args.redundancy} "
            f"(channel={args.code_length * args.redundancy} bits) --",
            flush=True,
        )
        runs = bw.run_benchmark(prompts, cfg)
        agg = bw._aggregate_runs(runs)
        row = _row(agg, layout, args.redundancy, args.code_length)
        rows.append(row)
        by_layout[layout] = row
        payloads.append(
            {
                "redundancy_layout": layout,
                "wm_bit_redundancy": args.redundancy,
                "aggregate": {
                    k: (float(v) if isinstance(v, float) else v) for k, v in row.items()
                },
                "per_run_master_ber": [float(x.master_ber_percent) for x in runs],
                "per_run_wm_gen_s": [float(x.seconds_watermarked_gen) for x in runs],
                "per_run_detect_s": [float(x.seconds_detect_total) for x in runs],
            }
        )
        if len(runs) > 1:
            bers = [float(x.master_ber_percent) for x in runs]
            print(
                f"   master_detect={row['master_detect_%']:.1f}%  "
                f"avg_BER={row['avg_BER_%']:.2f}%  "
                f"BER_stdev={statistics.pstdev(bers):.2f}  "
                f"wm_gen={row['avg_wm_gen_s']:.2f}s  detect={row['avg_detect_s']:.2f}s",
                flush=True,
            )

    print()
    _print_table(rows)

    if "depth" in by_layout and "block" in by_layout:
        depth = by_layout["depth"]
        block = by_layout["block"]
        print(
            f"\ndepth vs block at R={args.redundancy}: "
            f"detect {float(depth['master_detect_%']) - float(block['master_detect_%']):+.1f} pp, "
            f"BER {float(depth['avg_BER_%']) - float(block['avg_BER_%']):+.2f} pp "
            f"(negative BER delta = depth lower BER)"
        )

    if args.output is not None:
        payload = {
            "benchmark_kind": "redundancy_layout_comparison",
            "code_length": args.code_length,
            "wm_bit_redundancy": args.redundancy,
            "partition_mode": args.partition_mode,
            "modulus": args.modulus,
            "repeats_per_prompt": args.repeats,
            "layouts": layouts,
            "rows": rows,
            "details": payloads,
            **benchmark_io.runtime_metadata(llm_model_id=args.model_id),
        }
        out = benchmark_io.save_json(args.output, payload)
        print(f"\nWrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
