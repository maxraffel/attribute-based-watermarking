"""
Plot benchmark results from JSON files written by benchmark runners.

Examples:
  uv run python benchmark_plot.py fpr results/benchmark_fpr_vs_code_length.json
  uv run python benchmark_plot.py tpr results/benchmark_tpr_vs_wm_bit_redundancy.json --with-ci
  uv run python benchmark_plot.py label-matrix results/benchmark_label_conditioned_detection_matrix.json
  uv run python benchmark_plot.py prompt-matrix results/benchmark_prompt_conditioned_detection_matrix.json
  uv run python benchmark_plot.py ber results/benchmark_ber_diagnostics.json
  uv run python benchmark_plot.py auto results/some_run.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import benchmark_io


def _asym_errbar(y_vals: Sequence[float], lo: Sequence[float], hi: Sequence[float]) -> list[list[float]]:
    el: list[float] = []
    eh: list[float] = []
    for y, a, b in zip(y_vals, lo, hi):
        if y != y or a != a or b != b:
            el.append(float("nan"))
            eh.append(float("nan"))
        else:
            el.append(y - a)
            eh.append(b - y)
    return [el, eh]


def plot_fpr_vs_code_length(
    data: dict,
    *,
    output_png: Path | None = None,
    with_ci: bool = False,
    show: bool = True,
) -> Path:
    lengths = data["code_lengths"]
    scheme_all = data["scheme_fpr_all_runs"]
    scheme_x = data["scheme_fpr_x_matched_runs_only"]
    prc_rand = data["prc_random_detect_rate"]

    fig, ax = plt.subplots(figsize=(8, 5))
    if with_ci:
        ax.errorbar(
            lengths,
            scheme_all,
            yerr=_asym_errbar(
                scheme_all,
                data.get("scheme_fpr_all_ci_low", scheme_all),
                data.get("scheme_fpr_all_ci_high", scheme_all),
            ),
            fmt="o-",
            capsize=4,
            label="Scheme FPR (all runs)",
        )
        ax.errorbar(
            lengths,
            scheme_x,
            yerr=_asym_errbar(
                scheme_x,
                data.get("scheme_fpr_x_matched_ci_low", scheme_x),
                data.get("scheme_fpr_x_matched_ci_high", scheme_x),
            ),
            fmt="s--",
            capsize=4,
            label="Scheme FPR (x matched)",
        )
        ax.errorbar(
            lengths,
            prc_rand,
            yerr=_asym_errbar(
                prc_rand,
                data.get("prc_random_detect_rate_ci_low", prc_rand),
                data.get("prc_random_detect_rate_ci_high", prc_rand),
            ),
            fmt="^-",
            capsize=4,
            label="PRC baseline",
        )
    else:
        ax.plot(lengths, scheme_all, "o-", label="Scheme FPR (all runs)")
        ax.plot(lengths, scheme_x, "s--", label="Scheme FPR (x matched)")
        ax.plot(lengths, prc_rand, "^-", label="PRC baseline")

    ax.set_xlabel("Code length (logical PRC bits)")
    ax.set_ylabel("False positive rate")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    out = output_png or Path(str(data.get("_source_path", "benchmark_fpr_vs_code_length"))).with_suffix(".png")
    if with_ci and output_png is None:
        out = out.with_name(out.stem + "_ci.png")
    fig.savefig(out, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out


def plot_tpr_vs_redundancy(
    data: dict,
    *,
    output_png: Path | None = None,
    with_ci: bool = False,
    show: bool = True,
) -> Path:
    redundancies = data["wm_bit_redundancy"]
    code_length = int(data["code_length"])
    output_lens = [code_length * int(r) for r in redundancies]
    tpr_all = data["tpr_all_runs"]
    tpr_x = data["tpr_x_matched_runs_only"]

    fig, ax = plt.subplots(figsize=(8, 5))
    if with_ci:
        ax.errorbar(
            redundancies,
            tpr_all,
            yerr=_asym_errbar(
                tpr_all,
                data.get("tpr_all_runs_ci_low", tpr_all),
                data.get("tpr_all_runs_ci_high", tpr_all),
            ),
            fmt="o-",
            capsize=4,
            label="All runs",
        )
        ax.errorbar(
            redundancies,
            tpr_x,
            yerr=_asym_errbar(
                tpr_x,
                data.get("tpr_x_matched_ci_low", tpr_x),
                data.get("tpr_x_matched_ci_high", tpr_x),
            ),
            fmt="s--",
            capsize=4,
            label="x matched runs",
        )
    else:
        ax.plot(redundancies, tpr_all, "o-", label="All runs")
        ax.plot(redundancies, tpr_x, "s--", label="x matched runs")

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.0)
    ax.set_xticks(redundancies)
    ax.set_xticklabels([f"{r}\n({L} ch bits)" for r, L in zip(redundancies, output_lens)])
    ax.set_xlabel("WM bit redundancy / channel length")
    ax.set_ylabel("True positive rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    out = output_png or Path(str(data.get("_source_path", "benchmark_tpr_vs_wm_bit_redundancy"))).with_suffix(".png")
    if with_ci and output_png is None:
        out = out.with_name(out.stem + "_ci.png")
    fig.savefig(out, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out


def _matrix_heatmap(
    numerators: dict,
    denominators: dict,
    *,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    output_png: Path,
    show: bool,
) -> Path:
    num_df = pd.DataFrame(numerators).T.reindex(index=row_labels, columns=col_labels).fillna(0)
    den_df = pd.DataFrame(denominators).T.reindex(index=row_labels, columns=col_labels).fillna(0)
    pct = (num_df / den_df.replace(0, np.nan)) * 100.0

    fig, ax = plt.subplots(figsize=(max(6, len(col_labels) * 0.9), max(5, len(row_labels) * 0.7)))
    im = ax.imshow(pct.values, aspect="auto", cmap="viridis", vmin=0, vmax=100)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="detect rate (%)")
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_png


def plot_label_matrix(data: dict, *, output_png: Path | None = None, show: bool = True) -> Path:
    vocab = list(data["vocab"])
    out = output_png or Path(str(data.get("_source_path", "benchmark_label_conditioned_detection_matrix"))).with_suffix(".png")
    return _matrix_heatmap(
        data["numerators"],
        data["denominators"],
        row_labels=vocab,
        col_labels=vocab,
        title="Label-conditioned detection (%)",
        output_png=out,
        show=show,
    )


def plot_prompt_matrix(
    data: dict,
    *,
    output_png: Path | None = None,
    xmatch: bool = False,
    show: bool = True,
) -> Path:
    vocab = list(data["vocab"])
    cols = list(data["column_prompt_ids"])
    base = Path(str(data.get("_source_path", "benchmark_prompt_conditioned_detection_matrix")))
    suffix = "_xmatch" if xmatch else ""
    out = output_png or base.with_name(base.stem + suffix).with_suffix(".png")
    if xmatch:
        return _matrix_heatmap(
            data["numerators_attributes_match"],
            data["denominators_attributes_match"],
            row_labels=vocab,
            col_labels=cols,
            title="Prompt-conditioned detection — x matched only (%)",
            output_png=out,
            show=show,
        )
    return _matrix_heatmap(
        data["numerators"],
        data["denominators"],
        row_labels=vocab,
        col_labels=cols,
        title="Prompt-conditioned detection (%)",
        output_png=out,
        show=show,
    )


def plot_ber_diagnostics(data: dict, *, output_png: Path | None = None, show: bool = True) -> Path:
    results = data["results"]
    labels = [f"#{r['index'] + 1}" for r in results]
    stages = {
        "ch ids": [r["channel_ber_from_ids"] for r in results],
        "ch txt": [r["channel_ber_from_text"] for r in results],
        "logic": [r["logical_ber"] for r in results],
        "e2e": [r["end_to_end_ber_master"] for r in results],
    }

    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    for i, (name, vals) in enumerate(stages.items()):
        ax.bar(x + (i - 1.5) * width, vals, width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("BER (%)")
    ax.set_title("BER diagnostics by prompt")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out = output_png or Path(str(data.get("_source_path", "benchmark_ber_diagnostics"))).with_suffix(".png")
    fig.savefig(out, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out


def plot_from_file(
    kind: str,
    json_path: Path,
    *,
    output_png: Path | None = None,
    with_ci: bool = False,
    xmatch: bool = False,
    show: bool = True,
) -> Path:
    data = benchmark_io.load_json(json_path)
    data["_source_path"] = str(json_path)
    resolved = kind
    if kind == "auto":
        resolved = str(data.get("benchmark_kind", ""))
        kind_map = {
            benchmark_io.BENCHMARK_KIND_FPR_SWEEP: "fpr",
            benchmark_io.BENCHMARK_KIND_TPR_SWEEP: "tpr",
            benchmark_io.BENCHMARK_KIND_LABEL_MATRIX: "label-matrix",
            benchmark_io.BENCHMARK_KIND_PROMPT_MATRIX: "prompt-matrix",
            benchmark_io.BENCHMARK_KIND_BER: "ber",
        }
        resolved = kind_map.get(resolved, resolved)

    if resolved == "fpr":
        return plot_fpr_vs_code_length(data, output_png=output_png, with_ci=with_ci, show=show)
    if resolved == "tpr":
        return plot_tpr_vs_redundancy(data, output_png=output_png, with_ci=with_ci, show=show)
    if resolved == "label-matrix":
        return plot_label_matrix(data, output_png=output_png, show=show)
    if resolved == "prompt-matrix":
        return plot_prompt_matrix(data, output_png=output_png, xmatch=xmatch, show=show)
    if resolved == "ber":
        return plot_ber_diagnostics(data, output_png=output_png, show=show)
    raise ValueError(f"unsupported plot kind {kind!r} (resolved {resolved!r}) for {json_path}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "kind",
        choices=["auto", "fpr", "tpr", "label-matrix", "prompt-matrix", "ber"],
        help="Plot type (use auto to read benchmark_kind from JSON).",
    )
    p.add_argument("json_path", type=Path, help="Benchmark JSON file.")
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="PNG output path (default: same stem as JSON).",
    )
    p.add_argument(
        "--with-ci",
        action="store_true",
        help="For fpr/tpr sweeps, draw Wilson score error bars.",
    )
    p.add_argument(
        "--xmatch",
        action="store_true",
        help="For prompt-matrix, plot attributes-match subset.",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Save PNG without opening an interactive window.",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.json_path.is_file():
        print(f"file not found: {args.json_path}", file=sys.stderr)
        return 2
    out = plot_from_file(
        args.kind,
        args.json_path,
        output_png=args.output,
        with_ci=args.with_ci,
        xmatch=args.xmatch,
        show=not args.no_show,
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
