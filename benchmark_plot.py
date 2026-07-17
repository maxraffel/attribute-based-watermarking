"""
Plot benchmark results from JSON files written by benchmark runners.

Confidence intervals (when present in the JSON) are drawn as a translucent band
around each series line — not as error bars.

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


def _finite(vals: Sequence[float]) -> bool:
    return all(v == v for v in vals)


def _resolve_ci(
    data: dict,
    y_key: str,
    *,
    lo_key: str | None = None,
    hi_key: str | None = None,
) -> tuple[list[float] | None, list[float] | None]:
    """Return (ci_low, ci_high) lists when both keys exist and lengths match ``y``."""
    lo_k = lo_key or f"{y_key}_ci_low"
    hi_k = hi_key or f"{y_key}_ci_high"
    if lo_k not in data or hi_k not in data:
        # Common alternate naming: strip trailing series suffix patterns already covered.
        return None, None
    y = list(data[y_key])
    lo = list(data[lo_k])
    hi = list(data[hi_k])
    if len(lo) != len(y) or len(hi) != len(y):
        return None, None
    return lo, hi


def _should_draw_ci(with_ci: bool | None, lo: Sequence[float] | None, hi: Sequence[float] | None) -> bool:
    if lo is None or hi is None:
        return False
    if with_ci is False:
        return False
    # with_ci True or None (auto): draw when CI data is present
    return True


def _plot_line_with_ci_band(
    ax: plt.Axes,
    x: Sequence[float] | np.ndarray,
    y: Sequence[float],
    *,
    label: str,
    fmt: str = "o-",
    ci_low: Sequence[float] | None = None,
    ci_high: Sequence[float] | None = None,
    draw_ci: bool = True,
) -> None:
    """
    Plot a series line; when CI bounds are provided, shade a translucent band
    (thick soft border) around the line.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    (line,) = ax.plot(x_arr, y_arr, fmt, label=label, zorder=3)
    color = line.get_color()

    if not draw_ci or ci_low is None or ci_high is None:
        return
    lo = np.asarray(ci_low, dtype=float)
    hi = np.asarray(ci_high, dtype=float)
    if lo.shape != y_arr.shape or hi.shape != y_arr.shape:
        return
    if not (_finite(lo.tolist()) and _finite(hi.tolist()) and _finite(y_arr.tolist())):
        # Still shade finite segments where possible
        mask = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(y_arr) & np.isfinite(x_arr)
        if not np.any(mask):
            return
        x_arr, y_arr, lo, hi = x_arr[mask], y_arr[mask], lo[mask], hi[mask]

    # Soft wide stroke under the line (visual “border”), then CI envelope fill.
    ax.plot(
        x_arr,
        y_arr,
        color=color,
        alpha=0.18,
        linewidth=7.0,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=1,
    )
    ax.fill_between(x_arr, lo, hi, color=color, alpha=0.28, linewidth=0, zorder=2)


def plot_fpr_vs_code_length(
    data: dict,
    *,
    output_png: Path | None = None,
    with_ci: bool | None = None,
    show: bool = True,
) -> Path:
    lengths = data["code_lengths"]
    series = [
        (
            "scheme_fpr_all_runs",
            "Scheme FPR (all runs)",
            "o-",
            "scheme_fpr_all_ci_low",
            "scheme_fpr_all_ci_high",
        ),
        (
            "scheme_fpr_x_matched_runs_only",
            "Scheme FPR (x matched)",
            "s--",
            "scheme_fpr_x_matched_ci_low",
            "scheme_fpr_x_matched_ci_high",
        ),
        (
            "prc_random_detect_rate",
            "PRC baseline",
            "^-",
            "prc_random_detect_rate_ci_low",
            "prc_random_detect_rate_ci_high",
        ),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    any_ci = False
    for y_key, label, fmt, lo_k, hi_k in series:
        y = list(data[y_key])
        lo, hi = _resolve_ci(data, y_key, lo_key=lo_k, hi_key=hi_k)
        draw = _should_draw_ci(with_ci, lo, hi)
        any_ci = any_ci or draw
        _plot_line_with_ci_band(
            ax, lengths, y, label=label, fmt=fmt, ci_low=lo, ci_high=hi, draw_ci=draw
        )

    ax.set_xlabel("Code length (logical PRC bits)")
    ax.set_ylabel("False positive rate")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    out = output_png or Path(str(data.get("_source_path", "benchmark_fpr_vs_code_length"))).with_suffix(
        ".png"
    )
    if any_ci and output_png is None and with_ci is not False:
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
    with_ci: bool | None = None,
    show: bool = True,
) -> Path:
    # Prefer blowup key (newer JSON); fall back to redundancy.
    if "wm_channel_blowup" in data:
        x_vals = list(data["wm_channel_blowup"])
        x_label = "WM channel blowup / channel length"
    else:
        x_vals = list(data["wm_bit_redundancy"])
        x_label = "WM bit redundancy / channel length"
    code_length = int(data["code_length"])
    output_lens = [int(round(code_length * float(r))) for r in x_vals]
    tpr_all = list(data["tpr_all_runs"])
    tpr_x = list(data["tpr_x_matched_runs_only"])

    lo_all, hi_all = _resolve_ci(
        data, "tpr_all_runs", lo_key="tpr_all_runs_ci_low", hi_key="tpr_all_runs_ci_high"
    )
    lo_x, hi_x = _resolve_ci(
        data,
        "tpr_x_matched_runs_only",
        lo_key="tpr_x_matched_ci_low",
        hi_key="tpr_x_matched_ci_high",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    draw_all = _should_draw_ci(with_ci, lo_all, hi_all)
    draw_x = _should_draw_ci(with_ci, lo_x, hi_x)
    _plot_line_with_ci_band(
        ax,
        x_vals,
        tpr_all,
        label="All runs",
        fmt="o-",
        ci_low=lo_all,
        ci_high=hi_all,
        draw_ci=draw_all,
    )
    _plot_line_with_ci_band(
        ax,
        x_vals,
        tpr_x,
        label="x matched runs",
        fmt="s--",
        ci_low=lo_x,
        ci_high=hi_x,
        draw_ci=draw_x,
    )

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.0)
    ax.set_xticks(x_vals)
    ax.set_xticklabels([f"{r:g}\n({L} ch bits)" for r, L in zip(x_vals, output_lens)])
    ax.set_xlabel(x_label)
    ax.set_ylabel("True positive rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    default_stem = (
        "benchmark_tpr_vs_wm_channel_blowup"
        if "wm_channel_blowup" in data
        else "benchmark_tpr_vs_wm_bit_redundancy"
    )
    out = output_png or Path(str(data.get("_source_path", default_stem))).with_suffix(".png")
    if (draw_all or draw_x) and output_png is None and with_ci is not False:
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
    ci_low: dict | None = None,
    ci_high: dict | None = None,
    with_ci: bool | None = None,
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

    draw_ci = with_ci is not False and ci_low is not None and ci_high is not None
    if draw_ci:
        lo_df = pd.DataFrame(ci_low).T.reindex(index=row_labels, columns=col_labels)
        hi_df = pd.DataFrame(ci_high).T.reindex(index=row_labels, columns=col_labels)
        for i, row in enumerate(row_labels):
            for j, col in enumerate(col_labels):
                val = pct.values[i, j]
                if val != val:
                    continue
                lo = lo_df.loc[row, col]
                hi = hi_df.loc[row, col]
                if lo != lo or hi != hi:
                    text = f"{val:.0f}"
                else:
                    text = f"{val:.0f}\n[{100 * float(lo):.0f}-{100 * float(hi):.0f}]"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color="white" if val >= 50 else "black",
                    fontsize=7,
                )

    fig.colorbar(im, ax=ax, label="detect rate (%)")
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_png


def plot_label_matrix(
    data: dict,
    *,
    output_png: Path | None = None,
    with_ci: bool | None = None,
    show: bool = True,
) -> Path:
    vocab = list(data["vocab"])
    out = output_png or Path(
        str(data.get("_source_path", "benchmark_label_conditioned_detection_matrix"))
    ).with_suffix(".png")
    return _matrix_heatmap(
        data["numerators"],
        data["denominators"],
        row_labels=vocab,
        col_labels=vocab,
        title="Label-conditioned detection (%)",
        output_png=out,
        show=show,
        ci_low=data.get("rates_ci_low"),
        ci_high=data.get("rates_ci_high"),
        with_ci=with_ci,
    )


def plot_prompt_matrix(
    data: dict,
    *,
    output_png: Path | None = None,
    xmatch: bool = False,
    with_ci: bool | None = None,
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
            ci_low=data.get("rates_attributes_match_ci_low"),
            ci_high=data.get("rates_attributes_match_ci_high"),
            with_ci=with_ci,
        )
    return _matrix_heatmap(
        data["numerators"],
        data["denominators"],
        row_labels=vocab,
        col_labels=cols,
        title="Prompt-conditioned detection (%)",
        output_png=out,
        show=show,
        ci_low=data.get("rates_ci_low"),
        ci_high=data.get("rates_ci_high"),
        with_ci=with_ci,
    )


def plot_ber_diagnostics(
    data: dict,
    *,
    output_png: Path | None = None,
    with_ci: bool | None = None,
    show: bool = True,
) -> Path:
    results = data["results"]
    labels = [f"#{r['index'] + 1}" for r in results]
    stages = {
        "ch ids": [r["channel_ber_from_ids"] for r in results],
        "ch txt": [r["channel_ber_from_text"] for r in results],
        "logic": [r["logical_ber"] for r in results],
        "e2e": [r["end_to_end_ber_master"] for r in results],
    }
    x = np.arange(len(labels), dtype=float)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    for name, vals in stages.items():
        _plot_line_with_ci_band(ax, x, vals, label=name, fmt="o-", draw_ci=False)

    # Aggregate mean ± CI as a translucent horizontal band per stage (legend via proxy).
    aggregates = data.get("aggregates") or {}
    stage_agg_keys = {
        "ch ids": "channel_ber_from_ids",
        "ch txt": "channel_ber_from_text",
        "logic": "logical_ber",
        "e2e": "end_to_end_ber_master",
    }
    draw_agg = with_ci is not False and bool(aggregates)
    if draw_agg and len(x) > 0:
        # Match colors already assigned to stage lines.
        handles, legend_labels = ax.get_legend_handles_labels()
        color_by_label = {lab: h.get_color() for h, lab in zip(handles, legend_labels)}
        x0, x1 = float(x[0]), float(x[-1])
        for name, agg_key in stage_agg_keys.items():
            agg = aggregates.get(agg_key) or {}
            lo = agg.get("ci_low")
            hi = agg.get("ci_high")
            if lo is None or hi is None or lo != lo or hi != hi:
                continue
            color = color_by_label.get(name, "gray")
            ax.fill_between(
                [x0, x1],
                [float(lo), float(lo)],
                [float(hi), float(hi)],
                color=color,
                alpha=0.15,
                linewidth=0,
                zorder=0,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("BER (%)")
    ax.set_title("BER diagnostics by prompt")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out = output_png or Path(str(data.get("_source_path", "benchmark_ber_diagnostics"))).with_suffix(
        ".png"
    )
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
    with_ci: bool | None = None,
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
        return plot_label_matrix(data, output_png=output_png, with_ci=with_ci, show=show)
    if resolved == "prompt-matrix":
        return plot_prompt_matrix(
            data, output_png=output_png, xmatch=xmatch, with_ci=with_ci, show=show
        )
    if resolved == "ber":
        return plot_ber_diagnostics(data, output_png=output_png, with_ci=with_ci, show=show)
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
    p.set_defaults(with_ci=None)
    ci = p.add_mutually_exclusive_group()
    ci.add_argument(
        "--with-ci",
        dest="with_ci",
        action="store_const",
        const=True,
        help="Draw confidence bands/annotations when CI fields are present (default: auto).",
    )
    ci.add_argument(
        "--no-ci",
        dest="with_ci",
        action="store_const",
        const=False,
        help="Do not draw confidence intervals even if present in the JSON.",
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
