"""
JSON persistence for benchmark runs and sweep results.

Each saved file includes ``schema_version`` and ``benchmark_kind`` so ``benchmark_plot.py``
can reload and plot without re-running expensive LM trials.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, TypeVar

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

import model
import text_attributes

SCHEMA_VERSION = 1

BENCHMARK_KIND_POLICY = "policy_detection"
BENCHMARK_KIND_FPR_SWEEP = "fpr_vs_code_length"
BENCHMARK_KIND_TPR_SWEEP = "tpr_vs_wm_bit_redundancy"
BENCHMARK_KIND_LABEL_MATRIX = "label_conditioned_matrix"
BENCHMARK_KIND_PROMPT_MATRIX = "prompt_conditioned_matrix"
BENCHMARK_KIND_WATERMARK = "watermark_protocol"
BENCHMARK_KIND_BER = "ber_diagnostics"

T = TypeVar("T")


def in_notebook() -> bool:
    """True when running under IPython/Jupyter (including Colab)."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ in {
            "ZMQInteractiveShell",
            "Shell",
            "TerminalInteractiveShell",
        }
    except ImportError:
        return False


def use_plain_benchmark_tables() -> bool:
    """
    Plain fixed-width tables for benchmark summaries (default on).

    Rich ``Table`` output ellipsizes cells when many columns are present; plain tables
    stay readable in terminals and notebooks. Set ``BENCHMARK_PLAIN_TABLE=0`` to opt in
    to Rich tables (not recommended).
    """
    v = os.environ.get("BENCHMARK_PLAIN_TABLE", "").strip().lower()
    if v in ("0", "false", "no", "never"):
        return False
    return True


def print_plain_table(
    *,
    title: str | None,
    headers: Sequence[str],
    widths: Sequence[int],
    aligns: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> None:
    """Print an aligned ASCII table to stdout (no Rich truncation)."""
    if len(headers) != len(widths) or len(headers) != len(aligns):
        raise ValueError("headers, widths, and aligns must have the same length")
    if title:
        print()
        print(title)

    def _fmt(cells: Sequence[str]) -> str:
        out: list[str] = []
        for val, width, align in zip(cells, widths, aligns):
            a = ">" if align == ">" else "<"
            out.append(f"{val:{a}{width}}")
        return " ".join(out)

    header_line = _fmt(headers)
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        if len(row) != len(headers):
            raise ValueError(f"row length {len(row)} != {len(headers)} headers")
        print(_fmt(row))


def require_prc_extension() -> None:
    """Raise with build instructions if the Rust ``prc`` wheel is not installed."""
    import prc

    if callable(getattr(prc, "set_code_length", None)):
        return
    raise RuntimeError(
        "PRC native extension is not installed (imported prc has no set_code_length). "
        "Build it with:\n"
        "  uv sync --extra dev\n"
        "  uv run maturin develop --release -m prc/Cargo.toml"
    )


def make_benchmark_console() -> Console:
    """Console for any remaining Rich output (tables use plain ``print`` now)."""
    return make_progress_console()


def make_progress_console() -> Console:
    """
    Console sized to the **actual** terminal width so Rich progress ETA columns stay visible.

    ``make_benchmark_console`` previously forced min width 168, which is wider than many
    terminals and caused the time-remaining column to render off-screen.
    """
    try:
        if sys.stdout.isatty() and not in_notebook():
            w = max(shutil.get_terminal_size().columns, 40)
        else:
            w = 120
    except OSError:
        w = 120
    return Console(highlight=False, width=w, soft_wrap=False)


def iter_with_progress(
    items: Sequence[T],
    *,
    description: str,
    disable: bool = False,
) -> Iterator[T]:
    """Iterate benchmark trials with elapsed + ETA columns sized to the real terminal."""
    if disable:
        yield from items
        return

    progress_console = make_progress_console()
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=32),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(compact=True, elapsed_when_finished=False),
        console=progress_console,
        transient=True,
        refresh_per_second=10,
        expand=False,
    ) as progress:
        task_id = progress.add_task(description, total=len(items))
        for item in items:
            yield item
            progress.advance(task_id)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def runtime_metadata(*, llm_model_id: str | None = None) -> dict[str, Any]:
    return {
        "llm_model_id": llm_model_id or model.MODEL_ID,
        "classifier_model_id": text_attributes.get_scorer().model_id,
        "score_cutoff": text_attributes.SCORE_CUTOFF,
        "vocab": list(text_attributes.VOCABULARY),
        "sampling": dict(model.SAMPLING),
    }


def _rollup_to_dict(rollup: Any) -> dict[str, Any]:
    d = asdict(rollup)
    return d


def save_json(path: Path | str, payload: Mapping[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = {"schema_version": SCHEMA_VERSION, **dict(payload)}
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return out


def load_json(path: Path | str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def prompt_cases_to_json(cases: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"id": sid, "prompt": prompt} for sid, prompt in cases]


def save_policy_summary(
    path: Path | str,
    *,
    summary: Any,
    exit_code: int,
    runs: int,
    fresh_key_per_trial: bool,
    llm_model_id: str | None = None,
) -> Path:
    roll = {sid: _rollup_to_dict(r) for sid, r in summary.roll.items()}
    roll_xmatch = {sid: _rollup_to_dict(r) for sid, r in summary.roll_attributes_match.items()}
    return save_json(
        path,
        {
            "benchmark_kind": BENCHMARK_KIND_POLICY,
            "created_at": utc_now_iso(),
            "exit_code": int(exit_code),
            "config": {
                "modulus": summary.modulus,
                "code_length": summary.code_length,
                "wm_bit_redundancy": summary.wm_bit_redundancy,
                "runs_per_prompt": int(runs),
                "fresh_key_per_trial": bool(fresh_key_per_trial),
            },
            "runtime": runtime_metadata(llm_model_id=llm_model_id),
            "prompt_cases": prompt_cases_to_json(summary.prompt_cases),
            "strict_protocol_ok": bool(summary.strict_protocol_ok),
            "roll": roll,
            "roll_attributes_match": roll_xmatch,
        },
    )


def save_fpr_sweep(
    path: Path | str,
    *,
    payload: Mapping[str, Any],
    llm_model_id: str | None = None,
) -> Path:
    return save_json(
        path,
        {
            "benchmark_kind": BENCHMARK_KIND_FPR_SWEEP,
            "created_at": utc_now_iso(),
            "runtime": runtime_metadata(llm_model_id=llm_model_id),
            **dict(payload),
        },
    )


def save_tpr_sweep(
    path: Path | str,
    *,
    payload: Mapping[str, Any],
    llm_model_id: str | None = None,
) -> Path:
    return save_json(
        path,
        {
            "benchmark_kind": BENCHMARK_KIND_TPR_SWEEP,
            "created_at": utc_now_iso(),
            "runtime": runtime_metadata(llm_model_id=llm_model_id),
            **dict(payload),
        },
    )


def save_label_matrix(
    path: Path | str,
    *,
    matrix: Any,
    exit_code: int,
    llm_model_id: str | None = None,
) -> Path:
    return save_json(
        path,
        {
            "benchmark_kind": BENCHMARK_KIND_LABEL_MATRIX,
            "created_at": utc_now_iso(),
            "exit_code": int(exit_code),
            "runtime": runtime_metadata(llm_model_id=llm_model_id),
            "config": {
                "modulus": matrix.modulus,
                "code_length": matrix.code_length,
                "wm_bit_redundancy": matrix.wm_bit_redundancy,
                "runs_per_prompt": matrix.runs_per_prompt,
            },
            "prompt_cases": prompt_cases_to_json(matrix.prompt_cases),
            "strict_protocol_ok": bool(matrix.strict_protocol_ok),
            "vocab": list(matrix.vocab),
            "numerators": matrix.numerators,
            "denominators": matrix.denominators,
            "rates": matrix.rates,
        },
    )


def save_prompt_matrix(
    path: Path | str,
    *,
    matrix: Any,
    exit_code: int,
    llm_model_id: str | None = None,
) -> Path:
    return save_json(
        path,
        {
            "benchmark_kind": BENCHMARK_KIND_PROMPT_MATRIX,
            "created_at": utc_now_iso(),
            "exit_code": int(exit_code),
            "runtime": runtime_metadata(llm_model_id=llm_model_id),
            "config": {
                "modulus": matrix.modulus,
                "code_length": matrix.code_length,
                "wm_bit_redundancy": matrix.wm_bit_redundancy,
                "runs_per_prompt": matrix.runs_per_prompt,
            },
            "prompt_cases": prompt_cases_to_json(matrix.prompt_cases),
            "strict_protocol_ok": bool(matrix.strict_protocol_ok),
            "vocab": list(matrix.vocab),
            "column_prompt_ids": list(matrix.column_prompt_ids),
            "numerators": matrix.numerators,
            "denominators": matrix.denominators,
            "rates": matrix.rates,
            "numerators_attributes_match": matrix.numerators_attributes_match,
            "denominators_attributes_match": matrix.denominators_attributes_match,
            "rates_attributes_match": matrix.rates_attributes_match,
        },
    )


def save_watermark_benchmark(
    path: Path | str,
    *,
    config: Any,
    prompts: Sequence[str],
    runs: Sequence[Any],
    llm_model_id: str | None = None,
) -> Path:
    cfg = config if isinstance(config, dict) else asdict(config)
    run_rows = [asdict(r) if is_dataclass(r) else dict(r) for r in runs]
    return save_json(
        path,
        {
            "benchmark_kind": BENCHMARK_KIND_WATERMARK,
            "created_at": utc_now_iso(),
            "runtime": runtime_metadata(llm_model_id=llm_model_id),
            "config": cfg,
            "prompts": list(prompts),
            "runs": run_rows,
        },
    )


def save_ber_diagnostics(
    path: Path | str,
    *,
    summary: Any,
    llm_model_id: str | None = None,
) -> Path:
    cfg = asdict(summary.config)
    results = [asdict(r) for r in summary.results]
    for row in results:
        row["partition_error_breakdown"] = asdict(row["partition_error_breakdown"])
    return save_json(
        path,
        {
            "benchmark_kind": BENCHMARK_KIND_BER,
            "created_at": utc_now_iso(),
            "runtime": runtime_metadata(llm_model_id=llm_model_id),
            "config": cfg,
            "results": results,
        },
    )
