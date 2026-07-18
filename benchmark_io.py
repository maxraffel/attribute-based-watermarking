"""
JSON persistence for benchmark runs and sweep results.

Each saved file includes ``schema_version`` and ``benchmark_kind`` so ``benchmark_plot.py``
can reload and plot without re-running expensive LM trials.
"""

from __future__ import annotations

import json
import math
import shutil
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence, TypeVar

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
import randrecover
import text_attributes
import watermarking as wm

SCHEMA_VERSION = 2

BENCHMARK_KIND_POLICY = "policy_detection"
BENCHMARK_KIND_FPR_SWEEP = "fpr_vs_code_length"
BENCHMARK_KIND_TPR_SWEEP = "tpr_vs_wm_bit_redundancy"
BENCHMARK_KIND_LABEL_MATRIX = "label_conditioned_matrix"
BENCHMARK_KIND_PROMPT_MATRIX = "prompt_conditioned_matrix"
BENCHMARK_KIND_WATERMARK = "watermark_protocol"
BENCHMARK_KIND_BER = "ber_diagnostics"

# Default z for ~95% two-sided normal / Wilson intervals.
DEFAULT_CI_Z = 1.96


def wilson_score_interval(
    k: int,
    n: int,
    *,
    z: float = DEFAULT_CI_Z,
) -> tuple[float, float]:
    """Wilson score interval for binomial proportion ``k/n``."""
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"invalid wilson_score_interval args: k={k}, n={n}")
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    rad = (z / denom) * math.sqrt((p * (1.0 - p) + z2 / (4 * n)) / n)
    return (max(0.0, centre - rad), min(1.0, centre + rad))


def proportion_with_ci(
    k: int,
    n: int,
    *,
    z: float = DEFAULT_CI_Z,
) -> dict[str, float | int]:
    """Point estimate ``k/n`` plus Wilson ``ci_low`` / ``ci_high``."""
    if n <= 0:
        return {"rate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "k": int(k), "n": int(n)}
    lo, hi = wilson_score_interval(int(k), int(n), z=z)
    return {"rate": float(k) / float(n), "ci_low": lo, "ci_high": hi, "k": int(k), "n": int(n)}


def mean_with_ci(
    values: Sequence[float],
    *,
    z: float = DEFAULT_CI_Z,
) -> dict[str, float | int]:
    """Sample mean with normal-approx CI via SEM (``z * s / sqrt(n)``)."""
    xs = [float(v) for v in values if v == v]
    n = len(xs)
    if n == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0, "sem": float("nan")}
    mean = float(statistics.mean(xs))
    if n == 1:
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "n": 1, "sem": 0.0}
    sem = float(statistics.stdev(xs) / math.sqrt(n))
    return {
        "mean": mean,
        "ci_low": mean - z * sem,
        "ci_high": mean + z * sem,
        "n": n,
        "sem": sem,
    }


def rate_matrix_with_ci(
    numerators: Mapping[str, Mapping[str, int]],
    denominators: Mapping[str, Mapping[str, int]],
    *,
    z: float = DEFAULT_CI_Z,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Build rate / ci_low / ci_high matrices aligned to numerator keys."""
    rates: dict[str, dict[str, float]] = {}
    ci_low: dict[str, dict[str, float]] = {}
    ci_high: dict[str, dict[str, float]] = {}
    for row, cols in numerators.items():
        rates[row] = {}
        ci_low[row] = {}
        ci_high[row] = {}
        for col, num in cols.items():
            den = int(denominators.get(row, {}).get(col, 0))
            k = int(num)
            if den <= 0:
                rates[row][col] = float("nan")
                ci_low[row][col] = float("nan")
                ci_high[row][col] = float("nan")
            else:
                lo, hi = wilson_score_interval(k, den, z=z)
                rates[row][col] = k / den
                ci_low[row][col] = lo
                ci_high[row][col] = hi
    return rates, ci_low, ci_high

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
        "Install/build it with:\n"
        "  uv sync"
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
    """Iterate items with elapsed + ETA columns sized to the real terminal."""
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


@dataclass
class ProgressHandle:
    """Long-lived progress control for sweeps that span many protocol trials."""

    _advance: Callable[[int], None]
    _set_description: Callable[[str], None]

    def advance(self, n: int = 1) -> None:
        self._advance(int(n))

    def set_description(self, description: str) -> None:
        self._set_description(str(description))


@contextmanager
def progress_task(
    description: str,
    total: int,
    *,
    disable: bool = False,
) -> Iterator[ProgressHandle]:
    """Context manager for a single progress bar updated across nested work."""
    if disable or total <= 0:
        yield ProgressHandle(_advance=lambda _n: None, _set_description=lambda _s: None)
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
        task_id = progress.add_task(description, total=int(total))

        def _advance(n: int) -> None:
            progress.advance(task_id, n)

        def _set_description(text: str) -> None:
            progress.update(task_id, description=text)

        yield ProgressHandle(_advance=_advance, _set_description=_set_description)


def resolve_show_progress(*, quiet: bool, show_progress: bool | None) -> bool:
    """Trial progress defaults to on; ``quiet`` only suppresses tables/logs.

    Pass ``show_progress=False`` to disable bars (e.g. nested under a parent bar).
    """
    del quiet  # intentional: progress is independent of table quietness
    if show_progress is None:
        return True
    return bool(show_progress)


def baseline_max_new_tokens() -> int:
    """Token budget for baselines: burn-in + channel (matches ``watermarking.generate``)."""
    return int(wm.BURN_IN_TOKENS) + int(wm.SECURITY_PARAM) * int(wm.WM_BIT_REDUNDANCY)


@dataclass(frozen=True)
class BaselinePregenResult:
    """Aligned baseline texts for a list of prompts (1:1 with the input sequence)."""

    texts: tuple[str, ...]
    seconds: float
    batch_size: int
    max_new_tokens: int

    @property
    def amortized_seconds(self) -> float:
        n = len(self.texts)
        return (self.seconds / n) if n else 0.0


def pregenerate_baselines(
    prompts: Sequence[str],
    *,
    max_new_tokens: int | None = None,
    batch_size: int | None = None,
    quiet: bool = False,
    description: str = "Baselines",
    on_batch_done: Callable[[int], None] | None = None,
) -> BaselinePregenResult:
    """
    Prefetch baseline texts for every prompt in ``prompts`` (order-preserving).

    Uses batched HF ``generate`` with VRAM-adaptive micro-batches (OOM backoff).
    Progress advances by the number of baselines completed per micro-batch.
    ``on_batch_done(n)`` is always invoked when provided (e.g. parent sweep bars),
    including when ``quiet=True`` suppresses the local progress UI.
    """
    prompt_list = [str(p) for p in prompts]
    n = len(prompt_list)
    n_tokens = int(baseline_max_new_tokens() if max_new_tokens is None else max_new_tokens)
    if n == 0:
        return BaselinePregenResult(texts=(), seconds=0.0, batch_size=1, max_new_tokens=n_tokens)

    m, tok, device = model.load()
    t0 = time.perf_counter()

    max_input = 1
    sample_n = min(8, n)
    for p in prompt_list[:sample_n]:
        enc = randrecover.encode_prompt_for_generation(tok, p, "cpu")
        max_input = max(max_input, int(enc["input_ids"].shape[-1]))
    if batch_size is None:
        eff_bs = randrecover.suggest_baseline_batch_size(
            n, max_new_tokens=n_tokens, max_input_tokens=max_input
        )
    else:
        eff_bs = max(1, min(n, int(batch_size)))

    def _notify(k: int) -> None:
        if on_batch_done is not None:
            on_batch_done(int(k))

    if quiet:
        texts = randrecover.generate_baselines(
            m,
            tok,
            prompt_list,
            n_tokens,
            device,
            batch_size=eff_bs,
            on_batch_done=_notify if on_batch_done is not None else None,
        )
    else:
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
            task_id = progress.add_task(description, total=n)

            def _on_batch(k: int) -> None:
                progress.advance(task_id, int(k))
                _notify(k)

            texts = randrecover.generate_baselines(
                m,
                tok,
                prompt_list,
                n_tokens,
                device,
                batch_size=eff_bs,
                on_batch_done=_on_batch,
            )

    elapsed = time.perf_counter() - t0
    if not quiet:
        print(
            f"pregenerated {n} baselines in {elapsed:.2f}s "
            f"(max_new_tokens={n_tokens}, batch_size≈{eff_bs})"
        )
    return BaselinePregenResult(
        texts=tuple(texts),
        seconds=float(elapsed),
        batch_size=int(eff_bs),
        max_new_tokens=n_tokens,
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def runtime_metadata(*, llm_model_id: str | None = None) -> dict[str, Any]:
    return {
        "llm_model_id": llm_model_id or model.MODEL_ID,
        "classifier_model_id": text_attributes.get_scorer().model_id,
        "score_cutoff": text_attributes.SCORE_CUTOFF,
        "vocab": list(text_attributes.VOCABULARY),
        "sampling": dict(model.SAMPLING),
        "inference_dtype": model.inference_dtype_label(),
        "torch_compile": bool(model.TORCH_COMPILE),
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
    llm_model_id: str | None = None,
) -> Path:
    roll = {sid: _rollup_to_dict(r) for sid, r in summary.roll.items()}
    roll_xmatch = {sid: _rollup_to_dict(r) for sid, r in summary.roll_attributes_match.items()}

    # Lazy import avoids a cycle: policy_detection imports benchmark_io.
    from benchmark_policy_detection import micro_fpr_wilson, micro_tpr_wilson

    fpr, fpr_lo, fpr_hi = micro_fpr_wilson(summary.roll, summary.prompt_cases)
    tpr, tpr_lo, tpr_hi = micro_tpr_wilson(summary.roll, summary.prompt_cases)
    fpr_x, fpr_x_lo, fpr_x_hi = micro_fpr_wilson(
        summary.roll_attributes_match, summary.prompt_cases
    )
    tpr_x, tpr_x_lo, tpr_x_hi = micro_tpr_wilson(
        summary.roll_attributes_match, summary.prompt_cases
    )

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
            },
            "runtime": runtime_metadata(llm_model_id=llm_model_id),
            "prompt_cases": prompt_cases_to_json(summary.prompt_cases),
            "strict_protocol_ok": bool(summary.strict_protocol_ok),
            "roll": roll,
            "roll_attributes_match": roll_xmatch,
            "aggregates": {
                "fpr_all_runs": {
                    "rate": fpr,
                    "ci_low": fpr_lo,
                    "ci_high": fpr_hi,
                },
                "tpr_all_runs": {
                    "rate": tpr,
                    "ci_low": tpr_lo,
                    "ci_high": tpr_hi,
                },
                "fpr_x_matched_runs_only": {
                    "rate": fpr_x,
                    "ci_low": fpr_x_lo,
                    "ci_high": fpr_x_hi,
                },
                "tpr_x_matched_runs_only": {
                    "rate": tpr_x,
                    "ci_low": tpr_x_lo,
                    "ci_high": tpr_x_hi,
                },
            },
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
    rates, ci_low, ci_high = rate_matrix_with_ci(matrix.numerators, matrix.denominators)
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
            "rates": matrix.rates if getattr(matrix, "rates", None) is not None else rates,
            "rates_ci_low": ci_low,
            "rates_ci_high": ci_high,
        },
    )


def save_prompt_matrix(
    path: Path | str,
    *,
    matrix: Any,
    exit_code: int,
    llm_model_id: str | None = None,
) -> Path:
    rates, ci_low, ci_high = rate_matrix_with_ci(matrix.numerators, matrix.denominators)
    rates_x, ci_low_x, ci_high_x = rate_matrix_with_ci(
        matrix.numerators_attributes_match,
        matrix.denominators_attributes_match,
    )
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
            "rates": matrix.rates if getattr(matrix, "rates", None) is not None else rates,
            "rates_ci_low": ci_low,
            "rates_ci_high": ci_high,
            "numerators_attributes_match": matrix.numerators_attributes_match,
            "denominators_attributes_match": matrix.denominators_attributes_match,
            "rates_attributes_match": (
                matrix.rates_attributes_match
                if getattr(matrix, "rates_attributes_match", None) is not None
                else rates_x
            ),
            "rates_attributes_match_ci_low": ci_low_x,
            "rates_attributes_match_ci_high": ci_high_x,
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
    n = len(run_rows)

    def _bool_rate(key: str) -> dict[str, float | int]:
        k = sum(1 for r in run_rows if bool(r.get(key)))
        return proportion_with_ci(k, n)

    aggregates: dict[str, Any] = {
        "n_runs": n,
        "all_ok": _bool_rate("all_ok"),
        "master_detect_ok": _bool_rate("master_detect_ok"),
        "unconstrained_detect_ok": _bool_rate("unconstrained_detect_ok"),
        "neg_control_pass": _bool_rate("neg_control_pass"),
        "cprf_ok": _bool_rate("cprf_ok"),
        "attributes_match": _bool_rate("attributes_match"),
        "active_labels_match": _bool_rate("active_labels_match"),
        "recovery_ids_aligned": _bool_rate("recovery_ids_aligned"),
        "master_ber_percent": mean_with_ci(
            [float(r.get("master_ber_percent", float("nan"))) for r in run_rows]
        ),
    }
    return save_json(
        path,
        {
            "benchmark_kind": BENCHMARK_KIND_WATERMARK,
            "created_at": utc_now_iso(),
            "runtime": runtime_metadata(llm_model_id=llm_model_id),
            "config": cfg,
            "prompts": list(prompts),
            "runs": run_rows,
            "aggregates": aggregates,
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

    ber_keys = (
        "channel_ber_from_ids",
        "channel_ber_from_text",
        "retokenization_extra_ber",
        "logical_ber",
        "end_to_end_ber_master",
    )
    aggregates = {
        key: mean_with_ci([float(r[key]) for r in results]) for key in ber_keys
    }
    detect_n = len(results)
    aggregates["detect_oracle_key"] = proportion_with_ci(
        sum(1 for r in results if r.get("detect_oracle_key")), detect_n
    )
    aggregates["detect_actual_master"] = proportion_with_ci(
        sum(1 for r in results if r.get("detect_actual_master")), detect_n
    )

    return save_json(
        path,
        {
            "benchmark_kind": BENCHMARK_KIND_BER,
            "created_at": utc_now_iso(),
            "runtime": runtime_metadata(llm_model_id=llm_model_id),
            "config": cfg,
            "results": results,
            "aggregates": aggregates,
        },
    )
