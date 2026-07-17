"""
Policy-detection benchmark: same end-to-end flow as ``app.py`` (generate ΓåÆ verify ``derive_attributes`` ΓåÆ
issue unconstrained + one constrained key per closed-vocabulary label ΓåÆ CPRF seed checks ΓåÆ
``master_detect`` / ``detect`` on good transcript ΓåÆ negative-control ``master_detect`` on decoy).

Runs many trials over configurable prompts and code length with a Rich **progress bar** (transient).
Per-trial logging is minimal; results are **aggregated into tables**: per-prompt TPR/FNR/TNR/FPR for
policy ``detect`` vs recovered active-label expectation, counts of which protocol stages matched;
**the same policy table** computed only on runs where encode-time ``attributes`` equals verify-time
``derive_attributes`` (full vector); then mean wall time per pipeline stage.

CLI: ``--code-length``, ``--runs``, ``--modulus``, ``--reuse-key``, ``--wm-bit-redundancy``,
optional ``--llm-model``, repeatable ``--prompt-case id:prompt``.

``run_benchmark_label_conditioned_matrix`` builds a ``|V| ├ù |V|`` matrix (columns = verify-time attributed
labels); ``run_benchmark_prompt_conditioned_matrix`` builds ``|V| ├ù |P|`` (columns = benchmark prompt ids).
For notebooks / plotting: ``run_benchmark_with_summary(..., quiet=True)``
returns a ``BenchmarkRunSummary``;
``micro_fpr`` / ``micro_tpr`` pool per-label policy FPR/TPR; ``micro_fpr_wilson`` / ``micro_tpr_wilson``
add Wilson score ~95 percent intervals on those pooled proportions; ``wilson_score_interval`` applies likewise to Monte Carlo
rates; ``prc_random_detect_positive_rate`` estimates PRC ``detect`` acceptance on random bits against a random
PRC key (same spirit as ``testing.py``).
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from rich.console import Console
from rich.table import Table

import benchmark_io
import text_attributes
import model
import randrecover
import prc
import watermarking as wm
from pathlib import Path
from text_attributes import VOCABULARY, active_labels_from_attributes

make_benchmark_console = benchmark_io.make_benchmark_console


def _configure_benchmark(
    *,
    code_length: int,
    wm_bit_redundancy: int = 1,
    partition_mode: str = "static",
    redundancy_layout: str = "depth",
    llm_model_id: str | None = None,
) -> None:
    benchmark_io.require_prc_extension()
    wm.SECURITY_PARAM = code_length
    prc.set_code_length(code_length)
    wm.WM_BIT_REDUNDANCY = wm_bit_redundancy
    wm.set_partition_mode(partition_mode)
    wm.set_redundancy_layout(redundancy_layout)
    if llm_model_id:
        model.configure(model_id=llm_model_id)


def _use_plain_table() -> bool:
    return benchmark_io.use_plain_benchmark_tables()


DEFAULT_PROMPT_CASES: list[tuple[str, str]] = [
    (
        "medicine_stem_cell",
        "Explain how stem cell therapy is being used in regenerative medicine.",
    ),
    (
        "economics_min_wage",
        "Explain the economic effects of raising the minimum wage on employment and businesses.",
    ),
    (
        "art_surrealism",
        "Explain how Surrealist artists used dream imagery to challenge reality and logic.",
    ),
    (
        "software_breakthroughs",
        "Break down the most influential software breakthroughs in history.",
    ),
    (
        "sports_strategy_performance",
        "Explain the role of strategy and teamwork in achieving success in sports.",
    ),
    (
        "software_art_world",
        "Explain how software has transformed the art world.",
    ),
    (
        "sports_drake_maye",
        "Explain the economic nuance and impact of Drake Maye during his college football career at North Carolina.",
    ),
    (
        "medicine_software_practice",
        "Explain how software has transformed the practice of medicine.",
    ),
]

# Alias kept for notebook cells that import this name explicitly.
COMPREHENSIVE_PROMPT_CASES: list[tuple[str, str]] = DEFAULT_PROMPT_CASES



def parse_prompt_case(spec: str) -> tuple[str, str]:
    spec = spec.strip()
    if ":" not in spec:
        raise ValueError(f"prompt case must be 'id:prompt', got no colon: {spec!r}")
    sid, prompt = spec.split(":", 1)
    sid, prompt = sid.strip(), prompt.strip()
    if not sid or not prompt:
        raise ValueError(f"empty id or prompt in {spec!r}")
    return (sid, prompt)


def _ber_percent(secret: list[int], recovered: list[int]) -> float:
    n = len(secret)
    if n == 0:
        return 0.0
    m = min(len(recovered), n)
    errs = sum(1 for i in range(m) if int(secret[i]) != int(recovered[i]))
    errs += abs(n - len(recovered))
    return 100.0 * errs / n


def _rates(tp: int, fn: int, tn: int, fp: int) -> tuple[float, float, float, float]:
    """Return TPR, FNR, TNR, FPR in [0,1] or -1 if denominator zero."""
    tpr = (tp / (tp + fn)) if (tp + fn) > 0 else -1.0
    fnr = (fn / (tp + fn)) if (tp + fn) > 0 else -1.0
    tnr = (tn / (tn + fp)) if (tn + fp) > 0 else -1.0
    fpr = (fp / (tn + fp)) if (tn + fp) > 0 else -1.0
    return tpr, fnr, tnr, fpr


def _fmt_rate(x: float) -> str:
    if x < 0:
        return "n/a"
    return f"{100.0 * x:.2f}%"


@dataclass
class TimingTotals:
    t_setup: float = 0.0
    t_baseline_gen: float = 0.0
    t_wm_gen: float = 0.0
    t_derive_attributes: float = 0.0
    t_issue_keys: float = 0.0
    t_cprf_checks: float = 0.0
    t_master_good: float = 0.0
    t_detect_open: float = 0.0
    t_detect_per_label: float = 0.0
    t_negative_control: float = 0.0


@dataclass
class PromptRollup:
    """Aggregates over all runs for one prompt id."""

    runs: int = 0
    #: Per-label policy decisions: gold = (label in verify-time active set), pred = detect True.
    tp: int = 0
    fn: int = 0
    tn: int = 0
    fp: int = 0

    attributes_match: int = 0
    master_good: int = 0
    open_detect_good: int = 0
    unconstrained_cprf_ok: int = 0
    cprf_per_label_expect_ok: int = 0
    cprf_per_label_checks: int = 0
    control_correct: int = 0

    ber_sum: float = 0.0
    ber_max: float = 0.0

    #: When policy detect differs from attribute-based expectation ΓÇö likely cause buckets.
    mismatch_total: int = 0
    mismatch_cprf_heuristic_ok: int = 0  # seed_match aligned with CPRF-vs-attribute expectation
    mismatch_cprf_heuristic_bad: int = 0  # CPRF equality disagreed with attribute-based seed expectation
    mismatch_fn_with_matching_seeds: int = 0  # expect True, got False, seeds matched
    mismatch_fn_with_split_seeds: int = 0
    mismatch_fp_with_matching_seeds: int = 0  # expect False, got True, seeds matched (e.g. composite m)
    mismatch_fp_with_split_seeds: int = 0  # LDPC false positive suspect

    timings: TimingTotals = field(default_factory=TimingTotals)

    def add_run(
        self,
        *,
        word_stats: list[dict[str, Any]],
        attributes_match: bool,
        master_ok: bool,
        open_ok: bool,
        unconstrained_cprf_ok: bool,
        cprf_per_label_ok: int,
        cprf_per_label_n: int,
        control_ok: bool,
        ber: float,
        timings: TimingTotals,
    ) -> None:
        self.runs += 1
        if attributes_match:
            self.attributes_match += 1
        if master_ok:
            self.master_good += 1
        if open_ok:
            self.open_detect_good += 1
        if unconstrained_cprf_ok:
            self.unconstrained_cprf_ok += 1
        self.cprf_per_label_expect_ok += cprf_per_label_ok
        self.cprf_per_label_checks += cprf_per_label_n
        if control_ok:
            self.control_correct += 1
        self.ber_sum += ber
        self.ber_max = max(self.ber_max, ber)

        for row in word_stats:
            exp = bool(row["expect_detect"])
            got = bool(row["got_detect"])
            if exp and got:
                self.tp += 1
            elif exp and not got:
                self.fn += 1
            elif not exp and not got:
                self.tn += 1
            else:
                self.fp += 1

            if got != exp:
                self.mismatch_total += 1
                sm = bool(row["seed_match"])
                em = bool(row["expect_seed_match"])
                if sm == em:
                    self.mismatch_cprf_heuristic_ok += 1
                else:
                    self.mismatch_cprf_heuristic_bad += 1
                if exp and not got:
                    if sm:
                        self.mismatch_fn_with_matching_seeds += 1
                    else:
                        self.mismatch_fn_with_split_seeds += 1
                if not exp and got:
                    if sm:
                        self.mismatch_fp_with_matching_seeds += 1
                    else:
                        self.mismatch_fp_with_split_seeds += 1

        self.timings.t_setup += timings.t_setup
        self.timings.t_baseline_gen += timings.t_baseline_gen
        self.timings.t_wm_gen += timings.t_wm_gen
        self.timings.t_derive_attributes += timings.t_derive_attributes
        self.timings.t_issue_keys += timings.t_issue_keys
        self.timings.t_cprf_checks += timings.t_cprf_checks
        self.timings.t_master_good += timings.t_master_good
        self.timings.t_detect_open += timings.t_detect_open
        self.timings.t_detect_per_label += timings.t_detect_per_label
        self.timings.t_negative_control += timings.t_negative_control


@dataclass(frozen=True)
class BenchmarkRunSummary:
    """Aggregates after ``run_benchmark_with_summary`` (rollups are read-only for callers)."""

    roll: dict[str, PromptRollup]
    roll_attributes_match: dict[str, PromptRollup]
    prompt_cases: tuple[tuple[str, str], ...]
    vocab_n: int
    code_length: int
    wm_bit_redundancy: int
    modulus: int
    strict_protocol_ok: bool


@dataclass(frozen=True)
class LabelConditionedDetectionMatrix:
    """
    Matrix over closed vocabulary labels:
    - rows: constrained key label used for ``detect``
    - cols: labels active in verify-time attribution (can be multi-label per trial)
    - cell value: positive-detect count / conditioning count.
    """

    vocab: tuple[str, ...]
    numerators: dict[str, dict[str, int]]
    denominators: dict[str, dict[str, int]]
    rates: dict[str, dict[str, float]]
    prompt_cases: tuple[tuple[str, str], ...]
    runs_per_prompt: int
    code_length: int
    wm_bit_redundancy: int
    modulus: int
    strict_protocol_ok: bool


@dataclass(frozen=True)
class PromptConditionedDetectionMatrix:
    """
    Matrix ``|V| ├ù |P|`` over vocabulary rows and benchmark prompt ids as columns.

    - **rows:** constrained key label used for ``detect`` (same closed vocabulary as the label matrix).
    - **cols:** one column per ``prompt_cases`` entry (prompt id), in the order given.
    - **cell:** trials are attributed only to the prompt id for that run; denominators count
      ``detect`` opportunities on trials that had at least one verify-time attributed label
      (same skip rule as ``run_benchmark_label_conditioned_matrix`` when the active set is empty).

    ``numerators`` / ``denominators`` / ``rates`` pool **all** qualifying trials. The ``*_attributes_match`` fields
    mirror the same update rules but **only** on trials where encode-time ``attributes`` equals verify-time
    ``derive_attributes`` (full-vector match), like ``roll_attributes_match`` in ``run_benchmark_with_summary``.
    """

    vocab: tuple[str, ...]
    column_prompt_ids: tuple[str, ...]
    numerators: dict[str, dict[str, int]]
    denominators: dict[str, dict[str, int]]
    rates: dict[str, dict[str, float]]
    numerators_attributes_match: dict[str, dict[str, int]]
    denominators_attributes_match: dict[str, dict[str, int]]
    rates_attributes_match: dict[str, dict[str, float]]
    prompt_cases: tuple[tuple[str, str], ...]
    runs_per_prompt: int
    code_length: int
    wm_bit_redundancy: int
    modulus: int
    strict_protocol_ok: bool


def sum_confusion_counts(
    roll: dict[str, PromptRollup],
    prompt_cases: Sequence[tuple[str, str]],
) -> tuple[int, int, int, int]:
    """Sum TP/FN/TN/FP over every prompt id (micro pool over runs ├ù labels)."""
    tp = fn = tn = fp = 0
    for sid, _ in prompt_cases:
        r = roll[sid]
        tp += r.tp
        fn += r.fn
        tn += r.tn
        fp += r.fp
    return tp, fn, tn, fp


def micro_fpr(
    roll: dict[str, PromptRollup],
    prompt_cases: Sequence[tuple[str, str]],
) -> float:
    """Micro-averaged false positive rate for policy ``detect`` vs active-label gold."""
    tp, fn, tn, fp = sum_confusion_counts(roll, prompt_cases)
    return _rates(tp, fn, tn, fp)[3]


def micro_tpr(
    roll: dict[str, PromptRollup],
    prompt_cases: Sequence[tuple[str, str]],
) -> float:
    """Micro-averaged true positive rate for policy ``detect`` vs active-label gold."""
    tp, fn, tn, fp = sum_confusion_counts(roll, prompt_cases)
    return _rates(tp, fn, tn, fp)[0]


def wilson_score_interval(
    k: int,
    n: int,
    *,
    z: float = 1.96,
) -> tuple[float, float]:
    """Wilson score interval for binomial proportion ``k/n`` (default ``z`` = 1.96 Γëê two-sided 95%)."""
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


def micro_fpr_wilson(
    roll: dict[str, PromptRollup],
    prompt_cases: Sequence[tuple[str, str]],
    *,
    z: float = 1.96,
) -> tuple[float, float, float]:
    """Wilson-interval bounds on micro-FPR pooled over prompts (``FP / (TN+FP)``)."""
    tp, fn, tn, fp = sum_confusion_counts(roll, prompt_cases)
    n = tn + fp
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    lo, hi = wilson_score_interval(fp, n, z=z)
    return (fp / n, lo, hi)


def micro_tpr_wilson(
    roll: dict[str, PromptRollup],
    prompt_cases: Sequence[tuple[str, str]],
    *,
    z: float = 1.96,
) -> tuple[float, float, float]:
    """Wilson-interval bounds on micro-TPR pooled over prompts (``TP / (TP+FN)``)."""
    tp, fn, tn, fp = sum_confusion_counts(roll, prompt_cases)
    n = tp + fn
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    lo, hi = wilson_score_interval(tp, n, z=z)
    return (tp / n, lo, hi)


def prc_random_detect_positive_rate(
    code_length: int,
    n_trials: int,
    *,
    rng: random.Random | None = None,
) -> tuple[float, int]:
    """
    Empirical positive rate of ``prc.detect`` on **pure randomness at both ends**: one fixed random
    PRC user key ``s`` (from ``prc.key_gen_from_seed``), and each trial uses an independent length-``code_length``
    bit vector with i.i.d. fair bits (same construction as ``testing.py``).

    Returns ``(rate, false_positive_count)`` with ``rate = false_positive_count / n_trials``.
    """
    from hashlib import sha256

    import prc

    if code_length < 1:
        raise ValueError("code_length must be >= 1")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    prng = rng if rng is not None else random.Random()
    _configure_benchmark(code_length=code_length)
    s = prc.key_gen_from_seed(sha256(prng.randbytes(32)).digest())
    fp = 0
    for _ in range(n_trials):
        ri = prng.getrandbits(code_length)
        bits = [c == "1" for c in bin(ri)[2:].zfill(code_length)]
        if prc.detect(s, bits):
            fp += 1
    return fp / n_trials, fp


def run_benchmark_label_conditioned_matrix(
    *,
    prompt_cases: Sequence[tuple[str, str]],
    runs: int,
    modulus: int,
    code_length: int,
    fresh_key_per_trial: bool,
    console: Console,
    llm_model_id: str | None = None,
    wm_bit_redundancy: int = 1,
    partition_mode: str = "static",
    redundancy_layout: str = "depth",
    quiet: bool = False,
) -> tuple[int, LabelConditionedDetectionMatrix]:
    """
    Build a ``|V| x |V|`` matrix conditioned on verify-time active labels.

    For each trial, let ``A`` be the active-label set from verify-time ``derive_attributes``.
    For every column label ``c in A`` and every row label ``r in VOCABULARY``:
      - denominator[r,c] += 1
      - numerator[r,c] += 1 iff ``detect(issue([r]), wm_text)`` is True.
    """
    for name in ("httpx", "httpcore", "huggingface_hub", "urllib3", "text_attributes"):
        logging.getLogger(name).setLevel(logging.WARNING)

    if runs < 1:
        raise ValueError("runs must be >= 1")
    if code_length < 1:
        raise ValueError("code_length must be >= 1")
    if wm_bit_redundancy < 1:
        raise ValueError("wm_bit_redundancy must be >= 1")

    _configure_benchmark(
        code_length=code_length,
        wm_bit_redundancy=wm_bit_redundancy,
        partition_mode=partition_mode,
        redundancy_layout=redundancy_layout,
        llm_model_id=llm_model_id,
    )
    vocab = tuple(VOCABULARY)
    numerators: dict[str, dict[str, int]] = {
        r: {c: 0 for c in vocab} for r in vocab
    }
    denominators: dict[str, dict[str, int]] = {
        r: {c: 0 for c in vocab} for r in vocab
    }
    sk_shared: dict[str, Any] = {}
    sid_runs: dict[str, int] = {sid: 0 for sid, _ in prompt_cases}
    sid_master: dict[str, int] = {sid: 0 for sid, _ in prompt_cases}
    sid_open: dict[str, int] = {sid: 0 for sid, _ in prompt_cases}
    sid_ucprf: dict[str, int] = {sid: 0 for sid, _ in prompt_cases}
    sid_control: dict[str, int] = {sid: 0 for sid, _ in prompt_cases}

    if not quiet:
        console.print(
            f"matrix benchmark  code_length={wm.SECURITY_PARAM}  "
            f"wm_bit_redundancy={wm.WM_BIT_REDUNDANCY}  partition_mode={wm.PARTITION_MODE}  "
            f"redundancy_layout={wm.REDUNDANCY_LAYOUT}  "
            f"modulus={modulus}  runs={runs}  |V|={len(vocab)}  "
            f"keys={'fresh per trial' if fresh_key_per_trial else 'reuse per prompt id'}  "
            f"llm={model.MODEL_ID!r}"
        )

    trials = [(run_i, sid, prompt) for run_i in range(runs) for sid, prompt in prompt_cases]
    for _, sid, prompt in benchmark_io.iter_with_progress(
        trials,
        description="Benchmark matrix",
        disable=quiet,
    ):
        if fresh_key_per_trial:
            sk = wm.setup(modulus)
        else:
            if sk_shared.get(sid) is None:
                sk_shared[sid] = wm.setup(modulus)
            sk = sk_shared[sid]

        (
            word_stats,
            _attributes_match,
            master_ok,
            open_ok,
            _cprf_label_ok,
            _cprf_label_n,
            control_ok,
            _ber,
            _tt_inner,
            em_open_m,
        ) = run_one_trial(sk, prompt)
        sid_runs[sid] += 1
        if master_ok:
            sid_master[sid] += 1
        if open_ok:
            sid_open[sid] += 1
        if em_open_m:
            sid_ucprf[sid] += 1
        if control_ok:
            sid_control[sid] += 1

        got_by_row = {str(row["word"]): bool(row["got_detect"]) for row in word_stats}
        # Columns are the labels attributed to this trial's WM text (multi-label possible).
        attributed_cols = tuple(
            str(row["word"]) for row in word_stats if bool(row["expect_detect"])
        )
        if not attributed_cols:
            continue

        # Rule 1: for each attributed label column, +1 denominator in every row.
        for col in attributed_cols:
            for row in vocab:
                denominators[row][col] += 1

        # Rule 2: for each row-detection that is positive, +1 numerator in that row and
        # each attributed column only (i.e., same columns used above).
        for row in vocab:
            if not got_by_row.get(row, False):
                continue
            for col in attributed_cols:
                numerators[row][col] += 1

    all_ok = True
    for sid, _ in prompt_cases:
        n = sid_runs[sid]
        if n == 0:
            continue
        ok = (
            sid_master[sid] == n
            and sid_open[sid] == n
            and sid_ucprf[sid] == n
            and sid_control[sid] == n
        )
        all_ok = all_ok and ok

    rates: dict[str, dict[str, float]] = {r: {} for r in vocab}
    for r in vocab:
        for c in vocab:
            d = denominators[r][c]
            n = numerators[r][c]
            rates[r][c] = (n / d) if d > 0 else -1.0

    matrix = LabelConditionedDetectionMatrix(
        vocab=vocab,
        numerators=numerators,
        denominators=denominators,
        rates=rates,
        prompt_cases=tuple((sid, p) for sid, p in prompt_cases),
        runs_per_prompt=runs,
        code_length=code_length,
        wm_bit_redundancy=wm.WM_BIT_REDUNDANCY,
        modulus=modulus,
        strict_protocol_ok=all_ok,
    )
    return (0 if all_ok else 1, matrix)


def run_benchmark_prompt_conditioned_matrix(
    *,
    prompt_cases: Sequence[tuple[str, str]],
    runs: int,
    modulus: int,
    code_length: int,
    fresh_key_per_trial: bool,
    console: Console,
    llm_model_id: str | None = None,
    wm_bit_redundancy: int = 1,
    partition_mode: str = "static",
    redundancy_layout: str = "depth",
    quiet: bool = False,
) -> tuple[int, PromptConditionedDetectionMatrix]:
    """
    Build a ``|V| ├ù |P|`` matrix: rows are constrained-key labels, columns are benchmark prompt ids.

    For each trial on prompt id ``p``, let ``A`` be the verify-time active label set (non-empty
    required, same as the label-conditioned matrix). Then:

    - **Denominator pass:** for column ``p`` only, ``den[r,p] += 1`` for every vocabulary row ``r``.
    - **Numerator pass:** for each row ``r`` with positive ``detect``, ``num[r,p] += 1``.

    The returned matrix also includes ``*_attributes_match`` tallies over the same rules but **only** for
    trials where encode-time ``attributes`` equals verify-time ``derive_attributes`` (full vector).
    """
    for name in ("httpx", "httpcore", "huggingface_hub", "urllib3", "text_attributes"):
        logging.getLogger(name).setLevel(logging.WARNING)

    if runs < 1:
        raise ValueError("runs must be >= 1")
    if code_length < 1:
        raise ValueError("code_length must be >= 1")
    if wm_bit_redundancy < 1:
        raise ValueError("wm_bit_redundancy must be >= 1")

    _configure_benchmark(
        code_length=code_length,
        wm_bit_redundancy=wm_bit_redundancy,
        partition_mode=partition_mode,
        redundancy_layout=redundancy_layout,
        llm_model_id=llm_model_id,
    )
    vocab = tuple(VOCABULARY)
    col_ids = tuple(str(sid) for sid, _ in prompt_cases)
    numerators: dict[str, dict[str, int]] = {r: {p: 0 for p in col_ids} for r in vocab}
    denominators: dict[str, dict[str, int]] = {r: {p: 0 for p in col_ids} for r in vocab}
    numerators_attributes_match: dict[str, dict[str, int]] = {r: {p: 0 for p in col_ids} for r in vocab}
    denominators_attributes_match: dict[str, dict[str, int]] = {r: {p: 0 for p in col_ids} for r in vocab}
    sk_shared: dict[str, Any] = {}
    sid_runs: dict[str, int] = {sid: 0 for sid, _ in prompt_cases}
    sid_master: dict[str, int] = {sid: 0 for sid, _ in prompt_cases}
    sid_open: dict[str, int] = {sid: 0 for sid, _ in prompt_cases}
    sid_ucprf: dict[str, int] = {sid: 0 for sid, _ in prompt_cases}
    sid_control: dict[str, int] = {sid: 0 for sid, _ in prompt_cases}

    if not quiet:
        console.print(
            f"prompt-matrix benchmark  code_length={wm.SECURITY_PARAM}  "
            f"wm_bit_redundancy={wm.WM_BIT_REDUNDANCY}  partition_mode={wm.PARTITION_MODE}  "
            f"redundancy_layout={wm.REDUNDANCY_LAYOUT}  "
            f"modulus={modulus}  runs={runs}  |V|={len(vocab)}  "
            f"|P|={len(col_ids)}  keys={'fresh per trial' if fresh_key_per_trial else 'reuse per prompt id'}  "
            f"llm={model.MODEL_ID!r}"
        )

    trials = [(run_i, sid, prompt) for run_i in range(runs) for sid, prompt in prompt_cases]
    for _, sid, prompt in benchmark_io.iter_with_progress(
        trials,
        description="Benchmark prompt matrix",
        disable=quiet,
    ):
        if fresh_key_per_trial:
            sk = wm.setup(modulus)
        else:
            if sk_shared.get(sid) is None:
                sk_shared[sid] = wm.setup(modulus)
            sk = sk_shared[sid]

        (
            word_stats,
            attributes_match,
            master_ok,
            open_ok,
            _cprf_label_ok,
            _cprf_label_n,
            control_ok,
            _ber,
            _tt_inner,
            em_open_m,
        ) = run_one_trial(sk, prompt)
        sid_runs[sid] += 1
        if master_ok:
            sid_master[sid] += 1
        if open_ok:
            sid_open[sid] += 1
        if em_open_m:
            sid_ucprf[sid] += 1
        if control_ok:
            sid_control[sid] += 1

        got_by_row = {str(row["word"]): bool(row["got_detect"]) for row in word_stats}
        attributed_cols = tuple(
            str(row["word"]) for row in word_stats if bool(row["expect_detect"])
        )
        if not attributed_cols:
            continue

        col_sid = str(sid)
        for row in vocab:
            denominators[row][col_sid] += 1
        for row in vocab:
            if not got_by_row.get(row, False):
                continue
            numerators[row][col_sid] += 1

        if attributes_match:
            for row in vocab:
                denominators_attributes_match[row][col_sid] += 1
            for row in vocab:
                if not got_by_row.get(row, False):
                    continue
                numerators_attributes_match[row][col_sid] += 1

    all_ok = True
    for sid, _ in prompt_cases:
        n = sid_runs[sid]
        if n == 0:
            continue
        ok = (
            sid_master[sid] == n
            and sid_open[sid] == n
            and sid_ucprf[sid] == n
            and sid_control[sid] == n
        )
        all_ok = all_ok and ok

    rates: dict[str, dict[str, float]] = {r: {} for r in vocab}
    rates_attributes_match: dict[str, dict[str, float]] = {r: {} for r in vocab}
    for r in vocab:
        for p in col_ids:
            d = denominators[r][p]
            n = numerators[r][p]
            rates[r][p] = (n / d) if d > 0 else -1.0
            dx = denominators_attributes_match[r][p]
            nx = numerators_attributes_match[r][p]
            rates_attributes_match[r][p] = (nx / dx) if dx > 0 else -1.0

    matrix = PromptConditionedDetectionMatrix(
        vocab=vocab,
        column_prompt_ids=col_ids,
        numerators=numerators,
        denominators=denominators,
        rates=rates,
        numerators_attributes_match=numerators_attributes_match,
        denominators_attributes_match=denominators_attributes_match,
        rates_attributes_match=rates_attributes_match,
        prompt_cases=tuple((sid, p) for sid, p in prompt_cases),
        runs_per_prompt=runs,
        code_length=code_length,
        wm_bit_redundancy=wm.WM_BIT_REDUNDANCY,
        modulus=modulus,
        strict_protocol_ok=all_ok,
    )
    return (0 if all_ok else 1, matrix)


def derive_verify_attributes(wm_text: str, modulus: int) -> list[int]:
    return text_attributes.derive_attributes(
        wm_text,
        modulus,
        log_scores=False,
        scores_out={},
    )


def run_one_trial(
    sk: Any,
    prompt: str,
) -> tuple[
    list[dict[str, Any]],
    bool,
    bool,
    bool,
    int,
    int,
    bool,
    float,
    TimingTotals,
    bool,
]:
    """
    One full app.py-shaped trial. Returns word-level rows, flags, CPRF sub-counts, BER, timings,
    and whether unconstrained CPRF seeds matched.
    """
    tt = TimingTotals()

    out = wm.generate(sk, prompt)
    wm_text = out["generated_text_wm"]
    encode_attributes = list(out["attributes"])
    secret = list(out["prc_secret_bits"])
    t_baseline = float(out["seconds_baseline_gen"])
    t_wm = float(out["seconds_watermarked_gen"])
    tt.t_baseline_gen = t_baseline
    tt.t_wm_gen = t_wm

    t_d0 = time.perf_counter()
    verify_attributes = derive_verify_attributes(wm_text, sk.modulus)
    tt.t_derive_attributes = time.perf_counter() - t_d0

    active_wm = active_labels_from_attributes(verify_attributes, sk.modulus)
    active_set = set(active_wm)
    attributes_match = encode_attributes == verify_attributes

    t_k0 = time.perf_counter()
    dk_open = wm.issue(sk, [])
    dk_by_word = {w: wm.issue(sk, [w]) for w in VOCABULARY}
    tt.t_issue_keys = time.perf_counter() - t_k0

    attrs = list(verify_attributes)
    t_c0 = time.perf_counter()
    em_master = sk.eval(attrs)
    ec_open = dk_open.c_eval(attrs)
    em_open_m = em_master == ec_open
    cprf_per_label_ok = 0
    cprf_per_label_n = len(VOCABULARY)
    seed_match_by_word: dict[str, bool] = {}
    for w in VOCABULARY:
        dk = dk_by_word[w]
        expect_sm = w in active_set
        sm = em_master == dk.c_eval(attrs)
        seed_match_by_word[w] = sm
        if sm == expect_sm:
            cprf_per_label_ok += 1
    tt.t_cprf_checks = time.perf_counter() - t_c0

    t_m0 = time.perf_counter()
    recovered_wm = wm.recover_channel_bits(wm_text, generation_out=out)
    m_ok, m_bits = wm.master_detect(sk, wm_text, recovered_bits=recovered_wm)
    tt.t_master_good = time.perf_counter() - t_m0
    ber = _ber_percent(secret, m_bits)

    t_u0 = time.perf_counter()
    u_ok, _ = wm.detect(dk_open, wm_text, recovered_bits=recovered_wm)
    tt.t_detect_open = time.perf_counter() - t_u0

    word_stats: list[dict[str, Any]] = []
    t_pv0 = time.perf_counter()
    for w in VOCABULARY:
        expect_detect = w in active_set
        got, _ = wm.detect(dk_by_word[w], wm_text, recovered_bits=recovered_wm)
        word_stats.append(
            {
                "word": w,
                "expect_detect": expect_detect,
                "got_detect": got,
                "expect_seed_match": expect_detect,
                "seed_match": seed_match_by_word[w],
            }
        )
    tt.t_detect_per_label = time.perf_counter() - t_pv0

    t_nc0 = time.perf_counter()
    m, tok, device = model.load()
    wrong = randrecover.negative_control_transcript_like(
        wm_text,
        tok,
        device,
        n_bits=wm.SECURITY_PARAM * wm.WM_BIT_REDUNDANCY,
        model=m,
    )
    recovered_neg = wm.recover_channel_bits(wrong)
    ctrl_ok_raw, _ = wm.master_detect(sk, wrong, recovered_bits=recovered_neg)
    control_ok = not bool(ctrl_ok_raw)
    tt.t_negative_control = time.perf_counter() - t_nc0

    return (
        word_stats,
        attributes_match,
        bool(m_ok),
        bool(u_ok),
        cprf_per_label_ok,
        cprf_per_label_n,
        control_ok,
        ber,
        tt,
        em_open_m,
    )


def _mean_timings(r: PromptRollup) -> dict[str, float]:
    n = r.runs
    if n == 0:
        return {}
    t = r.timings
    keys = (
        "t_setup",
        "t_baseline_gen",
        "t_wm_gen",
        "t_derive_attributes",
        "t_issue_keys",
        "t_cprf_checks",
        "t_master_good",
        "t_detect_open",
        "t_detect_per_label",
        "t_negative_control",
    )
    return {k: getattr(t, k) / n for k in keys}


def _aggregate_timing_means(
    roll: dict[str, PromptRollup],
    prompt_cases: Sequence[tuple[str, str]],
) -> dict[str, float]:
    """Grand mean over all runs (all prompts): sum component / total runs."""
    total_runs = 0
    acc = {k: 0.0 for k in (
        "t_setup",
        "t_baseline_gen",
        "t_wm_gen",
        "t_derive_attributes",
        "t_issue_keys",
        "t_cprf_checks",
        "t_master_good",
        "t_detect_open",
        "t_detect_per_label",
        "t_negative_control",
    )}
    for sid, _ in prompt_cases:
        r = roll[sid]
        n = r.runs
        if n == 0:
            continue
        total_runs += n
        t = r.timings
        for k in acc:
            acc[k] += getattr(t, k)
    if total_runs == 0:
        return {**acc, "t_grand_avg": 0.0}
    for k in acc:
        acc[k] /= total_runs
    acc["t_grand_avg"] = sum(acc.values())
    return acc


def _sid_cell(sid: str, width: int = 22) -> str:
    if len(sid) <= width:
        return sid
    return sid[: width - 3] + "..."


def _print_timing_table_plain(
    roll: dict[str, PromptRollup],
    prompt_cases: Sequence[tuple[str, str]],
) -> None:
    def _rows(keys: Sequence[str]) -> list[list[str]]:
        out: list[list[str]] = []
        for sid, _ in prompt_cases:
            r = roll[sid]
            if r.runs == 0:
                continue
            m = _mean_timings(r)
            out.append([_sid_cell(sid)] + [f"{m[k]:.4f}" for k in keys])
        gm = _aggregate_timing_means(roll, prompt_cases)
        out.append(["ALL (mean/run)"] + [f"{gm[k]:.4f}" for k in keys])
        return out

    benchmark_io.print_plain_table(
        title="Mean wall time — generation & attributes (seconds per run)",
        headers=["prompt_id", "setup", "baseline", "wm_gen", "derive_attr"],
        widths=[22, 8, 10, 10, 11],
        aligns=["<", ">", ">", ">", ">"],
        rows=_rows(
            ("t_setup", "t_baseline_gen", "t_wm_gen", "t_derive_attributes"),
        ),
    )
    benchmark_io.print_plain_table(
        title="Mean wall time — keys & detection (seconds per run)",
        headers=["prompt_id", "issue", "cprf", "master", "open", "labels", "decoy"],
        widths=[22, 8, 8, 8, 8, 8, 8],
        aligns=["<", ">", ">", ">", ">", ">", ">"],
        rows=_rows(
            (
                "t_issue_keys",
                "t_cprf_checks",
                "t_master_good",
                "t_detect_open",
                "t_detect_per_label",
                "t_negative_control",
            ),
        ),
    )
    gm = _aggregate_timing_means(roll, prompt_cases)
    print(f"Sum of all listed stage means (per run): {gm['t_grand_avg']:.4f} s")


def _print_timing_rich_table(
    roll: dict[str, PromptRollup],
    prompt_cases: Sequence[tuple[str, str]],
    console: Console,
) -> None:
    console.print()
    table = Table(
        title="Mean wall time per pipeline stage (s; avg over runs)",
        expand=True,
    )
    table.add_column("prompt_id", style="dim", min_width=18, overflow="fold")
    for col in (
        "setup",
        "baseline",
        "wm_gen",
        "derive_attributes",
        "issue_keys",
        "cprf",
        "m_good",
        "det_open",
        "det_vocab",
        "neg_ctrl",
    ):
        table.add_column(col, justify="right", min_width=8, no_wrap=True, overflow="fold")

    for sid, _ in prompt_cases:
        r = roll[sid]
        if r.runs == 0:
            continue
        m = _mean_timings(r)
        table.add_row(
            sid,
            f"{m.get('t_setup', 0):.4f}",
            f"{m['t_baseline_gen']:.4f}",
            f"{m['t_wm_gen']:.4f}",
            f"{m['t_derive_attributes']:.4f}",
            f"{m['t_issue_keys']:.4f}",
            f"{m['t_cprf_checks']:.4f}",
            f"{m['t_master_good']:.4f}",
            f"{m['t_detect_open']:.4f}",
            f"{m['t_detect_per_label']:.4f}",
            f"{m['t_negative_control']:.4f}",
        )

    gm = _aggregate_timing_means(roll, prompt_cases)
    table.add_row(
        "[bold]ALL/run[/]",
        f"{gm['t_setup']:.4f}",
        f"{gm['t_baseline_gen']:.4f}",
        f"{gm['t_wm_gen']:.4f}",
        f"{gm['t_derive_attributes']:.4f}",
        f"{gm['t_issue_keys']:.4f}",
        f"{gm['t_cprf_checks']:.4f}",
        f"{gm['t_master_good']:.4f}",
        f"{gm['t_detect_open']:.4f}",
        f"{gm['t_detect_per_label']:.4f}",
        f"{gm['t_negative_control']:.4f}",
    )
    console.print(table)
    console.print(f"[dim]Sum of listed stage means per run:[/] {gm['t_grand_avg']:.4f} s")


def _print_plain_results(
    *,
    prompt_cases: Sequence[tuple[str, str]],
    roll: dict[str, PromptRollup],
    vocab_n: int,
    table_heading: str,
    print_legend: bool,
) -> None:
    def ratio(a: int, b: int) -> str:
        return f"{a}/{b}"

    rate_rows: list[list[str]] = []
    check_rows: list[list[str]] = []
    for sid, _ in prompt_cases:
        r = roll[sid]
        n = r.runs
        sid_disp = _sid_cell(sid)
        if n == 0:
            rate_rows.append([sid_disp, "0", "n/a", "n/a", "n/a", "n/a"])
            check_rows.append(
                [sid_disp, "0", "0/0", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"]
            )
            continue
        tpr, fnr, tnr, fpr = _rates(r.tp, r.fn, r.tn, r.fp)
        lcprf = (
            ratio(r.cprf_per_label_expect_ok, r.cprf_per_label_checks)
            if r.cprf_per_label_checks
            else "n/a"
        )
        mism = str(r.mismatch_total) if r.mismatch_total else "0"
        rate_rows.append(
            [sid_disp, str(n), _fmt_rate(tpr), _fmt_rate(fnr), _fmt_rate(tnr), _fmt_rate(fpr)]
        )
        check_rows.append(
            [
                sid_disp,
                str(n),
                ratio(r.attributes_match, n),
                ratio(r.master_good, n),
                ratio(r.open_detect_good, n),
                ratio(r.unconstrained_cprf_ok, n),
                lcprf,
                ratio(r.control_correct, n),
                f"{r.ber_sum / n:.2f}",
                f"{r.ber_max:.2f}",
                mism,
                str(r.mismatch_cprf_heuristic_bad),
                str(r.mismatch_fn_with_matching_seeds),
                str(r.mismatch_fp_with_split_seeds),
            ]
        )

    print()
    print(table_heading)
    benchmark_io.print_plain_table(
        title="Policy rates (pooled over runs x vocab labels)",
        headers=["prompt_id", "runs", "TPR", "FNR", "TNR", "FPR"],
        widths=[22, 5, 8, 8, 8, 8],
        aligns=["<", ">", ">", ">", ">", ">"],
        rows=rate_rows,
    )
    benchmark_io.print_plain_table(
        title="Protocol checks & mismatches",
        headers=[
            "prompt_id",
            "runs",
            "attr==",
            "master",
            "open",
            "uCPRF",
            "lCPRF",
            "decoy",
            "BER_avg",
            "BER_max",
            "mism",
            "dCP",
            "FN*",
            "FP*",
        ],
        widths=[22, 5, 7, 7, 7, 7, 7, 7, 8, 8, 5, 4, 4, 4],
        aligns=["<", ">", ">", ">", ">", ">", ">", ">", ">", ">", ">", ">", ">", ">"],
        rows=check_rows,
    )

    if print_legend:
        print()
        print(
            "Legend: TPR/FNR/TNR/FPR use gold = label in verify-time active set, pred = detect True. "
            "master/open/decoy = successes/runs. uCPRF = unconstrained CPRF seed match. "
            "lCPRF = per-label CPRF expectation hits/checks. "
            "BER_avg = mean master-path BER per run; BER_max = worst run. "
            "mism = attribute-vs-detect mismatches; dCP = CPRF heuristic disagreements; "
            "FN* = FN with matching seeds; FP* = FP with split seeds."
        )


def _print_rich_results(
    *,
    prompt_cases: Sequence[tuple[str, str]],
    roll: dict[str, PromptRollup],
    vocab_n: int,
    console: Console,
    table_title: str,
    print_legend: bool,
) -> None:
    console.print()
    table = Table(title=table_title, expand=True)
    table.add_column("prompt_id", style="dim", min_width=18, overflow="fold")
    table.add_column("runs", justify="right", min_width=4, no_wrap=True)
    for col in (
        "TPR",
        "FNR",
        "TNR",
        "FPR",
        "attr==",
        "master",
        "open",
        "uCPRF",
        "lCPRF",
        "ctrl",
        "BER_avg",
        "BER_max",
        "mism",
        "dCP",
        "FN*",
        "FP*",
    ):
        table.add_column(col, justify="right", min_width=7, no_wrap=True, overflow="fold")

    for sid, _ in prompt_cases:
        r = roll[sid]
        n = r.runs
        if n == 0:
            table.add_row(
                sid,
                "0",
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                "0/0",
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                "n/a",
            )
            continue
        tpr, fnr, tnr, fpr = _rates(r.tp, r.fn, r.tn, r.fp)
        lcprf = (
            f"{r.cprf_per_label_expect_ok}/{r.cprf_per_label_checks}"
            if r.cprf_per_label_checks
            else "n/a"
        )
        table.add_row(
            sid,
            str(n),
            _fmt_rate(tpr),
            _fmt_rate(fnr),
            _fmt_rate(tnr),
            _fmt_rate(fpr),
            f"{r.attributes_match}/{n}",
            f"{r.master_good}/{n}",
            f"{r.open_detect_good}/{n}",
            f"{r.unconstrained_cprf_ok}/{n}",
            lcprf,
            f"{r.control_correct}/{n}",
            f"{r.ber_sum / n:.2f}",
            f"{r.ber_max:.2f}",
            str(r.mismatch_total),
            str(r.mismatch_cprf_heuristic_bad),
            str(r.mismatch_fn_with_matching_seeds),
            str(r.mismatch_fp_with_split_seeds),
        )
    console.print(table)
    if print_legend:
        console.print(
            "[dim]Gold positive = label in verify-time active set; pred positive = detect True. "
            "uCPRF = unconstrained sk.eval==dk.c_eval. lCPRF = per-label CPRF expectation hits / checks. "
            "mism = attribute-vs-detect mismatches; dCP = mismatches where CPRF vs attribute expectation disagreed; "
            "FN* = FN with matching seeds (suspect LDPC); FP* = FP with split seeds (suspect LDPC).[/]"
        )


def _strict_protocol_ok(
    roll: dict[str, PromptRollup],
    prompt_cases: Sequence[tuple[str, str]],
) -> bool:
    all_ok = True
    for sid, _ in prompt_cases:
        r = roll[sid]
        n = r.runs
        if n == 0:
            continue
        ok = (
            r.master_good == n
            and r.open_detect_good == n
            and r.unconstrained_cprf_ok == n
            and r.control_correct == n
        )
        all_ok = all_ok and ok
    return all_ok


def _print_protocol_failure_details(
    *,
    roll: dict[str, PromptRollup],
    prompt_cases: Sequence[tuple[str, str]],
    plain: bool,
    console: Console,
) -> None:
    msg = (
        "Protocol checks did not pass on every run (require: master_detect good, detect open, "
        "unconstrained CPRF match, negative control rejects)."
    )
    if plain:
        print()
        print(msg)
        for sid, _ in prompt_cases:
            r = roll[sid]
            n = r.runs
            if n == 0:
                continue
            bad: list[str] = []
            if r.master_good < n:
                bad.append(f"master {r.master_good}/{n}")
            if r.open_detect_good < n:
                bad.append(f"open {r.open_detect_good}/{n}")
            if r.unconstrained_cprf_ok < n:
                bad.append(f"u_cprf {r.unconstrained_cprf_ok}/{n}")
            if r.control_correct < n:
                bad.append(f"control {r.control_correct}/{n}")
            if bad:
                print(f"  {sid}: " + ", ".join(bad))
    else:
        console.print()
        console.print(f"[bold red]{msg}[/]")
        for sid, _ in prompt_cases:
            r = roll[sid]
            n = r.runs
            if n == 0:
                continue
            bad: list[str] = []
            if r.master_good < n:
                bad.append(f"master {r.master_good}/{n}")
            if r.open_detect_good < n:
                bad.append(f"open {r.open_detect_good}/{n}")
            if r.unconstrained_cprf_ok < n:
                bad.append(f"u_cprf {r.unconstrained_cprf_ok}/{n}")
            if r.control_correct < n:
                bad.append(f"control {r.control_correct}/{n}")
            if bad:
                console.print(f"  [yellow]{sid}:[/] " + ", ".join(bad))


def run_benchmark_with_summary(
    *,
    prompt_cases: Sequence[tuple[str, str]],
    runs: int,
    modulus: int,
    code_length: int,
    fresh_key_per_trial: bool,
    console: Console,
    llm_model_id: str | None = None,
    wm_bit_redundancy: int = 1,
    partition_mode: str = "static",
    redundancy_layout: str = "depth",
    quiet: bool = False,
) -> tuple[int, BenchmarkRunSummary]:
    for name in ("httpx", "httpcore", "huggingface_hub", "urllib3", "text_attributes"):
        logging.getLogger(name).setLevel(logging.WARNING)

    _configure_benchmark(
        code_length=code_length,
        wm_bit_redundancy=wm_bit_redundancy,
        partition_mode=partition_mode,
        redundancy_layout=redundancy_layout,
        llm_model_id=llm_model_id,
    )
    roll: dict[str, PromptRollup] = {sid: PromptRollup() for sid, _ in prompt_cases}
    roll_attributes_match: dict[str, PromptRollup] = {sid: PromptRollup() for sid, _ in prompt_cases}
    sk_shared: dict[str, Any] = {}

    vocab_n = len(VOCABULARY)
    if not quiet:
        console.print(
            f"code_length={wm.SECURITY_PARAM}  wm_bit_redundancy={wm.WM_BIT_REDUNDANCY}  "
            f"partition_mode={wm.PARTITION_MODE}  "
            f"redundancy_layout={wm.REDUNDANCY_LAYOUT}  "
            f"channel_bits={wm.SECURITY_PARAM * wm.WM_BIT_REDUNDANCY}  modulus={modulus}  runs={runs}  |V|={vocab_n}  "
            f"keys={'fresh per trial' if fresh_key_per_trial else 'reuse per prompt id'}  "
            f"llm={model.MODEL_ID!r}"
        )

    trials = [(run_i, sid, prompt) for run_i in range(runs) for sid, prompt in prompt_cases]
    for _, sid, prompt in benchmark_io.iter_with_progress(
        trials,
        description="Benchmark",
        disable=quiet,
    ):
        t_setup0 = time.perf_counter()
        if fresh_key_per_trial:
            sk = wm.setup(modulus)
        else:
            if sk_shared.get(sid) is None:
                sk_shared[sid] = wm.setup(modulus)
            sk = sk_shared[sid]
        t_setup = time.perf_counter() - t_setup0

        (
            word_stats,
            attributes_match,
            master_ok,
            open_ok,
            cprf_label_ok,
            cprf_label_n,
            control_ok,
            ber,
            tt_inner,
            em_open_m,
        ) = run_one_trial(sk, prompt)
        tt_inner.t_setup = t_setup

        roll[sid].add_run(
            word_stats=word_stats,
            attributes_match=attributes_match,
            master_ok=master_ok,
            open_ok=open_ok,
            unconstrained_cprf_ok=em_open_m,
            cprf_per_label_ok=cprf_label_ok,
            cprf_per_label_n=cprf_label_n,
            control_ok=control_ok,
            ber=ber,
            timings=tt_inner,
        )
        if attributes_match:
            roll_attributes_match[sid].add_run(
                word_stats=word_stats,
                attributes_match=True,
                master_ok=master_ok,
                open_ok=open_ok,
                unconstrained_cprf_ok=em_open_m,
                cprf_per_label_ok=cprf_label_ok,
                cprf_per_label_n=cprf_label_n,
                control_ok=control_ok,
                ber=ber,
                timings=tt_inner,
            )

    all_ok = _strict_protocol_ok(roll, prompt_cases)
    summary = BenchmarkRunSummary(
        roll=roll,
        roll_attributes_match=roll_attributes_match,
        prompt_cases=tuple((sid, p) for sid, p in prompt_cases),
        vocab_n=vocab_n,
        code_length=code_length,
        wm_bit_redundancy=wm.WM_BIT_REDUNDANCY,
        modulus=modulus,
        strict_protocol_ok=all_ok,
    )

    if not quiet:
        _heading_all = (
            "Per-prompt aggregates: policy detection vs label classification "
            f"(runs x |V|={vocab_n} label decisions per run)"
        )
        _heading_attributes_match = (
            "Same metrics, attributes-match runs only "
            f"(encode-time attributes == verify-time attributes; runs x |V|={vocab_n})"
        )
        _print_plain_results(
            prompt_cases=prompt_cases,
            roll=roll,
            vocab_n=vocab_n,
            table_heading=_heading_all,
            print_legend=True,
        )
        _print_plain_results(
            prompt_cases=prompt_cases,
            roll=roll_attributes_match,
            vocab_n=vocab_n,
            table_heading=_heading_attributes_match,
            print_legend=False,
        )
        _print_timing_table_plain(roll, prompt_cases)

        if not all_ok:
            _print_protocol_failure_details(
                roll=roll, prompt_cases=prompt_cases, plain=True, console=console
            )

    return (0 if all_ok else 1, summary)


def run_benchmark(
    *,
    prompt_cases: Sequence[tuple[str, str]],
    runs: int,
    modulus: int,
    code_length: int,
    fresh_key_per_trial: bool,
    console: Console,
    llm_model_id: str | None = None,
    wm_bit_redundancy: int = 1,
    partition_mode: str = "static",
    redundancy_layout: str = "depth",
) -> int:
    code, _ = run_benchmark_with_summary(
        prompt_cases=prompt_cases,
        runs=runs,
        modulus=modulus,
        code_length=code_length,
        fresh_key_per_trial=fresh_key_per_trial,
        console=console,
        llm_model_id=llm_model_id,
        wm_bit_redundancy=wm_bit_redundancy,
        partition_mode=partition_mode,
        redundancy_layout=redundancy_layout,
        quiet=False,
    )
    return code


def run_fpr_vs_code_length_sweep(
    *,
    prompt_cases: Sequence[tuple[str, str]],
    code_lengths: Sequence[int],
    runs: int,
    modulus: int,
    wm_bit_redundancy: int,
    fresh_key_per_trial: bool,
    console: Console,
    llm_model_id: str | None = None,
    partition_mode: str = "static",
    redundancy_layout: str = "depth",
    prc_monte_carlo_trials: int = 100_000,
    rng: random.Random | None = None,
    quiet: bool = True,
) -> tuple[list[int], dict[str, list[float]], list[int]]:
    """
    Sweep logical ``code_length`` and return (lengths, metric dict, exit_codes).

    Metrics keys: ``scheme_fpr_all``, ``scheme_fpr_xmatch``, ``prc_random_fpr``,
    plus ``*_ci_low`` / ``*_ci_high`` Wilson bounds where applicable.
    """
    prng = rng if rng is not None else random.Random()
    lengths: list[int] = []
    scheme_fpr_all: list[float] = []
    scheme_fpr_xmatch: list[float] = []
    scheme_fpr_all_lo: list[float] = []
    scheme_fpr_all_hi: list[float] = []
    scheme_fpr_xmatch_lo: list[float] = []
    scheme_fpr_xmatch_hi: list[float] = []
    prc_random_fpr: list[float] = []
    prc_random_fpr_lo: list[float] = []
    prc_random_fpr_hi: list[float] = []
    exit_codes: list[int] = []

    for length in code_lengths:
        r_rand, fp_mc = prc_random_detect_positive_rate(
            int(length), int(prc_monte_carlo_trials), rng=prng
        )
        prc_random_fpr.append(float(r_rand))
        pl, ph = wilson_score_interval(fp_mc, int(prc_monte_carlo_trials))
        prc_random_fpr_lo.append(pl)
        prc_random_fpr_hi.append(ph)

        ex, summary = run_benchmark_with_summary(
            prompt_cases=prompt_cases,
            runs=int(runs),
            modulus=int(modulus),
            code_length=int(length),
            fresh_key_per_trial=fresh_key_per_trial,
            console=console,
            llm_model_id=llm_model_id,
            wm_bit_redundancy=int(wm_bit_redundancy),
            partition_mode=partition_mode,
            redundancy_layout=redundancy_layout,
            quiet=quiet,
        )
        lengths.append(int(length))
        exit_codes.append(int(ex))

        fa, fa_lo, fa_hi = micro_fpr_wilson(summary.roll, summary.prompt_cases)
        fx, fx_lo, fx_hi = micro_fpr_wilson(summary.roll_attributes_match, summary.prompt_cases)
        scheme_fpr_all.append(float(fa) if fa >= 0.0 else float("nan"))
        scheme_fpr_xmatch.append(float(fx) if fx >= 0.0 else float("nan"))
        scheme_fpr_all_lo.append(fa_lo if fa >= 0.0 else float("nan"))
        scheme_fpr_all_hi.append(fa_hi if fa >= 0.0 else float("nan"))
        scheme_fpr_xmatch_lo.append(fx_lo if fx >= 0.0 else float("nan"))
        scheme_fpr_xmatch_hi.append(fx_hi if fx >= 0.0 else float("nan"))

    metrics = {
        "scheme_fpr_all_runs": scheme_fpr_all,
        "scheme_fpr_x_matched_runs_only": scheme_fpr_xmatch,
        "scheme_fpr_all_ci_low": scheme_fpr_all_lo,
        "scheme_fpr_all_ci_high": scheme_fpr_all_hi,
        "scheme_fpr_x_matched_ci_low": scheme_fpr_xmatch_lo,
        "scheme_fpr_x_matched_ci_high": scheme_fpr_xmatch_hi,
        "prc_random_detect_rate": prc_random_fpr,
        "prc_random_detect_rate_ci_low": prc_random_fpr_lo,
        "prc_random_detect_rate_ci_high": prc_random_fpr_hi,
    }
    return lengths, metrics, exit_codes


def run_tpr_vs_wm_bit_redundancy_sweep(
    *,
    prompt_cases: Sequence[tuple[str, str]],
    wm_bit_redundancy_values: Sequence[int],
    code_length: int,
    runs: int,
    modulus: int,
    fresh_key_per_trial: bool,
    console: Console,
    llm_model_id: str | None = None,
    partition_mode: str = "static",
    redundancy_layout: str = "depth",
    quiet: bool = True,
) -> tuple[list[int], dict[str, list[float]], list[int]]:
    """Sweep ``wm_bit_redundancy`` at fixed logical ``code_length``."""
    redundancies: list[int] = []
    tpr_all: list[float] = []
    tpr_xmatch: list[float] = []
    tpr_all_lo: list[float] = []
    tpr_all_hi: list[float] = []
    tpr_xmatch_lo: list[float] = []
    tpr_xmatch_hi: list[float] = []
    exit_codes: list[int] = []

    for redundancy in wm_bit_redundancy_values:
        ex, summary = run_benchmark_with_summary(
            prompt_cases=prompt_cases,
            runs=int(runs),
            modulus=int(modulus),
            code_length=int(code_length),
            fresh_key_per_trial=fresh_key_per_trial,
            console=console,
            llm_model_id=llm_model_id,
            wm_bit_redundancy=int(redundancy),
            partition_mode=partition_mode,
            redundancy_layout=redundancy_layout,
            quiet=quiet,
        )
        redundancies.append(int(redundancy))
        exit_codes.append(int(ex))

        ta, ta_lo, ta_hi = micro_tpr_wilson(summary.roll, summary.prompt_cases)
        tx, tx_lo, tx_hi = micro_tpr_wilson(summary.roll_attributes_match, summary.prompt_cases)
        tpr_all.append(float(ta) if ta >= 0.0 else float("nan"))
        tpr_xmatch.append(float(tx) if tx >= 0.0 else float("nan"))
        tpr_all_lo.append(ta_lo if ta >= 0.0 else float("nan"))
        tpr_all_hi.append(ta_hi if ta >= 0.0 else float("nan"))
        tpr_xmatch_lo.append(tx_lo if tx >= 0.0 else float("nan"))
        tpr_xmatch_hi.append(tx_hi if tx >= 0.0 else float("nan"))

    metrics = {
        "tpr_all_runs": tpr_all,
        "tpr_x_matched_runs_only": tpr_xmatch,
        "tpr_all_runs_ci_low": tpr_all_lo,
        "tpr_all_runs_ci_high": tpr_all_hi,
        "tpr_x_matched_ci_low": tpr_xmatch_lo,
        "tpr_x_matched_ci_high": tpr_xmatch_hi,
    }
    return redundancies, metrics, exit_codes


def save_fpr_sweep_results(
    path: Path | str,
    *,
    code_lengths: Sequence[int],
    metrics: Mapping[str, Sequence[float]],
    exit_codes: Sequence[int],
    runs: int,
    modulus: int,
    wm_bit_redundancy: int,
    prc_monte_carlo_trials: int,
    prompt_cases: Sequence[tuple[str, str]],
    llm_model_id: str | None = None,
) -> Path:
    return benchmark_io.save_fpr_sweep(
        path,
        payload={
            "code_lengths": list(code_lengths),
            **{k: list(v) for k, v in metrics.items()},
            "benchmark_exit_codes": list(exit_codes),
            "runs_per_prompt": int(runs),
            "prc_monte_carlo_trials": int(prc_monte_carlo_trials),
            "modulus": int(modulus),
            "wm_bit_redundancy": int(wm_bit_redundancy),
            "prompt_cases": benchmark_io.prompt_cases_to_json(prompt_cases),
        },
        llm_model_id=llm_model_id,
    )


def save_tpr_sweep_results(
    path: Path | str,
    *,
    wm_bit_redundancy_values: Sequence[int],
    metrics: Mapping[str, Sequence[float]],
    exit_codes: Sequence[int],
    code_length: int,
    runs: int,
    modulus: int,
    prompt_cases: Sequence[tuple[str, str]],
    llm_model_id: str | None = None,
) -> Path:
    return benchmark_io.save_tpr_sweep(
        path,
        payload={
            "wm_bit_redundancy": list(wm_bit_redundancy_values),
            **{k: list(v) for k, v in metrics.items()},
            "benchmark_exit_codes": list(exit_codes),
            "code_length": int(code_length),
            "runs_per_prompt": int(runs),
            "modulus": int(modulus),
            "prompt_cases": benchmark_io.prompt_cases_to_json(prompt_cases),
        },
        llm_model_id=llm_model_id,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--code-length", type=int, default=300, metavar="N")
    p.add_argument("--modulus", type=int, default=1024)
    p.add_argument(
        "--reuse-key",
        action="store_true",
        help="Reuse one master key per prompt id across runs (default is fresh key every trial).",
    )
    p.add_argument(
        "--prompt-case",
        action="append",
        dest="prompt_cases",
        metavar="ID:PROMPT",
        help="Benchmark case (repeatable). First ':' separates id from prompt.",
    )
    p.add_argument(
        "--llm-model",
        dest="llm_model",
        metavar="HF_HUB_ID",
        default=None,
        help="Hugging Face hub id for the LM (``model.configure`` before load).",
    )
    p.add_argument(
        "--wm-bit-redundancy",
        type=int,
        default=1,
        metavar="R",
        help="Depth-interleave R replicas of each logical PRC bit on the token channel; recovery uses strict majority (ties->0).",
    )
    p.add_argument(
        "--partition-mode",
        choices=("static", "balanced"),
        default="static",
        help="Vocab partition scheme: static (original) or balanced (per-step softmax).",
    )
    p.add_argument(
        "--redundancy-layout",
        choices=("depth", "block"),
        default="depth",
        help="Channel replica layout: depth (interleaved passes) or block (contiguous).",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write JSON results to this path (for benchmark_plot.py).",
    )
    args = p.parse_args()
    if args.runs < 1 or args.code_length < 1:
        print("runs and code-length must be >= 1", file=sys.stderr)
        return 2
    if args.wm_bit_redundancy < 1:
        print("wm-bit-redundancy must be >= 1", file=sys.stderr)
        return 2

    if args.prompt_cases:
        try:
            cases = [parse_prompt_case(s) for s in args.prompt_cases]
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 2
    else:
        cases = list(DEFAULT_PROMPT_CASES)

    console = make_benchmark_console()
    exit_code, summary = run_benchmark_with_summary(
        prompt_cases=cases,
        runs=args.runs,
        modulus=args.modulus,
        code_length=args.code_length,
        fresh_key_per_trial=not args.reuse_key,
        console=console,
        llm_model_id=args.llm_model,
        wm_bit_redundancy=args.wm_bit_redundancy,
        partition_mode=args.partition_mode,
        redundancy_layout=args.redundancy_layout,
        quiet=False,
    )
    if args.output is not None:
        out = benchmark_io.save_policy_summary(
            args.output,
            summary=summary,
            exit_code=exit_code,
            runs=args.runs,
            fresh_key_per_trial=not args.reuse_key,
            llm_model_id=args.llm_model,
        )
        console.print(f"[dim]Wrote[/] {out}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
