"""
Policy-detection benchmark: same end-to-end flow as ``app.py`` (generate → verify
``derive_attributes`` → issue unconstrained + one constrained key per closed-vocabulary
label → CPRF seed checks → ``master_detect`` / ``detect`` on good transcript →
negative-control ``master_detect`` on decoy).

Runs many trials over configurable prompts and code length with a Rich progress bar
(transient). Per-trial logging is minimal; results are aggregated into plain tables:
per-prompt TPR/FNR/TNR/FPR for policy ``detect`` vs recovered active-label expectation,
counts of which protocol stages matched; the same policy table computed only on runs
where encode-time ``attributes`` equals verify-time ``derive_attributes`` (full vector);
then mean wall time per pipeline stage.

CLI: ``--code-length``, ``--runs``, ``--modulus``, ``--wm-bit-redundancy``,
``--torch-compile``, optional ``--llm-model``, repeatable ``--prompt-case id:prompt``.

``run_benchmark_label_conditioned_matrix`` builds a ``|V| x |V|`` matrix (columns =
verify-time attributed labels); ``run_benchmark_prompt_conditioned_matrix`` builds
``|V| x |P|`` (columns = benchmark prompt ids). For notebooks / plotting:
``run_benchmark_with_summary(..., quiet=True)`` returns a ``BenchmarkRunSummary`` with
tables suppressed but a per-trial progress bar still shown (pass ``show_progress=False``
to hide it).

``micro_fpr_wilson`` / ``micro_tpr_wilson`` add Wilson score ~95% intervals on pooled
policy FPR/TPR; ``wilson_score_interval`` applies likewise to Monte Carlo rates;
``prc_random_detect_positive_rate`` estimates PRC ``detect`` acceptance on random bits
against a random PRC key (same spirit as ``testing.py``).
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from rich.console import Console

import benchmark_io
import text_attributes
import model
import randrecover
import prc
import watermarking as wm
from pathlib import Path
from text_attributes import VOCABULARY, active_labels_from_attributes

make_benchmark_console = benchmark_io.make_benchmark_console

# Parent sweep bars count these LM phases per protocol trial (not score-only).
_SWEEP_PROGRESS_PHASES_PER_TRIAL = 3  # baseline, watermark generate, channel recover


def _configure_benchmark(
    *,
    code_length: int,
    wm_bit_redundancy: int = 1,
    burn_in_tokens: int = 100,
    llm_model_id: str | None = None,
    torch_compile: bool | None = None,
) -> None:
    benchmark_io.require_prc_extension()
    wm.SECURITY_PARAM = code_length
    prc.set_code_length(code_length)
    wm.WM_BIT_REDUNDANCY = wm_bit_redundancy
    wm.BURN_IN_TOKENS = burn_in_tokens
    kwargs: dict = {}
    if llm_model_id:
        kwargs["model_id"] = llm_model_id
    if torch_compile is not None:
        kwargs["torch_compile"] = torch_compile
    if kwargs:
        model.configure(**kwargs)


DEFAULT_PROMPT_CASES: list[tuple[str, str]] = [
    (
        "medicine_stem_cell",
        "Explain how stem cell therapy is being used in regenerative medicine."),
    (
        "economics_min_wage",
        "Explain the economic effects of raising the minimum wage on employment and businesses."),
    (
        "art_surrealism",
        "Explain how Surrealist artists used dream imagery to challenge reality and logic."),
    (
        "software_breakthroughs",
        "Break down the most influential software breakthroughs in history."),
    (
        "sports_strategy_performance",
        "Explain the role of strategy and teamwork in achieving success in sports."),
    (
        "software_art_world",
        "Explain how software has transformed the art world."),
    (
        "sports_drake_maye",
        "Explain the economic nuance and impact of Drake Maye during his college football career at North Carolina."),
    (
        "medicine_software_practice",
        "Explain how software has transformed the practice of medicine."),
]


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
    n_baseline_tokens: int = 0
    n_wm_tokens: int = 0
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
        timings: TimingTotals) -> None:
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
        self.timings.n_baseline_tokens += timings.n_baseline_tokens
        self.timings.n_wm_tokens += timings.n_wm_tokens
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

    ``numerators`` / ``denominators`` / ``rates`` pool **all** qualifying trials. The
    ``*_attributes_match`` fields mirror the same update rules but **only** on trials
    where encode-time ``attributes`` equals verify-time ``derive_attributes``
    (full-vector match).
    """

    vocab: tuple[str, ...]
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
    avg_master_ber: float = 0.0
    max_master_ber: float = 0.0
    n_ber_trials: int = 0


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
    avg_master_ber: float = 0.0
    max_master_ber: float = 0.0
    n_ber_trials: int = 0


def sum_confusion_counts(
    roll: dict[str, PromptRollup],
    prompt_cases: Sequence[tuple[str, str]]) -> tuple[int, int, int, int]:
    """Sum TP/FN/TN/FP over every prompt id (micro pool over runs × labels)."""
    tp = fn = tn = fp = 0
    for sid, _ in prompt_cases:
        r = roll[sid]
        tp += r.tp
        fn += r.fn
        tn += r.tn
        fp += r.fp
    return tp, fn, tn, fp


def wilson_score_interval(
    k: int,
    n: int,
    *,
    z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for binomial proportion ``k/n`` (default ``z`` ≈ two-sided 95%)."""
    return benchmark_io.wilson_score_interval(k, n, z=z)


def micro_fpr_wilson(
    roll: dict[str, PromptRollup],
    prompt_cases: Sequence[tuple[str, str]],
    *,
    z: float = 1.96) -> tuple[float, float, float]:
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
    z: float = 1.96) -> tuple[float, float, float]:
    """Wilson-interval bounds on micro-TPR pooled over prompts (``TP / (TP+FN)``)."""
    tp, fn, tn, fp = sum_confusion_counts(roll, prompt_cases)
    n = tp + fn
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    lo, hi = wilson_score_interval(tp, n, z=z)
    return (tp / n, lo, hi)


def micro_ber_stats(
    roll: dict[str, PromptRollup],
    prompt_cases: Sequence[tuple[str, str]],
) -> tuple[float, float, int]:
    """Pooled master-path BER over prompts: ``(avg_ber, max_ber, n_runs)``."""
    ber_sum = 0.0
    ber_max = 0.0
    n = 0
    for sid, _ in prompt_cases:
        r = roll[sid]
        if r.runs <= 0:
            continue
        ber_sum += r.ber_sum
        ber_max = max(ber_max, r.ber_max)
        n += r.runs
    if n <= 0:
        return (float("nan"), float("nan"), 0)
    return (ber_sum / n, ber_max, n)


def prc_random_detect_positive_rate(
    code_length: int,
    n_trials: int,
    *,
    rng: random.Random | None = None,
    quiet: bool = False,
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
    for _ in benchmark_io.iter_with_progress(
        range(n_trials),
        description="PRC random detect",
        disable=quiet,
    ):
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
    console: Console,
    llm_model_id: str | None = None,
    wm_bit_redundancy: int = 1,
    burn_in_tokens: int = 100,
    quiet: bool = False,
    show_progress: bool | None = None,
    trial_progress: benchmark_io.ProgressHandle | None = None,
    torch_compile: bool | None = None,
) -> tuple[int, LabelConditionedDetectionMatrix]:
    """
    Build a ``|V| x |V|`` matrix conditioned on verify-time active labels.

    For each trial, let ``A`` be the active-label set from verify-time ``derive_attributes``.
    For every column label ``c in A`` and every row label ``r in VOCABULARY``:
      - denominator[r,c] += 1
      - numerator[r,c] += 1 iff ``detect(issue([r]), wm_text)`` is True.

    The returned matrix also includes ``*_attributes_match`` tallies over the same rules
    but **only** for trials where encode-time ``attributes`` equals verify-time
    ``derive_attributes`` (full-vector match).
    """
    for name in ("httpx", "httpcore", "huggingface_hub", "urllib3", "text_attributes"):
        logging.getLogger(name).setLevel(logging.WARNING)

    if runs < 1:
        raise ValueError("runs must be >= 1")
    if code_length < 1:
        raise ValueError("code_length must be >= 1")
    if wm_bit_redundancy < 1:
        raise ValueError("wm_bit_redundancy must be >= 1")

    progress_on = benchmark_io.resolve_show_progress(
        quiet=quiet, show_progress=show_progress
    )

    _configure_benchmark(
        code_length=code_length,
        wm_bit_redundancy=wm_bit_redundancy,
        burn_in_tokens=burn_in_tokens,
        llm_model_id=llm_model_id,
        torch_compile=torch_compile,
    )
    vocab = tuple(VOCABULARY)
    numerators: dict[str, dict[str, int]] = {
        r: {c: 0 for c in vocab} for r in vocab
    }
    denominators: dict[str, dict[str, int]] = {
        r: {c: 0 for c in vocab} for r in vocab
    }
    numerators_attributes_match: dict[str, dict[str, int]] = {
        r: {c: 0 for c in vocab} for r in vocab
    }
    denominators_attributes_match: dict[str, dict[str, int]] = {
        r: {c: 0 for c in vocab} for r in vocab
    }
    sid_runs: dict[str, int] = {sid: 0 for sid, _ in prompt_cases}
    sid_master: dict[str, int] = {sid: 0 for sid, _ in prompt_cases}
    sid_open: dict[str, int] = {sid: 0 for sid, _ in prompt_cases}
    sid_ucprf: dict[str, int] = {sid: 0 for sid, _ in prompt_cases}
    sid_control: dict[str, int] = {sid: 0 for sid, _ in prompt_cases}
    ber_sum = 0.0
    ber_max = 0.0
    n_ber_trials = 0

    if not quiet:
        console.print(
            f"matrix benchmark  code_length={wm.SECURITY_PARAM}  "
            f"wm_bit_redundancy={wm.WM_BIT_REDUNDANCY}  "
            f"modulus={modulus}  runs={runs}  |V|={len(vocab)}  "
            f"keys=fresh per trial  "
            f"llm={model.MODEL_ID!r}"
        )

    trials, pregen = _prepare_trials_with_baselines(
        prompt_cases,
        runs,
        show_baseline_progress=progress_on and trial_progress is None,
        trial_progress=trial_progress,
    )
    amortized = pregen.amortized_seconds
    generated: list[_GeneratedTrial] = []
    if trial_progress is not None:
        trial_progress.set_description("Benchmark matrix generate")
    sks = [wm.setup(modulus) for _ in trials]
    prompts = [prompt for _, _, prompt in trials]

    def _on_wm_batch(k: int) -> None:
        if trial_progress is not None:
            trial_progress.advance(k)

    if progress_on and trial_progress is None:
        with benchmark_io.progress_task(
            "Benchmark matrix generate",
            len(trials),
            disable=False,
        ) as bar:
            generated = _generate_trials_batched(
                sks,
                prompts,
                baseline_texts=list(pregen.texts),
                baseline_gen_seconds=amortized,
                baseline_n_tokens=[int(t) for t in pregen.token_counts],
                on_batch_done=lambda k: bar.advance(k),
            )
    else:
        generated = _generate_trials_batched(
            sks,
            prompts,
            baseline_texts=list(pregen.texts),
            baseline_gen_seconds=amortized,
            baseline_n_tokens=[int(t) for t in pregen.token_counts],
            on_batch_done=_on_wm_batch if trial_progress is not None else None,
        )

    scored = _score_protocol_trials_batched(
        generated,
        show_progress=progress_on,
        trial_progress=trial_progress,
        description="Benchmark matrix",
    )
    for ((_, sid, _prompt), _baseline), (
        word_stats,
        attributes_match,
        master_ok,
        open_ok,
        _cprf_label_ok,
        _cprf_label_n,
        control_ok,
        ber,
        _tt_inner,
        em_open_m,
    ) in zip(zip(trials, pregen.texts), scored):
        sid_runs[sid] += 1
        if master_ok:
            sid_master[sid] += 1
        if open_ok:
            sid_open[sid] += 1
        if em_open_m:
            sid_ucprf[sid] += 1
        if control_ok:
            sid_control[sid] += 1
        ber_sum += float(ber)
        ber_max = max(ber_max, float(ber))
        n_ber_trials += 1

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

        if attributes_match:
            for col in attributed_cols:
                for row in vocab:
                    denominators_attributes_match[row][col] += 1
            for row in vocab:
                if not got_by_row.get(row, False):
                    continue
                for col in attributed_cols:
                    numerators_attributes_match[row][col] += 1

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
        for c in vocab:
            d = denominators[r][c]
            n = numerators[r][c]
            rates[r][c] = (n / d) if d > 0 else -1.0
            dx = denominators_attributes_match[r][c]
            nx = numerators_attributes_match[r][c]
            rates_attributes_match[r][c] = (nx / dx) if dx > 0 else -1.0

    avg_ber = (ber_sum / n_ber_trials) if n_ber_trials else 0.0
    if not quiet:
        console.print(
            f"  master BER avg={avg_ber:.2f}%  max={ber_max:.2f}%  "
            f"(n={n_ber_trials})"
        )

    matrix = LabelConditionedDetectionMatrix(
        vocab=vocab,
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
        avg_master_ber=avg_ber,
        max_master_ber=ber_max,
        n_ber_trials=n_ber_trials)
    return (0 if all_ok else 1, matrix)


def run_benchmark_prompt_conditioned_matrix(
    *,
    prompt_cases: Sequence[tuple[str, str]],
    runs: int,
    modulus: int,
    code_length: int,
    console: Console,
    llm_model_id: str | None = None,
    wm_bit_redundancy: int = 1,
    burn_in_tokens: int = 100,
    quiet: bool = False,
    show_progress: bool | None = None,
    trial_progress: benchmark_io.ProgressHandle | None = None,
    torch_compile: bool | None = None,
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

    progress_on = benchmark_io.resolve_show_progress(
        quiet=quiet, show_progress=show_progress
    )

    _configure_benchmark(
        code_length=code_length,
        wm_bit_redundancy=wm_bit_redundancy,
        burn_in_tokens=burn_in_tokens,
        llm_model_id=llm_model_id,
        torch_compile=torch_compile,
    )
    vocab = tuple(VOCABULARY)
    col_ids = tuple(str(sid) for sid, _ in prompt_cases)
    numerators: dict[str, dict[str, int]] = {r: {p: 0 for p in col_ids} for r in vocab}
    denominators: dict[str, dict[str, int]] = {r: {p: 0 for p in col_ids} for r in vocab}
    numerators_attributes_match: dict[str, dict[str, int]] = {r: {p: 0 for p in col_ids} for r in vocab}
    denominators_attributes_match: dict[str, dict[str, int]] = {r: {p: 0 for p in col_ids} for r in vocab}
    sid_runs: dict[str, int] = {sid: 0 for sid, _ in prompt_cases}
    sid_master: dict[str, int] = {sid: 0 for sid, _ in prompt_cases}
    sid_open: dict[str, int] = {sid: 0 for sid, _ in prompt_cases}
    sid_ucprf: dict[str, int] = {sid: 0 for sid, _ in prompt_cases}
    sid_control: dict[str, int] = {sid: 0 for sid, _ in prompt_cases}
    ber_sum = 0.0
    ber_max = 0.0
    n_ber_trials = 0

    if not quiet:
        console.print(
            f"prompt-matrix benchmark  code_length={wm.SECURITY_PARAM}  "
            f"wm_bit_redundancy={wm.WM_BIT_REDUNDANCY}  "
            f"modulus={modulus}  runs={runs}  |V|={len(vocab)}  "
            f"|P|={len(col_ids)}  keys=fresh per trial  "
            f"llm={model.MODEL_ID!r}"
        )

    trials, pregen = _prepare_trials_with_baselines(
        prompt_cases,
        runs,
        show_baseline_progress=progress_on and trial_progress is None,
        trial_progress=trial_progress,
    )
    amortized = pregen.amortized_seconds
    generated: list[_GeneratedTrial] = []
    if trial_progress is not None:
        trial_progress.set_description("Benchmark prompt matrix generate")
    sks = [wm.setup(modulus) for _ in trials]
    prompts = [prompt for _, _, prompt in trials]

    def _on_wm_batch(k: int) -> None:
        if trial_progress is not None:
            trial_progress.advance(k)

    if progress_on and trial_progress is None:
        with benchmark_io.progress_task(
            "Benchmark prompt matrix generate",
            len(trials),
            disable=False,
        ) as bar:
            generated = _generate_trials_batched(
                sks,
                prompts,
                baseline_texts=list(pregen.texts),
                baseline_gen_seconds=amortized,
                baseline_n_tokens=[int(t) for t in pregen.token_counts],
                on_batch_done=lambda k: bar.advance(k),
            )
    else:
        generated = _generate_trials_batched(
            sks,
            prompts,
            baseline_texts=list(pregen.texts),
            baseline_gen_seconds=amortized,
            baseline_n_tokens=[int(t) for t in pregen.token_counts],
            on_batch_done=_on_wm_batch if trial_progress is not None else None,
        )

    scored = _score_protocol_trials_batched(
        generated,
        show_progress=progress_on,
        trial_progress=trial_progress,
        description="Benchmark prompt matrix",
    )
    for ((_, sid, _prompt), _baseline), (
        word_stats,
        attributes_match,
        master_ok,
        open_ok,
        _cprf_label_ok,
        _cprf_label_n,
        control_ok,
        ber,
        _tt_inner,
        em_open_m,
    ) in zip(zip(trials, pregen.texts), scored):
        sid_runs[sid] += 1
        if master_ok:
            sid_master[sid] += 1
        if open_ok:
            sid_open[sid] += 1
        if em_open_m:
            sid_ucprf[sid] += 1
        if control_ok:
            sid_control[sid] += 1
        ber_sum += float(ber)
        ber_max = max(ber_max, float(ber))
        n_ber_trials += 1

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

    avg_ber = (ber_sum / n_ber_trials) if n_ber_trials else 0.0
    if not quiet:
        console.print(
            f"  master BER avg={avg_ber:.2f}%  max={ber_max:.2f}%  "
            f"(n={n_ber_trials})"
        )

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
        avg_master_ber=avg_ber,
        max_master_ber=ber_max,
        n_ber_trials=n_ber_trials)
    return (0 if all_ok else 1, matrix)


def derive_verify_attributes(wm_text: str, modulus: int) -> list[int]:
    return text_attributes.derive_attributes(
        wm_text,
        modulus,
        log_scores=False,
        scores_out={})


def _prepare_trials_with_baselines(
    prompt_cases: Sequence[tuple[str, str]],
    runs: int,
    *,
    show_baseline_progress: bool,
    trial_progress: benchmark_io.ProgressHandle | None = None,
) -> tuple[list[tuple[int, str, str]], benchmark_io.BaselinePregenResult]:
    """Build the trial grid and pregenerate one baseline per trial (order-matched)."""
    trials = [(run_i, sid, prompt) for run_i in range(runs) for sid, prompt in prompt_cases]
    # Nested under a parent sweep bar: suppress the child UI but still advance parent.
    child_bar = bool(show_baseline_progress) and trial_progress is None
    on_batch = (
        (lambda k: trial_progress.advance(k)) if trial_progress is not None else None
    )
    if trial_progress is not None:
        trial_progress.set_description("Baselines")
    pregen = benchmark_io.pregenerate_baselines(
        [prompt for _, _, prompt in trials],
        quiet=not child_bar,
        description="Baselines",
        on_batch_done=on_batch,
    )
    if len(pregen.texts) != len(trials):
        raise RuntimeError(
            f"baseline pregen length {len(pregen.texts)} != trials {len(trials)}"
        )
    if len(pregen.token_counts) != len(trials):
        raise RuntimeError(
            f"baseline token_counts length {len(pregen.token_counts)} != trials {len(trials)}"
        )
    return trials, pregen


@dataclass
class _GeneratedTrial:
    """WM generation payload for integrity-preserving (text + scheme) recovery."""

    sk: Any
    wm_text: str
    encode_attributes: list[Any]
    secret: list[int]
    tt: TimingTotals


def _generate_trial(
    sk: Any,
    prompt: str,
    *,
    baseline_text: str | None = None,
    baseline_gen_seconds: float | None = None,
    baseline_n_tokens: int | None = None,
) -> _GeneratedTrial:
    tt = TimingTotals()
    out = wm.generate(
        sk,
        prompt,
        baseline_text=baseline_text,
        baseline_n_tokens=baseline_n_tokens,
    )
    # Drop privileged generation metadata; recovery sees text + scheme only.
    wm_text = str(out["generated_text_wm"])
    encode_attributes = list(out["attributes"])
    secret = list(out["prc_secret_bits"])
    if baseline_gen_seconds is not None:
        t_baseline = float(baseline_gen_seconds)
    else:
        t_baseline = float(out["seconds_baseline_gen"])
    tt.t_baseline_gen = t_baseline
    tt.t_wm_gen = float(out["seconds_watermarked_gen"])
    tt.n_baseline_tokens = int(out["n_tokens_baseline"])
    tt.n_wm_tokens = int(out["n_tokens_watermarked"])
    del out
    return _GeneratedTrial(
        sk=sk,
        wm_text=wm_text,
        encode_attributes=encode_attributes,
        secret=secret,
        tt=tt,
    )


def _generate_trials_batched(
    sks: Sequence[Any],
    prompts: Sequence[str],
    *,
    baseline_texts: Sequence[str],
    baseline_gen_seconds: float,
    baseline_n_tokens: Sequence[int],
    on_batch_done: Any | None = None,
) -> list[_GeneratedTrial]:
    """CPU encode per trial (fresh keys), then one batched LM watermark generate."""
    from hashlib import sha256

    prompt_list = [str(p) for p in prompts]
    n = len(prompt_list)
    if len(sks) != n or len(baseline_texts) != n or len(baseline_n_tokens) != n:
        raise ValueError("sks/prompts/baselines/token_counts length mismatch")

    channel_bits_list: list[list[int]] = []
    attrs_list: list = []
    secret_list: list[list[int]] = []
    for sk, baseline in zip(sks, baseline_texts):
        attributes = text_attributes.derive_attributes(
            baseline, sk.modulus, log_scores=False
        )
        r = sk.eval(attributes)
        prc.set_code_length(wm.SECURITY_PARAM)
        bits = [
            1 if b else 0
            for b in prc.encode(prc.key_gen_from_seed(sha256(r).digest()))
        ]
        channel_bits_list.append(
            wm.expand_channel_bits(bits, wm.WM_BIT_REDUNDANCY)
        )
        attrs_list.append(attributes)
        secret_list.append(list(bits))

    outs = wm.generate_from_channel_bits_batch(
        prompt_list,
        channel_bits_list,
        on_batch_done=on_batch_done,
    )
    generated: list[_GeneratedTrial] = []
    for i, out in enumerate(outs):
        tt = TimingTotals()
        tt.t_baseline_gen = float(baseline_gen_seconds)
        tt.t_wm_gen = float(out["seconds_watermarked_gen"])
        tt.n_baseline_tokens = int(baseline_n_tokens[i])
        tt.n_wm_tokens = int(out["n_tokens_watermarked"])
        generated.append(
            _GeneratedTrial(
                sk=sks[i],
                wm_text=str(out["generated_text_wm"]),
                encode_attributes=list(attrs_list[i]),
                secret=list(secret_list[i]),
                tt=tt,
            )
        )
        del out
    return generated


def _score_generated_trial(
    generated: _GeneratedTrial,
    recovered_bits: Sequence[int],
    *,
    recover_seconds: float = 0.0,
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
    """Detect / CPRF / negative-control scoring for one generated trial."""
    sk = generated.sk
    wm_text = generated.wm_text
    encode_attributes = generated.encode_attributes
    secret = generated.secret
    tt = generated.tt
    recovered_wm = list(recovered_bits)

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
    m_ok, m_bits = wm.master_detect(sk, wm_text, recovered_bits=recovered_wm)
    tt.t_master_good = (time.perf_counter() - t_m0) + float(recover_seconds)
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
    _m, tok, device = model.load()
    wrong = randrecover.negative_control_transcript_like(
        wm_text,
        tok,
        device,
        n_bits=wm.SECURITY_PARAM * wm.WM_BIT_REDUNDANCY,
        model=_m,
    )
    # Decoy has no watermark channel; scheme-length uncorrelated bits.
    recovered_neg = randrecover.uncorrelated_bits_from_text(
        wrong, tok, n_bits=wm.SECURITY_PARAM * wm.WM_BIT_REDUNDANCY
    )
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


def _score_protocol_trials_batched(
    generated: Sequence[_GeneratedTrial],
    *,
    show_progress: bool,
    trial_progress: benchmark_io.ProgressHandle | None,
    description: str,
) -> list[
    tuple[
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
    ]
]:
    """Batch integrity recovery over generated texts, then score each trial."""
    texts = [g.wm_text for g in generated]
    n = len(texts)
    if n == 0:
        return []

    recover_disable = (not show_progress) or (trial_progress is not None)
    if trial_progress is not None:
        trial_progress.set_description(f"{description} recover")
    t0 = time.perf_counter()
    with benchmark_io.progress_task(
        f"{description} recover",
        n,
        disable=recover_disable,
    ) as recover_bar:

        def _on_recover_batch(k: int) -> None:
            recover_bar.advance(k)
            if trial_progress is not None:
                trial_progress.advance(k)

        bits_list = wm.recover_channel_bits_batch(
            texts, on_batch_done=_on_recover_batch
        )
    recover_wall = time.perf_counter() - t0
    amortize = recover_wall / n

    if trial_progress is not None:
        trial_progress.set_description(f"{description} score")

    scored: list[
        tuple[
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
        ]
    ] = []
    for g, bits in zip(
        benchmark_io.iter_with_progress(
            list(generated),
            description=f"{description} score",
            disable=not show_progress or trial_progress is not None,
        ),
        bits_list,
    ):
        scored.append(
            _score_generated_trial(g, bits, recover_seconds=amortize)
        )
    return scored


def _tokens_per_sec(n_tokens: float, seconds: float) -> float:
    if seconds <= 0.0 or n_tokens <= 0.0:
        return 0.0
    return float(n_tokens) / float(seconds)


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
        "t_negative_control")
    means = {k: getattr(t, k) / n for k in keys}
    means["n_baseline_tokens"] = t.n_baseline_tokens / n
    means["n_wm_tokens"] = t.n_wm_tokens / n
    means["tok_s_baseline"] = _tokens_per_sec(t.n_baseline_tokens, t.t_baseline_gen)
    means["tok_s_wm"] = _tokens_per_sec(t.n_wm_tokens, t.t_wm_gen)
    return means


def _aggregate_timing_means(
    roll: dict[str, PromptRollup],
    prompt_cases: Sequence[tuple[str, str]]) -> dict[str, float]:
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
        "t_negative_control")}
    n_baseline_tokens = 0
    n_wm_tokens = 0
    for sid, _ in prompt_cases:
        r = roll[sid]
        n = r.runs
        if n == 0:
            continue
        total_runs += n
        t = r.timings
        for k in acc:
            acc[k] += getattr(t, k)
        n_baseline_tokens += t.n_baseline_tokens
        n_wm_tokens += t.n_wm_tokens
    if total_runs == 0:
        return {
            **acc,
            "t_grand_avg": 0.0,
            "n_baseline_tokens": 0.0,
            "n_wm_tokens": 0.0,
            "tok_s_baseline": 0.0,
            "tok_s_wm": 0.0,
        }
    for k in acc:
        acc[k] /= total_runs
    acc["t_grand_avg"] = sum(acc.values())
    acc["n_baseline_tokens"] = n_baseline_tokens / total_runs
    acc["n_wm_tokens"] = n_wm_tokens / total_runs
    # Throughput from totals (not mean-of-rates) so amortized pregen is fair.
    total_bl_s = acc["t_baseline_gen"] * total_runs
    total_wm_s = acc["t_wm_gen"] * total_runs
    acc["tok_s_baseline"] = _tokens_per_sec(n_baseline_tokens, total_bl_s)
    acc["tok_s_wm"] = _tokens_per_sec(n_wm_tokens, total_wm_s)
    return acc


def _sid_cell(sid: str, width: int = 22) -> str:
    if len(sid) <= width:
        return sid
    return sid[: width - 3] + "..."


def _print_timing_table_plain(
    roll: dict[str, PromptRollup],
    prompt_cases: Sequence[tuple[str, str]]) -> None:
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

    def _gen_rows() -> list[list[str]]:
        out: list[list[str]] = []
        for sid, _ in prompt_cases:
            r = roll[sid]
            if r.runs == 0:
                continue
            m = _mean_timings(r)
            out.append(
                [
                    _sid_cell(sid),
                    f"{m['t_setup']:.4f}",
                    f"{m['t_baseline_gen']:.4f}",
                    f"{m['tok_s_baseline']:.1f}",
                    f"{m['t_wm_gen']:.4f}",
                    f"{m['tok_s_wm']:.1f}",
                    f"{m['t_derive_attributes']:.4f}",
                ]
            )
        gm = _aggregate_timing_means(roll, prompt_cases)
        out.append(
            [
                "ALL (mean/run)",
                f"{gm['t_setup']:.4f}",
                f"{gm['t_baseline_gen']:.4f}",
                f"{gm['tok_s_baseline']:.1f}",
                f"{gm['t_wm_gen']:.4f}",
                f"{gm['tok_s_wm']:.1f}",
                f"{gm['t_derive_attributes']:.4f}",
            ]
        )
        return out

    benchmark_io.print_plain_table(
        title="Mean wall time — generation & attributes (seconds per run; tok/s from totals)",
        headers=[
            "prompt_id",
            "setup",
            "baseline",
            "bl_tok/s",
            "wm_gen",
            "wm_tok/s",
            "derive_attr",
        ],
        widths=[22, 8, 10, 9, 10, 9, 11],
        aligns=["<", ">", ">", ">", ">", ">", ">"],
        rows=_gen_rows())
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
                "t_negative_control")))
    gm = _aggregate_timing_means(roll, prompt_cases)
    print(f"Sum of all listed stage means (per run): {gm['t_grand_avg']:.4f} s")
    print(
        f"Generation throughput (totals): baseline={gm['tok_s_baseline']:.1f} tok/s  "
        f"watermarked={gm['tok_s_wm']:.1f} tok/s"
    )


def _print_plain_results(
    *,
    prompt_cases: Sequence[tuple[str, str]],
    roll: dict[str, PromptRollup],
    vocab_n: int,
    table_heading: str,
    print_legend: bool) -> None:
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
        rows=rate_rows)
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
        rows=check_rows)

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


def _strict_protocol_ok(
    roll: dict[str, PromptRollup],
    prompt_cases: Sequence[tuple[str, str]]) -> bool:
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
) -> None:
    msg = (
        "Protocol checks did not pass on every run (require: master_detect good, detect open, "
        "unconstrained CPRF match, negative control rejects)."
    )
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


def run_benchmark_with_summary(
    *,
    prompt_cases: Sequence[tuple[str, str]],
    runs: int,
    modulus: int,
    code_length: int,
    console: Console,
    llm_model_id: str | None = None,
    wm_bit_redundancy: int = 1,
    burn_in_tokens: int = 100,
    quiet: bool = False,
    show_progress: bool | None = None,
    trial_progress: benchmark_io.ProgressHandle | None = None,
    torch_compile: bool | None = None,
) -> tuple[int, BenchmarkRunSummary]:
    for name in ("httpx", "httpcore", "huggingface_hub", "urllib3", "text_attributes"):
        logging.getLogger(name).setLevel(logging.WARNING)

    progress_on = benchmark_io.resolve_show_progress(
        quiet=quiet, show_progress=show_progress
    )

    _configure_benchmark(
        code_length=code_length,
        wm_bit_redundancy=wm_bit_redundancy,
        burn_in_tokens=burn_in_tokens,
        llm_model_id=llm_model_id,
        torch_compile=torch_compile,
    )
    roll: dict[str, PromptRollup] = {sid: PromptRollup() for sid, _ in prompt_cases}
    roll_attributes_match: dict[str, PromptRollup] = {sid: PromptRollup() for sid, _ in prompt_cases}

    vocab_n = len(VOCABULARY)
    if not quiet:
        console.print(
            f"code_length={wm.SECURITY_PARAM}  wm_bit_redundancy={wm.WM_BIT_REDUNDANCY}  "
            f"channel_bits={wm.SECURITY_PARAM * wm.WM_BIT_REDUNDANCY}  burn_in={wm.BURN_IN_TOKENS}  modulus={modulus}  runs={runs}  |V|={vocab_n}  "
            f"keys=fresh per trial  "
            f"llm={model.MODEL_ID!r}  dtype={model.inference_dtype_label()}  "
            f"torch_compile={'on' if model.TORCH_COMPILE else 'off'}"
        )

    trials, pregen = _prepare_trials_with_baselines(
        prompt_cases,
        runs,
        show_baseline_progress=progress_on and trial_progress is None,
        trial_progress=trial_progress,
    )
    amortized = pregen.amortized_seconds
    generated: list[_GeneratedTrial] = []
    setup_by_index: list[float] = []
    if trial_progress is not None:
        trial_progress.set_description("Benchmark generate")
    sks: list[Any] = []
    for _ in trials:
        t_setup0 = time.perf_counter()
        sks.append(wm.setup(modulus))
        setup_by_index.append(time.perf_counter() - t_setup0)
    prompts = [prompt for _, _, prompt in trials]

    def _on_wm_batch(k: int) -> None:
        if trial_progress is not None:
            trial_progress.advance(k)

    if progress_on and trial_progress is None:
        with benchmark_io.progress_task(
            "Benchmark generate",
            len(trials),
            disable=False,
        ) as bar:
            generated = _generate_trials_batched(
                sks,
                prompts,
                baseline_texts=list(pregen.texts),
                baseline_gen_seconds=amortized,
                baseline_n_tokens=[int(t) for t in pregen.token_counts],
                on_batch_done=lambda k: bar.advance(k),
            )
    else:
        generated = _generate_trials_batched(
            sks,
            prompts,
            baseline_texts=list(pregen.texts),
            baseline_gen_seconds=amortized,
            baseline_n_tokens=[int(t) for t in pregen.token_counts],
            on_batch_done=_on_wm_batch if trial_progress is not None else None,
        )

    scored = _score_protocol_trials_batched(
        generated,
        show_progress=progress_on,
        trial_progress=trial_progress,
        description="Benchmark",
    )
    for ((_, sid, _prompt), _baseline), setup_s, (
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
    ) in zip(zip(trials, pregen.texts), setup_by_index, scored):
        tt_inner.t_setup = setup_s

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
            timings=tt_inner)
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
                timings=tt_inner)

    all_ok = _strict_protocol_ok(roll, prompt_cases)
    summary = BenchmarkRunSummary(
        roll=roll,
        roll_attributes_match=roll_attributes_match,
        prompt_cases=tuple((sid, p) for sid, p in prompt_cases),
        vocab_n=vocab_n,
        code_length=code_length,
        wm_bit_redundancy=wm.WM_BIT_REDUNDANCY,
        modulus=modulus,
        strict_protocol_ok=all_ok)

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
            print_legend=True)
        _print_plain_results(
            prompt_cases=prompt_cases,
            roll=roll_attributes_match,
            vocab_n=vocab_n,
            table_heading=_heading_attributes_match,
            print_legend=False)
        _print_timing_table_plain(roll, prompt_cases)

        if not all_ok:
            _print_protocol_failure_details(
                roll=roll, prompt_cases=prompt_cases
            )

    return (0 if all_ok else 1, summary)


def run_benchmark(
    *,
    prompt_cases: Sequence[tuple[str, str]],
    runs: int,
    modulus: int,
    code_length: int,
    console: Console,
    llm_model_id: str | None = None,
    wm_bit_redundancy: int = 1,
    torch_compile: bool | None = None,
) -> int:
    code, _ = run_benchmark_with_summary(
        prompt_cases=prompt_cases,
        runs=runs,
        modulus=modulus,
        code_length=code_length,
        console=console,
        llm_model_id=llm_model_id,
        wm_bit_redundancy=wm_bit_redundancy,
        torch_compile=torch_compile,
        quiet=False)
    return code


def run_fpr_vs_code_length_sweep(
    *,
    prompt_cases: Sequence[tuple[str, str]],
    code_lengths: Sequence[int],
    runs: int,
    modulus: int,
    wm_bit_redundancy: int,
    console: Console,
    llm_model_id: str | None = None,
    prc_monte_carlo_trials: int = 100_000,
    rng: random.Random | None = None,
    quiet: bool = True,
    show_progress: bool | None = None,
    torch_compile: bool | None = None,
) -> tuple[list[int], dict[str, list[float]], list[int]]:
    """
    Sweep logical ``code_length`` and return (lengths, metric dict, exit_codes).

    Metrics keys: ``scheme_fpr_all``, ``scheme_fpr_xmatch``, ``prc_random_fpr``,
    ``master_ber_avg``, ``master_ber_max``, plus ``*_ci_low`` / ``*_ci_high`` Wilson
    bounds where applicable.

    Progress advances once per LM phase (baseline, watermark generate, channel
    recover) for every protocol trial across the whole sweep — not once per
    code length. ``quiet`` only suppresses per-length result tables.
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
    master_ber_avg: list[float] = []
    master_ber_max: list[float] = []
    exit_codes: list[int] = []

    length_list = list(code_lengths)
    trials_per_length = int(runs) * len(prompt_cases)
    total_units = (
        trials_per_length * len(length_list) * _SWEEP_PROGRESS_PHASES_PER_TRIAL
    )
    progress_on = True if show_progress is None else bool(show_progress)

    with benchmark_io.progress_task(
        "FPR sweep",
        total_units,
        disable=not progress_on or total_units <= 0,
    ) as sweep_progress:
        for length in length_list:
            sweep_progress.set_description(f"FPR sweep n={int(length)} · PRC MC")
            r_rand, fp_mc = prc_random_detect_positive_rate(
                int(length), int(prc_monte_carlo_trials), rng=prng, quiet=True
            )
            prc_random_fpr.append(float(r_rand))
            pl, ph = wilson_score_interval(fp_mc, int(prc_monte_carlo_trials))
            prc_random_fpr_lo.append(pl)
            prc_random_fpr_hi.append(ph)

            sweep_progress.set_description(f"FPR sweep n={int(length)}")
            ex, summary = run_benchmark_with_summary(
                prompt_cases=prompt_cases,
                runs=int(runs),
                modulus=int(modulus),
                code_length=int(length),
                console=console,
                llm_model_id=llm_model_id,
                wm_bit_redundancy=int(wm_bit_redundancy),
                quiet=quiet,
                show_progress=False,
                trial_progress=sweep_progress,
                torch_compile=torch_compile)
            lengths.append(int(length))
            exit_codes.append(int(ex))

            fa, fa_lo, fa_hi = micro_fpr_wilson(summary.roll, summary.prompt_cases)
            fx, fx_lo, fx_hi = micro_fpr_wilson(summary.roll_attributes_match, summary.prompt_cases)
            ber_avg, ber_max, _ = micro_ber_stats(summary.roll, summary.prompt_cases)
            scheme_fpr_all.append(float(fa) if fa >= 0.0 else float("nan"))
            scheme_fpr_xmatch.append(float(fx) if fx >= 0.0 else float("nan"))
            scheme_fpr_all_lo.append(fa_lo if fa >= 0.0 else float("nan"))
            scheme_fpr_all_hi.append(fa_hi if fa >= 0.0 else float("nan"))
            scheme_fpr_xmatch_lo.append(fx_lo if fx >= 0.0 else float("nan"))
            scheme_fpr_xmatch_hi.append(fx_hi if fx >= 0.0 else float("nan"))
            master_ber_avg.append(float(ber_avg))
            master_ber_max.append(float(ber_max))

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
        "master_ber_avg": master_ber_avg,
        "master_ber_max": master_ber_max,
    }
    return lengths, metrics, exit_codes


def run_tpr_vs_wm_bit_redundancy_sweep(
    *,
    prompt_cases: Sequence[tuple[str, str]],
    wm_bit_redundancy_values: Sequence[int],
    code_length: int,
    runs: int,
    modulus: int,
    console: Console,
    llm_model_id: str | None = None,
    quiet: bool = True,
    show_progress: bool | None = None,
    torch_compile: bool | None = None,
) -> tuple[list[int], dict[str, list[float]], list[int]]:
    """Sweep ``wm_bit_redundancy`` at fixed logical ``code_length``.

    Progress advances once per LM phase (baseline, watermark generate, channel
    recover) for every protocol trial across the whole sweep — not once per
    redundancy. ``quiet`` only suppresses per-point result tables.
    """
    redundancies: list[int] = []
    tpr_all: list[float] = []
    tpr_xmatch: list[float] = []
    tpr_all_lo: list[float] = []
    tpr_all_hi: list[float] = []
    tpr_xmatch_lo: list[float] = []
    tpr_xmatch_hi: list[float] = []
    master_ber_avg: list[float] = []
    master_ber_max: list[float] = []
    exit_codes: list[int] = []

    redundancy_list = list(wm_bit_redundancy_values)
    trials_per_point = int(runs) * len(prompt_cases)
    total_units = (
        trials_per_point * len(redundancy_list) * _SWEEP_PROGRESS_PHASES_PER_TRIAL
    )
    progress_on = True if show_progress is None else bool(show_progress)

    with benchmark_io.progress_task(
        "TPR sweep",
        total_units,
        disable=not progress_on or total_units <= 0,
    ) as sweep_progress:
        for redundancy in redundancy_list:
            sweep_progress.set_description(f"TPR sweep R={int(redundancy)}")
            ex, summary = run_benchmark_with_summary(
                prompt_cases=prompt_cases,
                runs=int(runs),
                modulus=int(modulus),
                code_length=int(code_length),
                console=console,
                llm_model_id=llm_model_id,
                wm_bit_redundancy=int(redundancy),
                quiet=quiet,
                show_progress=False,
                trial_progress=sweep_progress,
                torch_compile=torch_compile)
            redundancies.append(int(redundancy))
            exit_codes.append(int(ex))

            ta, ta_lo, ta_hi = micro_tpr_wilson(summary.roll, summary.prompt_cases)
            tx, tx_lo, tx_hi = micro_tpr_wilson(summary.roll_attributes_match, summary.prompt_cases)
            ber_avg, ber_max, _ = micro_ber_stats(summary.roll, summary.prompt_cases)
            tpr_all.append(float(ta) if ta >= 0.0 else float("nan"))
            tpr_xmatch.append(float(tx) if tx >= 0.0 else float("nan"))
            tpr_all_lo.append(ta_lo if ta >= 0.0 else float("nan"))
            tpr_all_hi.append(ta_hi if ta >= 0.0 else float("nan"))
            tpr_xmatch_lo.append(tx_lo if tx >= 0.0 else float("nan"))
            tpr_xmatch_hi.append(tx_hi if tx >= 0.0 else float("nan"))
            master_ber_avg.append(float(ber_avg))
            master_ber_max.append(float(ber_max))

    metrics = {
        "tpr_all_runs": tpr_all,
        "tpr_x_matched_runs_only": tpr_xmatch,
        "tpr_all_runs_ci_low": tpr_all_lo,
        "tpr_all_runs_ci_high": tpr_all_hi,
        "tpr_x_matched_ci_low": tpr_xmatch_lo,
        "tpr_x_matched_ci_high": tpr_xmatch_hi,
        "master_ber_avg": master_ber_avg,
        "master_ber_max": master_ber_max,
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
        "--prompt-case",
        action="append",
        dest="prompt_cases",
        metavar="ID:PROMPT",
        help="Benchmark case (repeatable). First ':' separates id from prompt.")
    p.add_argument(
        "--llm-model",
        dest="llm_model",
        metavar="HF_HUB_ID",
        default=None,
        help="Hugging Face hub id for the LM (``model.configure`` before load).")
    p.add_argument(
        "--wm-bit-redundancy",
        type=int,
        default=1,
        metavar="R",
        help="Depth-interleave R replicas of each logical PRC bit on the token channel; recovery uses strict majority (ties->0).")
    p.add_argument(
        "--burn-in-tokens",
        type=int,
        default=100,
        metavar="B",
        help="Unwatermarked warm-up tokens before the channel payload (default: 100).")
    p.add_argument(
        "--torch-compile",
        action="store_true",
        help="Apply torch.compile to the LM after load (first forwards are slower).")
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write JSON results to this path (for benchmark_plot.py).")
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
        console=console,
        llm_model_id=args.llm_model,
        wm_bit_redundancy=args.wm_bit_redundancy,
        burn_in_tokens=args.burn_in_tokens,
        torch_compile=True if args.torch_compile else None,
        quiet=False,
    )
    if args.output is not None:
        out = benchmark_io.save_policy_summary(
            args.output,
            summary=summary,
            exit_code=exit_code,
            runs=args.runs,
            llm_model_id=args.llm_model)
        console.print(f"[dim]Wrote[/] {out}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
