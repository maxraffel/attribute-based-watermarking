"""
Policy-detection benchmark: same end-to-end flow as ``app.py`` (generate → verify ``derive_x`` →
issue unconstrained + one constrained key per closed-vocabulary label → CPRF seed checks →
``master_detect`` / ``detect`` on good transcript → negative-control ``master_detect`` on decoy).

Runs many trials over configurable prompts and code length with a Rich **progress bar** (transient).
Per-trial logging is minimal; results are **aggregated into tables**: per-prompt TPR/FNR/TNR/FPR for
policy ``detect`` vs NLI-recovered attribute expectation, counts of which protocol stages matched;
**the same policy table** computed only on runs where encode-time ``attr_x`` equals verify-time
``derive_x`` (full vector); then mean wall time per pipeline stage.

CLI: ``--code-length``, ``--runs``, ``--modulus``, ``--reuse-key``, ``--wm-bit-redundancy``,
optional ``--llm-model``, repeatable ``--prompt-case id:prompt``. Env: ``BENCHMARK_PLAIN_TABLE``, ``BENCHMARK_CONSOLE_*``
(see ``make_benchmark_console`` / ``_use_plain_table``).

For notebooks / plotting: ``run_benchmark_with_summary(..., quiet=True)`` returns a ``BenchmarkRunSummary``;
``micro_fpr`` / ``micro_tpr`` pool per-label policy FPR/TPR; ``micro_fpr_wilson`` / ``micro_tpr_wilson``
add Wilson score ~95 percent intervals on those pooled proportions; ``wilson_score_interval`` applies likewise to Monte Carlo
rates; ``prc_random_detect_positive_rate`` estimates PRC ``detect`` acceptance on random bits against a random
PRC key (same spirit as ``testing.py``).
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from rich.console import Console
from rich.progress import track
from rich.table import Table

import attr_x_nli
import randrecover
import watermarking as wm
from closed_vocab import VOCABULARY, active_labels_from_verify_x


def make_benchmark_console() -> Console:
    floor_w = int(os.environ.get("BENCHMARK_CONSOLE_WIDTH", "168"))
    floor_h = int(os.environ.get("BENCHMARK_CONSOLE_HEIGHT", "120"))
    try:
        term = shutil.get_terminal_size()
        if sys.stdout.isatty():
            w = min(max(term.columns, 40), 240)
            h = min(max(term.lines, 10), 200)
        else:
            w = floor_w
            h = floor_h
    except OSError:
        w, h = floor_w, floor_h
    return Console(highlight=False, width=w, height=h)


def _use_plain_table() -> bool:
    v = os.environ.get("BENCHMARK_PLAIN_TABLE", "").strip().lower()
    if v in ("1", "true", "yes", "always"):
        return True
    if v in ("0", "false", "no", "never"):
        return False
    return not sys.stdout.isatty()


DEFAULT_PROMPT_CASES: list[tuple[str, str]] = [
    (
        "brady_econ_new_england",
        "Explain how Tom Brady's football legacy economically impacted the New England area."
    ),
]


COMPREHENSIVE_PROMPT_CASES: list[tuple[str, str]] = [
    (
        "brady_econ_new_england",
        "Explain how Tom Brady's football legacy economically impacted the New England area."
    ),
    (
        "picasso_analysis",
        "Explain the significance of Picasso's artwork in the art world."
    ),
    (
        "socrates_ethics",
        "Explain the ethical implications of Socrates's philosophy."
    ),
    (
        ""
    )
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
    t_derive_verify: float = 0.0
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

    x_perfect: int = 0
    master_good: int = 0
    open_detect_good: int = 0
    unconstrained_cprf_ok: int = 0
    cprf_per_label_expect_ok: int = 0
    cprf_per_label_checks: int = 0
    control_correct: int = 0

    ber_sum: float = 0.0

    #: When policy detect differs from attribute-based expectation — likely cause buckets.
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
        x_perfect: bool,
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
        if x_perfect:
            self.x_perfect += 1
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
        self.timings.t_derive_verify += timings.t_derive_verify
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
    roll_xmatch: dict[str, PromptRollup]
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


def sum_confusion_counts(
    roll: dict[str, PromptRollup],
    prompt_cases: Sequence[tuple[str, str]],
) -> tuple[int, int, int, int]:
    """Sum TP/FN/TN/FP over every prompt id (micro pool over runs × labels)."""
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
    """Micro-averaged false positive rate for policy ``detect`` vs NLI-active-set gold."""
    tp, fn, tn, fp = sum_confusion_counts(roll, prompt_cases)
    return _rates(tp, fn, tn, fp)[3]


def micro_tpr(
    roll: dict[str, PromptRollup],
    prompt_cases: Sequence[tuple[str, str]],
) -> float:
    """Micro-averaged true positive rate for policy ``detect`` vs NLI-active-set gold."""
    tp, fn, tn, fp = sum_confusion_counts(roll, prompt_cases)
    return _rates(tp, fn, tn, fp)[0]


def wilson_score_interval(
    k: int,
    n: int,
    *,
    z: float = 1.96,
) -> tuple[float, float]:
    """Wilson score interval for binomial proportion ``k/n`` (default ``z`` = 1.96 ≈ two-sided 95%)."""
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
    wm.set_prc_code_length(code_length)
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
    quiet: bool = False,
) -> tuple[int, LabelConditionedDetectionMatrix]:
    """
    Build a ``|V| x |V|`` matrix conditioned on verify-time active labels.

    For each trial, let ``A`` be the active-label set from verify-time ``derive_x``.
    For every column label ``c in A`` and every row label ``r in VOCABULARY``:
      - denominator[r,c] += 1
      - numerator[r,c] += 1 iff ``detect(issue_keyword_policy([r]), wm_text)`` is True.
    """
    for name in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)

    if llm_model_id is not None and llm_model_id.strip():
        wm.set_llm_model_id(llm_model_id.strip())

    if runs < 1:
        raise ValueError("runs must be >= 1")
    if code_length < 1:
        raise ValueError("code_length must be >= 1")
    if wm_bit_redundancy < 1:
        raise ValueError("wm_bit_redundancy must be >= 1")

    wm.set_prc_code_length(code_length)
    wm.set_wm_bit_redundancy(wm_bit_redundancy)
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
            f"wm_bit_redundancy={wm.WM_BIT_REDUNDANCY}  modulus={modulus}  runs={runs}  |V|={len(vocab)}  "
            f"keys={'fresh per trial' if fresh_key_per_trial else 'reuse per prompt id'}  "
            f"llm={wm.MODEL_ID!r}"
        )

    trials = [(run_i, sid, prompt) for run_i in range(runs) for sid, prompt in prompt_cases]
    for _, sid, prompt in track(
        trials,
        description="Benchmark matrix",
        console=console,
        transient=True,
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
            _x_perfect,
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


def _derive_x_verify(wm_text: str, modulus: int) -> list[int]:
    try:
        return attr_x_nli.derive_x(
            wm_text,
            modulus,
            log_nli_scores=False,
            nli_scores_out={},
        )
    except TypeError:
        return attr_x_nli.derive_x(wm_text, modulus)


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
    x_encode = list(out["attr_x"])
    secret = list(out["prc_secret_bits"])
    t_baseline = float(out["seconds_baseline_gen"])
    t_wm = float(out["seconds_watermarked_gen"])
    tt.t_baseline_gen = t_baseline
    tt.t_wm_gen = t_wm

    t_d0 = time.perf_counter()
    x_verify = _derive_x_verify(wm_text, sk.modulus)
    tt.t_derive_verify = time.perf_counter() - t_d0

    active_wm = active_labels_from_verify_x(x_verify, sk.modulus)
    active_set = set(active_wm)
    x_perfect = x_encode == x_verify

    t_k0 = time.perf_counter()
    dk_open = wm.issue_unconstrained(sk)
    dk_by_word = {w: wm.issue_keyword_policy(sk, [w]) for w in VOCABULARY}
    tt.t_issue_keys = time.perf_counter() - t_k0

    xl = list(x_verify)
    t_c0 = time.perf_counter()
    em_master = sk.eval(xl)
    ec_open = dk_open.c_eval(xl)
    em_open_m = em_master == ec_open
    cprf_per_label_ok = 0
    cprf_per_label_n = len(VOCABULARY)
    seed_match_by_word: dict[str, bool] = {}
    for w in VOCABULARY:
        dk = dk_by_word[w]
        expect_sm = w in active_set
        sm = em_master == dk.c_eval(xl)
        seed_match_by_word[w] = sm
        if sm == expect_sm:
            cprf_per_label_ok += 1
    tt.t_cprf_checks = time.perf_counter() - t_c0

    t_m0 = time.perf_counter()
    m_ok, m_bits = wm.master_detect(sk, wm_text)
    tt.t_master_good = time.perf_counter() - t_m0
    ber = _ber_percent(secret, m_bits)

    t_u0 = time.perf_counter()
    u_ok, _ = wm.detect(dk_open, wm_text)
    tt.t_detect_open = time.perf_counter() - t_u0

    word_stats: list[dict[str, Any]] = []
    t_pv0 = time.perf_counter()
    for w in VOCABULARY:
        expect_detect = w in active_set
        got, _ = wm.detect(dk_by_word[w], wm_text)
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
    assert wm.TOKENIZER is not None and wm.MODEL is not None
    wrong = randrecover.negative_control_transcript_like(
        wm_text,
        wm.TOKENIZER,
        wm.DEVICE,
        n_bits=wm.wm_channel_bits_length(),
        model=wm.MODEL,
    )
    ctrl_ok_raw, _ = wm.master_detect(sk, wrong)
    control_ok = not bool(ctrl_ok_raw)
    tt.t_negative_control = time.perf_counter() - t_nc0

    return (
        word_stats,
        x_perfect,
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
        "t_derive_verify",
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
        "t_derive_verify",
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


def _print_timing_table_plain(
    roll: dict[str, PromptRollup],
    prompt_cases: Sequence[tuple[str, str]],
) -> None:
    print()
    print("Mean wall time per pipeline stage (seconds; averaged over runs per prompt)")
    w = 18
    header = (
        f"{'prompt_id':<26}"
        + "".join(f"{k:>{w}}" for k in (
            "setup",
            "baseline",
            "wm_gen",
            "derive_x",
            "issue_keys",
            "cprf",
            "m_good",
            "det_open",
            "det_vocab",
            "neg_ctrl",
        ))
    )
    print(header)
    print("-" * len(header))

    for sid, _ in prompt_cases:
        r = roll[sid]
        if r.runs == 0:
            continue
        m = _mean_timings(r)
        sid_disp = sid if len(sid) <= 26 else sid[:23] + "..."
        parts = (
            m.get("t_setup", 0),
            m["t_baseline_gen"],
            m["t_wm_gen"],
            m["t_derive_verify"],
            m["t_issue_keys"],
            m["t_cprf_checks"],
            m["t_master_good"],
            m["t_detect_open"],
            m["t_detect_per_label"],
            m["t_negative_control"],
        )
        row = f"{sid_disp:<26}" + "".join(f"{p:>{w}.4f}" for p in parts)
        print(row)

    gm = _aggregate_timing_means(roll, prompt_cases)
    print("-" * len(header))
    print(
        f"{'ALL (mean / run)':<26}"
        + f"{gm['t_setup']:>{w}.4f}"
        + f"{gm['t_baseline_gen']:>{w}.4f}"
        + f"{gm['t_wm_gen']:>{w}.4f}"
        + f"{gm['t_derive_verify']:>{w}.4f}"
        + f"{gm['t_issue_keys']:>{w}.4f}"
        + f"{gm['t_cprf_checks']:>{w}.4f}"
        + f"{gm['t_master_good']:>{w}.4f}"
        + f"{gm['t_detect_open']:>{w}.4f}"
        + f"{gm['t_detect_per_label']:>{w}.4f}"
        + f"{gm['t_negative_control']:>{w}.4f}"
    )
    print(f"[footer] sum of listed stage means (per run): {gm['t_grand_avg']:.4f} s")


def _print_timing_rich_table(
    roll: dict[str, PromptRollup],
    prompt_cases: Sequence[tuple[str, str]],
    console: Console,
) -> None:
    console.print()
    table = Table(
        title="Mean wall time per pipeline stage (s; avg over runs)",
    )
    table.add_column("prompt_id", style="dim")
    for col in (
        "setup",
        "baseline",
        "wm_gen",
        "derive_x",
        "issue_keys",
        "cprf",
        "m_good",
        "det_open",
        "det_vocab",
        "neg_ctrl",
    ):
        table.add_column(col, justify="right")

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
            f"{m['t_derive_verify']:.4f}",
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
        f"{gm['t_derive_verify']:.4f}",
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
    print()
    print(table_heading)
    w_sid = 22
    line = (
        f"{'id':<{w_sid}} {'runs':>5} {'TPR%':>7} {'FNR%':>7} {'TNR%':>7} {'FPR%':>7} "
        f"{'x==':>7} {'mast':>6} {'open':>6} {'ucprf':>6} {'lcprf':>6} {'ctrl':>6} "
        f"{'BER':>8} {'mism':>5} {'m_cp':>4} {'FN_s':>4} {'FP_s':>4}"
    )
    print(line)
    print("-" * len(line))

    def ratio(a: int, b: int) -> str:
        return f"{a}/{b}"

    for sid, _ in prompt_cases:
        r = roll[sid]
        n = r.runs
        sid_disp = sid if len(sid) <= w_sid else sid[: w_sid - 3] + "..."
        if n == 0:
            print(
                f"{sid_disp:<{w_sid}} {n:5d} "
                f"{'n/a':>7} {'n/a':>7} {'n/a':>7} {'n/a':>7} "
                f"{'0/0':>7} {'n/a':>6} {'n/a':>6} {'n/a':>6} {'n/a':>6} {'n/a':>6} "
                f"{'n/a':>8} {'n/a':>5} {'n/a':>4} {'n/a':>4} {'n/a':>4}"
            )
            continue
        tpr, fnr, tnr, fpr = _rates(r.tp, r.fn, r.tn, r.fp)
        lcprf = (
            ratio(r.cprf_per_label_expect_ok, r.cprf_per_label_checks)
            if r.cprf_per_label_checks
            else "n/a"
        )
        mism = str(r.mismatch_total) if r.mismatch_total else "0"
        print(
            f"{sid_disp:<{w_sid}} {n:5d} "
            f"{_fmt_rate(tpr):>7} {_fmt_rate(fnr):>7} {_fmt_rate(tnr):>7} {_fmt_rate(fpr):>7} "
            f"{ratio(r.x_perfect, n):>7} "
            f"{ratio(r.master_good, n):>6} {ratio(r.open_detect_good, n):>6} "
            f"{ratio(r.unconstrained_cprf_ok, n):>6} {lcprf:>6} {ratio(r.control_correct, n):>6} "
            f"{r.ber_sum / n:8.2f} {mism:>5} {r.mismatch_cprf_heuristic_bad:>4} "
            f"{r.mismatch_fn_with_matching_seeds:>4} {r.mismatch_fp_with_split_seeds:>4}"
        )

    if print_legend:
        print()
        print(
            "Legend: TPR/FNR/TNR/FPR from per-(run, vocab label) gold = label in recovered NLI-active set; "
            "mast/open/ctrl = successes/runs; ucprf = unconstrained CPRF seed match; "
            "lcprf = fraction of per-label CPRF checks where sk.eval==dk.c_eval matched attribute-based expectation; "
            "mism = total attribute-vs-detect mismatches; m_cp = mismatches where CPRF seed equality disagreed "
            "with that expectation (composite modulus heuristic); FN_s = FN with seeds matching anyway (LDPC?); "
            "FP_s = FP with CPRF seeds split (possible LDPC false positive)."
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
    table = Table(title=table_title)
    table.add_column("prompt_id", style="dim")
    table.add_column("runs", justify="right")
    for col in ("TPR", "FNR", "TNR", "FPR", "x==", "master", "open", "uCPRF", "lCPRF", "ctrl", "BER%", "mism", "ΔCP", "FN•", "FP•"):
        table.add_column(col, justify="right")

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
            f"{r.x_perfect}/{n}",
            f"{r.master_good}/{n}",
            f"{r.open_detect_good}/{n}",
            f"{r.unconstrained_cprf_ok}/{n}",
            lcprf,
            f"{r.control_correct}/{n}",
            f"{r.ber_sum / n:.2f}",
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
            "mism = attribute-vs-detect mismatches; ΔCP = mismatches where CPRF vs attribute expectation disagreed; "
            "FN• = FN with matching seeds (suspect LDPC); FP• = FP with split seeds (suspect LDPC).[/]"
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
    quiet: bool = False,
) -> tuple[int, BenchmarkRunSummary]:
    for name in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)

    if llm_model_id is not None and llm_model_id.strip():
        wm.set_llm_model_id(llm_model_id.strip())

    wm.set_prc_code_length(code_length)
    wm.set_wm_bit_redundancy(wm_bit_redundancy)
    roll: dict[str, PromptRollup] = {sid: PromptRollup() for sid, _ in prompt_cases}
    roll_xmatch: dict[str, PromptRollup] = {sid: PromptRollup() for sid, _ in prompt_cases}
    sk_shared: dict[str, Any] = {}

    vocab_n = len(VOCABULARY)
    if not quiet:
        console.print(
            f"code_length={wm.SECURITY_PARAM}  wm_bit_redundancy={wm.WM_BIT_REDUNDANCY}  "
            f"channel_bits={wm.wm_channel_bits_length()}  modulus={modulus}  runs={runs}  |V|={vocab_n}  "
            f"keys={'fresh per trial' if fresh_key_per_trial else 'reuse per prompt id'}  "
            f"llm={wm.MODEL_ID!r}"
        )

    trials = [(run_i, sid, prompt) for run_i in range(runs) for sid, prompt in prompt_cases]
    for _, sid, prompt in track(
        trials,
        description="Benchmark",
        console=console,
        transient=True,
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
            x_perfect,
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
            x_perfect=x_perfect,
            master_ok=master_ok,
            open_ok=open_ok,
            unconstrained_cprf_ok=em_open_m,
            cprf_per_label_ok=cprf_label_ok,
            cprf_per_label_n=cprf_label_n,
            control_ok=control_ok,
            ber=ber,
            timings=tt_inner,
        )
        if x_perfect:
            roll_xmatch[sid].add_run(
                word_stats=word_stats,
                x_perfect=True,
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
        roll_xmatch=roll_xmatch,
        prompt_cases=tuple((sid, p) for sid, p in prompt_cases),
        vocab_n=vocab_n,
        code_length=code_length,
        wm_bit_redundancy=wm.WM_BIT_REDUNDANCY,
        modulus=modulus,
        strict_protocol_ok=all_ok,
    )

    plain = _use_plain_table()
    if not quiet:
        _heading_all = (
            "Per-prompt aggregates: policy detection vs NLI attribute (counts over runs × labels="
            + str(vocab_n)
            + ")"
        )
        _heading_x = (
            "Same metrics, restricted to runs where encode-time attr_x equals verify-time derive_x "
            f"(full vector; runs × |V|={vocab_n} label decisions per included run)"
        )
        if plain:
            _print_plain_results(
                prompt_cases=prompt_cases,
                roll=roll,
                vocab_n=vocab_n,
                table_heading=_heading_all,
                print_legend=True,
            )
            _print_plain_results(
                prompt_cases=prompt_cases,
                roll=roll_xmatch,
                vocab_n=vocab_n,
                table_heading=_heading_x,
                print_legend=False,
            )
        else:
            _print_rich_results(
                prompt_cases=prompt_cases,
                roll=roll,
                vocab_n=vocab_n,
                console=console,
                table_title=f"Per-prompt policy metrics (runs × |V|={vocab_n} label decisions per run)",
                print_legend=True,
            )
            _print_rich_results(
                prompt_cases=prompt_cases,
                roll=roll_xmatch,
                vocab_n=vocab_n,
                console=console,
                table_title=(
                    f"Per-prompt policy metrics — x matched only (runs × |V|={vocab_n} per included run)"
                ),
                print_legend=False,
            )

        if plain:
            _print_timing_table_plain(roll, prompt_cases)
        else:
            _print_timing_rich_table(roll, prompt_cases, console)

        if not all_ok:
            _print_protocol_failure_details(
                roll=roll, prompt_cases=prompt_cases, plain=plain, console=console
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
        quiet=False,
    )
    return code


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
        help="Hugging Face hub id for the causal LM (calls watermarking.set_llm_model_id).",
    )
    p.add_argument(
        "--wm-bit-redundancy",
        type=int,
        default=1,
        metavar="R",
        help="Repeat each logical PRC bit R times on the token channel; recovery uses strict majority (ties→0).",
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
    return run_benchmark(
        prompt_cases=cases,
        runs=args.runs,
        modulus=args.modulus,
        code_length=args.code_length,
        fresh_key_per_trial=not args.reuse_key,
        console=console,
        llm_model_id=args.llm_model,
        wm_bit_redundancy=args.wm_bit_redundancy,
    )


if __name__ == "__main__":
    sys.exit(main())
