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
optional ``--llm-model``, ``--no-chat-template`` (plain text completion encoding),
repeatable ``--prompt-case id:prompt``, or ``--c4-realnewslike`` to draw random ``allenai/c4``
``realnewslike`` snippets (snippet length via ``--c4-snippet-chars``).
Env: ``BENCHMARK_PLAIN_TABLE``, ``BENCHMARK_CONSOLE_*``
(see ``make_benchmark_console`` / ``_use_plain_table``).
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

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


def parse_prompt_case(spec: str) -> tuple[str, str]:
    spec = spec.strip()
    if ":" not in spec:
        raise ValueError(f"prompt case must be 'id:prompt', got no colon: {spec!r}")
    sid, prompt = spec.split(":", 1)
    sid, prompt = sid.strip(), prompt.strip()
    if not sid or not prompt:
        raise ValueError(f"empty id or prompt in {spec!r}")
    return (sid, prompt)


C4_HUB_ID = "allenai/c4"
C4_CONFIG_REALNEWSLIKE = "realnewslike"
C4_TRIAL_PROMPT_ID = "c4_realnewslike"


def _c4_realnewslike_row_stream(split: str):
    """Yield rows forever by re-opening the stream if a pass ends (streaming iterator)."""
    while True:
        from datasets import load_dataset

        ds = load_dataset(
            C4_HUB_ID,
            C4_CONFIG_REALNEWSLIKE,
            split=split,
            streaming=True,
        )
        yield from ds


def make_c4_realnewslike_prompt_sampler(
    *,
    snippet_chars: int,
    seed: int | None = None,
    split: str = "train",
    inter_sample_skip_max: int = 256,
) -> tuple[Callable[[], str], str]:
    """
    Random-window sampler over ``allenai/c4`` ``realnewslike`` documents (streaming).

    Each call returns ``snippet_chars`` characters from a whitespace-collapsed excerpt of a pseudo-
    random row's ``text`` field (uniform random contiguous window when the document is longer).
    Before each draw, skip ``Uniform(0, inter_sample_skip_max)`` dataset rows (when > 0) so trials
    are not always consecutive RealNews-like documents.
    """
    if snippet_chars < 1:
        raise ValueError("snippet_chars must be >= 1")
    if inter_sample_skip_max < 0:
        raise ValueError("inter_sample_skip_max must be >= 0")

    rng = random.Random(seed) if seed is not None else random.Random()
    rows = _c4_realnewslike_row_stream(split)

    def sample_prompt() -> str:
        if inter_sample_skip_max > 0:
            for _ in range(rng.randint(0, inter_sample_skip_max)):
                next(rows)
        attempts = 0
        max_attempts = 200
        while attempts < max_attempts:
            attempts += 1
            row = next(rows)
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            collapsed = " ".join(text.split())
            if not collapsed:
                continue
            if len(collapsed) <= snippet_chars:
                return collapsed
            start = rng.randrange(len(collapsed) - snippet_chars + 1)
            return collapsed[start : start + snippet_chars]

        raise RuntimeError(
            f"could not sample a non-empty snippet from {C4_HUB_ID} {C4_CONFIG_REALNEWSLIKE!r} "
            f"(split={split!r}) after {max_attempts} rows — check connectivity / dataset schema"
        )

    banner = (
        f"c4[{C4_CONFIG_REALNEWSLIKE} split={split!r}] len={snippet_chars}"
        f" inter_skip_max={inter_sample_skip_max}"
        + (f" rng_seed={seed}" if seed is not None else "")
    )
    return sample_prompt, banner


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
    *,
    use_chat_template: bool = True,
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

    out = wm.generate(sk, prompt, use_chat_template=use_chat_template)
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


def run_benchmark(
    *,
    runs: int,
    modulus: int,
    code_length: int,
    fresh_key_per_trial: bool,
    console: Console,
    prompt_cases: Sequence[tuple[str, str]] = (),
    llm_model_id: str | None = None,
    wm_bit_redundancy: int = 1,
    use_chat_template: bool = True,
    trial_prompt_sampler: Callable[[], str] | None = None,
    trial_prompt_case_id: str = C4_TRIAL_PROMPT_ID,
    sampler_banner: str | None = None,
) -> int:
    for name in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)

    if trial_prompt_sampler is not None and prompt_cases:
        raise ValueError("pass prompt_cases=() when trial_prompt_sampler is set")

    if trial_prompt_sampler is not None:
        prompt_cases_eff: list[tuple[str, str]] = [
            (trial_prompt_case_id, sampler_banner or "(random prompts)"),
        ]
        trials = [
            (i, trial_prompt_case_id, trial_prompt_sampler()) for i in range(runs)
        ]
    else:
        prompt_cases_eff = list(prompt_cases) if prompt_cases else list(DEFAULT_PROMPT_CASES)
        trials = [
            (run_i, sid, prompt)
            for run_i in range(runs)
            for sid, prompt in prompt_cases_eff
        ]

    if llm_model_id is not None and llm_model_id.strip():
        wm.set_llm_model_id(llm_model_id.strip())

    wm.set_prc_code_length(code_length)
    wm.set_wm_bit_redundancy(wm_bit_redundancy)
    roll: dict[str, PromptRollup] = {sid: PromptRollup() for sid, _ in prompt_cases_eff}
    roll_xmatch: dict[str, PromptRollup] = {sid: PromptRollup() for sid, _ in prompt_cases_eff}
    sk_shared: dict[str, Any] = {}

    vocab_n = len(VOCABULARY)
    banner_extra = f"  |  {sampler_banner}" if sampler_banner else ""
    console.print(
        f"code_length={wm.SECURITY_PARAM}  wm_bit_redundancy={wm.WM_BIT_REDUNDANCY}  "
        f"channel_bits={wm.wm_channel_bits_length()}  modulus={modulus}  runs={runs}  |V|={vocab_n}  "
        f"keys={'fresh per trial' if fresh_key_per_trial else 'reuse per prompt id'}  "
        f"prompt_encode={'chat' if use_chat_template else 'plain'}  "
        f"llm={wm.MODEL_ID!r}{banner_extra}"
    )
    for _, sid, prompt in track(
        trials,
        description="Benchmark",
        console=console,
        transient=True,
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
        ) = run_one_trial(sk, prompt, use_chat_template=use_chat_template)
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

    plain = _use_plain_table()
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
            prompt_cases=prompt_cases_eff,
            roll=roll,
            vocab_n=vocab_n,
            table_heading=_heading_all,
            print_legend=True,
        )
        _print_plain_results(
            prompt_cases=prompt_cases_eff,
            roll=roll_xmatch,
            vocab_n=vocab_n,
            table_heading=_heading_x,
            print_legend=False,
        )
    else:
        _print_rich_results(
            prompt_cases=prompt_cases_eff,
            roll=roll,
            vocab_n=vocab_n,
            console=console,
            table_title=f"Per-prompt policy metrics (runs × |V|={vocab_n} label decisions per run)",
            print_legend=True,
        )
        _print_rich_results(
            prompt_cases=prompt_cases_eff,
            roll=roll_xmatch,
            vocab_n=vocab_n,
            console=console,
            table_title=(
                f"Per-prompt policy metrics — x matched only (runs × |V|={vocab_n} per included run)"
            ),
            print_legend=False,
        )

    if plain:
        _print_timing_table_plain(roll, prompt_cases_eff)
    else:
        _print_timing_rich_table(roll, prompt_cases_eff, console)

    all_ok = True
    for sid, _ in prompt_cases_eff:
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

    if not all_ok:
        msg = (
            "Protocol checks did not pass on every run (require: master_detect good, detect open, "
            "unconstrained CPRF match, negative control rejects)."
        )
        if plain:
            print()
            print(msg)
            for sid, _ in prompt_cases_eff:
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
            for sid, _ in prompt_cases_eff:
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

    return 0 if all_ok else 1


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
        "--no-chat-template",
        action="store_true",
        help="Encode prompts as plain tokenizer text (skip chat template); text-completion style.",
    )
    p.add_argument(
        "--c4-realnewslike",
        action="store_true",
        help=(
            "Use random snippets from Hugging Face ``allenai/c4`` ``realnewslike`` as prompts "
            "(``--runs`` independent draws). Mutually exclusive with ``--prompt-case``."
        ),
    )
    p.add_argument(
        "--c4-snippet-chars",
        type=int,
        default=512,
        metavar="N",
        help="Character length of each random C4 excerpt (default: 512).",
    )
    p.add_argument(
        "--c4-seed",
        type=int,
        default=None,
        metavar="S",
        help="Optional RNG seed for C4 window placement and row skips (default: nondeterministic).",
    )
    p.add_argument(
        "--c4-split",
        default="train",
        metavar="SPLIT",
        help="C4 split passed to ``load_dataset`` (default: train).",
    )
    p.add_argument(
        "--c4-inter-sample-skip-max",
        type=int,
        default=256,
        metavar="K",
        help=(
            "Before each C4 prompt, skip up to K random rows in the stream (default: 256; use 0 "
            "to always take the next row)."
        ),
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
    if args.c4_snippet_chars < 1:
        print("c4-snippet-chars must be >= 1", file=sys.stderr)
        return 2
    if args.c4_inter_sample_skip_max < 0:
        print("c4-inter-sample-skip-max must be >= 0", file=sys.stderr)
        return 2
    if args.c4_realnewslike and args.prompt_cases:
        print(
            "--c4-realnewslike cannot be combined with --prompt-case (choose one prompt source)",
            file=sys.stderr,
        )
        return 2

    console = make_benchmark_console()
    if args.c4_realnewslike:
        try:
            sampler, banner = make_c4_realnewslike_prompt_sampler(
                snippet_chars=args.c4_snippet_chars,
                seed=args.c4_seed,
                split=args.c4_split.strip() or "train",
                inter_sample_skip_max=args.c4_inter_sample_skip_max,
            )
        except Exception as e:
            print(f"C4 sampler setup failed: {e}", file=sys.stderr)
            return 2
        return run_benchmark(
            prompt_cases=(),
            runs=args.runs,
            modulus=args.modulus,
            code_length=args.code_length,
            fresh_key_per_trial=not args.reuse_key,
            console=console,
            llm_model_id=args.llm_model,
            wm_bit_redundancy=args.wm_bit_redundancy,
            use_chat_template=not args.no_chat_template,
            trial_prompt_sampler=sampler,
            sampler_banner=banner,
        )

    cases: list[tuple[str, str]] = []
    if args.prompt_cases:
        try:
            cases.extend(parse_prompt_case(s) for s in args.prompt_cases)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 2
    if not cases:
        cases = list(DEFAULT_PROMPT_CASES)

    return run_benchmark(
        prompt_cases=cases,
        runs=args.runs,
        modulus=args.modulus,
        code_length=args.code_length,
        fresh_key_per_trial=not args.reuse_key,
        console=console,
        llm_model_id=args.llm_model,
        wm_bit_redundancy=args.wm_bit_redundancy,
        use_chat_template=not args.no_chat_template,
    )


if __name__ == "__main__":
    sys.exit(main())
