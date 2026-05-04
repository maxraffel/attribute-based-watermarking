"""
Watermarking benchmark: multiple prompts, repeated runs, one summary table per prompt.

For each prompt and run: ``generate`` → issue unconstrained, matching-policy, and unrelated
keys → ``master_detect`` plus three ``detect`` calls. Records success counts, BER (recovered vs
embedded PRC bits from ``generate``), timings (baseline vs watermarked generation from
``watermarking.generate``), and whether encode-time ``attr_x`` exactly equals verify-time
``derive_x`` on the watermarked text (count of perfect matches per prompt over runs).
Also reports **KPI/x**: among runs with a perfect ``x`` match, how often all detection KPIs
passed; and **total wall time** per operation summed over the whole benchmark (footer).

CLI: ``--code-length``, ``--runs``, ``--modulus``, ``--reuse-key``, optional ``--llm-model HF_HUB_ID``
(HF causal LM for ``watermarking``; overrides ``WATERMARK_LLM_ID`` for this process), and repeatable
``--prompt-case id:prompt`` (split on first ``:`` only). If no cases are given, two defaults are used.
Non-TTY / captured stdout (e.g. Colab ``subprocess.run(..., capture_output=True)``) prints a
**plain-text** results table automatically; set ``BENCHMARK_PLAIN_TABLE=0`` to force the Rich
table anyway, or ``=1`` to force plain text even on a TTY. ``BENCHMARK_CONSOLE_WIDTH`` /
``BENCHMARK_CONSOLE_HEIGHT`` tune the Rich console when stdout is not a TTY.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Sequence

from rich.console import Console
from rich.table import Table


def make_benchmark_console() -> Console:
    """
    When stdout is a **real TTY**, use the terminal's size (do not force a wide floor — that
    reflows badly in <168-column terminals). When stdout is **not** a TTY (piped / Colab
    ``capture_output``), set explicit dimensions so Rich's own ``print`` helpers still measure
    consistently; the results table itself switches to plain text (see ``_use_plain_table()``).
    """
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
    """Rich boxed tables ellipsize badly under captured/piped stdout; use fixed-width text."""
    v = os.environ.get("BENCHMARK_PLAIN_TABLE", "").strip().lower()
    if v in ("1", "true", "yes", "always"):
        return True
    if v in ("0", "false", "no", "never"):
        return False
    return not sys.stdout.isatty()


DEFAULT_PROMPT_CASES: list[tuple[str, str]] = [
    (
        "dcf_finance",
        "Explain how a DCF works in the context of finance."
    ),
    ("med_school", "Describe the best medical schools and their medical specialties."),
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


@dataclass
class PromptRollup:
    """Accumulated stats for one prompt id across runs."""

    runs: int = 0
    master_ok: int = 0
    open_ok: int = 0
    accept_ok: int = 0
    reject_correct: int = 0
    ber_sum: float = 0.0
    x_perfect_match: int = 0
    #: Runs where x matched *and* master/open/accept/reject KPIs all succeeded.
    full_kpi_when_x_perfect: int = 0
    t_baseline_sum: float = 0.0
    t_wm_sum: float = 0.0
    t_issue_sum: float = 0.0
    t_master_sum: float = 0.0
    t_open_sum: float = 0.0
    t_accept_sum: float = 0.0
    t_reject_sum: float = 0.0

    def add_run(
        self,
        *,
        master_ok: bool,
        open_ok: bool,
        accept_ok: bool,
        reject_got_true: bool,
        ber: float,
        x_perfect: bool,
        t_baseline: float,
        t_wm: float,
        t_issue: float,
        t_master: float,
        t_open: float,
        t_accept: float,
        t_reject: float,
    ) -> None:
        self.runs += 1
        if master_ok:
            self.master_ok += 1
        if open_ok:
            self.open_ok += 1
        if accept_ok:
            self.accept_ok += 1
        if not reject_got_true:
            self.reject_correct += 1
        self.ber_sum += ber
        if x_perfect:
            self.x_perfect_match += 1
        if x_perfect and master_ok and open_ok and accept_ok and (not reject_got_true):
            self.full_kpi_when_x_perfect += 1
        self.t_baseline_sum += t_baseline
        self.t_wm_sum += t_wm
        self.t_issue_sum += t_issue
        self.t_master_sum += t_master
        self.t_open_sum += t_open
        self.t_accept_sum += t_accept
        self.t_reject_sum += t_reject


def _fmt_kpi_rate_given_x(r: PromptRollup) -> str:
    """Full KPI successes among runs where encode-time x equals verify-time x."""
    nx = r.x_perfect_match
    if nx == 0:
        return "    n/a"
    return f"{r.full_kpi_when_x_perfect}/{nx}".rjust(7)


def _aggregate_timings(
    roll: dict[str, PromptRollup], prompt_cases: Sequence[tuple[str, str]]
) -> dict[str, float]:
    out = {k: 0.0 for k in (
        "t_baseline",
        "t_wm",
        "t_issue",
        "t_master",
        "t_open",
        "t_accept",
        "t_reject",
    )}
    for sid, _ in prompt_cases:
        r = roll[sid]
        if r.runs == 0:
            continue
        out["t_baseline"] += r.t_baseline_sum
        out["t_wm"] += r.t_wm_sum
        out["t_issue"] += r.t_issue_sum
        out["t_master"] += r.t_master_sum
        out["t_open"] += r.t_open_sum
        out["t_accept"] += r.t_accept_sum
        out["t_reject"] += r.t_reject_sum
    det = (
        out["t_master"] + out["t_open"] + out["t_accept"] + out["t_reject"]
    )
    out["t_detect_total"] = det
    out["t_grand"] = (
        out["t_baseline"]
        + out["t_wm"]
        + out["t_issue"]
        + det
    )
    return out


def _print_aggregate_timings_footer(
    *,
    roll: dict[str, PromptRollup],
    prompt_cases: Sequence[tuple[str, str]],
    console: Console,
    plain: bool,
) -> None:
    t = _aggregate_timings(roll, prompt_cases)
    lines = [
        "",
        "Total wall time (sum over every benchmark run; all prompts)",
        f"  baseline_generation (greedy):     {t['t_baseline']:10.3f} s",
        f"  watermarked_generation:           {t['t_wm']:10.3f} s",
        f"  issue_keys (3 policies):        {t['t_issue']:10.3f} s",
        f"  master_detect:                    {t['t_master']:10.3f} s",
        f"  detect (unconstrained):         {t['t_open']:10.3f} s",
        f"  detect (accept policy):         {t['t_accept']:10.3f} s",
        f"  detect (reject policy):           {t['t_reject']:10.3f} s",
        f"  detection subtotal (4 calls):   {t['t_detect_total']:10.3f} s",
        "  --------------------------------------------",
        f"  all measured operations:        {t['t_grand']:10.3f} s",
    ]
    if plain:
        print("\n".join(lines))
    else:
        console.print()
        for line in lines:
            if line.startswith("  all measured"):
                console.print("[bold]" + line + "[/]")
            else:
                console.print("[dim]" + line + "[/]")


def _print_plain_results_table(
    *,
    prompt_cases: Sequence[tuple[str, str]],
    roll: dict[str, PromptRollup],
) -> None:
    w_sid = 26
    print()
    print("Per-prompt averages over runs")
    header = (
        f"{'prompt_id':<{w_sid}} {'runs':>5} {'master':>7} {'open':>7} {'accept':>7} "
        f"{'rej_ok':>7} {'BER%':>7} {'x==':>7} {'KPI/x':>7} "
        f"{'t_bl':>9} {'t_wm':>9} {'t_key':>9} {'t_det':>9}"
    )
    print(header)
    print("-" * len(header))

    def ratio(k: int, n: int) -> str:
        return f"{k}/{n}"

    for sid, _ in prompt_cases:
        r = roll[sid]
        n = r.runs
        if n == 0:
            continue
        det_avg = (r.t_master_sum + r.t_open_sum + r.t_accept_sum + r.t_reject_sum) / n
        sid_disp = sid if len(sid) <= w_sid else sid[: w_sid - 3] + "..."
        kpi_x = _fmt_kpi_rate_given_x(r)
        print(
            f"{sid_disp:<{w_sid}} {n:5d} "
            f"{ratio(r.master_ok, n):>7} {ratio(r.open_ok, n):>7} {ratio(r.accept_ok, n):>7} "
            f"{ratio(r.reject_correct, n):>7} {r.ber_sum / n:7.2f} {ratio(r.x_perfect_match, n):>7} "
            f"{kpi_x:>7} "
            f"{r.t_baseline_sum / n:9.3f} {r.t_wm_sum / n:9.3f} {r.t_issue_sum / n:9.4f} {det_avg:9.4f}"
        )
    print()
    print(
        "Legend: master/open/accept = successes/runs; rej_ok = correct reject (expect reject=False); "
        "BER% = mean bit error vs embedded PRC (master recovery); x== = runs with full x match; "
        "KPI/x = full KPI successes among x-matched runs (n/a if no x match); "
        "t_bl / t_wm / t_key / t_det = mean seconds."
    )


def run_benchmark(
    *,
    prompt_cases: Sequence[tuple[str, str]],
    runs: int,
    modulus: int,
    code_length: int,
    fresh_key_per_trial: bool,
    console: Console,
    llm_model_id: str | None = None,
) -> int:
    import watermarking as wm
    import attr_x_nli
    from closed_vocab import (
        active_labels_from_verify_x,
        pick_unrelated_keyword_for_policy,
    )

    if llm_model_id is not None and llm_model_id.strip():
        wm.set_llm_model_id(llm_model_id.strip())

    wm.set_prc_code_length(code_length)
    roll: dict[str, PromptRollup] = {sid: PromptRollup() for sid, _ in prompt_cases}
    sk_shared: dict[str, object] = {}

    console.print(
        f"code_length={wm.SECURITY_PARAM}  modulus={modulus}  runs={runs}  "
        f"keys={'fresh per run' if fresh_key_per_trial else 'reuse per prompt id'}  "
        f"llm={wm.MODEL_ID!r}"
    )

    for _ in range(runs):
        for sid, prompt in prompt_cases:
            if fresh_key_per_trial:
                sk = wm.setup(modulus)
            else:
                if sk_shared.get(sid) is None:
                    sk_shared[sid] = wm.setup(modulus)
                sk = sk_shared[sid]

            out = wm.generate(sk, prompt)
            text = out["generated_text_wm"]
            x_gen = out["attr_x"]
            secret = out["prc_secret_bits"]
            t_bl = float(out["seconds_baseline_gen"])
            t_wm = float(out["seconds_watermarked_gen"])

            x_verify = attr_x_nli.derive_x(text, sk.modulus)
            active = active_labels_from_verify_x(x_verify, sk.modulus)
            x_perfect = list(x_gen) == list(x_verify)

            t0 = time.perf_counter()
            dk_open = wm.issue_unconstrained(sk)
            if active:
                dk_accept = wm.issue_keyword_policy(sk, list(active))
            else:
                dk_accept = wm.issue_keyword_policy(sk, [])
            unrelated = pick_unrelated_keyword_for_policy(
                x_verify, sk.modulus, set(active)
            )
            dk_reject = wm.issue_keyword_policy(sk, [unrelated])
            t_issue = time.perf_counter() - t0

            t0 = time.perf_counter()
            m_ok, m_bits = wm.master_detect(sk, text)
            t_m = time.perf_counter() - t0

            t0 = time.perf_counter()
            u_ok, _ = wm.detect(dk_open, text)
            t_u = time.perf_counter() - t0

            t0 = time.perf_counter()
            a_ok, _ = wm.detect(dk_accept, text)
            t_a = time.perf_counter() - t0

            t0 = time.perf_counter()
            r_ok, _ = wm.detect(dk_reject, text)
            t_r = time.perf_counter() - t0

            ber = _ber_percent(secret, m_bits)

            roll[sid].add_run(
                master_ok=bool(m_ok),
                open_ok=bool(u_ok),
                accept_ok=bool(a_ok),
                reject_got_true=bool(r_ok),
                ber=ber,
                x_perfect=x_perfect,
                t_baseline=t_bl,
                t_wm=t_wm,
                t_issue=t_issue,
                t_master=t_m,
                t_open=t_u,
                t_accept=t_a,
                t_reject=t_r,
            )

    all_ok = True
    for sid, _ in prompt_cases:
        r = roll[sid]
        n = r.runs
        if n == 0:
            continue
        all_ok = all_ok and (
            r.master_ok == n and r.open_ok == n and r.accept_ok == n and r.reject_correct == n
        )

    console.print()
    if _use_plain_table():
        _print_plain_results_table(prompt_cases=prompt_cases, roll=roll)
    else:
        table = Table(title="Per-prompt averages over runs")
        table.add_column("prompt_id", style="dim")
        table.add_column("runs", justify="right")
        table.add_column("master", justify="right")
        table.add_column("open", justify="right")
        table.add_column("accept", justify="right")
        table.add_column("rej_ok", justify="right")
        table.add_column("BER%", justify="right")
        table.add_column("x==", justify="right")
        table.add_column("KPI/x", justify="right")
        table.add_column("t_bl", justify="right")
        table.add_column("t_wm", justify="right")
        table.add_column("t_key", justify="right")
        table.add_column("t_det", justify="right")
        for sid, _ in prompt_cases:
            r = roll[sid]
            n = r.runs
            if n == 0:
                continue
            det_avg = (r.t_master_sum + r.t_open_sum + r.t_accept_sum + r.t_reject_sum) / n
            table.add_row(
                sid,
                str(n),
                f"{r.master_ok}/{n}",
                f"{r.open_ok}/{n}",
                f"{r.accept_ok}/{n}",
                f"{r.reject_correct}/{n}",
                f"{r.ber_sum / n:.2f}",
                f"{r.x_perfect_match}/{n}",
                _fmt_kpi_rate_given_x(r).strip(),
                f"{r.t_baseline_sum / n:.3f}",
                f"{r.t_wm_sum / n:.3f}",
                f"{r.t_issue_sum / n:.4f}",
                f"{det_avg:.4f}",
            )
        console.print(table)
        console.print(
            "[dim]master/open/accept = successes / runs; rej_ok = correct reject (expect False). "
            "BER% = mean bit error vs embedded PRC stream (master path recovery). "
            "x== = runs where encode-time attr_x equals verify-time derive_x (full vector). "
            "KPI/x = full KPI successes among x-matched runs (n/a if none). "
            "t_bl / t_wm = mean baseline vs watermarked gen seconds from generate(). "
            "t_key = mean issue time; t_det = mean total of four detection calls.[/]"
        )
    if not all_ok:
        if _use_plain_table():
            print()
            print(
                "Exiting with code 1: at least one run did not meet all KPI checks "
                "(master + unconstrained + policy-accept must succeed; policy-reject must fail)."
            )
            for sid, _ in prompt_cases:
                r = roll[sid]
                n = r.runs
                if n == 0:
                    continue
                bad: list[str] = []
                if r.master_ok < n:
                    bad.append(f"master {r.master_ok}/{n}")
                if r.open_ok < n:
                    bad.append(f"open {r.open_ok}/{n}")
                if r.accept_ok < n:
                    bad.append(f"accept {r.accept_ok}/{n}")
                if r.reject_correct < n:
                    bad.append(f"reject_false_positive {n - r.reject_correct}/{n}")
                if bad:
                    print(f"  {sid}: " + ", ".join(bad))
        else:
            console.print(
                "[bold red]Exiting with code 1:[/] at least one run did not meet all KPI checks "
                "(master + unconstrained + policy-accept must succeed; policy-reject must fail)."
            )
            for sid, _ in prompt_cases:
                r = roll[sid]
                n = r.runs
                if n == 0:
                    continue
                bad: list[str] = []
                if r.master_ok < n:
                    bad.append(f"master {r.master_ok}/{n}")
                if r.open_ok < n:
                    bad.append(f"open {r.open_ok}/{n}")
                if r.accept_ok < n:
                    bad.append(f"accept {r.accept_ok}/{n}")
                if r.reject_correct < n:
                    bad.append(f"reject_false_positive {n - r.reject_correct}/{n}")
                if bad:
                    console.print(f"  [yellow]{sid}:[/] " + ", ".join(bad))

    _print_aggregate_timings_footer(
        roll=roll,
        prompt_cases=prompt_cases,
        console=console,
        plain=_use_plain_table(),
    )

    return 0 if all_ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--code-length", type=int, default=300, metavar="N")
    p.add_argument("--modulus", type=int, default=1024)
    p.add_argument(
        "--reuse-key",
        action="store_true",
        help="Reuse one master key per prompt id across runs",
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
    args = p.parse_args()
    if args.runs < 1 or args.code_length < 1:
        print("runs and code-length must be >= 1", file=sys.stderr)
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
    )


if __name__ == "__main__":
    sys.exit(main())
