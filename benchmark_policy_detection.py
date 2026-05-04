"""
Constrained-key + PRC detection benchmark for one or more **prompt cases**.

Each **case** is ``(scenario_id, prompt)``. Per trial and per case: ``setup`` (optional reuse),
``generate``, then ``derive_x`` on the watermarked text to see which vocabulary labels are
**NLI-active**. The benchmark issues constrained keys for:

* all active labels together (one key),
* each active label alone,
* one **unrelated** single-label key (chosen so ``⟨f,x⟩ ≢ 0`` on verify-time ``x``),

and runs ``master_detect``, ``detect(unconstrained)``, and the policy checks above. Pass
``--prompt-case`` (repeatable) as ``id:first line of prompt...`` (only the **first** ``:`` splits
id from prompt); if omitted, two built-in defaults mirror the former A/B prompts.

Metrics are **binomial** per ``(scenario_id, test name)`` with Wilson 95% intervals, runtime
means per step, and optional failure diagnostics (``-v``).

PRC codeword length: ``--code-length`` → ``watermarking.set_prc_code_length``.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from detect_diagnostics import log_detect_mismatch

# Default cases (same prompts as the legacy two-scenario benchmark).
DEFAULT_PROMPT_CASES: list[tuple[str, str]] = [
    (
        "medical_procedure",
        "describe a common outpatient medical procedure "
        "and what the patient should expect before and after the visit.",
    ),
    (
        "med_school",
        "Describe the intersection of medicine and education.",
    ),
    (
        "medical_law",
        "Explain the intersection of medicine and law.",
    ),
    (
        "medical_software",
        "Explain the intersection of medicine and software.",
    ),
    (
        "medical_finance",
        "Explain the intersection of medicine and finance.",
    ),
]


@dataclass
class AggregatedTest:
    scenario: str
    test_name: str
    expect_true: bool
    successes: int = 0
    trials: int = 0

    def add(self, ok: bool) -> None:
        self.trials += 1
        if ok:
            self.successes += 1

    @property
    def rate(self) -> float:
        return self.successes / self.trials if self.trials else 0.0


@dataclass
class AggregatedTiming:
    scenario: str
    step: str
    total_seconds: float = 0.0
    samples: int = 0

    def add(self, seconds: float) -> None:
        self.total_seconds += seconds
        self.samples += 1

    @property
    def mean_seconds(self) -> float:
        return self.total_seconds / self.samples if self.samples else 0.0


def wilson_95(successes: int, n: int) -> tuple[float, float]:
    """Wilson score interval for binomial p, z=1.96."""
    if n == 0:
        return (0.0, 1.0)
    z = 1.96
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def parse_prompt_case(spec: str) -> tuple[str, str]:
    """``id:prompt text...`` — split on the first colon only."""
    spec = spec.strip()
    if ":" not in spec:
        raise ValueError(f"prompt case must be 'id:prompt', got no colon: {spec!r}")
    sid, prompt = spec.split(":", 1)
    sid = sid.strip()
    prompt = prompt.strip()
    if not sid or not prompt:
        raise ValueError(f"empty id or prompt in {spec!r}")
    return (sid, prompt)


def _timing_sort_key(key: tuple[str, str], scenario_order: dict[str, int]) -> tuple:
    scen, step = key
    if scen == "_trial_":
        return (1 << 20, scen, step)
    si = scenario_order.get(scen, 999)
    return (si, step)


def run_prompt_benchmark(
    *,
    prompt_cases: Sequence[tuple[str, str]],
    runs: int,
    modulus: int,
    code_length: int,
    fresh_key_per_trial: bool,
    console: Console,
    verbose: bool = False,
) -> int:
    """
    Run ``runs`` trials; each trial runs every prompt case in order (one wall-clock bucket per trial).

    Returns 0 iff every aggregated test passed on every trial.
    """
    import watermarking as wm
    import attr_x_nli
    from closed_vocab import (
        CPRF_ATTR_DIM,
        VOCABULARY,
        active_labels_from_verify_x,
        pick_unrelated_keyword_for_policy,
    )

    wm.set_prc_code_length(code_length)
    prc_code_length = wm.SECURITY_PARAM
    nli_bar = attr_x_nli.NLI_LABEL_ACTIVE_MIN_SCORE
    n_labels = len(VOCABULARY)
    scenario_order = {sid: i for i, (sid, _) in enumerate(prompt_cases)}

    cases_line = ", ".join(f"[cyan]{sid}[/]" for sid, _ in prompt_cases)
    console.print(
        Panel.fit(
            f"[bold]PRC codeword length[/]: [cyan]{prc_code_length}[/]\n"
            f"[bold]CPRF[/] modulus: [cyan]{modulus}[/]  ·  attr dim: [cyan]{CPRF_ATTR_DIM}[/] "
            f"(|V|={n_labels} + tail)\n"
            f"[bold]Cases[/]: {cases_line}\n"
            f"[bold]Trials[/]: [cyan]{runs}[/]  ·  "
            f"[bold]Key policy[/]: {'new master key each case per trial' if fresh_key_per_trial else 'reuse one key per case_id'}\n"
            f"[bold]NLI active bar[/]: [cyan]{nli_bar}[/]",
            title="Benchmark configuration",
        )
    )
    if runs < 30:
        console.print(
            "[dim]Note: with runs < 30, Wilson intervals are wide; use 50–200+ for tighter estimates.[/]"
        )

    agg: dict[tuple[str, str], AggregatedTest] = {}
    timing: dict[tuple[str, str], AggregatedTiming] = {}
    diag: Counter[tuple[str, str, str]] = Counter()

    def record(scenario: str, name: str, expected: bool, got: object) -> None:
        ok = got == expected
        key = (scenario, name)
        if key not in agg:
            agg[key] = AggregatedTest(
                scenario=scenario, test_name=name, expect_true=expected
            )
        agg[key].add(ok)

    def time_add(scenario: str, step: str, seconds: float) -> None:
        key = (scenario, step)
        if key not in timing:
            timing[key] = AggregatedTiming(scenario=scenario, step=step)
        timing[key].add(seconds)

    def run_detect(
        scenario: str,
        test_name: str,
        expect_true: bool,
        measure: Callable[[], object],
        *,
        x_encode: Sequence[int],
        wm_text: str,
        mod: int,
        baseline_text: str,
    ) -> None:
        t0 = time.perf_counter()
        got = measure()
        time_add(scenario, test_name, time.perf_counter() - t0)
        record(scenario, test_name, expect_true, got)
        if got != expect_true:
            log_detect_mismatch(
                scenario=scenario,
                test_name=test_name,
                expect_true=expect_true,
                got=got,
                x_encode=x_encode,
                wm_text=wm_text,
                modulus=mod,
                baseline_text=baseline_text,
                vocabulary=VOCABULARY,
                diag=diag,
                rich_console=console,
                verbose=verbose,
            )

    sk_shared: dict[str, object] = {}

    for _ in range(runs):
        trial_t0 = time.perf_counter()
        for scenario_id, prompt in prompt_cases:
            t0 = time.perf_counter()
            if fresh_key_per_trial:
                sk = wm.setup(modulus)
            else:
                if sk_shared.get(scenario_id) is None:
                    sk_shared[scenario_id] = wm.setup(modulus)
                sk = sk_shared[scenario_id]
            time_add(scenario_id, "setup", time.perf_counter() - t0)

            t0 = time.perf_counter()
            out = wm.generate(sk, prompt)
            time_add(scenario_id, "generate", time.perf_counter() - t0)
            text = out["generated_text_wm"]
            x_gen = out["attr_x"]
            bl = out["baseline_text"]

            x_verify = attr_x_nli.derive_x(text, sk.modulus)
            active = active_labels_from_verify_x(x_verify, sk.modulus)
            console.print(
                f"[dim]{scenario_id}: NLI-active on watermarked text: "
                f"[cyan]{', '.join(active) if active else '(none)'}[/][/]"
            )

            t0 = time.perf_counter()
            dk_open = wm.issue_unconstrained(sk)
            policy_keys: list[tuple[str, object]] = []
            if active:
                all_name = "detect(policy:" + "+".join(sorted(active)) + ")"
                policy_keys.append(
                    (all_name, wm.issue_keyword_policy(sk, list(active)))
                )
            for lab in sorted(active):
                policy_keys.append(
                    (f"detect(policy:{lab})", wm.issue_keyword_policy(sk, [lab]))
                )
            unrelated = pick_unrelated_keyword_for_policy(
                x_verify, sk.modulus, set(active)
            )
            dk_bad = wm.issue_keyword_policy(sk, [unrelated])
            time_add(scenario_id, "issue_keys", time.perf_counter() - t0)
            console.print(
                f"[dim]{scenario_id}: unrelated single-label policy: [cyan]{unrelated}[/][/]"
            )

            run_detect(
                scenario_id,
                "master_detect",
                True,
                lambda: wm.master_detect(sk, text),
                x_encode=x_gen,
                wm_text=text,
                mod=sk.modulus,
                baseline_text=bl,
            )
            run_detect(
                scenario_id,
                "detect(unconstrained)",
                True,
                lambda: wm.detect(dk_open, text),
                x_encode=x_gen,
                wm_text=text,
                mod=sk.modulus,
                baseline_text=bl,
            )

            if not active:
                console.print(
                    f"[yellow]{scenario_id}: no NLI-active labels — skipping combined / "
                    f"per-label policy detects (still running unrelated).[/]"
                )
            for test_name, dk in policy_keys:
                run_detect(
                    scenario_id,
                    test_name,
                    True,
                    lambda d=dk: wm.detect(d, text),
                    x_encode=x_gen,
                    wm_text=text,
                    mod=sk.modulus,
                    baseline_text=bl,
                )

            run_detect(
                scenario_id,
                "detect(unrelated_policy)",
                False,
                lambda: wm.detect(dk_bad, text),
                x_encode=x_gen,
                wm_text=text,
                mod=sk.modulus,
                baseline_text=bl,
            )

        time_add("_trial_", "wall", time.perf_counter() - trial_t0)

    # ---- aggregate tables ----
    table = Table(title="Aggregate results (Wilson 95% CI on success rate)")
    table.add_column("Scenario", style="dim")
    table.add_column("Test")
    table.add_column("Expect", justify="center")
    table.add_column("n", justify="right")
    table.add_column("OK", justify="right")
    table.add_column("Rate", justify="right")
    table.add_column("Wilson 95%", justify="left")

    def sort_key(k: tuple[str, str]) -> tuple[str, str]:
        return k

    for (scen, tname) in sorted(agg.keys(), key=sort_key):
        a = agg[(scen, tname)]
        lo, hi = wilson_95(a.successes, a.trials)
        table.add_row(
            scen,
            tname,
            "True" if a.expect_true else "False",
            str(a.trials),
            f"{a.successes}/{a.trials}",
            f"{100 * a.rate:.1f}%",
            f"[{100 * lo:.1f}%, {100 * hi:.1f}%]",
        )

    console.print()
    console.print(table)

    fa_table = Table(title="Error-type rates (for scaling runs)")
    fa_table.add_column("Scenario")
    fa_table.add_column("Test")
    fa_table.add_column("Type", style="yellow")
    fa_table.add_column("Count", justify="right")
    fa_table.add_column("Per trial", justify="right")

    for (scen, tname) in sorted(agg.keys(), key=sort_key):
        a = agg[(scen, tname)]
        if a.expect_true:
            misses = a.trials - a.successes
            if misses:
                fa_table.add_row(
                    scen, tname, "miss (expected True, got False)", str(misses), f"{misses}/{a.trials}"
                )
        else:
            false_alarms = a.successes
            if false_alarms:
                fa_table.add_row(
                    scen,
                    tname,
                    "false alarm (expected False, got True)",
                    str(false_alarms),
                    f"{false_alarms}/{a.trials}",
                )

    if len(fa_table.rows) > 0:
        console.print()
        console.print(fa_table)
    else:
        console.print("\n[green]No misses or false alarms across all aggregated tests.[/]")

    if diag:
        dtable = Table(
            title="Failure diagnosis (encode-time attr_x vs derive_x on watermarked text)"
        )
        dtable.add_column("Scenario", style="dim")
        dtable.add_column("Test")
        dtable.add_column("Kind", justify="left")
        dtable.add_column("Count", justify="right")
        kind_help = (
            "[dim]miss+x_drift: miss with differing x; miss+x_stable: miss with same x; "
            "fa+x_*: false alarm with x drift / stable.[/]"
        )
        for (scen, tname, kind), cnt in sorted(diag.items()):
            dtable.add_row(scen, tname, kind, str(cnt))
        console.print()
        console.print(dtable)
        console.print(kind_help)

    rt = Table(
        title="Runtime per step (mean = total / n; one sample per timed call per trial)"
    )
    rt.add_column("Scenario", style="dim")
    rt.add_column("Step")
    rt.add_column("n", justify="right")
    rt.add_column("Mean (s)", justify="right")
    rt.add_column("Total (s)", justify="right")

    for key in sorted(timing.keys(), key=lambda k: _timing_sort_key(k, scenario_order)):
        tm = timing[key]
        if tm.scenario == "_trial_":
            scen_disp = "—"
            step_disp = "trial wall (all cases in order)"
        else:
            scen_disp = tm.scenario
            step_disp = tm.step
        rt.add_row(
            scen_disp,
            step_disp,
            str(tm.samples),
            f"{tm.mean_seconds:.6f}",
            f"{tm.total_seconds:.3f}",
        )

    console.print()
    console.print(rt)

    all_full = all(a.successes == a.trials for a in agg.values())
    if all_full:
        console.print(Panel("[bold green]All tests passed on every trial.[/]", expand=False))
        return 0
    console.print(Panel("[bold red]Some tests did not pass on every trial — see table.[/]", expand=False))
    return 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=int, default=10, help="Trials (each trial runs every prompt case)")
    p.add_argument(
        "--code-length",
        type=int,
        default=300,
        metavar="N",
        help="PRC codeword length; passed to watermarking.set_prc_code_length",
    )
    p.add_argument("--modulus", type=int, default=1024, help="CPRF modulus for wm.setup")
    p.add_argument(
        "--reuse-key",
        action="store_true",
        help="Reuse one master key per case id across trials (default: new key each case per trial)",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="On detection failures, print longer baseline vs watermarked text excerpts",
    )
    p.add_argument(
        "--prompt-case",
        action="append",
        dest="prompt_cases",
        metavar="ID:PROMPT",
        help=(
            "Add a benchmark case (repeatable). Split on the first ':' only. "
            "If omitted, built-in default cases are used."
        ),
    )
    args = p.parse_args()
    if args.runs < 1:
        print("runs must be >= 1", file=sys.stderr)
        return 2
    if args.code_length < 1:
        print("code-length must be >= 1", file=sys.stderr)
        return 2

    if args.prompt_cases:
        try:
            prompt_cases = [parse_prompt_case(s) for s in args.prompt_cases]
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 2
    else:
        prompt_cases = list(DEFAULT_PROMPT_CASES)

    console = Console(highlight=False)
    return run_prompt_benchmark(
        prompt_cases=prompt_cases,
        runs=args.runs,
        modulus=args.modulus,
        code_length=args.code_length,
        fresh_key_per_trial=not args.reuse_key,
        console=console,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
