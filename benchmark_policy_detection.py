"""
Watermarking benchmark: multiple prompts, repeated runs, one summary table per prompt.

For each prompt and run: ``generate`` → issue unconstrained, matching-policy, and unrelated
keys → ``master_detect`` plus three ``detect`` calls. Records success counts, BER (recovered vs
embedded PRC bits from ``generate``), timings (baseline vs watermarked generation from
``watermarking.generate``), and agreement of encode-time ``attr_x`` prefix vs verify-time
``derive_x`` on the watermarked text.

CLI: ``--code-length``, ``--runs``, ``--modulus``, ``--reuse-key``, and repeatable
``--prompt-case id:prompt`` (split on first ``:`` only). If no cases are given, two defaults are used.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Sequence

from rich.console import Console
from rich.table import Table

DEFAULT_PROMPT_CASES: list[tuple[str, str]] = [
    (
        "medical_procedure",
        "In three short paragraphs, describe a common outpatient medical procedure "
        "and what the patient should expect before and after the visit.",
    ),
    ("med_school", "Describe the intersection of medicine and education."),
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


def _prefix_match_rate(x_enc: Sequence[int], x_ver: Sequence[int], n_prefix: int) -> float:
    if n_prefix <= 0:
        return 1.0
    m = min(len(x_enc), len(x_ver), n_prefix)
    if m == 0:
        return 0.0
    return sum(1 for i in range(m) if int(x_enc[i]) == int(x_ver[i])) / float(n_prefix)


@dataclass
class PromptRollup:
    """Accumulated stats for one prompt id across runs."""

    runs: int = 0
    master_ok: int = 0
    open_ok: int = 0
    accept_ok: int = 0
    reject_correct: int = 0
    ber_sum: float = 0.0
    x_prefix_match_sum: float = 0.0
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
        x_prefix_match: float,
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
        self.x_prefix_match_sum += x_prefix_match
        self.t_baseline_sum += t_baseline
        self.t_wm_sum += t_wm
        self.t_issue_sum += t_issue
        self.t_master_sum += t_master
        self.t_open_sum += t_open
        self.t_accept_sum += t_accept
        self.t_reject_sum += t_reject


def run_benchmark(
    *,
    prompt_cases: Sequence[tuple[str, str]],
    runs: int,
    modulus: int,
    code_length: int,
    fresh_key_per_trial: bool,
    console: Console,
) -> int:
    import watermarking as wm
    import attr_x_nli
    from closed_vocab import (
        VOCABULARY,
        active_labels_from_verify_x,
        pick_unrelated_keyword_for_policy,
    )

    wm.set_prc_code_length(code_length)
    n_prefix = len(VOCABULARY)
    roll: dict[str, PromptRollup] = {sid: PromptRollup() for sid, _ in prompt_cases}
    sk_shared: dict[str, object] = {}

    console.print(
        f"code_length={wm.SECURITY_PARAM}  modulus={modulus}  runs={runs}  "
        f"keys={'fresh per run' if fresh_key_per_trial else 'reuse per prompt id'}"
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
            x_match = _prefix_match_rate(x_gen, x_verify, n_prefix)

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
                x_prefix_match=x_match,
                t_baseline=t_bl,
                t_wm=t_wm,
                t_issue=t_issue,
                t_master=t_m,
                t_open=t_u,
                t_accept=t_a,
                t_reject=t_r,
            )

    table = Table(title="Per-prompt averages over runs")
    table.add_column("prompt_id", style="dim")
    table.add_column("runs", justify="right")
    table.add_column("master", justify="right")
    table.add_column("open", justify="right")
    table.add_column("accept", justify="right")
    table.add_column("rej_ok", justify="right")
    table.add_column("BER%", justify="right")
    table.add_column("x≡%", justify="right")
    table.add_column("t_bl", justify="right")
    table.add_column("t_wm", justify="right")
    table.add_column("t_key", justify="right")
    table.add_column("t_det", justify="right")

    all_ok = True
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
            f"{100.0 * r.x_prefix_match_sum / n:.1f}",
            f"{r.t_baseline_sum / n:.3f}",
            f"{r.t_wm_sum / n:.3f}",
            f"{r.t_issue_sum / n:.4f}",
            f"{det_avg:.4f}",
        )
        all_ok = all_ok and (
            r.master_ok == n and r.open_ok == n and r.accept_ok == n and r.reject_correct == n
        )

    console.print()
    console.print(table)
    console.print(
        "[dim]master/open/accept = successes / runs; rej_ok = correct reject (expect False). "
        "BER% = mean bit error vs embedded PRC stream (master path recovery). "
        "x≡% = mean fraction of prefix coords where encode-time attr_x matches verify-time derive_x. "
        "t_bl / t_wm = mean baseline vs watermarked gen seconds from generate(). "
        "t_key = mean issue time; t_det = mean total of four detection calls.[/]"
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

    console = Console(highlight=False)
    return run_benchmark(
        prompt_cases=cases,
        runs=args.runs,
        modulus=args.modulus,
        code_length=args.code_length,
        fresh_key_per_trial=not args.reuse_key,
        console=console,
    )


if __name__ == "__main__":
    sys.exit(main())
