"""
Single-run walkthrough of the same protocol shape as ``benchmark_policy_detection`` (one prompt),
with verbose Rich logging: texts, NLI-derived classifications, timings, recovered bits / BER, and
per-step pass/fail. Tweak the constants below; this file does not import the benchmark module.

Run: ``uv run python app.py``
"""

from __future__ import annotations

import logging
import time
from typing import List, Sequence

import attr_x_nli
import watermarking as wm
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from closed_vocab import VOCABULARY, active_labels_from_verify_x, pick_unrelated_keyword_for_policy

# --- customize here ---
MODULUS = 1024
CODE_LENGTH = 300
PROMPT = (
    "In three short paragraphs, describe a common outpatient medical procedure "
    "and what the patient should expect before and after the visit."
)


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


def _bits_preview(bits: List[int], max_len: int = 64) -> str:
    s = "".join(str(int(b) & 1) for b in bits[:max_len])
    if len(bits) > max_len:
        s += f"... ({len(bits)} total)"
    return s


def _log_text(c: Console, title: str, body: str, *, max_chars: int = 8000) -> None:
    c.rule(title, style="dim")
    if len(body) > max_chars:
        c.print(f"[dim](truncated to {max_chars} chars)[/]\n{body[:max_chars]}")
    else:
        c.print(body)


def _step(c: Console, name: str, got: object, expected: bool) -> bool:
    ok = bool(got) == bool(expected)
    tag = "[bold green]PASS[/]" if ok else "[bold red]FAIL[/]"
    c.print(f"  {tag}  {name}  [dim](got {bool(got)}, expect {expected})[/]")
    return ok


def main() -> int:
    for name in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    c = Console(highlight=False)
    n_prefix = len(VOCABULARY)
    all_ok = True

    c.print(
        Panel.fit(
            f"[bold]modulus[/] {MODULUS}  ·  [bold]code_length[/] {CODE_LENGTH}\n"
            f"[bold]vocab[/] |V|={n_prefix}  ·  [bold]NLI bar[/] {attr_x_nli.NLI_LABEL_ACTIVE_MIN_SCORE}",
            title="app.py protocol run",
        )
    )

    wm.set_prc_code_length(CODE_LENGTH)
    c.print(f"[dim]wm.SECURITY_PARAM =[/] {wm.SECURITY_PARAM}")

    c.rule("1) CPRF setup", style="cyan")
    sk = wm.setup(MODULUS)
    c.print(f"  Master key OK (modulus={sk.modulus}).")

    c.rule("2) Generate (baseline → attr_x → PRC → watermarked)", style="cyan")
    out = wm.generate(sk, PROMPT)
    baseline_text = out["baseline_text"]
    wm_text = out["generated_text_wm"]
    x_encode: List[int] = list(out["attr_x"])
    secret: List[int] = list(out["prc_secret_bits"])
    t_bl = float(out["seconds_baseline_gen"])
    t_wm = float(out["seconds_watermarked_gen"])
    _log_text(c, "Baseline (greedy) text", baseline_text)
    _log_text(c, "Watermarked generated text", wm_text)
    c.print(
        f"  [dim]seconds_baseline_gen[/]=[cyan]{t_bl:.4f}[/]  "
        f"[dim]seconds_watermarked_gen[/]=[cyan]{t_wm:.4f}[/]"
    )
    c.print(f"  [dim]encode-time attr_x (len {len(x_encode)}), prefix:[/] {x_encode[:n_prefix]}")
    c.print(f"  [dim]PRC secret bits (len {len(secret)}):[/] {_bits_preview(secret)}")

    c.rule("3) Verify-time x and NLI-active labels", style="cyan")
    x_verify = attr_x_nli.derive_x(wm_text, sk.modulus)
    active_wm = active_labels_from_verify_x(x_verify, sk.modulus)
    # Encode-time x is derive_x(baseline); same NLI view as re-running on baseline_text.
    active_bl = active_labels_from_verify_x(x_encode, sk.modulus)
    c.print(f"  [bold]NLI-active (baseline text):[/] {active_bl or '(none)'}")
    c.print(f"  [bold]NLI-active (watermarked text):[/] {active_wm or '(none)'}")
    c.print(f"  [dim]derive_x(wm) prefix:[/] {x_verify[:n_prefix]}")
    x_match = _prefix_match_rate(x_encode, x_verify, n_prefix)
    c.print(
        f"  [dim]encode vs verify prefix coordinate match rate:[/] [cyan]{100.0 * x_match:.1f}%[/]"
    )

    c.rule("4) Issue keys (unconstrained, accept=all active, reject=unrelated)", style="cyan")
    t0 = time.perf_counter()
    dk_open = wm.issue_unconstrained(sk)
    if active_wm:
        dk_accept = wm.issue_keyword_policy(sk, list(active_wm))
        c.print(f"  accept policy: [cyan]{', '.join(sorted(active_wm))}[/]")
    else:
        dk_accept = wm.issue_keyword_policy(sk, [])
        c.print("  [yellow]no NLI-active labels on WM text — accept key uses empty keyword list (f=0).[/]")
    unrelated = pick_unrelated_keyword_for_policy(x_verify, sk.modulus, set(active_wm))
    dk_reject = wm.issue_keyword_policy(sk, [unrelated])
    c.print(f"  reject policy single label: [cyan]{unrelated}[/]")
    t_keys = time.perf_counter() - t0
    c.print(f"  [dim]issue_keys_seconds[/]=[cyan]{t_keys:.5f}[/]")

    c.rule("5) Detection (4 calls)", style="cyan")

    t0 = time.perf_counter()
    m_ok, m_bits = wm.master_detect(sk, wm_text)
    t_m = time.perf_counter() - t0
    ber = _ber_percent(secret, m_bits)
    c.print(f"  [dim]master_detect[/] {t_m:.5f}s  BER={ber:.2f}%")
    c.print(f"    recovered bits: {_bits_preview(m_bits)}")
    all_ok &= _step(c, "master_detect (good transcript)", m_ok, True)

    t0 = time.perf_counter()
    u_ok, u_bits = wm.detect(dk_open, wm_text)
    t_u = time.perf_counter() - t0
    c.print(f"  [dim]detect(unconstrained)[/] {t_u:.5f}s")
    c.print(f"    recovered bits: {_bits_preview(u_bits)}")
    all_ok &= _step(c, "detect(unconstrained)", u_ok, True)

    t0 = time.perf_counter()
    a_ok, a_bits = wm.detect(dk_accept, wm_text)
    t_a = time.perf_counter() - t0
    c.print(f"  [dim]detect(accept policy)[/] {t_a:.5f}s")
    c.print(f"    recovered bits: {_bits_preview(a_bits)}")
    all_ok &= _step(c, "detect(accept policy)", a_ok, True)

    t0 = time.perf_counter()
    r_ok, r_bits = wm.detect(dk_reject, wm_text)
    t_r = time.perf_counter() - t0
    c.print(f"  [dim]detect(reject / unrelated policy)[/] {t_r:.5f}s")
    c.print(f"    recovered bits: {_bits_preview(r_bits)}")
    all_ok &= _step(c, "detect(reject policy) should be False", r_ok, False)

    det_total = t_m + t_u + t_a + t_r
    c.print(f"  [dim]total detection wall time:[/] [cyan]{det_total:.5f}[/]s")

    c.rule("6) Summary metrics (this run)", style="cyan")
    sum_table = Table(show_header=True, header_style="bold")
    sum_table.add_column("metric")
    sum_table.add_column("value", justify="right")
    sum_table.add_row("BER% (master path vs embedded)", f"{ber:.3f}")
    sum_table.add_row("x prefix match encode↔verify %", f"{100.0 * x_match:.2f}")
    sum_table.add_row("t_baseline_gen (s)", f"{t_bl:.4f}")
    sum_table.add_row("t_watermarked_gen (s)", f"{t_wm:.4f}")
    sum_table.add_row("t_issue_keys (s)", f"{t_keys:.5f}")
    sum_table.add_row("t_detect_total (s)", f"{det_total:.5f}")
    c.print(sum_table)

    c.rule("7) Negative control (wrong transcript)", style="cyan")
    wrong = "This is unrelated text used only as a negative control."
    _log_text(c, "Wrong transcript", wrong, max_chars=400)
    w_ok, w_bits = wm.master_detect(sk, wrong)
    c.print(f"  [dim]master_detect[/] → {bool(w_ok)}  recovered: {_bits_preview(w_bits)}")
    all_ok &= _step(c, "master_detect(wrong transcript) should be False", w_ok, False)

    c.print()
    if all_ok:
        c.print(Panel("[bold green]All protocol checks passed.[/]", expand=False))
        return 0
    c.print(Panel("[bold red]One or more checks failed — see FAIL lines above.[/]", expand=False))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
