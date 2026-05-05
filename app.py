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
import randrecover
import watermarking as wm
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from closed_vocab import (
    CPRF_ATTR_DIM,
    VOCABULARY,
    active_labels_from_verify_x,
    f_for_required_keywords,
    pick_unrelated_keyword_for_policy,
)

# --- customize here ---
MODULUS = 1024
CODE_LENGTH = 300
# Hub id for the watermark causal LM; ``None`` keeps ``watermarking`` default / notebook ``set_llm_model_id``.
LLM_MODEL_ID: str | None = None
PROMPT = (
            "Explain why Tom Brady is the GOAT, and why Drake Maye is an upcoming star in football, in a brief paragraph for each, comparing their football legacies."
)


def _ber_percent(secret: list[int], recovered: list[int]) -> float:
    n = len(secret)
    if n == 0:
        return 0.0
    m = min(len(recovered), n)
    errs = sum(1 for i in range(m) if int(secret[i]) != int(recovered[i]))
    errs += abs(n - len(recovered))
    return 100.0 * errs / n


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


def _dot_mod(f: Sequence[int], x: Sequence[int], modulus: int) -> int:
    """⟨f,x⟩ mod modulus over the overlapping prefix of f and x (CPRF dimension)."""
    n = min(len(f), len(x))
    return sum(int(f[i]) * int(x[i]) for i in range(n)) % modulus


def _log_f_dot_x_terms(
    c: Console,
    title: str,
    f: Sequence[int],
    x: Sequence[int],
    modulus: int,
) -> None:
    """Log ⟨f,x⟩ mod m and per-keyword contributions (algebraic diagnostic only)."""
    dot = _dot_mod(f, x, modulus)
    c.print(
        f"  [bold]{title}[/]  ⟨f,x⟩ mod m = [cyan]{dot}[/]"
        "  [dim](see CPRF Δ·⟨f,x⟩ note below — ⟨f,x⟩ alone does not fix eval vs c_eval)[/]"
    )
    n_pref = min(len(VOCABULARY), len(f), len(x))
    for i in range(n_pref):
        if int(f[i]) == 0:
            continue
        lab = VOCABULARY[i]
        xi = int(x[i]) % modulus
        term = (int(f[i]) * int(x[i])) % modulus
        c.print(f"    [dim]prefix[/] {lab!r}: f[{i}]={int(f[i])}, x[{i}] mod m = {xi}, term mod m = {term}")
    if len(f) > n_pref and any(int(f[j]) != 0 for j in range(n_pref, len(f))):
        tail_dot = sum(int(f[j]) * int(x[j]) for j in range(n_pref, min(len(f), len(x)))) % modulus
        c.print(f"    [dim]non-prefix f·x mod m (tail):[/] {tail_dot}")


def _log_cprf_seed_agreement(
    c: Console,
    sk: object,
    dk: object,
    x: Sequence[int],
    title: str,
    *,
    expect_seed_match: bool,
) -> tuple[bool, bool]:
    """
    Ground truth for PRC: same PRF input iff sk.eval(x) == dk.c_eval(x).

    For reject policies we expect ``expect_seed_match=False``. If the hashes still match,
    Δ·⟨f,x⟩≡0 (mod m) for this key's random Δ (common when m is composite); detection may still pass.
    """
    em = sk.eval(list(x))
    ec = dk.c_eval(list(x))
    match = em == ec
    c.print(f"  [bold]{title}[/]  sk.eval(x) == dk.c_eval(x) → [cyan]{match}[/]")
    c.print(f"    [dim]master SHA256-input layer…[/] {em.hex()[:24]}…")
    c.print(f"    [dim]c_eval …[/]                {ec.hex()[:24]}…")
    if match == expect_seed_match:
        c.print("  [bold green]PASS[/]  CPRF output pair matches expectation for this policy")
        return True, match
    if not expect_seed_match and match:
        c.print(
            "  [yellow]WARN[/]  Constrained key still matches master on this x: need "
            "Δ·⟨f,x⟩≢0 (mod m) to separate seeds; ⟨f,x⟩≢0 alone is not enough on composite m (e.g. 1024)."
        )
        c.print(
            "  [yellow]→[/]  PRC ``detect`` may still return True (same seed as watermark); "
            "try again or use a prime modulus if you need ⟨f,x⟩≠0 to imply separation."
        )
        return True, match
    c.print(
        "  [bold red]FAIL[/]  Expected CPRF agreement but sk.eval(x) ≠ dk.c_eval(x) "
        f"(expected match={expect_seed_match})"
    )
    return False, match


def main() -> int:
    for name in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    c = Console(highlight=False)
    n_prefix = len(VOCABULARY)
    all_ok = True

    if LLM_MODEL_ID:
        wm.set_llm_model_id(LLM_MODEL_ID.strip())

    c.print(
        Panel.fit(
            f"[bold]modulus[/] {MODULUS}  ·  [bold]code_length[/] {CODE_LENGTH}\n"
            f"[bold]LLM[/] {wm.MODEL_ID}\n"
            f"[bold]vocab[/] |V|={n_prefix}  ·  [bold]NLI prefix[/] multi-label (cutoff={attr_x_nli.NLI_MULTI_LABEL_SCORE_CUTOFF:g})",
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
    _log_text(c, "Baseline (HF sample) text", baseline_text)
    _log_text(c, "Watermarked generated text", wm_text)
    c.print(
        f"  [dim]seconds_baseline_gen[/]=[cyan]{t_bl:.4f}[/]  "
        f"[dim]seconds_watermarked_gen[/]=[cyan]{t_wm:.4f}[/]"
    )
    c.print(f"  [dim]encode-time attr_x (len {len(x_encode)}), prefix:[/] {x_encode[:n_prefix]}")
    c.print(f"  [dim]PRC secret bits (len {len(secret)}):[/] {_bits_preview(secret)}")

    c.rule("3) Verify-time x and NLI-active labels (multi-label cutoff)", style="cyan")
    wm_nli: dict[str, float] = {}
    try:
        x_verify = attr_x_nli.derive_x(
            wm_text,
            sk.modulus,
            log_nli_scores=False,
            nli_scores_out=wm_nli,
        )
    except TypeError:
        # Older ``attr_x_nli`` (positional ``derive_x`` only); skip paired NLI log.
        x_verify = attr_x_nli.derive_x(wm_text, sk.modulus)
    bl_scores = out.get("nli_label_scores_baseline") or {}
    if (
        hasattr(attr_x_nli, "log_pair_zero_shot_scores")
        and wm_nli
        and bl_scores
    ):
        attr_x_nli.log_pair_zero_shot_scores(
            baseline=bl_scores,
            watermarked=wm_nli,
        )
    active_wm = active_labels_from_verify_x(x_verify, sk.modulus)
    # Encode-time x is derive_x(baseline); same NLI view as re-running on baseline_text.
    active_bl = active_labels_from_verify_x(x_encode, sk.modulus)
    c.print(f"  [bold]NLI-active (baseline text):[/] {active_bl or '(none)'}")
    c.print(f"  [bold]NLI-active (watermarked text):[/] {active_wm or '(none)'}")
    c.print(f"  [dim]derive_x(wm) prefix:[/] {x_verify[:n_prefix]}")
    x_perfect = list(x_encode) == list(x_verify)
    # Do not nest style tags (e.g. [cyan][bold]…[/][/]) — Rich parses that poorly and looks broken.
    _xtag = "[bold green]yes[/]" if x_perfect else "[bold red]no[/]"
    c.print(
        "  [dim]encode-time attr_x equals verify-time derive_x (full vector):[/] " + _xtag
    )

    c.rule("4) Issue keys (unconstrained, accept=all active, reject=unrelated)", style="cyan")
    t0 = time.perf_counter()
    dk_open = wm.issue_unconstrained(sk)
    if active_wm:
        dk_accept = wm.issue_keyword_policy(sk, list(active_wm))
        c.print(f"  accept policy: [cyan]{', '.join(sorted(active_wm))}[/]")
    else:
        dk_accept = wm.issue_keyword_policy(sk, [])
        c.print("  [yellow]no NLI-active label on WM text — accept key uses empty keyword list (f=0).[/]")
    unrelated = pick_unrelated_keyword_for_policy(x_verify, sk.modulus, set(active_wm))
    dk_reject = wm.issue_keyword_policy(sk, [unrelated])
    c.print(f"  reject policy single label: [cyan]{unrelated}[/]")
    t_keys = time.perf_counter() - t0
    c.print(f"  [dim]issue_keys_seconds[/]=[cyan]{t_keys:.5f}[/]")

    c.rule("4b) CPRF algebra + sk.eval(x) vs dk.c_eval(x) (same x = derive_x on WM text)", style="cyan")
    m_sz = sk.modulus
    f_zero = [0] * CPRF_ATTR_DIM
    f_accept_vec = f_for_required_keywords(active_wm if active_wm else [])
    f_reject_vec = f_for_required_keywords([unrelated])
    c.print(
        "  [dim]Go CPRF: constrained z1 = z0 − f·Δ ⇒ inner-product term k_c = k_m − Δ·⟨f,x⟩ (mod m). "
        "Hashes match iff Δ·⟨f,x⟩≡0 — not merely ⟨f,x⟩≡0. Composite m (e.g. 1024) allows ⟨f,x⟩≠0 with "
        "Δ·⟨f,x⟩≡0, so rejection must be checked via byte equality below.[/]"
    )
    _log_f_dot_x_terms(c, "Unconstrained (f = 0)", f_zero, x_verify, m_sz)
    ok_open, _ = _log_cprf_seed_agreement(
        c, sk, dk_open, x_verify, "Unconstrained key", expect_seed_match=True
    )
    all_ok &= ok_open

    lab = ",".join(sorted(active_wm)) if active_wm else "∅"
    _log_f_dot_x_terms(c, f"Accept policy (required: [{lab}])", f_accept_vec, x_verify, m_sz)
    ok_acc, _ = _log_cprf_seed_agreement(
        c, sk, dk_accept, x_verify, "Accept policy key", expect_seed_match=True
    )
    all_ok &= ok_acc

    _log_f_dot_x_terms(c, f"Reject policy (one-hot: {unrelated!r})", f_reject_vec, x_verify, m_sz)
    ok_rej, reject_seed_match = _log_cprf_seed_agreement(
        c, sk, dk_reject, x_verify, "Reject policy key", expect_seed_match=False
    )
    all_ok &= ok_rej

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
    if reject_seed_match:
        c.print(
            "  [dim]Reject key used the same CPRF output as master (§4b WARN); "
            "PRC detect=True is expected — not a detector bug.[/]"
        )
        all_ok &= _step(
            c,
            "detect(reject policy) — CPRF did not diverge; expect same as master path",
            r_ok,
            True,
        )
    else:
        all_ok &= _step(c, "detect(reject policy) should be False (CPRF seed differs)", r_ok, False)
        if r_ok:
            c.print(
                "  [yellow]PRC returned True despite differing CPRF seeds — possible LDPC statistical false positive; "
                "rerun or tighten code parameters.[/]"
            )

    det_total = t_m + t_u + t_a + t_r
    c.print(f"  [dim]total detection wall time:[/] [cyan]{det_total:.5f}[/]s")

    c.rule("6) Summary metrics (this run)", style="cyan")
    sum_table = Table(show_header=True, header_style="bold")
    sum_table.add_column("metric")
    sum_table.add_column("value", justify="right")
    sum_table.add_row("BER% (master path vs embedded)", f"{ber:.3f}")
    sum_table.add_row("x encode↔verify (full vector match)", "yes" if x_perfect else "no")
    sum_table.add_row("t_baseline_gen (s)", f"{t_bl:.4f}")
    sum_table.add_row("t_watermarked_gen (s)", f"{t_wm:.4f}")
    sum_table.add_row("t_issue_keys (s)", f"{t_keys:.5f}")
    sum_table.add_row("t_detect_total (s)", f"{det_total:.5f}")
    c.print(sum_table)

    c.rule("7) Negative control (wrong transcript)", style="cyan")
    wrong = randrecover.negative_control_transcript_like(
        wm_text,
        wm.TOKENIZER,
        wm.DEVICE,
        n_bits=wm.SECURITY_PARAM,
        model=wm.MODEL,
    )
    c.print(
        f"  [dim]decoy length: {len(wrong)} chars (watermarked ref: {len(wm_text)}), "
        f"bit horizon {wm.SECURITY_PARAM}[/]"
    )
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
