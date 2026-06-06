"""
Single-run walkthrough of the same protocol shape as ``benchmark_policy_detection`` (one prompt),
with verbose Rich logging: texts, label classifications, timings, recovered bits / BER, and
per-step pass/fail. Issues one constrained CPRF key per closed-vocabulary label and compares
``detect`` to the expectation that the label is (or is not) in the verify-time recovered
attribute set. Tweak the constants below; this file does not import the benchmark module.

Run: ``uv run python app.py``
"""

from __future__ import annotations

import logging
import time
from typing import List, Sequence

import text_attributes
import model
import randrecover
import prc
import watermarking as wm
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from text_attributes import (
    CPRF_ATTR_DIM,
    SCORE_CUTOFF,
    VOCABULARY,
    active_labels_from_attributes,
    f_for_required_keywords,
)


# --- customize here ---
MODULUS = 1024
CODE_LENGTH = 100
WM_BIT_REDUNDANCY = 3  # token-channel repeats per logical PRC bit; recovery = strict majority (tie → 0)
MODEL_ID: str | None = None  # None → ``model.DEFAULT_MODEL_ID``
PROMPT = (
            "Explain how software has transformed the practice of medicine."
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


def _dot_mod(f: Sequence[int], attributes: Sequence[int], modulus: int) -> int:
    """⟨f,attributes⟩ mod modulus over the overlapping prefix of f and attributes."""
    n = min(len(f), len(attributes))
    return sum(int(f[i]) * int(attributes[i]) for i in range(n)) % modulus


def _log_f_dot_attributes_terms(
    c: Console,
    title: str,
    f: Sequence[int],
    attributes: Sequence[int],
    modulus: int,
) -> None:
    """Log ⟨f,attributes⟩ mod m and per-keyword contributions (algebraic diagnostic only)."""
    dot = _dot_mod(f, attributes, modulus)
    c.print(
        f"  [bold]{title}[/]  ⟨f,attributes⟩ mod m = [cyan]{dot}[/]"
        "  [dim](see CPRF Δ·⟨f,attributes⟩ note below — inner product alone does not fix eval vs c_eval)[/]"
    )
    n_pref = min(len(VOCABULARY), len(f), len(attributes))
    for i in range(n_pref):
        if int(f[i]) == 0:
            continue
        lab = VOCABULARY[i]
        ai = int(attributes[i]) % modulus
        term = (int(f[i]) * int(attributes[i])) % modulus
        c.print(f"    [dim]prefix[/] {lab!r}: f[{i}]={int(f[i])}, attributes[{i}] mod m = {ai}, term mod m = {term}")
    if len(f) > n_pref and any(int(f[j]) != 0 for j in range(n_pref, len(f))):
        tail_dot = sum(int(f[j]) * int(attributes[j]) for j in range(n_pref, min(len(f), len(attributes)))) % modulus
        c.print(f"    [dim]non-prefix f·attributes mod m (tail):[/] {tail_dot}")


def _log_cprf_seed_agreement(
    c: Console,
    sk: object,
    dk: object,
    attributes: Sequence[int],
    title: str,
    *,
    expect_seed_match: bool,
) -> tuple[bool, bool]:
    """
    Ground truth for PRC: same PRF input iff sk.eval(attributes) == dk.c_eval(attributes).

    For reject policies we expect ``expect_seed_match=False``. If the hashes still match,
    Δ·⟨f,attributes⟩≡0 (mod m) for this key's random Δ (common when m is composite); detection may still pass.
    """
    em = sk.eval(list(attributes))
    ec = dk.c_eval(list(attributes))
    match = em == ec
    c.print(f"  [bold]{title}[/]  sk.eval(attributes) == dk.c_eval(attributes) → [cyan]{match}[/]")
    c.print(f"    [dim]master SHA256-input layer…[/] {em.hex()[:24]}…")
    c.print(f"    [dim]c_eval …[/]                {ec.hex()[:24]}…")
    if match == expect_seed_match:
        c.print("  [bold green]PASS[/]  CPRF output pair matches expectation for this policy")
        return True, match
    if not expect_seed_match and match:
        c.print(
            "  [yellow]WARN[/]  Constrained key still matches master on this x: need "
            "Δ·⟨f,attributes⟩≢0 (mod m) to separate seeds; ⟨f,attributes⟩≢0 alone is not enough on composite m (e.g. 1024)."
        )
        c.print(
            "  [yellow]→[/]  PRC ``detect`` may still return True (same seed as watermark); "
            "try again or use a prime modulus if you need ⟨f,attributes⟩≠0 to imply separation."
        )
        return True, match
    c.print(
        "  [bold red]FAIL[/]  Expected CPRF agreement but sk.eval(attributes) ≠ dk.c_eval(attributes) "
        f"(expected match={expect_seed_match})"
    )
    return False, match


def main() -> int:
    for name in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
    # In notebooks, logging is often preconfigured so ``basicConfig`` is ignored.
    # ``force=True`` guarantees INFO-level label score tables appear consistently.
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )

    c = Console(highlight=False)
    n_prefix = len(VOCABULARY)
    all_ok = True

    if MODEL_ID:
        model.configure(model_id=MODEL_ID)

    c.print(
        Panel.fit(
            f"[bold]modulus[/] {MODULUS}  ·  [bold]code_length[/] {CODE_LENGTH}  ·  "
            f"[bold]wm_bit_redundancy[/] {WM_BIT_REDUNDANCY}  (channel {CODE_LENGTH * WM_BIT_REDUNDANCY} bits)\n"
            f"[bold]LLM[/] {model.MODEL_ID}\n"
            f"[bold]sampling[/] temperature={model.SAMPLING['temperature']}  "
            f"top_p={model.SAMPLING['top_p']}  "
            f"top_k={model.SAMPLING['top_k']}\n"
            f"[bold]vocab[/] |V|={n_prefix}  ·  [bold]label prefix[/] multi-label (cutoff={SCORE_CUTOFF:g})",
            title="app.py protocol run",
        )
    )

    wm.SECURITY_PARAM = CODE_LENGTH
    prc.set_code_length(CODE_LENGTH)
    wm.WM_BIT_REDUNDANCY = WM_BIT_REDUNDANCY
    c.print(
        f"[dim]wm.SECURITY_PARAM =[/] {wm.SECURITY_PARAM}  "
        f"[dim]wm.WM_BIT_REDUNDANCY =[/] {wm.WM_BIT_REDUNDANCY}  "
        f"[dim]channel bits =[/] {wm.SECURITY_PARAM * wm.WM_BIT_REDUNDANCY}"
    )

    c.rule("1) CPRF setup", style="cyan")
    sk = wm.setup(MODULUS)
    c.print(f"  Master key OK (modulus={sk.modulus}).")

    c.rule("2) Generate (baseline → attributes → PRC → watermarked)", style="cyan")
    out = wm.generate(sk, PROMPT)
    baseline_text = out["baseline_text"]
    wm_text = out["generated_text_wm"]
    encode_attributes: List[int] = list(out["attributes"])
    secret: List[int] = list(out["prc_secret_bits"])
    t_bl = float(out["seconds_baseline_gen"])
    t_wm = float(out["seconds_watermarked_gen"])
    _log_text(c, "Baseline (HF sample) text", baseline_text)
    _log_text(c, "Watermarked generated text", wm_text)
    c.print(
        f"  [dim]seconds_baseline_gen[/]=[cyan]{t_bl:.4f}[/]  "
        f"[dim]seconds_watermarked_gen[/]=[cyan]{t_wm:.4f}[/]"
    )
    c.print(f"  [dim]encode-time attributes (len {len(encode_attributes)}), prefix:[/] {encode_attributes[:n_prefix]}")
    c.print(f"  [dim]PRC secret bits (len {len(secret)}):[/] {_bits_preview(secret)}")

    c.rule("3) Verify-time attributes and active labels (multi-label cutoff)", style="cyan")
    wm_scores: dict[str, float] = {}
    verify_attributes = text_attributes.derive_attributes(
        wm_text,
        sk.modulus,
        log_scores=False,
        scores_out=wm_scores,
    )
    bl_scores = out.get("label_scores_baseline") or {}
    if wm_scores and bl_scores:
        text_attributes.log_paired_label_scores(
            baseline=bl_scores,
            watermarked=wm_scores,
        )
    active_wm = active_labels_from_attributes(verify_attributes, sk.modulus)
    active_bl = active_labels_from_attributes(encode_attributes, sk.modulus)
    c.print(f"  [bold]Active labels (baseline text):[/] {active_bl or '(none)'}")
    c.print(f"  [bold]Active labels (watermarked text):[/] {active_wm or '(none)'}")
    c.print(f"  [dim]verify-time attributes prefix:[/] {verify_attributes[:n_prefix]}")
    attributes_match = list(encode_attributes) == list(verify_attributes)
    _xtag = "[bold green]yes[/]" if attributes_match else "[bold red]no[/]"
    c.print(
        "  [dim]encode-time attributes equal verify-time attributes (full vector):[/] " + _xtag
    )

    c.rule("4) Issue keys (unconstrained + one constrained key per vocab label)", style="cyan")
    t0 = time.perf_counter()
    dk_open = wm.issue(sk, [])
    active_set = set(active_wm)
    dk_by_word = {w: wm.issue(sk, [w]) for w in VOCABULARY}
    c.print(
        f"  [bold]Verify-time active labels (recovered attribute):[/] "
        f"{sorted(active_wm) if active_wm else '(none)'}"
    )
    c.print(
        "  Each other key requires exactly one label; detection should succeed when that "
        "label appears in the recovered attribute set above."
    )
    t_keys = time.perf_counter() - t0
    c.print(f"  [dim]issue_keys_seconds[/]=[cyan]{t_keys:.5f}[/]  ([cyan]{1 + len(VOCABULARY)}[/] keys)")

    c.rule("4b) CPRF algebra + sk.eval(attributes) vs dk.c_eval(attributes) (verify-time text)", style="cyan")
    m_sz = sk.modulus
    f_zero = [0] * CPRF_ATTR_DIM
    c.print(
        "  [dim]Go CPRF: constrained z1 = z0 − f·Δ ⇒ inner-product term k_c = k_m − Δ·⟨f,attributes⟩ (mod m). "
        "Hashes match iff Δ·⟨f,attributes⟩≡0 — not merely ⟨f,attributes⟩≡0. Composite m (e.g. 1024) allows ⟨f,attributes⟩≠0 with "
        "Δ·⟨f,attributes⟩≡0, so match expectation uses byte equality below. For one-hot f on label w, "
        "⟨f,attributes⟩≡0 when w is active (prefix coordinate ≡ 0 mod m).[/]"
    )
    _log_f_dot_attributes_terms(c, "Unconstrained (f = 0)", f_zero, verify_attributes, m_sz)
    ok_open, _ = _log_cprf_seed_agreement(
        c, sk, dk_open, verify_attributes, "Unconstrained key", expect_seed_match=True
    )
    all_ok &= ok_open

    policy_table = Table(show_header=True, header_style="bold", title="Per-label constrained keys (verify attributes)")
    policy_table.add_column("required label")
    policy_table.add_column("in recovered attr", justify="center")
    policy_table.add_column("expect seed match", justify="center")
    policy_table.add_column("sk.eval==dk.c_eval", justify="center")
    seed_match_by_word: dict[str, bool] = {}
    for w in VOCABULARY:
        f_w = f_for_required_keywords([w])
        expect_match = w in active_set
        _log_f_dot_attributes_terms(c, f"Policy requires [{w!r}]", f_w, verify_attributes, m_sz)
        ok_w, sm = _log_cprf_seed_agreement(
            c,
            sk,
            dk_by_word[w],
            verify_attributes,
            f"Constrained key (required: {w!r})",
            expect_seed_match=expect_match,
        )
        all_ok &= ok_w
        seed_match_by_word[w] = sm
        policy_table.add_row(
            w,
            "yes" if w in active_set else "no",
            "yes" if expect_match else "no",
            "yes" if sm else "no",
        )
    c.print(policy_table)

    c.rule("5) Detection (master + unconstrained + one detect per vocab label)", style="cyan")

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

    det_total = t_m + t_u
    det_rows: list[tuple[str, str, str, str, str]] = []
    for w in VOCABULARY:
        expect_ok = w in active_set
        t0 = time.perf_counter()
        w_ok, w_bits = wm.detect(dk_by_word[w], wm_text)
        t_w = time.perf_counter() - t0
        det_total += t_w
        c.print(f"  [dim]detect(required={w!r})[/] {t_w:.5f}s")
        c.print(f"    recovered bits: {_bits_preview(w_bits)}")
        step_label = (
            f"detect(required={w!r}) — expect {'True' if expect_ok else 'False'} "
            f"(label {'in' if expect_ok else 'not in'} recovered attribute)"
        )
        all_ok &= _step(c, step_label, w_ok, expect_ok)
        if w_ok != expect_ok:
            sm = seed_match_by_word[w]
            c.print(
                "  [yellow]Outcome differs from attribute-based expectation[/] "
                f"(sk.eval==dk.c_eval was {sm} for this key; composite modulus / LDPC may apply — see step 4b)."
            )
        det_rows.append(
            (
                w,
                "yes" if w in active_set else "no",
                "True" if expect_ok else "False",
                "True" if w_ok else "False",
                "[green]PASS[/]" if bool(w_ok) == bool(expect_ok) else "[red]FAIL[/]",
            )
        )

    det_table = Table(show_header=True, header_style="bold", title="Per-label detection vs recovered attribute")
    det_table.add_column("required label")
    det_table.add_column("in recovered attr", justify="center")
    det_table.add_column("expect detect", justify="center")
    det_table.add_column("got detect", justify="center")
    det_table.add_column("check", justify="center")
    for row in det_rows:
        det_table.add_row(*row)
    c.print(det_table)
    c.print(f"  [dim]total detection wall time:[/] [cyan]{det_total:.5f}[/]s")

    c.rule("6) Summary metrics (this run)", style="cyan")
    sum_table = Table(show_header=True, header_style="bold")
    sum_table.add_column("metric")
    sum_table.add_column("value", justify="right")
    sum_table.add_row("BER% (master path vs embedded)", f"{ber:.3f}")
    sum_table.add_row("attributes encode↔verify (full vector match)", "yes" if attributes_match else "no")
    sum_table.add_row("t_baseline_gen (s)", f"{t_bl:.4f}")
    sum_table.add_row("t_watermarked_gen (s)", f"{t_wm:.4f}")
    sum_table.add_row("t_issue_keys (s)", f"{t_keys:.5f}")
    sum_table.add_row("t_detect_total (s)", f"{det_total:.5f}")
    c.print(sum_table)

    c.rule("7) Negative control (wrong transcript)", style="cyan")
    m, tok, device = model.load()
    wrong = randrecover.negative_control_transcript_like(
        wm_text,
        tok,
        device,
        n_bits=wm.SECURITY_PARAM * wm.WM_BIT_REDUNDANCY,
        model=m,
    )
    c.print(
        f"  [dim]decoy length: {len(wrong)} chars (watermarked ref: {len(wm_text)}), "
        f"channel bit horizon {wm.SECURITY_PARAM * wm.WM_BIT_REDUNDANCY}[/]"
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
