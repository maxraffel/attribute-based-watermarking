"""
Single-run walkthrough of the same protocol shape as ``benchmark_policy_detection`` (one prompt),
with Rich summaries: config, generated text, label scores, CPRF/policy checks, detection, and
timings. Issues one constrained CPRF key per closed-vocabulary label and compares ``detect`` to
the verify-time recovered attribute set.

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
    _active_mask_from_scores,
    active_labels_from_attributes,
    derive_attributes,
    f_for_required_keywords,
)

# --- customize here ---
MODULUS = 1024
CODE_LENGTH = 100
WM_BIT_REDUNDANCY = 3  # token-channel repeats per PRC bit; recovery = strict majority (tie → 0)
# ``static`` = original seeded partitions; ``balanced`` = per-step softmax-balanced masks
PARTITION_MODE = "balanced"
# ``depth`` = R interleaved codeword passes; ``block`` = each bit repeated R times contiguously
REDUNDANCY_LAYOUT = "depth"
MODEL_ID: str | None = None  # None → ``model.DEFAULT_MODEL_ID``
PROMPT = (
    # '''Explain the economic nuance and impact of Drake Maye during his college football career at North Carolina.'''
    # "Write a program to reverse a linked list then provide an analysis of its time complexity."
    "Explain how software has transformed the art world."
)

TEXT_EXCERPT_CHARS = 400


def _ber_percent(secret: list[int], recovered: list[int]) -> float:
    n = len(secret)
    if n == 0:
        return 0.0
    m = min(len(recovered), n)
    errs = sum(1 for i in range(m) if int(secret[i]) != int(recovered[i]))
    errs += abs(n - len(recovered))
    return 100.0 * errs / n


def _excerpt(text: str, max_chars: int = TEXT_EXCERPT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _print_text(c: Console, title: str, body: str) -> None:
    c.print(f"  [bold]{title}[/] [dim]({len(body)} chars)[/]")
    c.print(f"  {body}")


def _pass_cell(ok: bool) -> str:
    return "[bold green]PASS[/]" if ok else "[bold red]FAIL[/]"


def _dot_mod(f: Sequence[int], attributes: Sequence[int], modulus: int) -> int:
    n = min(len(f), len(attributes))
    return sum(int(f[i]) * int(attributes[i]) for i in range(n)) % modulus


def _cprf_seeds_match(sk: object, dk: object, attributes: Sequence[int]) -> bool:
    return sk.eval(list(attributes)) == dk.c_eval(list(attributes))


def _cprf_agreement_table(
    sk: object,
    dk_open: object,
    dk_by_word: dict[str, object],
    verify_attributes: Sequence[int],
    active_set: set[str],
) -> tuple[Table, bool]:
    table = Table(title="CPRF eval vs c_eval on verify-time attributes", header_style="bold")
    table.add_column("policy")
    table.add_column("label active", justify="center")
    table.add_column("⟨f,a⟩ mod m", justify="right")
    table.add_column("expect match", justify="center")
    table.add_column("seed match", justify="center")
    table.add_column("check", justify="center")

    all_ok = True
    f_zero = [0] * CPRF_ATTR_DIM
    sm_open = _cprf_seeds_match(sk, dk_open, verify_attributes)
    ok_open = sm_open is True
    all_ok &= ok_open
    table.add_row(
        "(unconstrained)",
        "—",
        str(_dot_mod(f_zero, verify_attributes, sk.modulus)),
        "yes",
        "yes" if sm_open else "no",
        _pass_cell(ok_open),
    )

    for w in VOCABULARY:
        f_w = f_for_required_keywords([w])
        expect_match = w in active_set
        sm = _cprf_seeds_match(sk, dk_by_word[w], verify_attributes)
        ok_w = sm == expect_match
        all_ok &= ok_w
        table.add_row(
            w,
            "yes" if w in active_set else "no",
            str(_dot_mod(f_w, verify_attributes, sk.modulus)),
            "yes" if expect_match else "no",
            "yes" if sm else "no",
            _pass_cell(ok_w),
        )
    return table, all_ok


def _label_scores_table(
    baseline: dict[str, float],
    watermarked: dict[str, float],
    *,
    cutoff: float,
) -> Table:
    b_act = _active_mask_from_scores(baseline, cutoff)
    w_act = _active_mask_from_scores(watermarked, cutoff)
    table = Table(
        title=f"Label scores (baseline vs watermarked; active if score >= {cutoff:g})",
        show_header=True,
        header_style="bold",
    )
    table.add_column("label", style="dim")
    table.add_column("baseline", justify="right")
    table.add_column("wm", justify="right")
    table.add_column("delta", justify="right")
    table.add_column("b", justify="center")
    table.add_column("w", justify="center")
    for i, lab in enumerate(VOCABULARY):
        sb = float(baseline.get(lab, 0.0))
        sw = float(watermarked.get(lab, 0.0))
        b_y = "y" if i < len(b_act) and b_act[i] else "n"
        w_y = "y" if i < len(w_act) and w_act[i] else "n"
        table.add_row(lab, f"{sb:.4f}", f"{sw:.4f}", f"{sw - sb:+.4f}", b_y, w_y)
    return table


def main() -> int:
    for name in ("httpx", "httpcore", "huggingface_hub", "urllib3", "text_attributes"):
        logging.getLogger(name).setLevel(logging.WARNING)

    c = Console(highlight=False)
    all_ok = True

    if MODEL_ID:
        model.configure(model_id=MODEL_ID)

    c.print(
        Panel.fit(
            f"[bold]modulus[/] {MODULUS}  ·  [bold]code_length[/] {CODE_LENGTH}  ·  "
            f"[bold]wm_bit_redundancy[/] {WM_BIT_REDUNDANCY}  "
            f"(channel {CODE_LENGTH * WM_BIT_REDUNDANCY} bits)\n"
            f"[bold]partition_mode[/] {PARTITION_MODE}  ·  "
            f"[bold]redundancy_layout[/] {REDUNDANCY_LAYOUT}\n"
            f"[bold]LLM[/] {model.MODEL_ID}\n"
            f"[bold]sampling[/] temperature={model.SAMPLING['temperature']}  "
            f"top_p={model.SAMPLING['top_p']}  top_k={model.SAMPLING['top_k']}\n"
            f"[bold]classifier[/] {text_attributes.get_scorer().model_id}\n"
            f"[bold]vocab[/] |V|={len(VOCABULARY)}  ·  [bold]score cutoff[/] {SCORE_CUTOFF:g}",
            title="Protocol run",
        )
    )

    wm.SECURITY_PARAM = CODE_LENGTH
    prc.set_code_length(CODE_LENGTH)
    wm.WM_BIT_REDUNDANCY = WM_BIT_REDUNDANCY
    wm.set_partition_mode(PARTITION_MODE)
    wm.set_redundancy_layout(REDUNDANCY_LAYOUT)

    c.rule("1) Setup & generate", style="cyan")
    sk = wm.setup(MODULUS)
    c.print(f"  CPRF master key OK  [dim](modulus={sk.modulus}, dim={CPRF_ATTR_DIM})[/]")
    c.print(f"  [dim]prompt:[/] {_excerpt(PROMPT, 120)}")

    out = wm.generate(sk, PROMPT)
    baseline_text = out["baseline_text"]
    wm_text = out["generated_text_wm"]
    encode_attributes: List[int] = list(out["attributes"])
    secret: List[int] = list(out["prc_secret_bits"])
    sacrificed_bits = int(out.get("sacrificed_bits", 0))
    natural_partition_choices = int(out.get("natural_partition_choices", 0))
    recovery_stalls = int(out.get("recovery_stalls", 0))
    retok_replacements = int(out.get("retok_replacements", 0))
    recovery_ids_aligned = bool(out.get("recovery_ids_aligned", False))
    t_bl = float(out["seconds_baseline_gen"])
    t_wm = float(out["seconds_watermarked_gen"])

    _print_text(c, "Baseline text", baseline_text)
    _print_text(c, "Watermarked text", wm_text)
    c.print(
        f"  [dim]generate[/] baseline={t_bl:.3f}s  watermarked={t_wm:.3f}s  "
        f"PRC payload={len(secret)} logical bits  "
        f"sacrificed_bits={sacrificed_bits}  natural_partition={natural_partition_choices}  "
        f"recovery_stalls={recovery_stalls}  retok_replacements={retok_replacements}  "
        f"recovery_ids_aligned={'yes' if recovery_ids_aligned else 'no'}"
    )

    c.rule("2) Label classification", style="cyan")
    wm_scores: dict[str, float] = {}
    verify_attributes = derive_attributes(
        wm_text, sk.modulus, log_scores=False, scores_out=wm_scores
    )
    bl_scores = out.get("label_scores_baseline") or {}
    if bl_scores and wm_scores:
        c.print(_label_scores_table(bl_scores, wm_scores, cutoff=SCORE_CUTOFF))

    active_bl = active_labels_from_attributes(encode_attributes, sk.modulus)
    active_wm = active_labels_from_attributes(verify_attributes, sk.modulus)
    attributes_match = list(encode_attributes) == list(verify_attributes)
    c.print(
        f"  [bold]Active labels[/] baseline={active_bl or '(none)'}  "
        f"watermarked={active_wm or '(none)'}  "
        f"encode/verify match={'[green]yes[/]' if attributes_match else '[red]no[/]'}"
    )

    c.rule("3) CPRF keys & seed agreement", style="cyan")
    t0 = time.perf_counter()
    dk_open = wm.issue(sk, [])
    dk_by_word = {w: wm.issue(sk, [w]) for w in VOCABULARY}
    t_keys = time.perf_counter() - t0
    active_set = set(active_wm)

    cprf_table, cprf_ok = _cprf_agreement_table(
        sk, dk_open, dk_by_word, verify_attributes, active_set
    )
    all_ok &= cprf_ok
    n_keys = 1 + len(VOCABULARY)
    if cprf_ok:
        c.print(
            f"  CPRF keys & seed agreement {_pass_cell(True)}  "
            f"[dim]({n_keys} keys, {t_keys:.3f}s)[/]"
        )
    else:
        c.print(cprf_table)
        c.print(f"  [dim]issue_keys[/] {t_keys:.3f}s  ({n_keys} keys)")

    c.rule("4) Detection", style="cyan")
    det_table = Table(title="Watermark detect vs attribute expectation", header_style="bold")
    det_table.add_column("key / path")
    det_table.add_column("label active", justify="center")
    det_table.add_column("expect", justify="center")
    det_table.add_column("got", justify="center")
    det_table.add_column("BER%", justify="right")
    det_table.add_column("sacrificed", justify="right")
    det_table.add_column("natural_part", justify="right")
    det_table.add_column("time", justify="right")
    det_table.add_column("check", justify="center")

    t0 = time.perf_counter()
    recovered_wm = wm.recover_channel_bits(wm_text, generation_out=out)
    t_recover = time.perf_counter() - t0

    t0 = time.perf_counter()
    m_ok, m_bits = wm.master_detect(sk, wm_text, recovered_bits=recovered_wm)
    t_m = time.perf_counter() - t0
    ber = _ber_percent(secret, m_bits)
    all_ok &= bool(m_ok)
    det_table.add_row(
        "master (good transcript)",
        "—",
        "True",
        "True" if m_ok else "False",
        f"{ber:.2f}",
        str(sacrificed_bits),
        str(natural_partition_choices),
        f"{t_recover + t_m:.3f}s",
        _pass_cell(m_ok),
    )

    t0 = time.perf_counter()
    u_ok, _ = wm.detect(dk_open, wm_text, recovered_bits=recovered_wm)
    t_u = time.perf_counter() - t0
    all_ok &= u_ok is True
    det_table.add_row(
        "unconstrained",
        "—",
        "True",
        "True" if u_ok else "False",
        "—",
        "—",
        "—",
        f"{t_u:.3f}s",
        _pass_cell(u_ok),
    )

    det_total = t_recover + t_m + t_u
    for w in VOCABULARY:
        expect_ok = w in active_set
        t0 = time.perf_counter()
        w_ok, _ = wm.detect(dk_by_word[w], wm_text, recovered_bits=recovered_wm)
        t_w = time.perf_counter() - t0
        det_total += t_w
        all_ok &= bool(w_ok) == bool(expect_ok)
        det_table.add_row(
            f"required={w!r}",
            "yes" if w in active_set else "no",
            "True" if expect_ok else "False",
            "True" if w_ok else "False",
            "—",
            "—",
            "—",
            f"{t_w:.3f}s",
            _pass_cell(w_ok == expect_ok),
        )

    m, tok, device = model.load()
    wrong = randrecover.negative_control_transcript_like(
        wm_text,
        tok,
        device,
        n_bits=wm.SECURITY_PARAM * wm.WM_BIT_REDUNDANCY,
        model=m,
    )
    t0 = time.perf_counter()
    recovered_neg = wm.recover_channel_bits(wrong)
    neg_ok, _ = wm.master_detect(sk, wrong, recovered_bits=recovered_neg)
    t_neg = time.perf_counter() - t0
    neg_pass = not bool(neg_ok)
    all_ok &= neg_pass
    det_total += t_neg
    det_table.add_row(
        "master (decoy transcript)",
        "—",
        "False",
        "True" if neg_ok else "False",
        "—",
        "—",
        "—",
        f"{t_neg:.3f}s",
        _pass_cell(neg_pass),
    )

    c.print(det_table)
    c.print(f"  [dim]detection total[/] {det_total:.3f}s")

    c.print()
    if all_ok:
        c.print(Panel("[bold green]All protocol checks passed.[/]", expand=False))
        return 0
    c.print(Panel("[bold red]One or more checks failed — see tables above.[/]", expand=False))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
