"""Compare encode-time ``attr_x`` to ``derive_x`` on a transcript when a detection check fails."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional, Sequence

from rich.console import Console
from rich.markup import escape

logger = logging.getLogger(__name__)


def label_prefix_mismatch_indices(
    x_encode: Sequence[int], x_wm: Sequence[int], n_labels: int
) -> list[int]:
    return [i for i in range(min(n_labels, len(x_encode), len(x_wm))) if x_encode[i] != x_wm[i]]


def log_detect_mismatch(
    *,
    scenario: str,
    test_name: str,
    expect_true: bool,
    got: object,
    x_encode: Sequence[int],
    wm_text: str,
    modulus: int,
    baseline_text: str,
    vocabulary: Sequence[str],
    verbose: bool = False,
    diag: Optional[Counter[tuple[str, str, str]]] = None,
    rich_console: Optional[Console] = None,
) -> None:
    """
    Log whether NLI-derived ``x`` on ``wm_text`` matches encode-time ``x_encode`` (baseline path).

    If ``rich_console`` is set, emit Rich markup there; otherwise log plain text at WARNING.
    When ``diag`` is set, increment ``(scenario, test_name, kind)`` for benchmark aggregation.
    """
    import attr_x_nli

    x_wm = attr_x_nli.derive_x(wm_text, modulus)
    x_match = list(x_encode) == list(x_wm)
    n_labels = len(vocabulary)
    mis_i = label_prefix_mismatch_indices(x_encode, x_wm, n_labels)
    mis_labels = [vocabulary[i] for i in mis_i]

    got_b = bool(got)
    if expect_true and not got_b:
        bucket = "miss+x_stable" if x_match else "miss+x_drift"
    elif (not expect_true) and got_b:
        bucket = "fa+x_stable" if x_match else "fa+x_drift"
    else:
        return

    if diag is not None:
        diag[(scenario, test_name, bucket)] += 1

    plain_core = (
        f"expected {expect_true} got {got_b}. "
        + (
            "encode_x vs wm_x: MATCH — likely PRC / bit recovery / LDPC, not NLI label drift."
            if x_match
            else (
                "encode_x vs wm_x: MISMATCH — NLI prefix likely drifted vs greedy baseline; "
                f"labels differing in absence-bit: {mis_labels}"
            )
        )
    )
    plain = f"Failure diagnostics {scenario} / {test_name}: {plain_core}"

    if rich_console is not None:
        mis_labels_s = escape(", ".join(mis_labels))
        drift_hint = (
            "[cyan]encode_x vs wm_x: MATCH[/] — failure is likely PRC / bit recovery / LDPC, not NLI label drift."
            if x_match
            else (
                "[yellow]encode_x vs wm_x: MISMATCH[/] — NLI-derived prefix likely drifted vs greedy baseline; "
                f"labels differing in absence-bit: {mis_labels_s}"
            )
        )
        rich_console.print(
            f"[bold red]Failure diagnostics[/] [dim]{escape(scenario)}[/] / [bold]{escape(test_name)}[/]: "
            f"expected [cyan]{expect_true}[/] got [cyan]{got_b}[/]. {drift_hint}"
        )
        if verbose:
            clip = 220
            rich_console.print("[dim]baseline (encode) excerpt:[/]")
            rich_console.print(repr(baseline_text[:clip]), markup=False)
            rich_console.print("[dim]transcript excerpt:[/]")
            rich_console.print(repr(wm_text[:clip]), markup=False)
    else:
        logger.warning(plain)
        if verbose:
            clip = 220
            logger.info("baseline (encode) excerpt: %r", baseline_text[:clip])
            logger.info("transcript excerpt: %r", wm_text[:clip])
