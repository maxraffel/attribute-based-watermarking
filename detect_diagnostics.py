"""Compare encode-time attributes to verify-time attributes when a detection check fails."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional, Sequence

from rich.console import Console
from rich.markup import escape

logger = logging.getLogger(__name__)


def attributes_prefix_mismatch_indices(
    encode_attributes: Sequence[int],
    verify_attributes: Sequence[int],
    n_labels: int,
) -> list[int]:
    n = min(n_labels, len(encode_attributes), len(verify_attributes))
    return [i for i in range(n) if encode_attributes[i] != verify_attributes[i]]


def log_detect_mismatch(
    *,
    scenario: str,
    test_name: str,
    expect_true: bool,
    got: object,
    encode_attributes: Sequence[int],
    wm_text: str,
    modulus: int,
    baseline_text: str,
    vocabulary: Sequence[str],
    verbose: bool = False,
    diag: Optional[Counter[tuple[str, str, str]]] = None,
    rich_console: Optional[Console] = None,
) -> None:
    """
    Log whether verify-time attributes on ``wm_text`` match encode-time attributes (baseline path).

    If ``rich_console`` is set, emit Rich markup there; otherwise log plain text at WARNING.
    When ``diag`` is set, increment ``(scenario, test_name, kind)`` for benchmark aggregation.
    """
    import text_attributes

    verify_attributes = text_attributes.derive_attributes(wm_text, modulus)
    attributes_match = list(encode_attributes) == list(verify_attributes)
    n_labels = len(vocabulary)
    mis_i = attributes_prefix_mismatch_indices(encode_attributes, verify_attributes, n_labels)
    mis_labels = [vocabulary[i] for i in mis_i]

    got_b = bool(got)
    if expect_true and not got_b:
        bucket = "miss+attr_stable" if attributes_match else "miss+attr_drift"
    elif (not expect_true) and got_b:
        bucket = "fa+attr_stable" if attributes_match else "fa+attr_drift"
    else:
        return

    if diag is not None:
        diag[(scenario, test_name, bucket)] += 1

    plain_core = (
        f"expected {expect_true} got {got_b}. "
        + (
            "encode vs verify attributes: MATCH — likely PRC / bit recovery / LDPC, not label drift."
            if attributes_match
            else (
                "encode vs verify attributes: MISMATCH — label prefix likely drifted vs greedy baseline; "
                f"labels differing in absence-bit: {mis_labels}"
            )
        )
    )
    plain = f"Failure diagnostics {scenario} / {test_name}: {plain_core}"

    if rich_console is not None:
        mis_labels_s = escape(", ".join(mis_labels))
        drift_hint = (
            "[cyan]encode vs verify attributes: MATCH[/] — failure is likely PRC / bit recovery / LDPC, not label drift."
            if attributes_match
            else (
                "[yellow]encode vs verify attributes: MISMATCH[/] — label-derived prefix likely drifted vs greedy baseline; "
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
