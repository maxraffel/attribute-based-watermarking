"""CPRF attribute x: zero-shot MNLI scores over ``closed_vocab.VOCABULARY`` + fixed random tail.

Prefix bits (``multi_label=True``, sigmoid per label): coordinate ``i`` is **0** when label ``i``
is **active** (score ≥ ``NLI_MULTI_LABEL_SCORE_CUTOFF``), else **1**. If no label meets the cutoff,
the argmax label alone is marked active (deterministic tie-break: smallest ``VOCABULARY`` index).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, List, Mapping, MutableMapping, Optional

import torch
from transformers import pipeline

from closed_vocab import ATTR_TAIL_DIM, VOCABULARY

NLI_MODEL_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
# Deterministic tail bytes (independent of text/prefix); same for every ``derive_x`` call.
_FIXED_TAIL_SEED = b"watermarking-for-llm/attr-x/fixed-tail/v1\x00"

# Hugging Face ``hypothesis_template`` for zero-shot labels; ``{}`` is replaced by each candidate.
NLI_HYPOTHESIS_TEMPLATE = "{} is the primary subject of this text."

# Per-label sigmoid scores when ``multi_label=True``; used as the active threshold for prefix bits.
NLI_MULTI_LABEL_SCORE_CUTOFF = 0.5

logger = logging.getLogger(__name__)

_classifier: Optional[Any] = None


def _winner_index(score_by_label: Mapping[str, float]) -> int | None:
    """Smallest VOCABULARY index among labels tied for the maximum zero-shot score (deterministic)."""
    labels = list(VOCABULARY)
    if not labels:
        return None
    scored = [float(score_by_label.get(w, 0.0)) for w in labels]
    mx = max(scored)
    return min(i for i, s in enumerate(scored) if s == mx)


def _active_mask_from_scores(score_by_label: Mapping[str, float], cutoff: float) -> list[bool]:
    """
    Active labels for prefix bits: score ≥ ``cutoff``, or if none qualify, argmax-only fallback.
    """
    labels = list(VOCABULARY)
    if not labels:
        return []
    scored = [float(score_by_label.get(w, 0.0)) for w in labels]
    active = [s >= cutoff for s in scored]
    if not any(active):
        win = _winner_index(score_by_label)
        if win is not None:
            active = [False] * len(labels)
            active[win] = True
    return active


def _prefix_absence_bits_from_scores(score_by_label: Mapping[str, float], cutoff: float) -> List[int]:
    """Prefix coordinates: 0 = label active, 1 = inactive (same convention as former single-label path)."""
    return [0 if a else 1 for a in _active_mask_from_scores(score_by_label, cutoff)]


def _format_zero_shot_scores_table(
    scores: Mapping[str, float],
    *,
    cutoff: float,
    row_indent: str = "  ",
) -> str:
    """Aligned rows in ``VOCABULARY`` order for log output (matches closed-vocab prefix layout)."""
    labels = list(VOCABULARY)
    col_w = max(len(lab) for lab in labels) if labels else 8
    col_w = max(col_w, 10)
    active = _active_mask_from_scores(scores, cutoff)
    act_hdr = f"active(>={cutoff:g})"
    lines = [
        f"{row_indent}{'label':<{col_w}}  {'score':>7}  {act_hdr}",
        f"{row_indent}{'-' * col_w}  {'-----':>7}  {'-' * len(act_hdr)}",
    ]
    for i, lab in enumerate(labels):
        s = float(scores.get(lab, 0.0))
        yn = "yes" if i < len(active) and active[i] else "no"
        lines.append(f"{row_indent}{lab:<{col_w}}  {s:>7.4f}  {yn}")
    return "\n".join(lines)


def _format_paired_zero_shot_table(
    baseline: Mapping[str, float],
    watermarked: Mapping[str, float],
    *,
    cutoff: float,
    row_indent: str = "  ",
) -> str:
    """
    One table: greedy baseline vs watermarked scores per label, delta, and y/n active
    (score ≥ ``cutoff`` per side, with same fallback as ``_active_mask_from_scores``; columns ``b`` / ``w``).
    """
    labels = list(VOCABULARY)
    col_w = max(len(lab) for lab in labels) if labels else 8
    col_w = max(col_w, 10)
    b_act = _active_mask_from_scores(baseline, cutoff)
    w_act = _active_mask_from_scores(watermarked, cutoff)
    lines = [
        f"{row_indent}{'label':<{col_w}}  {'baseline':>8}  {'wm':>8}  {'Δ':>8}  {'b':>3}  {'w':>3}",
        f"{row_indent}{'-' * col_w}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 3}  {'-' * 3}",
    ]
    for i, lab in enumerate(labels):
        sb = float(baseline.get(lab, 0.0))
        sw = float(watermarked.get(lab, 0.0))
        delta = sw - sb
        b_y = "y" if i < len(b_act) and b_act[i] else "n"
        w_y = "y" if i < len(w_act) and w_act[i] else "n"
        lines.append(
            f"{row_indent}{lab:<{col_w}}  {sb:>8.4f}  {sw:>8.4f}  {delta:+8.4f}  {b_y:>3}  {w_y:>3}"
        )
    return "\n".join(lines)


def _pipeline_device() -> int:
    return 0 if torch.cuda.is_available() else -1


def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "zero-shot-classification",
            model=NLI_MODEL_ID,
            device=_pipeline_device(),
        )
    return _classifier


def _scores_and_absence_bits(text: str, cutoff: float) -> tuple[dict[str, float], List[int]]:
    """
    Run zero-shot NLI on ``text``; return rounded per-label scores (``VOCABULARY`` order in the dict)
    and prefix absence bits: **0** for each label whose sigmoid score is ≥ ``cutoff`` (multi-label
    active), **1** otherwise. If no label reaches ``cutoff``, the argmax label uses **0** only.
    """
    clf = _get_classifier()
    premise = (text or " ").strip() or " "
    if len(premise) > 6000:
        premise = premise[:6000]

    raw = clf(
        premise,
        list(VOCABULARY),
        multi_label=True,
        hypothesis_template=NLI_HYPOTHESIS_TEMPLATE,
    )
    score_by_label = {lab: float(s) for lab, s in zip(raw["labels"], raw["scores"])}

    final_scores = {w: round(score_by_label.get(w, 0.0), 4) for w in VOCABULARY}
    out = _prefix_absence_bits_from_scores(score_by_label, cutoff)
    return final_scores, out


def log_pair_zero_shot_scores(
    *,
    baseline: Mapping[str, float],
    watermarked: Mapping[str, float],
    nli_score_cutoff: float | None = None,
) -> None:
    """Log greedy baseline vs watermarked NLI scores in one aligned comparison table."""
    co = NLI_MULTI_LABEL_SCORE_CUTOFF if nli_score_cutoff is None else nli_score_cutoff
    msg = (
        "Zero-shot label scores — greedy vs watermarked (prefix: multi-label active per side; "
        f"cutoff={co:g}; b/w = active y/n)\n"
        f"{_format_paired_zero_shot_table(baseline, watermarked, cutoff=co)}"
    )
    logger.info(msg)


def _fixed_tail(n: int, modulus: int) -> List[int]:
    raw = hashlib.shake_256(_FIXED_TAIL_SEED).digest(2 * n)
    return [
        int.from_bytes(raw[i * 2 : (i + 1) * 2], "big", signed=False) % modulus
        for i in range(n)
    ]


def derive_x(
    text: str,
    modulus: int,
    *,
    log_nli_scores: bool = True,
    nli_scores_out: Optional[MutableMapping[str, float]] = None,
    nli_score_cutoff: float | None = None,
) -> List[int]:
    cutoff = NLI_MULTI_LABEL_SCORE_CUTOFF if nli_score_cutoff is None else nli_score_cutoff
    final_scores, prefix = _scores_and_absence_bits(text, cutoff)
    if log_nli_scores:
        logger.info(
            "Zero-shot label scores (prefix: multi-label active, cutoff=%g)\n%s",
            cutoff,
            _format_zero_shot_scores_table(final_scores, cutoff=cutoff),
        )
    if nli_scores_out is not None:
        nli_scores_out.clear()
        nli_scores_out.update(final_scores)
    tail = _fixed_tail(ATTR_TAIL_DIM, modulus)
    combined = prefix + tail
    return [v % modulus for v in combined]
