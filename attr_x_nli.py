"""CPRF attribute x: zero-shot label scores over ``closed_vocab.VOCABULARY`` + fixed random tail."""

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

# With ``multi_label=True``, each label gets a score in [0, 1]. Label is active (absence_bit 0)
# iff score >= this bar. Raise → stricter; lower → more labels active.
NLI_LABEL_ACTIVE_MIN_SCORE: float = 0.9

# Hugging Face ``hypothesis_template`` for zero-shot labels; ``{}`` is replaced by each candidate.
NLI_HYPOTHESIS_TEMPLATE = "{} is the primary subject of this text."

logger = logging.getLogger(__name__)

_classifier: Optional[Any] = None


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


def _scores_and_absence_bits(text: str) -> tuple[dict[str, float], List[int]]:
    """
    Run zero-shot NLI on ``text``; return rounded per-label scores (VOCABULARY order in the dict)
    and prefix absence bits (0 = label active at or above ``NLI_LABEL_ACTIVE_MIN_SCORE``).
    """
    clf = _get_classifier()
    premise = (text or " ").strip() or " "
    if len(premise) > 6000:
        premise = premise[:6000]

    raw = clf(
        premise,
        list(VOCABULARY),
        multi_label=True,
        # hypothesis_template=NLI_HYPOTHESIS_TEMPLATE,
    )
    score_by_label = {lab: float(s) for lab, s in zip(raw["labels"], raw["scores"])}

    final_scores = {w: round(score_by_label.get(w, 0.0), 4) for w in VOCABULARY}
    out: List[int] = []
    for w in VOCABULARY:
        score = score_by_label.get(w, 0.0)
        bit = 0 if score >= NLI_LABEL_ACTIVE_MIN_SCORE else 1
        out.append(bit)
    return final_scores, out


def log_pair_zero_shot_scores(
    *,
    baseline: Mapping[str, float],
    watermarked: Mapping[str, float],
) -> None:
    """One INFO line with both score maps for side-by-side log comparison."""
    logger.info(
        "Zero-shot label scores — baseline: %s | watermarked: %s",
        dict(baseline),
        dict(watermarked),
    )


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
) -> List[int]:
    final_scores, prefix = _scores_and_absence_bits(text)
    if log_nli_scores:
        logger.info("Zero-shot label scores: %s", final_scores)
    if nli_scores_out is not None:
        nli_scores_out.clear()
        nli_scores_out.update(final_scores)
    tail = _fixed_tail(ATTR_TAIL_DIM, modulus)
    combined = prefix + tail
    return [v % modulus for v in combined]
