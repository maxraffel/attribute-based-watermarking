"""Closed vocabulary, label classification, and CPRF attribute vector ``x`` derivation."""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from typing import List, Mapping, MutableMapping, Optional, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from model import require_cuda_device

logger = logging.getLogger(__name__)

# --- closed vocabulary (CPRF prefix) ---

VOCABULARY: List[str] = [
    "medicine", "economics", "art", "software", "sports",
]

ATTR_TAIL_DIM = 32
CPRF_ATTR_DIM = len(VOCABULARY) + ATTR_TAIL_DIM

# Multi-label activation threshold on normalized scores in [0, 1].
SCORE_CUTOFF = 0.001

# Query side of each (query, document) pair passed to the label scorer.
LABEL_QUERY_TEMPLATE = "About the topic of {label}"

_FIXED_TAIL_SEED = b"watermarking-for-llm/attr-x/fixed-tail/v1\x00"
_MAX_DOCUMENT_CHARS = 6000


# --- label scorer interface ---


class LabelScorer(ABC):
    """Score how strongly each label applies to a document (higher = more active)."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        ...

    @abstractmethod
    def score_labels(self, document: str, labels: Sequence[str]) -> dict[str, float]:
        """Return per-label scores in ``[0, 1]`` (implementation-defined normalization)."""


class BGERerankerScorer(LabelScorer):
    """``BAAI/bge-reranker-v2-m3`` cross-encoder relevance scores, sigmoid-normalized."""

    MODEL_ID = "BAAI/bge-reranker-v2-m3"
    MAX_LENGTH = 512

    def __init__(self) -> None:
        self._model: AutoModelForSequenceClassification | None = None
        self._tokenizer: AutoTokenizer | None = None
        self._device: str | None = None

    @property
    def model_id(self) -> str:
        return self.MODEL_ID

    def _load(self) -> None:
        if self._model is not None:
            return
        device = require_cuda_device()
        self._device = device
        logger.info("Loading label scorer %s on GPU...", self.MODEL_ID)
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_ID,
        ).eval().to(device)

    @torch.no_grad()
    def score_labels(self, document: str, labels: Sequence[str]) -> dict[str, float]:
        self._load()
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._device is not None

        text = (document or " ").strip() or " "
        if len(text) > _MAX_DOCUMENT_CHARS:
            text = text[:_MAX_DOCUMENT_CHARS]

        pairs = [
            [LABEL_QUERY_TEMPLATE.format(label=label), text]
            for label in labels
        ]
        inputs = self._tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.MAX_LENGTH,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        logits = self._model(**inputs, return_dict=True).logits.view(-1).float()
        probs = torch.sigmoid(logits).tolist()
        return {label: float(prob) for label, prob in zip(labels, probs)}


_default_scorer: LabelScorer | None = None


def get_scorer() -> LabelScorer:
    global _default_scorer
    if _default_scorer is None:
        _default_scorer = BGERerankerScorer()
    return _default_scorer


# --- CPRF vocabulary helpers ---


def f_for_required_keywords(required: Sequence[str]) -> List[int]:
    """``f[i]=1`` on prefix indices for each required label; tail coordinates are zero."""
    idx = {w.casefold(): i for i, w in enumerate(VOCABULARY)}
    f = [0] * CPRF_ATTR_DIM
    for r in required:
        k = r.casefold().strip()
        if k in idx:
            f[idx[k]] = 1
    return f


def active_labels_from_attributes(attributes: Sequence[int], modulus: int) -> List[str]:
    """
    Labels marked **active** on the transcript used to build ``attributes``.

    Prefix entries are absence bits: active label ⇒ coordinate ``≡ 0 (mod modulus)``.
    """
    out: List[str] = []
    for i, w in enumerate(VOCABULARY):
        if i >= len(attributes):
            break
        if int(attributes[i]) % modulus == 0:
            out.append(w)
    return out


# --- score → attribute vector ---


def _active_mask_from_scores(score_by_label: Mapping[str, float], cutoff: float) -> list[bool]:
    labels = list(VOCABULARY)
    if not labels:
        return []
    scored = [float(score_by_label.get(w, 0.0)) for w in labels]
    return [s >= cutoff for s in scored]


def _prefix_absence_bits_from_scores(score_by_label: Mapping[str, float], cutoff: float) -> List[int]:
    return [0 if a else 1 for a in _active_mask_from_scores(score_by_label, cutoff)]


def format_label_scores_table(
    scores: Mapping[str, float],
    *,
    cutoff: float,
    row_indent: str = "  ",
) -> str:
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


def _fixed_tail(n: int, modulus: int) -> List[int]:
    raw = hashlib.shake_256(_FIXED_TAIL_SEED).digest(2 * n)
    return [
        int.from_bytes(raw[i * 2 : (i + 1) * 2], "big", signed=False) % modulus
        for i in range(n)
    ]


def classify_text(text: str, *, cutoff: float | None = None) -> dict[str, float]:
    """Score every vocabulary label for ``text``; returns rounded label → score."""
    co = SCORE_CUTOFF if cutoff is None else cutoff
    scorer = get_scorer()
    raw = scorer.score_labels(text, list(VOCABULARY))
    return {w: round(raw.get(w, 0.0), 4) for w in VOCABULARY}


def derive_attributes(
    text: str,
    modulus: int,
    *,
    log_scores: bool = True,
    scores_out: Optional[MutableMapping[str, float]] = None,
    score_cutoff: float | None = None,
) -> List[int]:
    """
    Derive CPRF attribute vector from ``text``: prefix from label classification + fixed tail.

    Prefix coordinate ``i`` is **0** when label ``i`` is active (score ≥ cutoff), else **1**.
    If no label meets the cutoff, all prefix coordinates are **1** (no active labels).
    """
    cutoff = SCORE_CUTOFF if score_cutoff is None else score_cutoff
    scorer = get_scorer()
    score_by_label = scorer.score_labels(text, list(VOCABULARY))
    final_scores = {w: round(score_by_label.get(w, 0.0), 4) for w in VOCABULARY}
    prefix = _prefix_absence_bits_from_scores(score_by_label, cutoff)

    if log_scores:
        logger.info(
            "Label classification (classifier=%s; cutoff=%g)\n%s",
            scorer.model_id,
            cutoff,
            format_label_scores_table(final_scores, cutoff=cutoff),
        )
    if scores_out is not None:
        scores_out.clear()
        scores_out.update(final_scores)

    tail = _fixed_tail(ATTR_TAIL_DIM, modulus)
    return [v % modulus for v in prefix + tail]
