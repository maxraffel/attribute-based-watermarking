"""Closed vocabulary labels and CPRF constraint vectors (prefix-only; tail is zero-padded)."""

from __future__ import annotations

from typing import List, Sequence

# Single project-wide label list (CPRF inner-product prefix length).
VOCABULARY: List[str] = [
    "medicine", "law", "technology", "finance", "art", "sports", "cooking", "environment", "philosophy"
]

# Hash-derived coordinates appended after NLI label bits (ignored by keyword constraints).
ATTR_TAIL_DIM = 32

CPRF_ATTR_DIM = len(VOCABULARY) + ATTR_TAIL_DIM


def f_for_required_keywords(required: Sequence[str]) -> List[int]:
    """
    f[i]=1 on prefix indices for each required label in VOCABULARY; unknown strings ignored.
    Tail coordinates are zero so constraining ignores the non-label part of x.
    """
    idx = {w.casefold(): i for i, w in enumerate(VOCABULARY)}
    f = [0] * CPRF_ATTR_DIM
    for r in required:
        k = r.casefold().strip()
        if k in idx:
            f[idx[k]] = 1
    return f


def issue_constrained_key_for_keywords(sk, required: Sequence[str]):
    """Constrained key: every listed (known) required label must NLI-match (prefix of x only)."""
    return sk.constrain(f_for_required_keywords(required))
