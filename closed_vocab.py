"""CPRF attribute x: absence bits per global VOCABULARY; keyword policies use indicator f."""

from __future__ import annotations

from typing import List, Sequence

# Single project-wide word list (CPRF dimension = len(VOCABULARY)).
VOCABULARY: List[str] = [
    "life", "meaning", "python", "water", "mark", "model", "language",
    "answer", "question", "text", "code", "data", "learning", "network",
    "system", "time", "world", "science", "research", "signal",
]


def _tokens(text: str) -> List[str]:
    return "".join(ch if ch.isalnum() else " " for ch in text.casefold()).split()


def _present(word: str, text: str) -> bool:
    w = word.casefold().strip()
    return bool(w) and w in _tokens(text)


def derive_x(text: str, modulus: int) -> List[int]:
    """x[i] = 0 if VOCABULARY[i] appears in text, else 1 (mod modulus)."""
    return [(0 if _present(w, text) else 1) % modulus for w in VOCABULARY]


def f_for_required_keywords(required: Sequence[str]) -> List[int]:
    """f[i]=1 for each required word that appears in VOCABULARY; unknown strings ignored."""
    idx = {w.casefold(): i for i, w in enumerate(VOCABULARY)}
    f = [0] * len(VOCABULARY)
    for r in required:
        k = r.casefold().strip()
        if k in idx:
            f[idx[k]] = 1
    return f


def issue_constrained_key_for_keywords(sk, required: Sequence[str]):
    """Constrained key: policy is that every listed (known) required word appears in the text."""
    return sk.constrain(f_for_required_keywords(required))
