"""Closed vocabulary labels and CPRF constraint vectors (prefix-only; tail is zero-padded)."""

from __future__ import annotations

from typing import List, Sequence, Set

# Single project-wide label list (CPRF inner-product prefix length).
VOCABULARY: List[str] = [
    "medicine", "law", "software", "business", "art", "football", "cooking", "philosophy", "carpentry", "architecture"
]

# Fixed random coordinates appended after label bits (ignored by keyword constraints).
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
    """Constrained key: every listed (known) required label must be NLI-active on x's prefix (score cutoff)."""
    return sk.constrain(f_for_required_keywords(required))


def active_labels_from_verify_x(x_wm: Sequence[int], modulus: int) -> List[str]:
    """
    Vocabulary labels NLI marks **active** (multi-label score ≥ cutoff; see ``attr_x_nli``) on the
    transcript used to build ``x_wm``.

    Matches ``attr_x_nli.derive_x``: prefix entries are absence bits reduced mod ``modulus``;
    active label ⇒ coordinate ``≡ 0 (mod modulus)`` (zero, one, or more labels).
    """
    out: List[str] = []
    for i, w in enumerate(VOCABULARY):
        if i >= len(x_wm):
            break
        if int(x_wm[i]) % modulus == 0:
            out.append(w)
    return out


def pick_unrelated_keyword_for_policy(
    x_wm: Sequence[int],
    modulus: int,
    exclude: Set[str] | Sequence[str],
) -> str:
    """
    Pick one vocabulary label not in ``exclude`` for a **negative** single-label policy test.

    Verifiers use ``x_wm`` from ``derive_x(watermarked_text, modulus)``. Prefer a label whose
    prefix coordinate is non-zero mod ``modulus`` so ``⟨f,x⟩ ≢ 0`` for one-hot ``f``, avoiding
    accidental ``detect=True`` when NLI already marks that label as active on the transcript.
    """
    ex = {e.casefold() for e in exclude}
    xd = [int(v) % modulus for v in x_wm]
    for i, w in enumerate(VOCABULARY):
        if w.casefold() in ex:
            continue
        if i < len(xd) and xd[i] != 0:
            return w
    for w in VOCABULARY:
        if w.casefold() in ex:
            continue
        f = f_for_required_keywords([w])
        dot = sum(f[j] * xd[j] for j in range(min(len(f), len(xd)))) % modulus
        if dot != 0:
            return w
    for w in VOCABULARY:
        if w.casefold() not in ex:
            return w
    return VOCABULARY[-1]
