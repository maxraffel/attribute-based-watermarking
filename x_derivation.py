"""
Derive the CPRF attribute vector x from unwatermarked model output + KeyBERT keyphrases.

Security / reproducibility model (intended use):

1) **Replicability (same x everywhere):** Anyone who can reconstruct the *same* canonical
   inputs gets the same x: normalized UTF-8 text, a pinned KeyBERT model + extract_keywords
   parameters, and a versioned IKM (input keying material) layout. Unwatermarked generation
   should use greedy decoding (temperature 0) so the reference string is stable for a fixed
   model and prompt.

2) **Binding:** x is a deterministic function of (reference text, keyphrases, optional salt).
   Edits to the reference text or different KeyBERT versions usually change x (avalanche via
   SHAKE-256), which re-targets the constrained CPRF and PRC chain.

3) **Optional application salt:** Set env `CPRF_X_SALT` to a hex string of extra secret bytes
   that are mixed into the IKM. Verifiers need the same salt (or the same pre-derived x) to
   match. Without a salt, x is a *public* function of the text + phrases: fine when the
   attribute itself is public but you still want stable residues mod the modulus.

4) **Randomness / uniformity:** IKM is expanded with SHAKE-256 (NIST, domain-separated by
   construction). Each component uses two bytes, reduced mod the CPRF modulus (here 1024).
   This is not a uniform distribution on Z/1024Z for tiny moduli, but is standard and
   sufficient for a synthetic attribute vector in this construction.

5) **What KeyBERT is doing:** It surfaces salient n-grams (unigrams and bigrams by default) as
   a compact semantic digest of the reference answer. Sorting by keyword string stabilizes
   order where scores might tie across runs.
"""
from __future__ import annotations

import hashlib
import os
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from keybert import KeyBERT

# Pinned: changing any of these changes the derived x.
KEYBERT_MODEL_NAME = "all-MiniLM-L6-v2"
KEYPHRASE_NGRAM_RANGE = (1, 2)
KEYPHRASE_TOP_N = 32
IKM_PREFIX = b"watermarking-for-llm/attr-x/ikm/v1\x00"

_DEFAULT_KEYBERT: Optional[KeyBERT] = None


def _optional_app_salt() -> bytes:
    raw = os.environ.get("CPRF_X_SALT", "").strip()
    if not raw:
        return b""
    try:
        return bytes.fromhex(raw)
    except ValueError as e:
        raise ValueError("CPRF_X_SALT, if set, must be a hex string of bytes") from e


def _normalize(s: str) -> str:
    return unicodedata.normalize("NFKC", s).strip()


def get_keybert() -> KeyBERT:
    global _DEFAULT_KEYBERT
    if _DEFAULT_KEYBERT is None:
        _DEFAULT_KEYBERT = KeyBERT(model=KEYBERT_MODEL_NAME)
    return _DEFAULT_KEYBERT


@dataclass
class XDerivationInfo:
    keyphrases: List[Tuple[str, float]]
    ikm_fp16: str  # sha256(ikm)[:16] hex — safe to log, identifies derivation inputs


def _canonical_keyphrases(kws: Sequence[Tuple[str, float]]) -> str:
    # Sort by case-folded key so order does not depend on KeyBERT's ranking jitter.
    lines = [f"{_normalize(k)}\t{float(score):.8f}" for k, score in kws]
    lines.sort(key=lambda line: line.split("\t", 1)[0].casefold())
    return "\n".join(lines)


def build_ikm(
    unwatermarked_text: str, keyphrases: Sequence[Tuple[str, float]], extra_secret: bytes
) -> bytes:
    text_b = _normalize(unwatermarked_text).encode("utf-8", errors="strict")
    kpb = _canonical_keyphrases(keyphrases).encode("utf-8", errors="strict")
    return (
        IKM_PREFIX
        + b"sec_len:"
        + len(extra_secret).to_bytes(2, "big", signed=False)
        + extra_secret
        + b"text_len:"
        + len(text_b).to_bytes(8, "big", signed=False)
        + text_b
        + b"\x00keyphrases:\n"
        + kpb
    )


def ikm_to_x(ikm: bytes, code_len: int, modulus: int) -> List[int]:
    n_bytes = 2 * code_len
    raw = hashlib.shake_256(ikm).digest(n_bytes)
    return [
        int.from_bytes(raw[i * 2 : (i + 1) * 2], "big", signed=False) % modulus
        for i in range(code_len)
    ]


def extract_keyphrases(
    unwatermarked_text: str, *, keybert: Optional[KeyBERT] = None
) -> List[Tuple[str, float]]:
    """Run KeyBERT with pinned hyperparameters; returns (phrase, score) pairs."""
    kb = keybert or get_keybert()
    text = _normalize(unwatermarked_text) or " "
    raw = kb.extract_keywords(
        text,
        keyphrase_ngram_range=KEYPHRASE_NGRAM_RANGE,
        stop_words="english",
        use_mmr=False,
        top_n=KEYPHRASE_TOP_N,
    )
    out: List[Tuple[str, float]] = []
    for item in raw:
        if isinstance(item, tuple) and len(item) >= 2:
            out.append((str(item[0]), float(item[1])))
    return out


def derive_x(
    unwatermarked_text: str,
    code_len: int,
    modulus: int,
    *,
    keybert: Optional[KeyBERT] = None,
) -> Tuple[List[int], XDerivationInfo]:
    kws = extract_keyphrases(unwatermarked_text, keybert=keybert)
    # print(f"Keyphrases: {kws}")
    ikm = build_ikm(unwatermarked_text, kws, _optional_app_salt())
    fp = hashlib.sha256(ikm).hexdigest()[:16]
    return ikm_to_x(ikm, code_len, modulus), XDerivationInfo(keyphrases=kws, ikm_fp16=fp)
