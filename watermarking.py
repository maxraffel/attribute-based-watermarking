import time
from hashlib import sha256
from typing import List, Optional, Tuple

import torch
import cprf
import prc
import randrecover
from transformers import AutoModelForCausalLM, AutoTokenizer

from attr_x_nli import derive_x
from closed_vocab import CPRF_ATTR_DIM, issue_constrained_key_for_keywords


def _derive_x(text: str, modulus: int, **kwargs) -> List[int]:
    """Call ``derive_x`` with optional kwargs (newer ``attr_x_nli``); fall back for older checkouts."""
    if not kwargs:
        return derive_x(text, modulus)
    try:
        return derive_x(text, modulus, **kwargs)
    except TypeError:
        return derive_x(text, modulus)


DEFAULT_LLM_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_ID: str = DEFAULT_LLM_MODEL_ID
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

TOKENIZER: Optional[AutoTokenizer] = None
MODEL: Optional[AutoModelForCausalLM] = None

SECURITY_PARAM = 300


def _load_llm() -> None:
    """Load ``TOKENIZER`` and ``MODEL`` for the current ``MODEL_ID`` (no-op if already loaded)."""
    global TOKENIZER, MODEL
    if MODEL is not None and TOKENIZER is not None:
        return
    print(f"Loading causal LM {MODEL_ID!r} on {DEVICE}")
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
    MODEL = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)


def _ensure_llm() -> None:
    if MODEL is None or TOKENIZER is None:
        _load_llm()


def set_llm_model_id(model_id: str) -> None:
    """
    Select the Hugging Face hub id for the watermark causal LM.

    Call before ``generate`` / ``detect`` / ``master_detect`` if you need a non-default model
    (e.g. from a notebook snippet). Reloads weights when ``model_id`` differs from the loaded id.
    """
    global MODEL_ID, MODEL, TOKENIZER
    mid = model_id.strip()
    if not mid:
        raise ValueError("LLM model id must be a non-empty string")

    if MODEL is None and TOKENIZER is None:
        MODEL_ID = mid
        _load_llm()
        return

    if mid == MODEL_ID:
        return

    print(f"Switching causal LM {MODEL_ID!r} -> {mid!r}")
    MODEL_ID = mid
    MODEL = None
    TOKENIZER = None
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    _load_llm()


def set_prc_code_length(n: int) -> None:
    """
    Set the PRC codeword length and the greedy-baseline / bit-recovery horizon (``SECURITY_PARAM``).
    Also calls ``prc.set_code_length(n)`` so the Rust LDPC instance matches.
    """
    global SECURITY_PARAM
    if n < 1:
        raise ValueError("PRC code length must be positive")
    SECURITY_PARAM = n
    prc.set_code_length(n)


def _baseline(prompt: str) -> str:
    _ensure_llm()
    assert MODEL is not None and TOKENIZER is not None
    return randrecover.generate_baseline(
        MODEL, TOKENIZER, prompt, SECURITY_PARAM, DEVICE
    )


def setup(modulus: int) -> cprf.MasterKey:
    return cprf.keygen(modulus, CPRF_ATTR_DIM)


def issue(sk: cprf.MasterKey, f: List[int]) -> cprf.ConstrainedKey:
    return sk.constrain(f)


def issue_unconstrained(sk: cprf.MasterKey) -> cprf.ConstrainedKey:
    """Constrained key with f = 0 (always agrees with master eval for any x)."""
    return issue(sk, [0] * CPRF_ATTR_DIM)


def issue_keyword_policy(sk: cprf.MasterKey, required: List[str]) -> cprf.ConstrainedKey:
    """
    Constrained key: inner product uses only the label prefix of ``x``.
    ``detect`` / ``master_detect`` rebuild ``x`` from the watermarked transcript (zero-shot
    prefix on that text + fixed tail), so policies apply to that reconstructed attribute.
    """
    return issue_constrained_key_for_keywords(sk, required)


def attr_x_for_prompt(sk: cprf.MasterKey, prompt: str) -> List[int]:
    return derive_x(_baseline(prompt), sk.modulus)


def generate(sk: cprf.MasterKey, prompt: str) -> dict:
    t0 = time.perf_counter()
    baseline = _baseline(prompt)
    t1 = time.perf_counter()
    baseline_nli: dict[str, float] = {}
    x = _derive_x(
        baseline,
        sk.modulus,
        log_nli_scores=False,
        nli_scores_out=baseline_nli,
    )
    r = sk.eval(x)
    prc.set_code_length(SECURITY_PARAM)
    c = prc.encode(prc.key_gen_from_seed(sha256(r).digest()))
    bits = [1 if b else 0 for b in c]
    t2 = time.perf_counter()
    assert MODEL is not None and TOKENIZER is not None
    out = randrecover.generate_with_watermark(MODEL, TOKENIZER, prompt, bits, DEVICE)
    t3 = time.perf_counter()
    out["attr_x"] = x
    out["baseline_text"] = baseline
    out["nli_label_scores_baseline"] = dict(baseline_nli)
    out["seconds_baseline_gen"] = t1 - t0
    out["seconds_watermarked_gen"] = t3 - t2
    out["prc_secret_bits"] = list(bits)
    return out


def detect(dk: cprf.ConstrainedKey, watermarked_text: str) -> Tuple[bool, List[int]]:
    """
    Recover PRC bits. Rebuilds ``x`` from ``watermarked_text`` via ``derive_x`` (zero-shot prefix + fixed tail).
    This must match the ``x`` used in ``generate`` (from the greedy baseline) for detection to succeed.

    Returns ``(prc_ok, recovered_bits)`` where ``recovered_bits`` are ``0``/``1`` integers of length
    ``SECURITY_PARAM`` (for logging / BER).
    """
    _ensure_llm()
    assert TOKENIZER is not None
    x = _derive_x(watermarked_text, dk.modulus, log_nli_scores=False)
    prc.set_code_length(SECURITY_PARAM)
    # CPRF seed is sha256(commonEval···); constrained vs master outputs match iff Δ·⟨f,x⟩≡0 (mod m),
    # not merely ⟨f,x⟩≡0 on composite modulus — compare dk.c_eval(x) to sk.eval(x) when debugging policies.
    recovered_s = prc.key_gen_from_seed(sha256(dk.c_eval(x)).digest())
    bits, _ = randrecover.recover_bitstream_from_text(watermarked_text, TOKENIZER, DEVICE)
    bits = (bits + [0] * SECURITY_PARAM)[:SECURITY_PARAM]
    bits_int = [1 if b else 0 for b in bits]
    ok = prc.detect(recovered_s, [bool(b) for b in bits])
    return ok, bits_int


def master_detect(sk: cprf.MasterKey, watermarked_text: str) -> Tuple[bool, List[int]]:
    """Same as ``detect`` but using the master key (oracle verifier)."""
    _ensure_llm()
    assert TOKENIZER is not None
    x = _derive_x(watermarked_text, sk.modulus, log_nli_scores=False)
    prc.set_code_length(SECURITY_PARAM)
    s = prc.key_gen_from_seed(sha256(sk.eval(x)).digest())
    bits, _ = randrecover.recover_bitstream_from_text(watermarked_text, TOKENIZER, DEVICE)
    bits = (bits + [0] * SECURITY_PARAM)[:SECURITY_PARAM]
    bits_int = [1 if b else 0 for b in bits]
    ok = prc.detect(s, [bool(b) for b in bits])
    return ok, bits_int
