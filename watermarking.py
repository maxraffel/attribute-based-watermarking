import time
from hashlib import sha256
from typing import List, Tuple

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

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading with device: ", DEVICE)

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
MODEL = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)

SECURITY_PARAM = 300


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
    x = _derive_x(watermarked_text, sk.modulus, log_nli_scores=False)
    prc.set_code_length(SECURITY_PARAM)
    s = prc.key_gen_from_seed(sha256(sk.eval(x)).digest())
    bits, _ = randrecover.recover_bitstream_from_text(watermarked_text, TOKENIZER, DEVICE)
    bits = (bits + [0] * SECURITY_PARAM)[:SECURITY_PARAM]
    bits_int = [1 if b else 0 for b in bits]
    ok = prc.detect(s, [bool(b) for b in bits])
    return ok, bits_int
