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

# Hugging Face merges ``model.generation_config`` into every ``model.generate`` call and into
# ``randrecover``'s incremental path (``_prepare_generation_config``). Set keys here to override
# the model card (e.g. Llama-3.2-1B-Instruct defaults: temperature≈0.6, top_k=50, top_p=0.9).
GENERATION_SAMPLING_OVERRIDES: dict = {
    "temperature": 0.3,
    "top_p": 1.0,
    "top_k": 0,  # HF: 0 often means disable top-k
}

TOKENIZER: Optional[AutoTokenizer] = None
MODEL: Optional[AutoModelForCausalLM] = None

SECURITY_PARAM = 300
# Each logical PRC bit is embedded WM_BIT_REDUNDANCY times on the token channel; recovery uses strict majority per group.
WM_BIT_REDUNDANCY = 1


def wm_channel_bits_length() -> int:
    """
    New-token horizon for **both** baseline sampling and watermarked embedding: logical
    ``SECURITY_PARAM`` × ``WM_BIT_REDUNDANCY`` (one payload token per physical channel bit).
    """
    return int(SECURITY_PARAM) * int(WM_BIT_REDUNDANCY)


def set_wm_bit_redundancy(r: int) -> None:
    """
    Set repetition factor for the partition channel (default ``1`` = legacy one sample per logical bit).

    Each logical PRC bit is expanded to ``r`` identical channel bits before ``generate_with_watermark``;
    ``detect`` / ``master_detect`` recover ``SECURITY_PARAM * r`` raw bits and fold with strict majority
    (ties become ``0``) before ``prc.detect``. Baseline sampling uses the same new-token count as the
    watermarked path (``wm_channel_bits_length()``).
    """
    global WM_BIT_REDUNDANCY
    ri = int(r)
    if ri < 1:
        raise ValueError("WM_BIT_REDUNDANCY must be >= 1")
    WM_BIT_REDUNDANCY = ri


def _expand_bits_for_wm_channel(logical: List[int], r: int) -> List[int]:
    return [int(b) & 1 for b in logical for _ in range(int(r))]


def _majority_fold_channel_bits(raw: List[int], r: int, n_logical: int) -> List[int]:
    """``n_logical`` groups of ``r`` raw bits → logical bits; strict majority, tie → ``0``."""
    r = int(r)
    need = int(n_logical) * r
    padded = [int(x) & 1 for x in raw] + [0] * max(0, need - len(raw))
    padded = padded[:need]
    out: List[int] = []
    for i in range(int(n_logical)):
        chunk = padded[i * r : (i + 1) * r]
        s = sum(chunk)
        out.append(1 if 2 * s > r else 0)
    return out


def _load_llm() -> None:
    """Load ``TOKENIZER`` and ``MODEL`` for the current ``MODEL_ID`` (no-op if already loaded)."""
    global TOKENIZER, MODEL
    if MODEL is not None and TOKENIZER is not None:
        return
    print(f"Loading causal LM {MODEL_ID!r} on {DEVICE}")
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
    MODEL = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
    for attr, value in GENERATION_SAMPLING_OVERRIDES.items():
        setattr(MODEL.generation_config, attr, value)
    if not MODEL.config.is_encoder_decoder:
        TOKENIZER.padding_side = "left"


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
    Set the **logical** PRC codeword length (``SECURITY_PARAM``): LDPC block size and the length of
    ``prc.encode`` / ``prc.detect`` bit vectors. Baseline and watermarked LM paths both use
    ``wm_channel_bits_length()`` = ``SECURITY_PARAM * WM_BIT_REDUNDANCY`` new tokens. Also calls
    ``prc.set_code_length(n)``.
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
        MODEL, TOKENIZER, prompt, wm_channel_bits_length(), DEVICE
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
    channel_bits = _expand_bits_for_wm_channel(bits, WM_BIT_REDUNDANCY)
    t2 = time.perf_counter()
    assert MODEL is not None and TOKENIZER is not None
    out = randrecover.generate_with_watermark(MODEL, TOKENIZER, prompt, channel_bits, DEVICE)
    t3 = time.perf_counter()
    out["attr_x"] = x
    out["baseline_text"] = baseline
    out["nli_label_scores_baseline"] = dict(baseline_nli)
    out["seconds_baseline_gen"] = t1 - t0
    out["seconds_watermarked_gen"] = t3 - t2
    out["prc_secret_bits"] = list(bits)
    out["wm_bit_redundancy"] = int(WM_BIT_REDUNDANCY)
    out["wm_channel_bits"] = list(channel_bits)
    return out


def detect(dk: cprf.ConstrainedKey, watermarked_text: str) -> Tuple[bool, List[int]]:
    """
    Recover PRC bits. Rebuilds ``x`` from ``watermarked_text`` via ``derive_x`` (zero-shot prefix + fixed tail).
    This must match the ``x`` used in ``generate`` (from the sampling baseline) for detection to succeed.

    Returns ``(prc_ok, recovered_bits)`` where ``recovered_bits`` are ``0``/``1`` integers of length
    ``SECURITY_PARAM`` (logical; majority-folded when ``WM_BIT_REDUNDANCY`` > 1).
    """
    _ensure_llm()
    assert TOKENIZER is not None
    x = _derive_x(watermarked_text, dk.modulus, log_nli_scores=False)
    prc.set_code_length(SECURITY_PARAM)
    # CPRF seed is sha256(commonEval···); constrained vs master outputs match iff Δ·⟨f,x⟩≡0 (mod m),
    # not merely ⟨f,x⟩≡0 on composite modulus — compare dk.c_eval(x) to sk.eval(x) when debugging policies.
    recovered_s = prc.key_gen_from_seed(sha256(dk.c_eval(x)).digest())
    assert MODEL is not None
    raw, _ = randrecover.recover_bitstream_from_text(
        watermarked_text, TOKENIZER, DEVICE, model=MODEL
    )
    need = wm_channel_bits_length()
    raw = (raw + [0] * need)[:need]
    bits = _majority_fold_channel_bits(raw, WM_BIT_REDUNDANCY, SECURITY_PARAM)
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
    assert MODEL is not None
    raw, _ = randrecover.recover_bitstream_from_text(
        watermarked_text, TOKENIZER, DEVICE, model=MODEL
    )
    need = wm_channel_bits_length()
    raw = (raw + [0] * need)[:need]
    bits = _majority_fold_channel_bits(raw, WM_BIT_REDUNDANCY, SECURITY_PARAM)
    bits_int = [1 if b else 0 for b in bits]
    ok = prc.detect(s, [bool(b) for b in bits])
    return ok, bits_int
