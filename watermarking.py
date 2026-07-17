import time
from hashlib import sha256
from typing import Any, Dict, List, Sequence, Tuple

import cprf
import prc
import randrecover

import model
from text_attributes import derive_attributes, VOCABULARY, CPRF_ATTR_DIM

SECURITY_PARAM = 300
WM_BIT_REDUNDANCY = 1
# ``depth``: R full passes of the codeword (replicas of bit i at i, i+n, …).
# ``block``: each bit repeated R times contiguously (replicas of bit i at iR..iR+R-1).
REDUNDANCY_LAYOUT = "depth"
_VALID_REDUNDANCY_LAYOUTS = frozenset({"depth", "block"})
# ``static``: seeded unigram vocab partitions (original).
# ``balanced``: per-step masks from post-processor next-token probabilities.
PARTITION_MODE = "static"
_VALID_PARTITION_MODES = frozenset({"static", "balanced"})


def set_partition_mode(mode: str) -> str:
    """Set and return the active partition mode (``static`` or ``balanced``)."""
    global PARTITION_MODE
    key = str(mode).strip().lower()
    if key not in _VALID_PARTITION_MODES:
        raise ValueError(
            f"PARTITION_MODE must be one of {sorted(_VALID_PARTITION_MODES)}, got {mode!r}"
        )
    PARTITION_MODE = key
    return PARTITION_MODE


def set_redundancy_layout(layout: str) -> str:
    """Set and return the channel redundancy layout (``depth`` or ``block``)."""
    global REDUNDANCY_LAYOUT
    key = str(layout).strip().lower()
    if key not in _VALID_REDUNDANCY_LAYOUTS:
        raise ValueError(
            f"REDUNDANCY_LAYOUT must be one of {sorted(_VALID_REDUNDANCY_LAYOUTS)}, got {layout!r}"
        )
    REDUNDANCY_LAYOUT = key
    return REDUNDANCY_LAYOUT


def recover_channel_bits(
    watermarked_text: str,
    *,
    generation_out: Dict[str, Any] | None = None,
    prompt: str | None = None,
) -> List[int]:
    """Recover raw channel bits using the active ``PARTITION_MODE``.

    Call this once per transcript and pass the result to ``detect`` /
    ``master_detect`` via ``recovered_bits`` so multi-key detection does not
    repeat model rewind (especially costly in ``balanced`` mode).

    Balanced recovery requires ``generation_out`` (prompt + model token ids).
    Text-only inputs (e.g. decoy / negative controls) always use the cheap
    static partition extractor — a full LM rewind is neither needed nor faithful
    for text that was not produced by balanced watermarked generation.
    """
    del prompt  # reserved for callers; balanced text-only rewind is intentionally unused
    m, tok, device = model.load()
    if PARTITION_MODE == "balanced" and generation_out is not None:
        raw, _, _ = randrecover.recover_bitstream_balanced_from_generation(
            m, tok, generation_out, device
        )
        return list(raw)

    raw, _ = randrecover.recover_bitstream_from_text(
        watermarked_text, tok, device, model=m
    )
    return list(raw)


def expand_channel_bits(
    bits: Sequence[int],
    redundancy: int,
    layout: str | None = None,
) -> List[int]:
    """Expand a length-n codeword to ``n * R`` channel bits.

    ``depth`` (default): emit R full passes of the codeword. Replicas of logical
    bit ``i`` sit at ``i, i+n, …, i+(R-1)n``, so a short burst corrupts at most
    one replica per bit.

    ``block``: repeat each bit R times in place (``b0`` R times, then ``b1``, …).
    A burst of length R can wipe every replica of one logical bit.
    """
    layout_key = (REDUNDANCY_LAYOUT if layout is None else layout).strip().lower()
    if layout_key not in _VALID_REDUNDANCY_LAYOUTS:
        raise ValueError(
            f"layout must be one of {sorted(_VALID_REDUNDANCY_LAYOUTS)}, got {layout!r}"
        )
    if redundancy <= 1:
        return [int(b) for b in bits]
    if layout_key == "depth":
        out: List[int] = []
        for _ in range(redundancy):
            out.extend(int(b) for b in bits)
        return out
    out = []
    for b in bits:
        out.extend([int(b)] * redundancy)
    return out


def interleave_repetitions(bits: Sequence[int], redundancy: int) -> List[int]:
    """Depth-layout expand (compat alias for ``expand_channel_bits(..., layout='depth')``)."""
    return expand_channel_bits(bits, redundancy, layout="depth")


def majority_deinterleave(
    raw: Sequence[int],
    code_length: int,
    redundancy: int,
    layout: str | None = None,
) -> List[int]:
    """Strict-majority vote over R replicas (tie → 0)."""
    layout_key = (REDUNDANCY_LAYOUT if layout is None else layout).strip().lower()
    if layout_key not in _VALID_REDUNDANCY_LAYOUTS:
        raise ValueError(
            f"layout must be one of {sorted(_VALID_REDUNDANCY_LAYOUTS)}, got {layout!r}"
        )
    need = code_length * redundancy
    padded = (list(raw) + [0] * need)[:need]
    if redundancy <= 1:
        return [int(padded[i]) for i in range(code_length)]
    bits: List[int] = []
    for i in range(code_length):
        if layout_key == "depth":
            votes = sum(int(padded[i + r * code_length]) for r in range(redundancy))
        else:
            base = i * redundancy
            votes = sum(int(padded[base + r]) for r in range(redundancy))
        bits.append(1 if 2 * votes > redundancy else 0)
    return bits


def setup(modulus: int) -> cprf.MasterKey:
    return cprf.keygen(modulus, CPRF_ATTR_DIM)


def issue(sk: cprf.MasterKey, keywords: List[str]) -> cprf.ConstrainedKey:
    idx = {w.casefold(): i for i, w in enumerate(VOCABULARY)}
    f = [0] * CPRF_ATTR_DIM
    for kw in keywords:
        i = idx.get(kw.casefold().strip())
        if i is not None:
            f[i] = 1
    return sk.constrain(f)


def generate(sk: cprf.MasterKey, prompt: str, *, baseline_text: str | None = None) -> dict:
    m, tok, device = model.load()
    n_channel = SECURITY_PARAM * WM_BIT_REDUNDANCY

    if baseline_text is None:
        t0 = time.perf_counter()
        baseline = randrecover.generate_baseline(m, tok, prompt, n_channel, device)
        seconds_baseline_gen = time.perf_counter() - t0
    else:
        baseline = baseline_text
        seconds_baseline_gen = 0.0

    baseline_scores: dict[str, float] = {}
    attributes = derive_attributes(
        baseline, sk.modulus, log_scores=False, scores_out=baseline_scores
    )
    r = sk.eval(attributes)
    prc.set_code_length(SECURITY_PARAM)
    bits = [1 if b else 0 for b in prc.encode(prc.key_gen_from_seed(sha256(r).digest()))]
    channel_bits = expand_channel_bits(bits, WM_BIT_REDUNDANCY)
    t2 = time.perf_counter()
    if PARTITION_MODE == "balanced":
        out = randrecover.generate_with_watermark_balanced(
            m, tok, prompt, channel_bits, device
        )
    else:
        out = randrecover.generate_with_watermark(m, tok, prompt, channel_bits, device)
    seconds_watermarked_gen = time.perf_counter() - t2

    # --- logging ---
    out["attributes"] = attributes
    out["baseline_text"] = baseline
    out["label_scores_baseline"] = dict(baseline_scores)
    out["seconds_baseline_gen"] = seconds_baseline_gen
    out["seconds_watermarked_gen"] = seconds_watermarked_gen
    out["prc_secret_bits"] = list(bits)
    out["wm_bit_redundancy"] = WM_BIT_REDUNDANCY
    out["wm_channel_bits"] = list(channel_bits)
    out["partition_mode"] = PARTITION_MODE
    out["redundancy_layout"] = REDUNDANCY_LAYOUT
    return out


def detect(
    dk: cprf.ConstrainedKey,
    watermarked_text: str,
    *,
    generation_out: Dict[str, Any] | None = None,
    prompt: str | None = None,
    recovered_bits: Sequence[int] | None = None,
) -> Tuple[bool, List[int]]:
    attributes = derive_attributes(watermarked_text, dk.modulus, log_scores=False)
    prc.set_code_length(SECURITY_PARAM)
    recovered_s = prc.key_gen_from_seed(sha256(dk.c_eval(attributes)).digest())
    raw = (
        list(recovered_bits)
        if recovered_bits is not None
        else recover_channel_bits(
            watermarked_text, generation_out=generation_out, prompt=prompt
        )
    )
    bits_int = majority_deinterleave(raw, SECURITY_PARAM, WM_BIT_REDUNDANCY)
    ok = prc.detect(recovered_s, [bool(b) for b in bits_int])
    return ok, bits_int


def master_detect(
    sk: cprf.MasterKey,
    watermarked_text: str,
    *,
    generation_out: Dict[str, Any] | None = None,
    prompt: str | None = None,
    recovered_bits: Sequence[int] | None = None,
) -> Tuple[bool, List[int]]:
    attributes = derive_attributes(watermarked_text, sk.modulus, log_scores=False)
    prc.set_code_length(SECURITY_PARAM)
    s = prc.key_gen_from_seed(sha256(sk.eval(attributes)).digest())
    raw = (
        list(recovered_bits)
        if recovered_bits is not None
        else recover_channel_bits(
            watermarked_text, generation_out=generation_out, prompt=prompt
        )
    )
    bits_int = majority_deinterleave(raw, SECURITY_PARAM, WM_BIT_REDUNDANCY)
    ok = prc.detect(s, [bool(b) for b in bits_int])
    return ok, bits_int
