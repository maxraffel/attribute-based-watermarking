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
# Free (unwatermarked) model tokens before the channel payload. Prompt-free
# recovery warms up on this prefix; separate decode+concat isolates retokenization.
BURN_IN_TOKENS = 100


def recover_channel_bits(
    watermarked_text: str,
    *,
    generation_out: Dict[str, Any] | None = None,
    prompt: str | None = None,
) -> List[int]:
    """Recover raw channel bits via prompt-free balanced rewind.

    Call this once per transcript and pass the result to ``detect`` /
    ``master_detect`` via ``recovered_bits`` so multi-key detection does not
    repeat model rewind.

    With ``generation_out``, recovery uses the cascade-isolated character split
    (``burn_in_char_len``) and does **not** need the original prompt. Generation
    builds the same prompt-free partitions while still sampling under the prompt.
    Text-only decoys without generation metadata receive uncorrelated bits.
    """
    del prompt  # intentionally unused: recovery is prompt-free
    m, tok, device = model.load()
    if generation_out is not None:
        raw, _, _ = randrecover.recover_bitstream_from_generation(
            m, tok, generation_out, device
        )
        return list(raw)

    n_bits = SECURITY_PARAM * WM_BIT_REDUNDANCY
    return randrecover.uncorrelated_bits_from_text(watermarked_text, tok, n_bits=n_bits)


def expand_channel_bits(bits: Sequence[int], redundancy: int) -> List[int]:
    """Expand a length-n codeword to ``n * R`` channel bits (depth layout).

    Emits R full passes of the codeword. Replicas of logical bit ``i`` sit at
    ``i, i+n, …, i+(R-1)n``, so a short burst corrupts at most one replica per bit.
    """
    if redundancy <= 1:
        return [int(b) for b in bits]
    out: List[int] = []
    for _ in range(redundancy):
        out.extend(int(b) for b in bits)
    return out


def interleave_repetitions(bits: Sequence[int], redundancy: int) -> List[int]:
    """Alias for ``expand_channel_bits`` (depth-interleaved replicas)."""
    return expand_channel_bits(bits, redundancy)


def majority_deinterleave(
    raw: Sequence[int],
    code_length: int,
    redundancy: int,
) -> List[int]:
    """Strict-majority vote over R depth-layout replicas (tie → 0)."""
    need = code_length * redundancy
    padded = (list(raw) + [0] * need)[:need]
    if redundancy <= 1:
        return [int(padded[i]) for i in range(code_length)]
    bits: List[int] = []
    for i in range(code_length):
        votes = sum(int(padded[i + r * code_length]) for r in range(redundancy))
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
    # Match watermarked length: burn-in + channel tokens.
    n_baseline = BURN_IN_TOKENS + n_channel

    if baseline_text is None:
        t0 = time.perf_counter()
        baseline = randrecover.generate_baseline(m, tok, prompt, n_baseline, device)
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
    out = randrecover.generate_with_watermark(
        m,
        tok,
        prompt,
        channel_bits,
        device,
        burn_in_tokens=BURN_IN_TOKENS,
    )
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
    out["burn_in_tokens"] = BURN_IN_TOKENS
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
