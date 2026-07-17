import time
from hashlib import sha256
from typing import List, Sequence, Tuple

import cprf
import prc
import randrecover

import model
from text_attributes import derive_attributes, VOCABULARY, CPRF_ATTR_DIM

SECURITY_PARAM = 300
WM_BIT_REDUNDANCY = 1


def interleave_repetitions(bits: Sequence[int], redundancy: int) -> List[int]:
    """Depth-interleave R replicas of a length-n codeword onto the token channel.

    Emits R full passes of the codeword (``b0..bn-1`` repeated R times), so
    replicas of logical bit ``i`` land at channel indices ``i, i+n, …, i+(R-1)n``.
    That spacing matches the bursty error pattern of the randrecovery channel
    (retokenization spans, sacrificed slots, local sampling correlation): a
    contiguous error run corrupts at most one replica per logical bit instead
    of wiping a consecutive repeat block.
    """
    if redundancy <= 1:
        return [int(b) for b in bits]
    out: List[int] = []
    for _ in range(redundancy):
        out.extend(int(b) for b in bits)
    return out


def majority_deinterleave(
    raw: Sequence[int], code_length: int, redundancy: int
) -> List[int]:
    """Strict-majority vote over depth-interleaved replicas (tie → 0)."""
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
    channel_bits = interleave_repetitions(bits, WM_BIT_REDUNDANCY)
    t2 = time.perf_counter()
    out = randrecover.generate_with_watermark(
        m, tok, prompt, channel_bits, device, tokenizer_id=model.MODEL_ID
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
    return out


def recover_channel_bits(watermarked_text: str) -> List[int]:
    """Tokenize text once and recover the raw watermark channel bits."""
    m, tok, device = model.load()
    raw, _ = randrecover.recover_bitstream_from_text(
        watermarked_text, tok, device, model=m, tokenizer_id=model.MODEL_ID
    )
    return list(raw)


def detect(
    dk: cprf.ConstrainedKey,
    watermarked_text: str,
    *,
    raw_bits: Sequence[int] | None = None,
) -> Tuple[bool, List[int]]:
    """
    Detect with a constrained key.

    Pass ``raw_bits`` from ``recover_channel_bits`` when checking many keys on the
    same text so partition recovery is not repeated.
    """
    attributes = derive_attributes(watermarked_text, dk.modulus, log_scores=False)
    prc.set_code_length(SECURITY_PARAM)
    recovered_s = prc.key_gen_from_seed(sha256(dk.c_eval(attributes)).digest())
    raw = list(raw_bits) if raw_bits is not None else recover_channel_bits(watermarked_text)
    bits_int = majority_deinterleave(raw, SECURITY_PARAM, WM_BIT_REDUNDANCY)
    ok = prc.detect(recovered_s, [bool(b) for b in bits_int])
    return ok, bits_int


def master_detect(
    sk: cprf.MasterKey,
    watermarked_text: str,
    *,
    raw_bits: Sequence[int] | None = None,
) -> Tuple[bool, List[int]]:
    """Master-key detect; see ``detect`` for ``raw_bits`` reuse."""
    attributes = derive_attributes(watermarked_text, sk.modulus, log_scores=False)
    prc.set_code_length(SECURITY_PARAM)
    s = prc.key_gen_from_seed(sha256(sk.eval(attributes)).digest())
    raw = list(raw_bits) if raw_bits is not None else recover_channel_bits(watermarked_text)
    bits_int = majority_deinterleave(raw, SECURITY_PARAM, WM_BIT_REDUNDANCY)
    ok = prc.detect(s, [bool(b) for b in bits_int])
    return ok, bits_int
