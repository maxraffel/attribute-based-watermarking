import time
from hashlib import sha256
from typing import List, Tuple

import cprf
import prc
import randrecover

import model
from text_attributes import derive_attributes, VOCABULARY, CPRF_ATTR_DIM

SECURITY_PARAM = 300
WM_BIT_REDUNDANCY = 1


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


def generate(sk: cprf.MasterKey, prompt: str) -> dict:
    m, tok, device = model.load()
    n_channel = SECURITY_PARAM * WM_BIT_REDUNDANCY

    t0 = time.perf_counter()
    baseline = randrecover.generate_baseline(m, tok, prompt, n_channel, device)
    t1 = time.perf_counter()
    baseline_scores: dict[str, float] = {}
    attributes = derive_attributes(
        baseline, sk.modulus, log_scores=False, scores_out=baseline_scores
    )
    r = sk.eval(attributes)
    prc.set_code_length(SECURITY_PARAM)
    bits = [1 if b else 0 for b in prc.encode(prc.key_gen_from_seed(sha256(r).digest()))]
    channel_bits = [b for b in bits for _ in range(WM_BIT_REDUNDANCY)]
    t2 = time.perf_counter()
    out = randrecover.generate_with_watermark(m, tok, prompt, channel_bits, device)
    t3 = time.perf_counter()

    # --- logging ---
    out["attributes"] = attributes
    out["baseline_text"] = baseline
    out["label_scores_baseline"] = dict(baseline_scores)
    out["seconds_baseline_gen"] = t1 - t0
    out["seconds_watermarked_gen"] = t3 - t2
    out["prc_secret_bits"] = list(bits)
    out["wm_bit_redundancy"] = WM_BIT_REDUNDANCY
    out["wm_channel_bits"] = list(channel_bits)
    return out


def detect(dk: cprf.ConstrainedKey, watermarked_text: str) -> Tuple[bool, List[int]]:
    m, tok, device = model.load()
    attributes = derive_attributes(watermarked_text, dk.modulus, log_scores=False)
    prc.set_code_length(SECURITY_PARAM)
    recovered_s = prc.key_gen_from_seed(sha256(dk.c_eval(attributes)).digest())
    raw, _ = randrecover.recover_bitstream_from_text(
        watermarked_text, tok, device, model=m
    )
    need = SECURITY_PARAM * WM_BIT_REDUNDANCY
    raw = (raw + [0] * need)[:need]
    r = WM_BIT_REDUNDANCY
    bits = []
    for i in range(SECURITY_PARAM):
        chunk = raw[i * r : (i + 1) * r]
        bits.append(1 if 2 * sum(chunk) > r else 0)
    bits_int = [1 if b else 0 for b in bits]
    ok = prc.detect(recovered_s, [bool(b) for b in bits])
    return ok, bits_int


def master_detect(sk: cprf.MasterKey, watermarked_text: str) -> Tuple[bool, List[int]]:
    m, tok, device = model.load()
    attributes = derive_attributes(watermarked_text, sk.modulus, log_scores=False)
    prc.set_code_length(SECURITY_PARAM)
    s = prc.key_gen_from_seed(sha256(sk.eval(attributes)).digest())
    raw, _ = randrecover.recover_bitstream_from_text(
        watermarked_text, tok, device, model=m
    )
    need = SECURITY_PARAM * WM_BIT_REDUNDANCY
    raw = (raw + [0] * need)[:need]
    r = WM_BIT_REDUNDANCY
    bits = []
    for i in range(SECURITY_PARAM):
        chunk = raw[i * r : (i + 1) * r]
        bits.append(1 if 2 * sum(chunk) > r else 0)
    bits_int = [1 if b else 0 for b in bits]
    ok = prc.detect(s, [bool(b) for b in bits])
    return ok, bits_int
