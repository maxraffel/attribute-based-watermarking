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
# recovery warms up on this many retokenized prefix tokens; generation
# stabilizes the published burn-in so its retokenization has this length.
BURN_IN_TOKENS = 100


def recover_channel_bits(watermarked_text: str) -> List[int]:
    """Recover raw channel bits via prompt-free balanced rewind.

    Uses only ``watermarked_text`` plus scheme constants
    (``BURN_IN_TOKENS``, ``SECURITY_PARAM``, ``WM_BIT_REDUNDANCY``) and the
    loaded LM/tokenizer.

    Call once per transcript and pass the result to ``detect`` /
    ``master_detect`` via ``recovered_bits`` so multi-key detection does not
    repeat model rewind.
    """
    return recover_channel_bits_batch([watermarked_text])[0]


def recover_channel_bits_batch(
    watermarked_texts: Sequence[str],
    *,
    batch_size: int | None = None,
    on_batch_done: Any | None = None,
) -> List[List[int]]:
    """Batched integrity-preserving recovery for many watermarked texts."""
    m, tok, device = model.load()
    n_bits = SECURITY_PARAM * WM_BIT_REDUNDANCY
    results = randrecover.recover_bitstreams_from_watermarked_texts(
        m,
        tok,
        list(watermarked_texts),
        device,
        burn_in_tokens=BURN_IN_TOKENS,
        n_channel_bits=n_bits,
        batch_size=batch_size,
        on_batch_done=on_batch_done,
    )
    return [list(bits) for bits, _tokens, _gaps in results]


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


def _tokens_per_sec(n_tokens: int, seconds: float) -> float:
    if seconds <= 0.0 or n_tokens <= 0:
        return 0.0
    return float(n_tokens) / float(seconds)


def generate(
    sk: cprf.MasterKey,
    prompt: str,
    *,
    baseline_text: str | None = None,
    baseline_n_tokens: int | None = None,
) -> dict:
    return generate_batch(
        sk,
        [prompt],
        baseline_texts=[baseline_text] if baseline_text is not None else None,
        baseline_n_tokens=(
            [baseline_n_tokens] if baseline_n_tokens is not None else None
        ),
        batch_size=1,
    )[0]


def generate_batch(
    sk: cprf.MasterKey,
    prompts: Sequence[str],
    *,
    baseline_texts: Sequence[str | None] | None = None,
    baseline_n_tokens: Sequence[int | None] | None = None,
    batch_size: int | None = None,
    on_batch_done: Any | None = None,
) -> list[dict]:
    """Batched protocol generate (baselines optional; watermark LM path is batched)."""
    prompt_list = [str(p) for p in prompts]
    n = len(prompt_list)
    if n == 0:
        return []

    m, tok, device = model.load()
    n_channel = SECURITY_PARAM * WM_BIT_REDUNDANCY
    n_baseline = BURN_IN_TOKENS + n_channel

    if baseline_texts is None:
        t0 = time.perf_counter()
        texts, token_counts = randrecover.generate_baselines(
            m, tok, prompt_list, n_baseline, device, batch_size=batch_size
        )
        seconds_baseline_each = (time.perf_counter() - t0) / max(n, 1)
        baselines = list(texts)
        baseline_toks = [int(t) for t in token_counts]
        baseline_secs = [seconds_baseline_each] * n
    else:
        if len(baseline_texts) != n:
            raise ValueError("baseline_texts length must match prompts")
        baselines = []
        baseline_toks = []
        baseline_secs = []
        for i, bt in enumerate(baseline_texts):
            if bt is None:
                t0 = time.perf_counter()
                text, n_tok = randrecover.generate_baseline(
                    m, tok, prompt_list[i], n_baseline, device
                )
                baselines.append(text)
                baseline_toks.append(int(n_tok))
                baseline_secs.append(time.perf_counter() - t0)
            else:
                baselines.append(str(bt))
                baseline_secs.append(0.0)
                if baseline_n_tokens is not None and baseline_n_tokens[i] is not None:
                    baseline_toks.append(int(baseline_n_tokens[i]))
                else:
                    baseline_toks.append(int(n_baseline))

    channel_bits_list: list[list[int]] = []
    attrs_list: list = []
    scores_list: list[dict[str, float]] = []
    secret_list: list[list[int]] = []
    for baseline in baselines:
        baseline_scores: dict[str, float] = {}
        attributes = derive_attributes(
            baseline, sk.modulus, log_scores=False, scores_out=baseline_scores
        )
        r = sk.eval(attributes)
        prc.set_code_length(SECURITY_PARAM)
        bits = [
            1 if b else 0
            for b in prc.encode(prc.key_gen_from_seed(sha256(r).digest()))
        ]
        channel_bits = expand_channel_bits(bits, WM_BIT_REDUNDANCY)
        attrs_list.append(attributes)
        scores_list.append(dict(baseline_scores))
        secret_list.append(list(bits))
        channel_bits_list.append(list(channel_bits))

    t2 = time.perf_counter()
    wm_outs = randrecover.generate_with_watermarks(
        m,
        tok,
        prompt_list,
        channel_bits_list,
        device,
        burn_in_tokens=BURN_IN_TOKENS,
        batch_size=batch_size,
        on_batch_done=on_batch_done,
    )
    seconds_wm_total = time.perf_counter() - t2
    seconds_wm_each = seconds_wm_total / max(n, 1)

    results: list[dict] = []
    for i, out in enumerate(wm_outs):
        n_tokens_watermarked = len(out["burn_in_ids"]) + len(out["wm_suffix_ids"])
        out = dict(out)
        out["attributes"] = attrs_list[i]
        out["baseline_text"] = baselines[i]
        out["label_scores_baseline"] = scores_list[i]
        out["seconds_baseline_gen"] = float(baseline_secs[i])
        out["seconds_watermarked_gen"] = float(seconds_wm_each)
        out["n_tokens_baseline"] = int(baseline_toks[i])
        out["n_tokens_watermarked"] = int(n_tokens_watermarked)
        out["tokens_per_sec_baseline"] = _tokens_per_sec(
            baseline_toks[i], baseline_secs[i]
        )
        out["tokens_per_sec_watermarked"] = _tokens_per_sec(
            n_tokens_watermarked, seconds_wm_each
        )
        out["prc_secret_bits"] = list(secret_list[i])
        out["wm_bit_redundancy"] = WM_BIT_REDUNDANCY
        out["wm_channel_bits"] = list(channel_bits_list[i])
        out["burn_in_tokens"] = BURN_IN_TOKENS
        results.append(out)
    return results


def generate_from_channel_bits_batch(
    prompts: Sequence[str],
    channel_bitstreams: Sequence[Sequence[int]],
    *,
    batch_size: int | None = None,
    on_batch_done: Any | None = None,
) -> list[dict]:
    """Batched LM watermark path only (no CPRF/PRC). Timing amortized per item."""
    prompt_list = [str(p) for p in prompts]
    bits_list = [list(bits) for bits in channel_bitstreams]
    n = len(prompt_list)
    if n == 0:
        return []
    if len(bits_list) != n:
        raise ValueError("prompts and channel_bitstreams must have the same length")
    m, tok, device = model.load()
    t0 = time.perf_counter()
    outs = randrecover.generate_with_watermarks(
        m,
        tok,
        prompt_list,
        bits_list,
        device,
        burn_in_tokens=BURN_IN_TOKENS,
        batch_size=batch_size,
        on_batch_done=on_batch_done,
    )
    elapsed = time.perf_counter() - t0
    each = elapsed / max(n, 1)
    results: list[dict] = []
    for out in outs:
        out = dict(out)
        n_tok = len(out["burn_in_ids"]) + len(out["wm_suffix_ids"])
        out["seconds_watermarked_gen"] = float(each)
        out["n_tokens_watermarked"] = int(n_tok)
        out["tokens_per_sec_watermarked"] = _tokens_per_sec(n_tok, each)
        out["burn_in_tokens"] = BURN_IN_TOKENS
        results.append(out)
    return results


def detect(
    dk: cprf.ConstrainedKey,
    watermarked_text: str,
    *,
    recovered_bits: Sequence[int] | None = None,
) -> Tuple[bool, List[int]]:
    attributes = derive_attributes(watermarked_text, dk.modulus, log_scores=False)
    prc.set_code_length(SECURITY_PARAM)
    recovered_s = prc.key_gen_from_seed(sha256(dk.c_eval(attributes)).digest())
    raw = (
        list(recovered_bits)
        if recovered_bits is not None
        else recover_channel_bits(watermarked_text)
    )
    bits_int = majority_deinterleave(raw, SECURITY_PARAM, WM_BIT_REDUNDANCY)
    ok = prc.detect(recovered_s, [bool(b) for b in bits_int])
    return ok, bits_int


def master_detect(
    sk: cprf.MasterKey,
    watermarked_text: str,
    *,
    recovered_bits: Sequence[int] | None = None,
) -> Tuple[bool, List[int]]:
    attributes = derive_attributes(watermarked_text, sk.modulus, log_scores=False)
    prc.set_code_length(SECURITY_PARAM)
    s = prc.key_gen_from_seed(sha256(sk.eval(attributes)).digest())
    raw = (
        list(recovered_bits)
        if recovered_bits is not None
        else recover_channel_bits(watermarked_text)
    )
    bits_int = majority_deinterleave(raw, SECURITY_PARAM, WM_BIT_REDUNDANCY)
    ok = prc.detect(s, [bool(b) for b in bits_int])
    return ok, bits_int
