"""Helpers for reasoning / chain-of-thought (CoT) completions used in watermarking.

Many models wrap internal reasoning in delimiter tags. When present, the greedy baseline
split lets us prefill the same CoT token sequence before watermarked continuation so the
answer is conditioned on identical reasoning.
"""

from __future__ import annotations

from typing import Sequence

# End-of-reasoning markers (search in order; earliest closing tag in the string wins).
# Built with concatenation so editors/agents do not strip literal XML-like substrings.
_THINK = "think"
_REDACT_THINK = "redacted_thinking"
REASONING_END_MARKERS: tuple[str, ...] = (
    "</" + _THINK + ">",  # common Qwen3 / DeepSeek-R1 closing tags
    "</" + _REDACT_THINK + ">",
    "<|/" + _THINK + "|>",  # some chat templates
    "<|/" + _REDACT_THINK + "|>",
)


def split_completion_at_reasoning_end(completion: str) -> tuple[str, str]:
    """
    Split assistant completion into (cot_prefix, answer_suffix).

    ``cot_prefix`` includes everything through the matched end marker (inclusive).
    If no marker is found, returns ("", completion) so the whole string is treated as
    the post-prompt continuation for both NLI and PRC recovery (legacy path).
    """
    if not completion:
        return "", completion

    best_end: int | None = None
    best_start: int | None = None
    for marker in REASONING_END_MARKERS:
        if not marker:
            continue
        pos = completion.find(marker)
        if pos == -1:
            continue
        end = pos + len(marker)
        if best_start is None or pos < best_start:
            best_start = pos
            best_end = end

    if best_end is None:
        return "", completion

    cot = completion[:best_end]
    answer = completion[best_end:].lstrip()
    return cot, answer


def find_cot_token_boundary(
    gen_ids: Sequence[int],
    tokenizer,
    *,
    skip_special_tokens: bool = True,
) -> int | None:
    """
    Return ``k`` such that ``decode(gen_ids[:k])`` matches the CoT prefix from
    ``split_completion_at_reasoning_end`` on the full decoded completion.

    ``k`` is the number of *generated* tokens (prefix of ``gen_ids``) to prefill before
    watermarked sampling. Returns ``None`` if no reasoning block is detected or no
    token boundary aligns.
    """
    gen_ids = list(gen_ids)
    if not gen_ids:
        return None

    completion = tokenizer.decode(gen_ids, skip_special_tokens=skip_special_tokens)
    cot, answer = split_completion_at_reasoning_end(completion)
    if not cot or not answer:
        return None

    cot_stripped = cot.rstrip()

    for k in range(1, len(gen_ids) + 1):
        prefix = tokenizer.decode(gen_ids[:k], skip_special_tokens=skip_special_tokens)
        if prefix == cot:
            return k
        if prefix.rstrip() == cot_stripped:
            return k

    # Local search around a length ratio estimate (handles rare decode/id mismatches).
    pos = len(cot)
    comp_len = max(len(completion), 1)
    k0 = max(1, min(len(gen_ids), int(len(gen_ids) * pos / comp_len)))
    for radius in range(0, min(len(gen_ids), 48) + 1):
        for sign in (-1, 1):
            kn = k0 + sign * radius
            if kn < 1 or kn > len(gen_ids):
                continue
            prefix = tokenizer.decode(gen_ids[:kn], skip_special_tokens=skip_special_tokens)
            if prefix == cot or prefix.rstrip() == cot_stripped:
                return kn

    return None


def prc_recovery_text(watermarked_full_completion: str) -> str:
    """
    Text slice used for PRC bit recovery: answer-only when a reasoning block is present,
    otherwise the full completion (same as legacy ``recover_bitstream_from_text`` input).
    """
    cot, ans = split_completion_at_reasoning_end(watermarked_full_completion)
    if cot and ans:
        return ans
    if cot and not ans:
        return watermarked_full_completion
    return watermarked_full_completion


__all__ = [
    "REASONING_END_MARKERS",
    "split_completion_at_reasoning_end",
    "find_cot_token_boundary",
    "prc_recovery_text",
]
