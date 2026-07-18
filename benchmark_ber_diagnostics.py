"""
Decompose watermark BER into channel recovery, retokenization, majority vote, attributes,
and PRC-detection stages.

For each prompt the script runs one full generate/detect cycle and reports BER at:
  1. Token channel (embedded channel bits vs recovered raw bits)
  2. Retokenization gap (recovery from ``input_ids`` vs from decoded text)
  3. Logical PRC payload (majority-voted bits vs secret)
  4. CPRF key mismatch (encode-time vs verify-time attributes)
  5. PRC ``detect`` under oracle keys/bits vs the actual master path

Run:
  uv run python benchmark_ber_diagnostics.py
  uv run python benchmark_ber_diagnostics.py --prompts prompts.txt --code-length 100
  uv run python benchmark_ber_diagnostics.py --verbose
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Sequence

import benchmark_io
import model
import prc
import randrecover
import text_attributes
import watermarking as wm

from text_attributes import VOCABULARY, active_labels_from_attributes, derive_attributes

DEFAULT_PROMPTS: tuple[str, ...] = (
    "Explain what NFL Madden is.",
    (
        "Explain the economic nuance and impact of Drake Maye during his college "
        "football career at North Carolina."
    ),
    "Summarize the plot of Romeo and Juliet.",
)


@dataclass(frozen=True)
class BerDiagConfig:
    modulus: int = 1024
    code_length: int = 100
    wm_bit_redundancy: int = 3
    burn_in_tokens: int = 100
    model_id: str | None = None
    #: ``None`` leaves ``model.TORCH_COMPILE`` unchanged (Colab shared settings).
    torch_compile: bool | None = None
    quiet: bool = False


@dataclass
@dataclass
class BerPromptDiagnostic:
    index: int
    prompt: str
    n_channel_embedded: int
    n_channel_from_ids: int
    n_channel_from_text: int
    n_payload_tokens: int
    channel_ber_from_ids: float
    channel_ber_from_text: float
    retokenization_extra_ber: float
    retokenization_mismatch_count: int
    logical_ber: float
    attributes_match: bool
    cprf_seed_match: bool
    prefix_coords_differ: int
    encode_active: list[str]
    verify_active: list[str]
    detect_oracle_both: bool
    detect_oracle_key: bool
    detect_oracle_bits: bool
    detect_actual_master: bool
    truncated_recovery: bool
    majority_vote_corrections: int
    majority_vote_regressions: int
    tie_votes: int
    end_to_end_ber_master: float
    end_to_end_detect_master: bool
    channel_error_indices: list[int] = field(default_factory=list)
    logical_error_indices: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class BerDiagnosticsSummary:
    config: BerDiagConfig
    results: list[BerPromptDiagnostic]

    @property
    def n(self) -> int:
        return len(self.results)

    def _avg(self, attr: str) -> float:
        if not self.results:
            return 0.0
        return sum(getattr(r, attr) for r in self.results) / self.n

    def _max(self, attr: str) -> float:
        if not self.results:
            return 0.0
        return max(getattr(r, attr) for r in self.results)


def _ber_percent(secret: Sequence[int], recovered: Sequence[int]) -> float:
    n = len(secret)
    if n == 0:
        return 0.0
    m = min(len(recovered), n)
    errs = sum(1 for i in range(m) if int(secret[i]) != int(recovered[i]))
    errs += abs(n - len(recovered))
    return 100.0 * errs / n


def _align_bits(secret: Sequence[int], recovered: Sequence[int]) -> tuple[list[int], list[int]]:
    n = len(secret)
    rec = list(recovered)
    if len(rec) < n:
        rec = rec + [0] * (n - len(rec))
    return list(secret), rec[:n]


def _bit_errors(secret: Sequence[int], recovered: Sequence[int]) -> list[int]:
    s, r = _align_bits(secret, recovered)
    return [i for i in range(len(s)) if s[i] != r[i]]


def _logical_bits(
    raw: Sequence[int],
    code_length: int,
    redundancy: int,
) -> list[int]:
    return wm.majority_deinterleave(raw, code_length, redundancy)


def _replica_indices(bit_index: int, code_length: int, redundancy: int) -> list[int]:
    return [bit_index + r * code_length for r in range(redundancy)]


def _majority_vote_analysis(
    channel_bits: Sequence[int],
    raw: Sequence[int],
    code_length: int,
    redundancy: int,
) -> tuple[int, int, int]:
    """Return (corrections, regressions, tie_votes) relative to embedded logical bits."""
    if redundancy <= 1:
        return 0, 0, 0
    secret_logical = _logical_bits(channel_bits, code_length, redundancy)
    raw_logical = _logical_bits(raw, code_length, redundancy)
    corrections = regressions = tie_votes = 0
    need = code_length * redundancy
    padded = (list(raw) + [0] * need)[:need]
    for i in range(code_length):
        idxs = _replica_indices(i, code_length, redundancy)
        votes = [int(padded[j]) for j in idxs]
        if 2 * sum(votes) == redundancy:
            tie_votes += 1
        embedded = secret_logical[i]
        recovered = raw_logical[i]
        chunk_err = any(
            j < len(channel_bits) and int(channel_bits[j]) != int(padded[j]) for j in idxs
        )
        if chunk_err and embedded == recovered:
            corrections += 1
        if embedded != recovered:
            regressions += 1
    return corrections, regressions, tie_votes


def _prc_detect(eval_bytes: object, logical_bits: Sequence[int], code_length: int) -> bool:
    prc.set_code_length(code_length)
    key = prc.key_gen_from_seed(sha256(eval_bytes).digest())
    return bool(prc.detect(key, [bool(b) for b in logical_bits]))


def _configure(cfg: BerDiagConfig) -> None:
    wm.SECURITY_PARAM = cfg.code_length
    prc.set_code_length(cfg.code_length)
    wm.WM_BIT_REDUNDANCY = cfg.wm_bit_redundancy
    wm.BURN_IN_TOKENS = cfg.burn_in_tokens
    kwargs: dict = {}
    if cfg.model_id:
        kwargs["model_id"] = cfg.model_id
    if cfg.torch_compile is not None:
        kwargs["torch_compile"] = cfg.torch_compile
    if kwargs:
        model.configure(**kwargs)


def _load_prompts(path: Path | None) -> list[str]:
    if path is None:
        return list(DEFAULT_PROMPTS)
    lines = path.read_text(encoding="utf-8").splitlines()
    prompts = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def _bit_at(raw: Sequence[int], k: int) -> int | None:
    if k >= len(raw):
        return None
    return int(raw[k])


def _retokenization_mismatch_count(
    channel_bits: Sequence[int],
    raw_from_ids: Sequence[int],
    raw_from_text: Sequence[int],
) -> int:
    """Count channel positions where ids recovery is correct but text recovery is not."""
    n = 0
    for k, embedded in enumerate(channel_bits):
        ids_bit = _bit_at(raw_from_ids, k)
        text_bit = _bit_at(raw_from_text, k)
        if text_bit is None or int(text_bit) == int(embedded):
            continue
        if ids_bit is not None and int(ids_bit) == int(embedded):
            n += 1
    return n


def diagnose_prompt(
    sk: object,
    prompt: str,
    index: int,
    cfg: BerDiagConfig,
    *,
    baseline_text: str | None = None,
) -> BerPromptDiagnostic:
    m, tok, device = model.load()
    out = wm.generate(sk, prompt, baseline_text=baseline_text)

    encode_attributes = list(out["attributes"])
    wm_text = out["generated_text_wm"]
    secret: list[int] = list(out["prc_secret_bits"])
    channel_bits: list[int] = list(out["wm_channel_bits"])

    verify_attributes = derive_attributes(wm_text, sk.modulus, log_scores=False)
    encode_active = active_labels_from_attributes(encode_attributes, sk.modulus)
    verify_active = active_labels_from_attributes(verify_attributes, sk.modulus)

    prefix_n = min(len(VOCABULARY), len(encode_attributes), len(verify_attributes))
    prefix_diff = sum(
        1 for i in range(prefix_n) if int(encode_attributes[i]) != int(verify_attributes[i])
    )
    r_enc = sk.eval(encode_attributes)
    r_ver = sk.eval(verify_attributes)

    raw_from_ids, _, _ = randrecover.recover_bitstream_from_generation(
        m, tok, out, device, prefer_text_split=False
    )
    raw_from_text, _, _ = randrecover.recover_bitstream_from_text(
        m,
        tok,
        wm_text,
        device,
        burn_in_char_len=int(out["burn_in_char_len"]),
    )

    logical_from_text = _logical_bits(
        raw_from_text, cfg.code_length, cfg.wm_bit_redundancy
    )
    retok_mismatches = _retokenization_mismatch_count(
        channel_bits, raw_from_ids, raw_from_text
    )

    vote_corr, vote_reg, tie_votes = _majority_vote_analysis(
        channel_bits,
        raw_from_text,
        cfg.code_length,
        cfg.wm_bit_redundancy,
    )

    detect_oracle_both = _prc_detect(r_enc, secret, cfg.code_length)
    detect_oracle_key = _prc_detect(r_enc, logical_from_text, cfg.code_length)
    detect_oracle_bits = _prc_detect(r_ver, secret, cfg.code_length)
    master_ok, master_bits = wm.master_detect(
        sk, wm_text, recovered_bits=raw_from_text
    )

    ch_ids = _ber_percent(channel_bits, raw_from_ids)
    ch_text = _ber_percent(channel_bits, raw_from_text)
    ch_err_idx = _bit_errors(channel_bits, raw_from_text)
    log_err_idx = _bit_errors(secret, logical_from_text)

    notes: list[str] = []
    if not detect_oracle_both:
        notes.append("PRC sanity failed: encode key + perfect secret bits did not detect")
    if ch_ids == 0.0 and ch_text > 0:
        notes.append("All channel error is from decode/retokenize (ids recovery is clean)")
    elif ch_text > ch_ids:
        notes.append(
            f"Retokenization adds {ch_text - ch_ids:.1f}% channel BER "
            f"({ch_ids:.1f}% ids -> {ch_text:.1f}% text)"
        )
    if retok_mismatches > 0:
        notes.append(
            f"{retok_mismatches} channel errors from retokenization "
            f"(ids recovery correct, text recovery wrong)"
        )
    if prefix_diff > 0:
        notes.append(f"{prefix_diff} prefix attribute coords differ (encode vs verify)")
    if detect_oracle_key and not master_ok:
        notes.append("Recovered bits OK with encode key, but master detect failed (attribute drift)")
    elif not detect_oracle_key and master_ok:
        notes.append("Master detect passed despite logical BER (PRC threshold absorbed errors)")
    if vote_corr > 0:
        notes.append(f"Majority vote corrected {vote_corr} logical bit(s)")
    if vote_reg > 0 and cfg.wm_bit_redundancy > 1:
        notes.append(f"Majority vote failed on {vote_reg} logical bit(s)")

    return BerPromptDiagnostic(
        index=index,
        prompt=prompt,
        n_channel_embedded=len(channel_bits),
        n_channel_from_ids=len(raw_from_ids),
        n_channel_from_text=len(raw_from_text),
        n_payload_tokens=len(raw_from_ids),
        channel_ber_from_ids=ch_ids,
        channel_ber_from_text=ch_text,
        retokenization_extra_ber=max(0.0, ch_text - ch_ids),
        retokenization_mismatch_count=retok_mismatches,
        logical_ber=_ber_percent(secret, logical_from_text),
        attributes_match=encode_attributes == verify_attributes,
        cprf_seed_match=r_enc == r_ver,
        prefix_coords_differ=prefix_diff,
        encode_active=encode_active,
        verify_active=verify_active,
        detect_oracle_both=detect_oracle_both,
        detect_oracle_key=detect_oracle_key,
        detect_oracle_bits=detect_oracle_bits,
        detect_actual_master=bool(master_ok),
        truncated_recovery=len(raw_from_text) < len(channel_bits),
        majority_vote_corrections=vote_corr,
        majority_vote_regressions=vote_reg,
        tie_votes=tie_votes,
        end_to_end_ber_master=_ber_percent(secret, master_bits),
        end_to_end_detect_master=bool(master_ok),
        channel_error_indices=ch_err_idx,
        logical_error_indices=log_err_idx,
        notes=notes,
    )


def run_ber_diagnostics(
    prompts: Sequence[str],
    config: BerDiagConfig | None = None,
) -> BerDiagnosticsSummary:
    cfg = config or BerDiagConfig()
    _configure(cfg)
    sk = wm.setup(cfg.modulus)
    prompt_list = list(prompts)
    pregen = benchmark_io.pregenerate_baselines(
        prompt_list,
        quiet=cfg.quiet,
        description="Baselines",
    )
    results: list[BerPromptDiagnostic] = []
    for (i, prompt), baseline in zip(
        benchmark_io.iter_with_progress(
            list(enumerate(prompt_list)),
            description="BER diagnostics",
            disable=cfg.quiet,
        ),
        pregen.texts,
    ):
        results.append(
            diagnose_prompt(sk, prompt, i, cfg, baseline_text=baseline)
        )
    return BerDiagnosticsSummary(config=cfg, results=results)


def _prompt_preview(prompt: str, max_chars: int = 50) -> str:
    if len(prompt) <= max_chars:
        return prompt
    return prompt[: max_chars - 3].rstrip() + "..."


def _yn(ok: bool) -> str:
    return "yes" if ok else "no"


def print_ber_diagnostics(
    summary: BerDiagnosticsSummary,
    *,
    verbose: bool = False,
) -> None:
    cfg = summary.config

    print()
    print("=== BER diagnostics ===")
    print(f"prompts: {summary.n}")
    print(
        f"code_length: {cfg.code_length}  wm_bit_redundancy: {cfg.wm_bit_redundancy}  "
        f"burn_in_tokens: {cfg.burn_in_tokens}  "
        f"modulus: {cfg.modulus}"
    )
    print(f"LLM: {model.MODEL_ID}")
    print(f"classifier: {text_attributes.get_scorer().model_id}")
    print(
        f"inference_dtype: {model.inference_dtype_label()}  "
        f"torch_compile: {'yes' if model.TORCH_COMPILE else 'no'}"
    )

    print()
    print("-- Stage BER (avg / max) --")
    print(
        f"  1. Channel from ids={summary._avg('channel_ber_from_ids'):.2f}%/"
        f"{summary._max('channel_ber_from_ids'):.2f}%  "
        f"from text={summary._avg('channel_ber_from_text'):.2f}%/"
        f"{summary._max('channel_ber_from_text'):.2f}%  "
        f"(retokenize gap={summary._avg('retokenization_extra_ber'):.2f}%/"
        f"{summary._max('retokenization_extra_ber'):.2f}%)"
    )
    print(
        f"  2. Logical PRC payload: {summary._avg('logical_ber'):.2f}%/"
        f"{summary._max('logical_ber'):.2f}%"
    )
    print(
        f"  3. End-to-end (master path): {summary._avg('end_to_end_ber_master'):.2f}%/"
        f"{summary._max('end_to_end_ber_master'):.2f}%"
    )

    n = summary.n or 1
    print()
    print("-- Error sources --")
    print(
        f"  retokenization mismatches (ids OK, text wrong): "
        f"{sum(r.retokenization_mismatch_count for r in summary.results)}"
    )
    print(
        f"  truncated recovery: "
        f"{sum(1 for r in summary.results if r.truncated_recovery)}/{summary.n}"
    )
    print(
        f"  attribute vector mismatch: "
        f"{sum(1 for r in summary.results if not r.attributes_match)}/{summary.n}  "
        f"CPRF seed mismatch: "
        f"{sum(1 for r in summary.results if not r.cprf_seed_match)}/{summary.n}"
    )
    if cfg.wm_bit_redundancy > 1:
        print(
            f"  majority vote corrections={sum(r.majority_vote_corrections for r in summary.results)}  "
            f"regressions={sum(r.majority_vote_regressions for r in summary.results)}  "
            f"ties={sum(r.tie_votes for r in summary.results)}"
        )

    print()
    print("-- PRC detect scenarios --")
    print(
        f"  oracle_both={sum(r.detect_oracle_both for r in summary.results)}/{n}  "
        f"oracle_key={sum(r.detect_oracle_key for r in summary.results)}/{n}  "
        f"oracle_bits={sum(r.detect_oracle_bits for r in summary.results)}/{n}  "
        f"actual={sum(r.detect_actual_master for r in summary.results)}/{n}"
    )

    ber_rows: list[list[str]] = []
    for r in summary.results:
        ber_rows.append(
            [
                str(r.index + 1),
                _prompt_preview(r.prompt, 36),
                f"{r.channel_ber_from_ids:.1f}",
                f"{r.channel_ber_from_text:.1f}",
                f"{r.logical_ber:.1f}",
                f"{r.end_to_end_ber_master:.1f}",
                str(r.retokenization_mismatch_count),
                _yn(r.attributes_match),
                _yn(r.cprf_seed_match),
                _yn(r.detect_oracle_key),
                _yn(r.detect_actual_master),
            ]
        )

    benchmark_io.print_plain_table(
        title="Per-prompt BER and detect flags",
        headers=["#", "prompt", "ch_ids", "ch_txt", "logic", "e2e", "retok", "attrs", "seed", "o_key", "actual"],
        widths=[3, 36, 7, 7, 7, 7, 6, 6, 5, 6, 7],
        aligns=[">", "<", ">", ">", ">", ">", ">", ">", ">", ">", ">"],
        rows=ber_rows,
    )

    flagged = [r for r in summary.results if r.notes]
    if flagged:
        print()
        print("-- Insights --")
        for r in flagged:
            print(f"  #{r.index + 1} {_prompt_preview(r.prompt, 40)}")
            for note in r.notes:
                print(f"    - {note}")

    if verbose and summary.results:
        print()
        print("-- Verbose: channel / logical error indices --")
        for r in summary.results:
            print(
                f"  #{r.index + 1} channel[{len(r.channel_error_indices)}]: "
                f"{r.channel_error_indices[:24]}"
            )
            print(
                f"     logical[{len(r.logical_error_indices)}]: "
                f"{r.logical_error_indices[:24]}"
            )

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decompose watermark BER by error source.")
    p.add_argument("--prompts", type=Path, help="One prompt per line.")
    p.add_argument("--modulus", type=int, default=1024)
    p.add_argument("--code-length", type=int, default=100)
    p.add_argument("--wm-bit-redundancy", type=int, default=1)
    p.add_argument(
        "--burn-in-tokens",
        type=int,
        default=100,
        help="Unwatermarked warm-up tokens before the channel payload (default: 100).",
    )
    p.add_argument("--model-id", default=None)
    p.add_argument(
        "--torch-compile",
        action="store_true",
        help="Apply torch.compile to the LM after load (first forwards are slower).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Disable the progress bar.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print channel and logical bit error indices per prompt.",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write JSON results for benchmark_plot.py.",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    for name in ("httpx", "httpcore", "huggingface_hub", "urllib3", "text_attributes"):
        logging.getLogger(name).setLevel(logging.WARNING)

    args = _parse_args(argv)
    cfg = BerDiagConfig(
        modulus=args.modulus,
        code_length=args.code_length,
        wm_bit_redundancy=args.wm_bit_redundancy,
        burn_in_tokens=args.burn_in_tokens,
        model_id=args.model_id,
        torch_compile=True if args.torch_compile else None,
        quiet=bool(args.quiet),
    )
    summary = run_ber_diagnostics(_load_prompts(args.prompts), cfg)
    print_ber_diagnostics(summary, verbose=args.verbose)
    if args.output is not None:
        out = benchmark_io.save_ber_diagnostics(
            args.output,
            summary=summary,
            llm_model_id=cfg.model_id,
        )
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
