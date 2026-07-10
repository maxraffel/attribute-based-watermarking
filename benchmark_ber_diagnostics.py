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
    model_id: str | None = None


@dataclass
class PartitionErrorBreakdown:
    """Mutually exclusive primary causes for text-recovery vs embedded channel mismatches."""

    total_text_channel_errors: int
    retokenization_mismatch: int
    stochastic_p_roll: int
    fallback_sampling: int
    enforced_anomaly: int
    stochastic_rolls_total: int
    stochastic_rolls_wrong_half: int


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
    partition_masks_match: bool
    channel_errors_at_generation: int
    channel_errors_at_recovery: int
    partition_error_breakdown: PartitionErrorBreakdown
    mapping_mismatches: int
    partition_cell_mismatches: int
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


def _logical_bits(raw: Sequence[int], code_length: int, redundancy: int) -> list[int]:
    need = code_length * redundancy
    padded = (list(raw) + [0] * need)[:need]
    out: list[int] = []
    for i in range(code_length):
        chunk = padded[i * redundancy : (i + 1) * redundancy]
        out.append(1 if 2 * sum(chunk) > redundancy else 0)
    return out


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
        chunk = padded[i * redundancy : (i + 1) * redundancy]
        if 2 * sum(chunk) == redundancy:
            tie_votes += 1
        embedded = secret_logical[i]
        recovered = raw_logical[i]
        chunk_err = any(
            int(channel_bits[i * redundancy + j]) != int(padded[i * redundancy + j])
            for j in range(redundancy)
            if i * redundancy + j < len(channel_bits)
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
    if cfg.model_id:
        model.configure(model_id=cfg.model_id)


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


def _analyze_partition_error_causes(
    channel_bits: Sequence[int],
    raw_from_ids: Sequence[int],
    raw_from_text: Sequence[int],
    channel_steps: Sequence[randrecover.WatermarkChannelStep],
) -> PartitionErrorBreakdown:
    """
    Classify each text-recovery channel error by primary cause.

    - ``retokenization_mismatch``: generation ids recover the embedded bit, but text does not
    - ``stochastic_p_roll``: generation already wrong because the ``2q`` roll followed mass ``p``
      instead of enforcing the secret bit
    - ``fallback_sampling``: generation used unmasked full-vocab fallback sampling
    - ``enforced_anomaly``: generation wrong despite enforcing the secret bit (unexpected)
    """
    steps_by_index = {s.bit_index: s for s in channel_steps}
    retok = stochastic = fallback = anomaly = 0
    stochastic_rolls_total = stochastic_rolls_wrong = 0

    for s in channel_steps:
        if not s.used_enforce_branch:
            stochastic_rolls_total += 1
            if s.bit_implied_by_token is not None and s.bit_implied_by_token != s.secret_bit:
                stochastic_rolls_wrong += 1

    text_errors = 0
    for k, embedded in enumerate(channel_bits):
        ids_bit = _bit_at(raw_from_ids, k)
        text_bit = _bit_at(raw_from_text, k)
        if text_bit is None or int(text_bit) == int(embedded):
            continue
        text_errors += 1
        ids_ok = ids_bit is not None and int(ids_bit) == int(embedded)
        if ids_ok:
            retok += 1
            continue
        step = steps_by_index.get(k)
        if step is not None and step.used_fallback:
            fallback += 1
        elif step is not None and not step.used_enforce_branch:
            stochastic += 1
        else:
            anomaly += 1

    return PartitionErrorBreakdown(
        total_text_channel_errors=text_errors,
        retokenization_mismatch=retok,
        stochastic_p_roll=stochastic,
        fallback_sampling=fallback,
        enforced_anomaly=anomaly,
        stochastic_rolls_total=stochastic_rolls_total,
        stochastic_rolls_wrong_half=stochastic_rolls_wrong,
    )


def diagnose_prompt(
    sk: object,
    prompt: str,
    index: int,
    cfg: BerDiagConfig,
) -> BerPromptDiagnostic:
    m, tok, device = model.load()
    out = wm.generate(sk, prompt)

    encode_attributes = list(out["attributes"])
    wm_text = out["generated_text_wm"]
    secret: list[int] = list(out["prc_secret_bits"])
    channel_bits: list[int] = list(out["wm_channel_bits"])
    partition_dim = int(out.get("partition_vocab_dim") or m.config.vocab_size)
    special_ids = set(getattr(tok, "all_special_ids", []))

    verify_attributes = derive_attributes(wm_text, sk.modulus, log_scores=False)
    encode_active = active_labels_from_attributes(encode_attributes, sk.modulus)
    verify_active = active_labels_from_attributes(verify_attributes, sk.modulus)

    prefix_n = min(len(VOCABULARY), len(encode_attributes), len(verify_attributes))
    prefix_diff = sum(
        1 for i in range(prefix_n) if int(encode_attributes[i]) != int(verify_attributes[i])
    )
    r_enc = sk.eval(encode_attributes)
    r_ver = sk.eval(verify_attributes)

    ids_suffix = out["input_ids_wm"][0].tolist()
    raw_from_ids, _ = randrecover.recover_bitstream(
        ids_suffix, partition_dim, device, special_ids
    )
    raw_from_text, _ = randrecover.recover_bitstream_from_text(
        wm_text, tok, device, model=m, partition_vocab_size=partition_dim
    )

    logical_from_text = _logical_bits(
        raw_from_text, cfg.code_length, cfg.wm_bit_redundancy
    )

    audit = randrecover.audit_partition_replication_tokenwise(out, tok, device, model=m)
    channel_steps: list[randrecover.WatermarkChannelStep] = list(out.get("channel_steps") or [])
    partition_breakdown = _analyze_partition_error_causes(
        channel_bits, raw_from_ids, raw_from_text, channel_steps
    )

    gen_errs = rec_errs = map_mismatches = cell_mismatches = 0
    for row in audit.rows:
        k = row.bit_index
        if k < len(channel_bits):
            emb = int(channel_bits[k])
            if (
                row.bit_implied_by_generation_mask is not None
                and row.bit_implied_by_generation_mask != emb
            ):
                gen_errs += 1
            if row.bit_from_recovery_pipeline is not None and row.bit_from_recovery_pipeline != emb:
                rec_errs += 1
        if row.mapping_entry_agrees_with_token_id is False:
            map_mismatches += 1
        cell_mismatches += row.num_vocab_cells_mismatched

    vote_corr, vote_reg, tie_votes = _majority_vote_analysis(
        channel_bits, raw_from_text, cfg.code_length, cfg.wm_bit_redundancy
    )

    detect_oracle_both = _prc_detect(r_enc, secret, cfg.code_length)
    detect_oracle_key = _prc_detect(r_enc, logical_from_text, cfg.code_length)
    detect_oracle_bits = _prc_detect(r_ver, secret, cfg.code_length)
    master_ok, master_bits = wm.master_detect(sk, wm_text)

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
    if not audit.all_partitions_recreated:
        notes.append("Generation vs recovery partition masks differ (vocab dim mismatch?)")
    if gen_errs > 0:
        notes.append(
            f"{gen_errs} tokens landed outside secret half at generation "
            f"(p-roll={partition_breakdown.stochastic_p_roll}, "
            f"fallback={partition_breakdown.fallback_sampling}, "
            f"anomaly={partition_breakdown.enforced_anomaly})"
        )
    if partition_breakdown.retokenization_mismatch > 0:
        notes.append(
            f"{partition_breakdown.retokenization_mismatch} channel errors from retokenization "
            f"(ids recovery correct, text recovery wrong)"
        )
    if partition_breakdown.stochastic_rolls_total > 0:
        notes.append(
            f"{partition_breakdown.stochastic_rolls_total} generation steps used follow-mass roll "
            f"({partition_breakdown.stochastic_rolls_wrong_half} landed on wrong half)"
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
        n_payload_tokens=audit.num_payload_tokens,
        channel_ber_from_ids=ch_ids,
        channel_ber_from_text=ch_text,
        retokenization_extra_ber=max(0.0, ch_text - ch_ids),
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
        partition_masks_match=audit.all_partitions_recreated,
        channel_errors_at_generation=gen_errs,
        channel_errors_at_recovery=rec_errs,
        partition_error_breakdown=partition_breakdown,
        mapping_mismatches=map_mismatches,
        partition_cell_mismatches=cell_mismatches,
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
    results = [diagnose_prompt(sk, p, i, cfg) for i, p in enumerate(prompts)]
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
    console: object | None = None,
    verbose: bool = False,
) -> None:
    del console
    cfg = summary.config

    print()
    print("=== BER diagnostics ===")
    print(f"prompts: {summary.n}")
    print(
        f"code_length: {cfg.code_length}  wm_bit_redundancy: {cfg.wm_bit_redundancy}  "
        f"modulus: {cfg.modulus}"
    )
    print(f"LLM: {model.MODEL_ID}")
    print(f"classifier: {text_attributes.get_scorer().model_id}")

    print()
    print("-- Stage BER (averages) --")
    print(
        f"  1. Channel from ids={summary._avg('channel_ber_from_ids'):.2f}%  "
        f"from text={summary._avg('channel_ber_from_text'):.2f}%  "
        f"(retokenize gap={summary._avg('retokenization_extra_ber'):.2f}%)"
    )
    print(f"  2. Logical PRC payload: {summary._avg('logical_ber'):.2f}%")
    print(f"  3. End-to-end (master path): {summary._avg('end_to_end_ber_master'):.2f}%")

    print()
    print("-- Partition error causes (text recovery vs embedded) --")
    print(
        f"  total text channel errors: "
        f"{sum(r.partition_error_breakdown.total_text_channel_errors for r in summary.results)}"
    )
    print(
        f"  retokenization mismatch (ids OK, text wrong): "
        f"{sum(r.partition_error_breakdown.retokenization_mismatch for r in summary.results)}"
    )
    print(
        f"  stochastic p-roll: "
        f"{sum(r.partition_error_breakdown.stochastic_p_roll for r in summary.results)}"
    )
    print(
        f"  fallback sampling: "
        f"{sum(r.partition_error_breakdown.fallback_sampling for r in summary.results)}"
    )
    print(
        f"  enforced anomaly: "
        f"{sum(r.partition_error_breakdown.enforced_anomaly for r in summary.results)}"
    )
    print(
        f"  follow-mass steps: "
        f"{sum(r.partition_error_breakdown.stochastic_rolls_total for r in summary.results)}  "
        f"wrong half: "
        f"{sum(r.partition_error_breakdown.stochastic_rolls_wrong_half for r in summary.results)}"
    )

    print()
    print("-- Error sources --")
    n = summary.n or 1
    print(
        f"  generation half-space misses: "
        f"{sum(r.channel_errors_at_generation for r in summary.results)} tokens"
    )
    print(
        f"  recovery bit mismatches: "
        f"{sum(r.channel_errors_at_recovery for r in summary.results)} tokens"
    )
    print(f"  retokenization mapping mismatches: {sum(r.mapping_mismatches for r in summary.results)}")
    print(f"  partition mask cell mismatches: {sum(r.partition_cell_mismatches for r in summary.results)}")
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
    err_rows: list[list[str]] = []
    for r in summary.results:
        pb = r.partition_error_breakdown
        ber_rows.append(
            [
                str(r.index + 1),
                _prompt_preview(r.prompt, 36),
                f"{r.channel_ber_from_ids:.1f}",
                f"{r.channel_ber_from_text:.1f}",
                f"{r.logical_ber:.1f}",
                f"{r.end_to_end_ber_master:.1f}",
                _yn(r.attributes_match),
                _yn(r.cprf_seed_match),
                _yn(r.detect_oracle_key),
                _yn(r.detect_actual_master),
            ]
        )
        err_rows.append(
            [
                str(r.index + 1),
                _prompt_preview(r.prompt, 36),
                str(pb.stochastic_p_roll),
                str(pb.retokenization_mismatch),
                str(r.channel_errors_at_generation),
                str(r.channel_errors_at_recovery),
            ]
        )

    benchmark_io.print_plain_table(
        title="Per-prompt BER and detect flags",
        headers=["#", "prompt", "ch_ids", "ch_txt", "logic", "e2e", "attrs", "seed", "o_key", "actual"],
        widths=[3, 36, 7, 7, 7, 7, 6, 5, 6, 7],
        aligns=[">", "<", ">", ">", ">", ">", ">", ">", ">", ">"],
        rows=ber_rows,
    )
    benchmark_io.print_plain_table(
        title="Per-prompt error counts",
        headers=["#", "prompt", "p_roll", "retok", "gen_err", "rec_err"],
        widths=[3, 36, 7, 7, 8, 8],
        aligns=[">", "<", ">", ">", ">", ">"],
        rows=err_rows,
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
    p.add_argument("--model-id", default=None)
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
        model_id=args.model_id,
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
