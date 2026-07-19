"""
Batch benchmark: run the ``app.py`` watermark protocol over a prompt list, repeating
each prompt a configurable number of times. Reports per-prompt aggregates and a
combined summary across all runs.

Run:
  uv run python benchmark_watermark.py
  uv run python benchmark_watermark.py --prompts prompts.txt --repeats 3 --code-length 100
"""

from __future__ import annotations

import argparse
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import benchmark_io
import model
import prc
import randrecover
import text_attributes
import watermarking as wm

from text_attributes import (
    SCORE_CUTOFF,
    VOCABULARY,
    active_labels_from_attributes,
    derive_attributes,
)

DEFAULT_PROMPTS: tuple[str, ...] = (
    "Explain how software has transformed the art world.",
    "Explain the economic nuance and impact of Drake Maye during his college football career at North Carolina.",
    "Explain how software has transformed the practice of medicine.",
    "Explain how stem cell therapy is being used in regenerative medicine.",
    "Explain the economic effects of raising the minimum wage on employment and businesses.",
    "Explain how Surrealist artists used dream imagery to challenge reality and logic.",
    "Break down the most influential software breakthroughs in history.",
    "Explain the role of strategy and teamwork in achieving success in sports",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    modulus: int = 1024
    code_length: int = 100
    wm_bit_redundancy: int = 7
    burn_in_tokens: int = 100
    model_id: str | None = None
    repeats_per_prompt: int = 1
    #: ``None`` leaves ``model.TORCH_COMPILE`` unchanged (Colab shared settings).
    torch_compile: bool | None = None
    quiet: bool = False


@dataclass
class ProtocolRunResult:
    prompt_index: int
    repeat_index: int
    prompt: str
    all_ok: bool
    master_detect_ok: bool
    unconstrained_detect_ok: bool
    neg_control_pass: bool
    cprf_ok: bool
    attributes_match: bool
    active_labels_match: bool
    master_ber_percent: float
    natural_partition_choices: int
    retok_replacements: int
    recovery_ids_aligned: bool
    label_policy_ok: dict[str, bool] = field(default_factory=dict)
    encode_active: list[str] = field(default_factory=list)
    verify_active: list[str] = field(default_factory=list)
    seconds_baseline_gen: float = 0.0
    seconds_watermarked_gen: float = 0.0
    n_tokens_baseline: int = 0
    n_tokens_watermarked: int = 0
    seconds_issue_keys: float = 0.0
    seconds_detect_total: float = 0.0
    baseline_text: str = ""


def _ber_percent(secret: list[int], recovered: list[int]) -> float:
    n = len(secret)
    if n == 0:
        return 0.0
    m = min(len(recovered), n)
    errs = sum(1 for i in range(m) if int(secret[i]) != int(recovered[i]))
    errs += abs(n - len(recovered))
    return 100.0 * errs / n


def _cprf_seeds_match(sk: object, dk: object, attributes: Sequence[int]) -> bool:
    return sk.eval(list(attributes)) == dk.c_eval(list(attributes))


def _cprf_agreement_ok(
    sk: object,
    dk_open: object,
    dk_by_word: dict[str, object],
    verify_attributes: Sequence[int],
    active_set: set[str],
) -> bool:
    if not _cprf_seeds_match(sk, dk_open, verify_attributes):
        return False
    for w in VOCABULARY:
        expect_match = w in active_set
        sm = _cprf_seeds_match(sk, dk_by_word[w], verify_attributes)
        if sm != expect_match:
            return False
    return True


def _prompt_preview(prompt: str, max_chars: int = 56) -> str:
    if len(prompt) <= max_chars:
        return prompt
    return prompt[: max_chars - 3].rstrip() + "..."


def _configure_watermarking(config: BenchmarkConfig) -> None:
    wm.SECURITY_PARAM = config.code_length
    prc.set_code_length(config.code_length)
    wm.WM_BIT_REDUNDANCY = config.wm_bit_redundancy
    wm.BURN_IN_TOKENS = config.burn_in_tokens
    kwargs: dict = {}
    if config.model_id:
        kwargs["model_id"] = config.model_id
    if config.torch_compile is not None:
        kwargs["torch_compile"] = config.torch_compile
    if kwargs:
        model.configure(**kwargs)


def _load_prompts(path: Path | None) -> list[str]:
    if path is None:
        return list(DEFAULT_PROMPTS)
    lines = path.read_text(encoding="utf-8").splitlines()
    prompts = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


@dataclass
class _GeneratedProtocol:
    prompt_index: int
    repeat_index: int
    prompt: str
    wm_text: str
    baseline_text: str
    encode_attributes: list
    secret: list[int]
    natural_partition_choices: int
    retok_replacements: int
    recovery_ids_aligned: bool
    seconds_baseline_gen: float
    seconds_watermarked_gen: float
    n_tokens_baseline: int
    n_tokens_watermarked: int


def _generate_protocol_once(
    sk: object,
    *,
    prompt_index: int,
    repeat_index: int,
    prompt: str,
    baseline_text: str | None = None,
    baseline_n_tokens: int | None = None,
) -> _GeneratedProtocol:
    out = wm.generate(
        sk,
        prompt,
        baseline_text=baseline_text,
        baseline_n_tokens=baseline_n_tokens,
    )
    generated = _GeneratedProtocol(
        prompt_index=prompt_index,
        repeat_index=repeat_index,
        prompt=prompt,
        wm_text=str(out["generated_text_wm"]),
        baseline_text=str(out["baseline_text"]),
        encode_attributes=list(out["attributes"]),
        secret=list(out["prc_secret_bits"]),
        natural_partition_choices=int(out.get("natural_partition_choices", 0)),
        retok_replacements=int(out.get("retok_replacements", 0)),
        recovery_ids_aligned=bool(out.get("recovery_ids_aligned", False)),
        seconds_baseline_gen=float(out["seconds_baseline_gen"]),
        seconds_watermarked_gen=float(out["seconds_watermarked_gen"]),
        n_tokens_baseline=int(out["n_tokens_baseline"]),
        n_tokens_watermarked=int(out["n_tokens_watermarked"]),
    )
    del out  # integrity: recovery must not see privileged generation metadata
    return generated


def _score_protocol_once(
    sk: object,
    dk_open: object,
    dk_by_word: dict[str, object],
    generated: _GeneratedProtocol,
    recovered_bits: Sequence[int],
    *,
    recover_seconds: float = 0.0,
    neg_control_model_bundle: tuple[object, object, str] | None = None,
) -> ProtocolRunResult:
    """Score one generated trial with integrity-path recovered channel bits."""
    all_ok = True
    wm_text = generated.wm_text
    encode_attributes = generated.encode_attributes
    secret = generated.secret
    recovered_wm = list(recovered_bits)

    verify_attributes = derive_attributes(wm_text, sk.modulus, log_scores=False)
    encode_active = active_labels_from_attributes(encode_attributes, sk.modulus)
    verify_active = active_labels_from_attributes(verify_attributes, sk.modulus)
    attributes_match = list(encode_attributes) == list(verify_attributes)
    active_labels_match = encode_active == verify_active
    all_ok &= attributes_match

    t0 = time.perf_counter()
    cprf_ok = _cprf_agreement_ok(
        sk, dk_open, dk_by_word, verify_attributes, set(verify_active)
    )
    t_keys = time.perf_counter() - t0
    all_ok &= cprf_ok

    active_set = set(verify_active)
    label_policy_ok: dict[str, bool] = {}

    t0 = time.perf_counter()
    m_ok, m_bits = wm.master_detect(sk, wm_text, recovered_bits=recovered_wm)
    t_m = (time.perf_counter() - t0) + float(recover_seconds)
    master_ber = _ber_percent(secret, m_bits)
    all_ok &= bool(m_ok)

    t0 = time.perf_counter()
    u_ok, _ = wm.detect(dk_open, wm_text, recovered_bits=recovered_wm)
    t_u = time.perf_counter() - t0
    all_ok &= u_ok is True

    det_total = t_m + t_u
    for w in VOCABULARY:
        expect_ok = w in active_set
        t0 = time.perf_counter()
        w_ok, _ = wm.detect(dk_by_word[w], wm_text, recovered_bits=recovered_wm)
        det_total += time.perf_counter() - t0
        ok_w = bool(w_ok) == bool(expect_ok)
        label_policy_ok[w] = ok_w
        all_ok &= ok_w

    if neg_control_model_bundle is None:
        neg_control_model_bundle = model.load()
    m, tok, device = neg_control_model_bundle
    wrong = randrecover.negative_control_transcript_like(
        wm_text,
        tok,
        device,
        n_bits=wm.SECURITY_PARAM * wm.WM_BIT_REDUNDANCY,
        model=m,
    )
    t0 = time.perf_counter()
    recovered_neg = randrecover.uncorrelated_bits_from_text(
        wrong, tok, n_bits=wm.SECURITY_PARAM * wm.WM_BIT_REDUNDANCY
    )
    neg_ok, _ = wm.master_detect(sk, wrong, recovered_bits=recovered_neg)
    det_total += time.perf_counter() - t0
    neg_control_pass = not bool(neg_ok)
    all_ok &= neg_control_pass

    return ProtocolRunResult(
        prompt_index=generated.prompt_index,
        repeat_index=generated.repeat_index,
        prompt=generated.prompt,
        all_ok=all_ok,
        master_detect_ok=bool(m_ok),
        unconstrained_detect_ok=bool(u_ok),
        neg_control_pass=neg_control_pass,
        cprf_ok=cprf_ok,
        attributes_match=attributes_match,
        active_labels_match=active_labels_match,
        master_ber_percent=master_ber,
        natural_partition_choices=generated.natural_partition_choices,
        retok_replacements=generated.retok_replacements,
        recovery_ids_aligned=generated.recovery_ids_aligned,
        label_policy_ok=label_policy_ok,
        encode_active=encode_active,
        verify_active=verify_active,
        seconds_baseline_gen=generated.seconds_baseline_gen,
        seconds_watermarked_gen=generated.seconds_watermarked_gen,
        n_tokens_baseline=generated.n_tokens_baseline,
        n_tokens_watermarked=generated.n_tokens_watermarked,
        seconds_issue_keys=t_keys,
        seconds_detect_total=det_total,
        baseline_text=generated.baseline_text,
    )


def run_benchmark(
    prompts: Sequence[str],
    config: BenchmarkConfig | None = None,
) -> list[ProtocolRunResult]:
    cfg = config or BenchmarkConfig()
    _configure_watermarking(cfg)

    model.load()
    sk = wm.setup(cfg.modulus)
    dk_open = wm.issue(sk, [])
    dk_by_word = {w: wm.issue(sk, [w]) for w in VOCABULARY}
    neg_bundle = model.load()

    trials = [
        (prompt_index, repeat_index, prompt)
        for prompt_index, prompt in enumerate(prompts)
        for repeat_index in range(cfg.repeats_per_prompt)
    ]

    # Independent baseline per trial (batched).
    pregen = benchmark_io.pregenerate_baselines(
        [prompt for _, _, prompt in trials],
        quiet=cfg.quiet,
        description="Baselines",
    )
    trial_baselines = list(pregen.texts)
    trial_baseline_tokens = list(pregen.token_counts)

    # Amortize so sum of per-trial baseline times equals pregen wall time.
    amortized = pregen.seconds / max(len(trials), 1)

    n = len(trials)
    generated: list[_GeneratedProtocol] = []
    t0 = time.perf_counter()
    with benchmark_io.progress_task(
        "Watermark generate",
        n,
        disable=cfg.quiet or n <= 0,
    ) as wm_bar:
        outs = wm.generate_batch(
            sk,
            [prompt for _, _, prompt in trials],
            baseline_texts=trial_baselines,
            baseline_n_tokens=trial_baseline_tokens,
            on_batch_done=lambda k: wm_bar.advance(k),
        )
    wm_wall = time.perf_counter() - t0
    wm_amortized = wm_wall / max(n, 1)

    for (prompt_index, repeat_index, prompt), baseline_text, n_bl_tok, out in zip(
        trials,
        trial_baselines,
        trial_baseline_tokens,
        outs,
    ):
        generated.append(
            _GeneratedProtocol(
                prompt_index=prompt_index,
                repeat_index=repeat_index,
                prompt=prompt,
                wm_text=str(out["generated_text_wm"]),
                baseline_text=str(baseline_text),
                encode_attributes=list(out["attributes"]),
                secret=list(out["prc_secret_bits"]),
                natural_partition_choices=int(out.get("natural_partition_choices", 0)),
                retok_replacements=int(out.get("retok_replacements", 0)),
                recovery_ids_aligned=bool(out.get("recovery_ids_aligned", False)),
                seconds_baseline_gen=float(amortized),
                seconds_watermarked_gen=float(wm_amortized),
                n_tokens_baseline=int(n_bl_tok),
                n_tokens_watermarked=int(out["n_tokens_watermarked"]),
            )
        )
        del out  # integrity: recovery must not see privileged generation metadata

    n = len(generated)
    t0 = time.perf_counter()
    with benchmark_io.progress_task(
        "Watermark recover",
        n,
        disable=cfg.quiet or n <= 0,
    ) as recover_bar:
        bits_list = wm.recover_channel_bits_batch(
            [g.wm_text for g in generated],
            on_batch_done=lambda k: recover_bar.advance(k),
        )
    recover_amortized = (time.perf_counter() - t0) / max(n, 1)

    results: list[ProtocolRunResult] = []
    for g, bits in zip(
        benchmark_io.iter_with_progress(
            generated,
            description="Watermark score",
            disable=cfg.quiet,
        ),
        bits_list,
    ):
        result = _score_protocol_once(
            sk,
            dk_open,
            dk_by_word,
            g,
            bits,
            recover_seconds=recover_amortized,
            neg_control_model_bundle=neg_bundle,
        )
        result.seconds_baseline_gen = amortized
        results.append(result)
    return results


@dataclass
class RunAggregate:
    n_runs: int
    all_ok_rate: float
    master_detect_rate: float
    unconstrained_detect_rate: float
    neg_control_pass_rate: float
    cprf_ok_rate: float
    attributes_match_rate: float
    active_labels_match_rate: float
    recovery_ids_aligned_rate: float
    avg_master_ber: float
    max_master_ber: float
    avg_natural_partition: float
    avg_replacements: float
    avg_baseline_gen_s: float
    avg_watermarked_gen_s: float
    avg_baseline_tok_s: float
    avg_watermarked_tok_s: float
    avg_detect_s: float
    label_policy_rates: dict[str, float] = field(default_factory=dict)


def _tokens_per_sec(n_tokens: float, seconds: float) -> float:
    if seconds <= 0.0 or n_tokens <= 0.0:
        return 0.0
    return float(n_tokens) / float(seconds)


def _aggregate_runs(runs: Sequence[ProtocolRunResult]) -> RunAggregate:
    n = len(runs)
    if n == 0:
        return RunAggregate(
            n_runs=0,
            all_ok_rate=0.0,
            master_detect_rate=0.0,
            unconstrained_detect_rate=0.0,
            neg_control_pass_rate=0.0,
            cprf_ok_rate=0.0,
            attributes_match_rate=0.0,
            active_labels_match_rate=0.0,
            recovery_ids_aligned_rate=0.0,
            avg_master_ber=0.0,
            max_master_ber=0.0,
            avg_natural_partition=0.0,
            avg_replacements=0.0,
            avg_baseline_gen_s=0.0,
            avg_watermarked_gen_s=0.0,
            avg_baseline_tok_s=0.0,
            avg_watermarked_tok_s=0.0,
            avg_detect_s=0.0,
        )

    def _rate(pred) -> float:
        return sum(1 for r in runs if pred(r)) / n

    label_policy_rates: dict[str, float] = {}
    for w in VOCABULARY:
        label_policy_rates[w] = sum(1 for r in runs if r.label_policy_ok.get(w, False)) / n

    total_bl_s = sum(r.seconds_baseline_gen for r in runs)
    total_wm_s = sum(r.seconds_watermarked_gen for r in runs)
    total_bl_tok = sum(r.n_tokens_baseline for r in runs)
    total_wm_tok = sum(r.n_tokens_watermarked for r in runs)

    return RunAggregate(
        n_runs=n,
        all_ok_rate=_rate(lambda r: r.all_ok),
        master_detect_rate=_rate(lambda r: r.master_detect_ok),
        unconstrained_detect_rate=_rate(lambda r: r.unconstrained_detect_ok),
        neg_control_pass_rate=_rate(lambda r: r.neg_control_pass),
        cprf_ok_rate=_rate(lambda r: r.cprf_ok),
        attributes_match_rate=_rate(lambda r: r.attributes_match),
        active_labels_match_rate=_rate(lambda r: r.active_labels_match),
        recovery_ids_aligned_rate=_rate(lambda r: r.recovery_ids_aligned),
        avg_master_ber=statistics.mean(r.master_ber_percent for r in runs),
        max_master_ber=max(r.master_ber_percent for r in runs),
        avg_natural_partition=statistics.mean(r.natural_partition_choices for r in runs),
        avg_replacements=statistics.mean(r.retok_replacements for r in runs),
        avg_baseline_gen_s=statistics.mean(r.seconds_baseline_gen for r in runs),
        avg_watermarked_gen_s=statistics.mean(r.seconds_watermarked_gen for r in runs),
        avg_baseline_tok_s=_tokens_per_sec(total_bl_tok, total_bl_s),
        avg_watermarked_tok_s=_tokens_per_sec(total_wm_tok, total_wm_s),
        avg_detect_s=statistics.mean(r.seconds_detect_total for r in runs),
        label_policy_rates=label_policy_rates,
    )


def _pct(rate: float) -> str:
    return f"{100.0 * rate:.1f}%"


def _print_config_block(
    prompts: Sequence[str],
    config: BenchmarkConfig,
    overall: RunAggregate,
) -> None:
    print()
    print("=== Watermark benchmark ===")
    print(f"prompts: {len(prompts)}  repeats/prompt: {config.repeats_per_prompt}  total runs: {overall.n_runs}")
    print(
        f"modulus: {config.modulus}  code_length: {config.code_length}  "
        f"wm_bit_redundancy: {config.wm_bit_redundancy}  "
        f"burn_in_tokens: {config.burn_in_tokens}  "
        f"channel: {config.code_length * config.wm_bit_redundancy} bits (depth-interleaved, balanced partition)"
    )
    print(f"LLM: {model.MODEL_ID}")
    print(
        f"sampling: temperature={model.SAMPLING['temperature']}  "
        f"top_p={model.SAMPLING['top_p']}  top_k={model.SAMPLING['top_k']}"
    )
    print(f"classifier: {text_attributes.get_scorer().model_id}")
    print(f"vocab |V|={len(VOCABULARY)}  score cutoff: {SCORE_CUTOFF:g}")
    print(
        f"inference_dtype: {model.inference_dtype_label()}  "
        f"torch_compile: {'yes' if model.TORCH_COMPILE else 'no'}"
    )


def _print_per_prompt_plain(
    prompts: Sequence[str],
    runs: Sequence[ProtocolRunResult],
) -> None:
    rows: list[list[str]] = []
    for prompt_index, prompt in enumerate(prompts):
        prompt_runs = [r for r in runs if r.prompt_index == prompt_index]
        agg = _aggregate_runs(prompt_runs)
        preview = _prompt_preview(prompt, 40)
        rows.append(
            [
                str(prompt_index + 1),
                preview,
                str(agg.n_runs),
                _pct(agg.all_ok_rate),
                _pct(agg.master_detect_rate),
                _pct(agg.unconstrained_detect_rate),
                _pct(agg.neg_control_pass_rate),
                _pct(agg.attributes_match_rate),
                f"{agg.avg_master_ber:.2f}",
                f"{agg.max_master_ber:.2f}",
                f"{agg.avg_natural_partition:.0f}",
                _pct(agg.recovery_ids_aligned_rate),
                f"{agg.avg_baseline_gen_s + agg.avg_watermarked_gen_s:.1f}s",
                f"{agg.avg_baseline_tok_s:.0f}/{agg.avg_watermarked_tok_s:.0f}",
                f"{agg.avg_detect_s:.1f}s",
            ]
        )
    benchmark_io.print_plain_table(
        title="Per-prompt aggregates",
        headers=[
            "#",
            "prompt",
            "runs",
            "pass",
            "master",
            "open",
            "decoy",
            "attrs",
            "BER_avg",
            "BER_max",
            "nat",
            "aligned",
            "gen",
            "tok/s",
            "det",
        ],
        widths=[3, 40, 5, 7, 7, 7, 7, 7, 8, 8, 4, 7, 6, 9, 6],
        aligns=[">", "<", ">", ">", ">", ">", ">", ">", ">", ">", ">", ">", ">", ">", ">"],
        rows=rows,
    )


def _print_aggregate_plain(agg: RunAggregate, *, title: str) -> None:
    print()
    print(title)
    print(f"  runs: {agg.n_runs}")
    print(f"  all checks pass: {_pct(agg.all_ok_rate)}")
    print(
        f"  master detect: {_pct(agg.master_detect_rate)}  "
        f"unconstrained: {_pct(agg.unconstrained_detect_rate)}  "
        f"decoy reject: {_pct(agg.neg_control_pass_rate)}"
    )
    print(
        f"  CPRF agreement: {_pct(agg.cprf_ok_rate)}  "
        f"attrs match: {_pct(agg.attributes_match_rate)}  "
        f"labels match: {_pct(agg.active_labels_match_rate)}"
    )
    print(f"  recovery_ids_aligned: {_pct(agg.recovery_ids_aligned_rate)}")
    print(
        f"  avg master BER: {agg.avg_master_ber:.2f}%  "
        f"max master BER: {agg.max_master_ber:.2f}%  "
        f"avg natural_part: {agg.avg_natural_partition:.1f}"
    )
    print(
        f"  avg replacements: {agg.avg_replacements:.1f}"
    )
    print(
        f"  avg gen: baseline={agg.avg_baseline_gen_s:.2f}s ({agg.avg_baseline_tok_s:.1f} tok/s)  "
        f"wm={agg.avg_watermarked_gen_s:.2f}s ({agg.avg_watermarked_tok_s:.1f} tok/s)  "
        f"avg detect={agg.avg_detect_s:.2f}s"
    )
    policy_bits = "  ".join(
        f"{w}={_pct(agg.label_policy_rates.get(w, 0.0))}" for w in VOCABULARY
    )
    print(f"  label policy: {policy_bits}")


def _print_per_run_plain(runs: Sequence[ProtocolRunResult]) -> None:
    rows: list[list[str]] = []
    for r in runs:
        rows.append(
            [
                str(r.prompt_index + 1),
                str(r.repeat_index + 1),
                "PASS" if r.all_ok else "FAIL",
                "ok" if r.master_detect_ok else "no",
                "ok" if r.unconstrained_detect_ok else "no",
                "ok" if r.neg_control_pass else "no",
                f"{r.master_ber_percent:.2f}",
                str(r.natural_partition_choices),
                "yes" if r.recovery_ids_aligned else "no",
            ]
        )
    benchmark_io.print_plain_table(
        title="Per-run detail",
        headers=["#", "rep", "check", "master", "open", "decoy", "BER%", "nat", "aligned"],
        widths=[3, 4, 6, 7, 5, 6, 7, 4, 7],
        aligns=[">", ">", ">", ">", ">", ">", ">", ">", ">"],
        rows=rows,
    )


def print_summary(
    prompts: Sequence[str],
    runs: list[ProtocolRunResult],
    config: BenchmarkConfig,
    *,
    verbose: bool = False,
) -> None:
    overall = _aggregate_runs(runs)

    _print_config_block(prompts, config, overall)
    _print_per_prompt_plain(prompts, runs)
    _print_aggregate_plain(overall, title="=== All prompts combined ===")

    attrs_match_runs = [r for r in runs if r.attributes_match]
    attrs_excluded = overall.n_runs - len(attrs_match_runs)
    attrs_only = _aggregate_runs(attrs_match_runs)
    _print_aggregate_plain(
        attrs_only,
        title=(
            f"=== All prompts combined (attributes match only; "
            f"excluded {attrs_excluded} of {overall.n_runs}) ==="
        ),
    )

    if verbose:
        _print_per_run_plain(runs)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch-run the app.py watermark protocol over multiple prompts."
    )
    p.add_argument(
        "--prompts",
        type=Path,
        help="Text file with one prompt per line (# comments and blank lines ignored).",
    )
    p.add_argument("--repeats", type=int, default=1, help="Runs per prompt (default: 1).")
    p.add_argument("--modulus", type=int, default=1024)
    p.add_argument("--code-length", type=int, default=100)
    p.add_argument("--wm-bit-redundancy", type=int, default=5)
    p.add_argument(
        "--burn-in-tokens",
        type=int,
        default=100,
        help="Unwatermarked warm-up tokens before the channel payload (default: 100).",
    )
    p.add_argument("--model-id", default=None, help="Override LM hub id (default from model.py).")
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
        "-v",
        action="store_true",
        help="Print per-run detail table in addition to aggregates.",
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
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    prompts = _load_prompts(args.prompts)
    config = BenchmarkConfig(
        modulus=args.modulus,
        code_length=args.code_length,
        wm_bit_redundancy=args.wm_bit_redundancy,
        burn_in_tokens=args.burn_in_tokens,
        model_id=args.model_id,
        repeats_per_prompt=args.repeats,
        torch_compile=True if args.torch_compile else None,
        quiet=bool(args.quiet),
    )

    runs = run_benchmark(prompts, config)
    print_summary(prompts, runs, config, verbose=args.verbose)

    if args.output is not None:
        out = benchmark_io.save_watermark_benchmark(
            args.output,
            config=config,
            prompts=prompts,
            runs=runs,
            llm_model_id=config.model_id,
        )
        print(f"Wrote {out}")

    overall = _aggregate_runs(runs)
    return 0 if overall.all_ok_rate == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
