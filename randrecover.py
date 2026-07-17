from __future__ import annotations

import hashlib
import inspect
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList

_REPO_ROOT = Path(__file__).resolve().parent
_PACKAGED_WEIGHTS_DIR = _REPO_ROOT / "data" / "partition_weights"
_WEIGHTS_CACHE_DIR = _REPO_ROOT / ".cache"

# Normalized unigram weights over the partition vocabulary (CPU float32).
_PARTITION_WEIGHTS: torch.Tensor | None = None
_PARTITION_WEIGHTS_KEY: str | None = None


def _partition_seed(seed_index: int) -> int:
    return int(hashlib.md5(f"{seed_index}".encode()).hexdigest()[:16], 16)


def partition_weights_key(
    tokenizer: AutoTokenizer | None = None,
    vocab_size: int = 0,
    *,
    tokenizer_id: str | None = None,
) -> str:
    """Stable filename stem for packaged / cached partition weights."""
    name = tokenizer_id
    if not name and tokenizer is not None:
        name = getattr(tokenizer, "name_or_path", None) or type(tokenizer).__name__
    if not name:
        name = "unknown"
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))
    return f"{safe}_v{int(vocab_size)}"


def packaged_partition_weights_path(key: str) -> Path:
    return _PACKAGED_WEIGHTS_DIR / f"{key}.pt"


def _weights_cache_path(key: str) -> Path:
    return _WEIGHTS_CACHE_DIR / f"partition_weights_{key}.pt"


def configure_partition_weights(
    weights: torch.Tensor,
    *,
    key: str | None = None,
) -> torch.Tensor:
    """Install normalized non-negative partition weights (copied to CPU)."""
    global _PARTITION_WEIGHTS, _PARTITION_WEIGHTS_KEY
    if weights.ndim != 1:
        raise ValueError(f"partition weights must be 1-D, got shape {tuple(weights.shape)}")
    w = weights.detach().to(dtype=torch.float32, device="cpu").clamp(min=0.0)
    total = float(w.sum().item())
    if total <= 0.0:
        raise ValueError("partition weights must sum to a positive value")
    _PARTITION_WEIGHTS = w / total
    _PARTITION_WEIGHTS_KEY = key
    return _PARTITION_WEIGHTS


def clear_partition_weights() -> None:
    global _PARTITION_WEIGHTS, _PARTITION_WEIGHTS_KEY
    _PARTITION_WEIGHTS = None
    _PARTITION_WEIGHTS_KEY = None


def get_partition_weights(vocab_size: int | None = None) -> torch.Tensor | None:
    """Return the active weight vector, optionally checking ``vocab_size``."""
    if _PARTITION_WEIGHTS is None:
        return None
    if vocab_size is not None and int(_PARTITION_WEIGHTS.numel()) != int(vocab_size):
        raise ValueError(
            f"partition weights length {_PARTITION_WEIGHTS.numel()} "
            f"!= vocab_size {vocab_size}"
        )
    return _PARTITION_WEIGHTS


def load_partition_weights_file(path: Path | str) -> Tuple[torch.Tensor, dict[str, Any]]:
    """Load a packaged artifact (raw tensor or ``{weights, ...}`` dict)."""
    p = Path(path)
    obj = torch.load(p, map_location="cpu", weights_only=False)
    meta: dict[str, Any] = {"path": str(p)}
    if isinstance(obj, torch.Tensor):
        weights = obj
    elif isinstance(obj, Mapping) and "weights" in obj:
        weights = obj["weights"]
        meta.update({k: v for k, v in obj.items() if k != "weights"})
    else:
        raise ValueError(f"unrecognized partition weights file format at {p}")
    if not isinstance(weights, torch.Tensor) or weights.ndim != 1:
        raise ValueError(f"partition weights in {p} must be a 1-D tensor")
    return weights.to(dtype=torch.float32), meta


def save_partition_weights_artifact(
    path: Path | str,
    counts_or_weights: torch.Tensor,
    *,
    tokenizer_id: str,
    vocab_size: int,
    corpus: str,
    num_tokens_counted: int,
    num_documents: int = 0,
    laplace: float = 1.0,
    key: str | None = None,
    already_normalized: bool = False,
) -> Path:
    """Write a packaged ``.pt`` dict; returns the output path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = counts_or_weights.detach().to(dtype=torch.float32, device="cpu").clamp(min=0.0)
    if int(raw.numel()) != int(vocab_size):
        raise ValueError(
            f"counts length {raw.numel()} != vocab_size {vocab_size}"
        )
    if already_normalized:
        weights = raw / max(float(raw.sum().item()), 1e-12)
    else:
        smoothed = raw + float(laplace)
        weights = smoothed / smoothed.sum()
    payload = {
        "weights": weights,
        "tokenizer_id": tokenizer_id,
        "vocab_size": int(vocab_size),
        "corpus": corpus,
        "num_tokens_counted": int(num_tokens_counted),
        "num_documents": int(num_documents),
        "laplace": float(laplace),
        "key": key or partition_weights_key(vocab_size=vocab_size, tokenizer_id=tokenizer_id),
    }
    torch.save(payload, p)
    return p


def count_unigram_tokens(
    tokenizer: AutoTokenizer,
    vocab_size: int,
    texts: Iterable[str],
    *,
    max_tokens: int | None = None,
    chunk_chars: int = 2000,
    progress_every: int = 250_000,
) -> Tuple[torch.Tensor, int, int]:
    """
    Count tokenizer ids over ``texts``.

    Returns ``(counts, num_tokens, num_documents)`` with no smoothing applied.
    """
    counts = torch.zeros(int(vocab_size), dtype=torch.float32)
    n_tokens = 0
    n_docs = 0
    v = int(vocab_size)
    for text in texts:
        n_docs += 1
        if not text or not str(text).strip():
            continue
        s = str(text)
        for start in range(0, len(s), chunk_chars):
            chunk = s[start : start + chunk_chars]
            if not chunk.strip():
                continue
            enc = tokenizer(chunk, add_special_tokens=False)
            ids = enc["input_ids"]
            if ids and isinstance(ids[0], list):
                ids = [t for row in ids for t in row]
            for tid in ids:
                t = int(tid)
                if 0 <= t < v:
                    counts[t] += 1.0
                    n_tokens += 1
                    if max_tokens is not None and n_tokens >= max_tokens:
                        if progress_every and n_tokens % progress_every < len(ids):
                            print(f"  counted {n_tokens} tokens...", flush=True)
                        return counts, n_tokens, n_docs
            if progress_every and n_tokens > 0 and n_tokens % progress_every < chunk_chars:
                # cheap progress: when crossing multiples is hard; print periodically by docs
                pass
        if progress_every and n_docs % 200 == 0:
            print(f"  docs={n_docs} tokens={n_tokens}", flush=True)
        if max_tokens is not None and n_tokens >= max_tokens:
            break
    return counts, n_tokens, n_docs


def ensure_partition_weights(
    vocab_size: int,
    tokenizer: AutoTokenizer | None = None,
    *,
    tokenizer_id: str | None = None,
    weights_path: Path | str | None = None,
) -> torch.Tensor:
    """
    Load module-level partition weights for ``vocab_size`` (no corpus recount).

    Resolution order:
      1. Already configured in-memory weights for this vocab / key
      2. Explicit ``weights_path`` if given
      3. Packaged repo artifact ``data/partition_weights/<key>.pt``
      4. Legacy ``.cache/partition_weights_<key>.pt`` (local only)

    Raises ``FileNotFoundError`` if nothing is found — run
    ``scripts/build_partition_weights.py`` once and commit the artifact.
    """
    global _PARTITION_WEIGHTS, _PARTITION_WEIGHTS_KEY
    v = int(vocab_size)
    key = partition_weights_key(tokenizer, v, tokenizer_id=tokenizer_id)

    if _PARTITION_WEIGHTS is not None and int(_PARTITION_WEIGHTS.numel()) == v:
        if _PARTITION_WEIGHTS_KEY in (None, key):
            return _PARTITION_WEIGHTS

    candidates: list[Path] = []
    if weights_path is not None:
        candidates.append(Path(weights_path))
    candidates.append(packaged_partition_weights_path(key))
    candidates.append(_weights_cache_path(key))

    for path in candidates:
        if not path.is_file():
            continue
        weights, meta = load_partition_weights_file(path)
        if int(weights.numel()) != v:
            raise ValueError(
                f"partition weights at {path} have length {weights.numel()}, "
                f"expected vocab_size={v}"
            )
        file_key = meta.get("key") if isinstance(meta.get("key"), str) else key
        configure_partition_weights(weights, key=str(file_key))
        return _PARTITION_WEIGHTS  # type: ignore[return-value]

    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"No packaged partition weights for key={key!r} (vocab_size={v}). "
        f"Searched: {searched}. "
        f"Build once with: uv run python scripts/build_partition_weights.py "
        f"--model-id {tokenizer_id or key.rsplit('_v', 1)[0]} "
        f"--vocab-size {v}"
    )


def _partition_order_and_cutoff(
    vocab_size: int, seed_index: int
) -> Tuple[torch.Tensor, int]:
    """Seeded random token order and inclusive cutoff index for set A."""
    v = int(vocab_size)
    weights = get_partition_weights(v)
    if weights is None:
        raise RuntimeError(
            "partition weights are not configured; call ensure_partition_weights(...) "
            "before building partitions (generate/recover paths do this automatically)"
        )
    # randperm with a CUDA generator is unreliable across devices; permute on CPU.
    g = torch.Generator(device="cpu")
    g.manual_seed(_partition_seed(seed_index))
    order = torch.randperm(v, generator=g, device="cpu")
    cum = torch.cumsum(weights[order], dim=0)
    # Shortest prefix with mass >= half, then prefer the nearer of that prefix or
    # the one before it so a single heavy token cannot overshoot too far.
    half = 0.5 * float(cum[-1].item())
    cutoff = int(torch.searchsorted(cum, torch.tensor(half, dtype=cum.dtype)).item())
    if cutoff >= v:
        cutoff = v - 1
    if cutoff > 0:
        over = float(cum[cutoff].item()) - half
        under = half - float(cum[cutoff - 1].item())
        if over > under:
            cutoff -= 1
    return order, cutoff


def get_vectorized_partition(vocab_size: int, device: str, seed_index: int) -> torch.BoolTensor:
    """
    Deterministic boolean mask over the vocabulary for one bit index.

    Membership is a seeded random permutation of token ids; set A is the shortest
    prefix of that order whose static unigram weight is at least half the total.
    That keeps masks looking random while balancing estimated probability mass.
    """
    order, cutoff = _partition_order_and_cutoff(vocab_size, seed_index)
    mask = torch.zeros(int(vocab_size), dtype=torch.bool, device="cpu")
    mask[order[: cutoff + 1]] = True
    dev = torch.device(device) if isinstance(device, str) else device
    return mask.to(device=dev)


def partition_bit_for_token(vocab_size: int, seed_index: int, token_id: int) -> int:
    """
    Recover the watermark bit implied by ``token_id`` at ``seed_index``.

    Same partition as ``get_vectorized_partition``, but only answers membership for
    one id (no full boolean mask allocation — used by detection).
    """
    v = int(vocab_size)
    tid = int(token_id)
    if tid < 0 or tid >= v:
        return 1
    order, cutoff = _partition_order_and_cutoff(v, seed_index)
    pos = int((order == tid).nonzero(as_tuple=True)[0].item())
    return 0 if pos <= cutoff else 1


def resolve_partition_vocab_size_for_recovery(
    tokenizer: AutoTokenizer,
    *,
    model: torch.nn.Module | None = None,
    partition_vocab_size: int | None = None,
) -> int:
    v = partition_vocab_size
    if v is None and model is not None:
        cfg = getattr(model, "config", None)
        if cfg is None:
            raise ValueError("model has no config; pass partition_vocab_size explicitly")
        txt = getattr(cfg, "text_config", None)
        base_vocab = getattr(txt or cfg, "vocab_size", None)
        if base_vocab is None:
            raise ValueError("model.config has no vocab_size; pass partition_vocab_size explicitly")
        v = int(base_vocab)
    if v is None:
        v = int(getattr(tokenizer, "vocab_size", len(tokenizer)))
    return int(v)


def _tokenizer_pad_token_id(tokenizer: AutoTokenizer) -> int:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        raise ValueError("tokenizer has no pad_token_id or eos_token_id for generation")
    return int(pad_id)


def encode_prompt_for_generation(
    tokenizer: AutoTokenizer,
    user_prompt: str,
    device: str | torch.device,
) -> Dict[str, torch.Tensor]:
    dev = torch.device(device) if isinstance(device, str) else device
    if getattr(tokenizer, "chat_template", None) is None:
        batch = tokenizer(user_prompt, return_tensors="pt")
    else:
        messages = [{"role": "user", "content": user_prompt}]
        out = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        if isinstance(out, torch.Tensor):
            input_ids = out
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            batch = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids, dtype=torch.long),
            }
        else:
            batch = dict(out)
            if "attention_mask" not in batch and "input_ids" in batch:
                batch["attention_mask"] = torch.ones_like(batch["input_ids"], dtype=torch.long)
    return {k: v.to(dev) for k, v in batch.items()}


def _prepare_sampling_logits_processor_bundle(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt_inputs: Dict[str, torch.Tensor],
    *,
    decode_horizon: int,
    generation_extra: Dict[str, Any] | None = None,
) -> Tuple[Any, LogitsProcessorList]:
    device = prompt_inputs["input_ids"].device
    pad_id = _tokenizer_pad_token_id(tokenizer)
    prep_kwargs: Dict[str, Any] = {
        **prompt_inputs,
        "max_new_tokens": decode_horizon,
        "do_sample": True,
        "pad_token_id": pad_id,
        "use_cache": True,
    }
    if generation_extra:
        prep_kwargs.update(generation_extra)

    generation_config, model_kwargs = model._prepare_generation_config(None, **prep_kwargs)
    inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(
        None,
        generation_config.bos_token_id,
        model_kwargs,
    )
    kwargs_has_attention_mask = model_kwargs.get("attention_mask") is not None
    model._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

    accepts_attention_mask = "attention_mask" in set(inspect.signature(model.forward).parameters.keys())
    if (
        not kwargs_has_attention_mask
        and accepts_attention_mask
        and not model.config.is_encoder_decoder
    ):
        model_kwargs["attention_mask"] = model._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config, model_kwargs
        )

    if model.config.is_encoder_decoder:
        raise NotImplementedError("encoder-decoder models are not supported for watermarked incremental decode")
    if model_input_name != "input_ids":
        model_kwargs.pop("input_ids")
    input_ids_seq_length = int(inputs_tensor.shape[1])

    has_default_max_length = (
        prep_kwargs.get("max_length") is None
        and model.generation_config.max_length is None
    )
    has_default_min_length = (
        prep_kwargs.get("min_length") is None
        and model.generation_config.min_length is None
    )
    generation_config = model._prepare_generated_length(
        generation_config=generation_config,
        has_default_max_length=has_default_max_length,
        has_default_min_length=has_default_min_length,
        model_input_name=model_input_name,
        input_ids_length=input_ids_seq_length,
        inputs_tensor=inputs_tensor,
    )

    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=inputs_tensor,
        device=inputs_tensor.device,
        model_kwargs=model_kwargs,
    )
    return generation_config, logits_processor


def _scores_to_next_token_probs(
    logits_last: torch.Tensor,
    input_ids_for_proc: torch.Tensor,
    logits_processor: LogitsProcessorList,
) -> torch.Tensor:
    next_logits = logits_last.unsqueeze(0).to(copy=True, dtype=torch.float32, device=input_ids_for_proc.device)
    next_scores = logits_processor(input_ids_for_proc, next_logits)
    return F.softmax(next_scores, dim=-1).squeeze(0)


def _choose_watermark_partition_half(secret_bit: int, p: float, q: float) -> Tuple[bool, bool]:
    """Return ``(choose_set_A, used_enforce_branch)``."""
    if random.random() < (2 * q):
        return (secret_bit == 0), True
    return (p > 0.5), False


def _mask_disallowed_half_probs(
    probs: torch.Tensor,
    mask_A: torch.BoolTensor,
    *,
    choose_set_A: bool,
) -> torch.Tensor:
    if choose_set_A:
        return torch.where(mask_A, probs, torch.zeros_like(probs))
    return torch.where(mask_A, torch.zeros_like(probs), probs)


def _watermark_step_probs(
    logits_last: torch.Tensor,
    input_ids_for_proc: torch.Tensor,
    logits_processor: LogitsProcessorList,
    mask_A: torch.BoolTensor,
    secret_bit: int,
) -> Tuple[torch.Tensor, bool]:
    base_probs = _scores_to_next_token_probs(logits_last, input_ids_for_proc, logits_processor)
    p = float(base_probs[mask_A].sum().item())
    q = min(p, 1.0 - p)
    choose_A, used_enforce = _choose_watermark_partition_half(int(secret_bit), p, q)
    return _mask_disallowed_half_probs(base_probs, mask_A, choose_set_A=choose_A), used_enforce


def _decode_suffix_for_recovery(tokenizer: AutoTokenizer, suffix_ids: List[int]) -> str:
    """Same string ``recover_bitstream_from_text`` will tokenize (suffix only, skip specials)."""
    return tokenizer.decode(suffix_ids, skip_special_tokens=True)


def _recovery_non_special_token_ids(
    tokenizer: AutoTokenizer,
    suffix_ids: List[int],
    special_ids: set[int],
) -> List[int]:
    """Non-special token ids from the detect-time decode→tokenize path."""
    suffix_text = _decode_suffix_for_recovery(tokenizer, suffix_ids)
    retok_ids = tokenizer(suffix_text, return_tensors="pt")["input_ids"][0].tolist()
    return [t for t in retok_ids if t not in special_ids]


def _update_recovery_state_after_token(
    *,
    prev_retok_ids_ns: List[int],
    retok_ids_ns: List[int],
    sacrificed_bits: int,
) -> Tuple[List[int], int, int, int, bool]:
    """
    Advance the recovery token snapshot after one generated token.

    The next partition index is always ``len(prev_retok_ids_ns)`` — the slot
    recovery will assign to the next committed payload token. Never advance that
    index independently of how the recovery stream actually changed.
    """
    if retok_ids_ns == prev_retok_ids_ns:
        return prev_retok_ids_ns, sacrificed_bits, 0, 0, True

    prev_len = len(prev_retok_ids_ns)
    new_len = len(retok_ids_ns)

    if new_len > prev_len:
        delta = new_len - prev_len
        sacrificed_delta = delta - 1
        if sacrificed_delta:
            sacrificed_bits += sacrificed_delta
        return list(retok_ids_ns), sacrificed_bits, 0, sacrificed_delta, False

    if new_len == prev_len:
        return list(retok_ids_ns), sacrificed_bits, 1, 0, False

    sacrificed_delta = prev_len - new_len
    sacrificed_bits += sacrificed_delta
    return list(retok_ids_ns), sacrificed_bits, 0, sacrificed_delta, False


def generate_with_watermark(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    secret_bitstream: List[int],
    device: str = "cpu",
    *,
    generation_extra: Dict[str, Any] | None = None,
    tokenizer_id: str | None = None,
) -> Dict:
    """Sample tokens until the recovery stream has one payload token per secret bit.

    Partition index ``k`` is always ``len(prev_retok_ids_ns)`` — the same index
    ``recover_bitstream`` uses for the ``k``-th non-special token at detect time.
    """
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    inputs = encode_prompt_for_generation(tokenizer, prompt, device)
    input_ids_wm = inputs["input_ids"].clone()
    prompt_len = int(inputs["input_ids"].shape[1])
    attn_mask = inputs.get("attention_mask", None)

    _, logits_processor = _prepare_sampling_logits_processor_bundle(
        model,
        tokenizer,
        inputs,
        decode_horizon=len(secret_bitstream),
        generation_extra=generation_extra,
    )

    prev_retok_ids_ns: List[int] = []
    sacrificed_bits = 0
    natural_partition_choices = 0
    retok_replacements = 0
    recovery_stalls = 0
    partition_vocab_dim: int | None = None
    model_kwargs: Dict[str, Any] = {"use_cache": True}
    if attn_mask is not None:
        model_kwargs["attention_mask"] = attn_mask
    step_input_ids = input_ids_wm
    max_generation_steps = len(secret_bitstream) * 8
    generation_steps = 0

    while len(prev_retok_ids_ns) < len(secret_bitstream):
        generation_steps += 1
        if generation_steps > max_generation_steps:
            raise RuntimeError(
                f"watermarked generation stalled: {len(prev_retok_ids_ns)} recovery slots filled "
                f"of {len(secret_bitstream)} after {max_generation_steps} steps "
                f"(recovery_stalls={recovery_stalls})"
            )

        recovery_slot = len(prev_retok_ids_ns)
        with torch.no_grad():
            outputs = model(input_ids=step_input_ids, **model_kwargs)
            logits_wm = outputs.logits[0, -1, :]
            d_logits = int(logits_wm.shape[-1])
            if partition_vocab_dim is None:
                partition_vocab_dim = d_logits
                ensure_partition_weights(
                    partition_vocab_dim, tokenizer, tokenizer_id=tokenizer_id
                )
            elif partition_vocab_dim != d_logits:
                raise RuntimeError(
                    f"logits vocab width changed mid-decode ({d_logits} vs {partition_vocab_dim})"
                )
            model_kwargs["past_key_values"] = outputs.past_key_values

            mask_A = get_vectorized_partition(partition_vocab_dim, device, recovery_slot)
            secret_bit = int(secret_bitstream[recovery_slot])
            probs_wm, used_enforce = _watermark_step_probs(
                logits_wm, input_ids_wm, logits_processor, mask_A, secret_bit
            )
            if not used_enforce:
                natural_partition_choices += 1
            if torch.isfinite(probs_wm).all() and float(probs_wm.sum().item()) > 0:
                probs_wm = probs_wm.clamp(min=0)
                probs_wm = probs_wm / probs_wm.sum()
                next_token_id_wm = torch.multinomial(probs_wm, num_samples=1)
            else:
                probs_fb = _scores_to_next_token_probs(logits_wm, input_ids_wm, logits_processor)
                next_token_id_wm = torch.multinomial(probs_fb, num_samples=1)

        input_ids_wm = torch.cat([input_ids_wm, next_token_id_wm.unsqueeze(0)], dim=-1)
        step_input_ids = next_token_id_wm.unsqueeze(0)
        if "attention_mask" in model_kwargs:
            model_kwargs["attention_mask"] = torch.cat(
                [
                    model_kwargs["attention_mask"],
                    torch.ones((1, 1), dtype=model_kwargs["attention_mask"].dtype, device=device),
                ],
                dim=-1,
            )

        try:
            suffix_ids = input_ids_wm[0, prompt_len:].tolist()
            retok_ids_ns = _recovery_non_special_token_ids(tokenizer, suffix_ids, special_ids)
            prev_retok_ids_ns, sacrificed_bits, replacements, _, stalled = _update_recovery_state_after_token(
                prev_retok_ids_ns=prev_retok_ids_ns,
                retok_ids_ns=retok_ids_ns,
                sacrificed_bits=sacrificed_bits,
            )
            retok_replacements += replacements
            if stalled:
                recovery_stalls += 1
        except Exception:
            recovery_stalls += 1

    suffix_ids = input_ids_wm[0, prompt_len:].tolist()
    generated_text_wm = _decode_suffix_for_recovery(tokenizer, suffix_ids)
    recovery_ns = _recovery_non_special_token_ids(tokenizer, suffix_ids, special_ids)
    gen_ns = [t for t in suffix_ids if t not in special_ids]
    recovery_ids_aligned = prev_retok_ids_ns == recovery_ns
    recovery_input_ids = tokenizer(generated_text_wm, return_tensors="pt")["input_ids"]

    return {
        "prompt_text": prompt,
        "generated_text_wm": generated_text_wm,
        "input_ids_wm": recovery_input_ids,
        "secret_bitstream": secret_bitstream,
        "sacrificed_bits": sacrificed_bits,
        "natural_partition_choices": natural_partition_choices,
        "recovery_stalls": recovery_stalls,
        "retok_replacements": retok_replacements,
        "recovery_ids_aligned": recovery_ids_aligned,
        "recovery_slots_committed": len(prev_retok_ids_ns),
        "incremental_ids_aligned": gen_ns == recovery_ns,
    }


def generate_baseline(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: str = "cpu",
) -> str:
    inputs = encode_prompt_for_generation(tokenizer, prompt, device)
    prompt_len = int(inputs["input_ids"].shape[1])
    pad_id = _tokenizer_pad_token_id(tokenizer)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            pad_token_id=pad_id,
        )
    gen_ids = out[0, prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def recover_bitstream(
    full_sequence_ids: List[int],
    vocab_size: int,
    device: str,
    special_ids: set,
    *,
    tokenizer: AutoTokenizer | None = None,
    tokenizer_id: str | None = None,
) -> Tuple[List[int], List[int]]:
    del device  # partitions are built on CPU; kept for API compatibility
    ensure_partition_weights(int(vocab_size), tokenizer, tokenizer_id=tokenizer_id)
    recovered_bits, recovered_tokens = [], []
    filtered_ids = [tid for tid in full_sequence_ids if tid not in special_ids]
    v = int(vocab_size)

    for bit_idx, actual_token_id in enumerate(filtered_ids):
        recovered_bits.append(partition_bit_for_token(v, bit_idx, int(actual_token_id)))
        recovered_tokens.append(actual_token_id)

    return recovered_bits, recovered_tokens


def recover_bitstream_from_text(
    full_text: str,
    tokenizer: AutoTokenizer,
    device: str,
    *,
    model: torch.nn.Module | None = None,
    partition_vocab_size: int | None = None,
    tokenizer_id: str | None = None,
) -> Tuple[List[int], List[int]]:
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    enc = tokenizer(full_text, return_tensors="pt")
    vocab_size = resolve_partition_vocab_size_for_recovery(
        tokenizer, model=model, partition_vocab_size=partition_vocab_size
    )
    return recover_bitstream(
        full_sequence_ids=enc["input_ids"][0].tolist(),
        vocab_size=vocab_size,
        device=device,
        special_ids=special_ids,
        tokenizer=tokenizer,
        tokenizer_id=tokenizer_id,
    )


def negative_control_transcript_like(
    reference_text: str,
    tokenizer: AutoTokenizer,
    device: str,
    *,
    n_bits: int,
    model: torch.nn.Module | None = None,
    phrase: str = "Unrelated decoy text used only as a negative control. ",
) -> str:
    """Grow decoy text until it has enough non-special tokens (no partition work)."""
    del device, model  # API compatibility with callers that pass LM/device
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    ref_chars = max(len(reference_text), 1)
    s = ""
    while True:
        s += phrase
        if len(s) < ref_chars:
            continue
        ids = tokenizer(s, add_special_tokens=False)["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = [t for row in ids for t in row]
        n = sum(1 for t in ids if int(t) not in special_ids)
        if n >= n_bits:
            return s


__all__ = [
    "get_vectorized_partition",
    "partition_bit_for_token",
    "configure_partition_weights",
    "clear_partition_weights",
    "get_partition_weights",
    "partition_weights_key",
    "packaged_partition_weights_path",
    "load_partition_weights_file",
    "save_partition_weights_artifact",
    "count_unigram_tokens",
    "ensure_partition_weights",
    "resolve_partition_vocab_size_for_recovery",
    "encode_prompt_for_generation",
    "generate_with_watermark",
    "generate_baseline",
    "recover_bitstream",
    "recover_bitstream_from_text",
    "negative_control_transcript_like",
]
