import torch
from transformers import AutoTokenizer
import hashlib
import random
from typing import Dict, List, Tuple

def get_vectorized_partition(vocab_size: int, device: str, seed_index: int) -> torch.BoolTensor:
    """Generates a boolean mask for the entire vocabulary using a deterministic seed."""
    seed_string = f"{seed_index}"
    seed = int(hashlib.md5(seed_string.encode()).hexdigest()[:16], 16)
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.rand(vocab_size, generator=g, device=device) > 0.5

def generate_with_watermark(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    secret_bitstream: List[int],
    device: str = "cpu",
    track_token_alignment: bool = False,
) -> Dict:
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids_wm = inputs["input_ids"].clone()
    prompt_len = inputs["input_ids"].shape[1]
    attn_mask = inputs.get("attention_mask", None)

    bit_index = 0
    bit_index_to_token_id = [-1] * len(secret_bitstream)
    running_mean_p, p_count = 0.0, 0

    def _sample_modified_token(probs, mask_A, p, q, random_bit):
        modified = probs.clone()
        choose_set_A = (random_bit == 0) if random.random() < (2 * q) else (p > 0.5)
        if choose_set_A:
            modified[~mask_A] = 0
        else:
            modified[mask_A] = 0
        return torch.multinomial(modified / modified.sum(), num_samples=1)

    # Incremental decoding with KV cache avoids full-sequence forward each step.
    model_kwargs = {"use_cache": True}
    if attn_mask is not None:
        model_kwargs["attention_mask"] = attn_mask
    step_input_ids = input_ids_wm

    # Force generation until bitstream is exhausted
    while bit_index < len(secret_bitstream):
        with torch.no_grad():
            outputs = model(input_ids=step_input_ids, **model_kwargs)
            logits_wm = outputs.logits[0, -1, :]
            probs_wm = torch.softmax(logits_wm, dim=-1)
            model_kwargs["past_key_values"] = outputs.past_key_values

        mask_A = get_vectorized_partition(probs_wm.shape[-1], device, bit_index)
        p = probs_wm[mask_A].sum().item()
        p_count += 1
        running_mean_p += (p - running_mean_p) / p_count
        
        next_token_id_wm = _sample_modified_token(
            probs_wm, mask_A, p, min(p, 1 - p), secret_bitstream[bit_index]
        )
        tid = int(next_token_id_wm.item())
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

        if track_token_alignment:
            if "_retok_offset" not in locals():
                _retok_offset = 0
            try:
                gen_text_so_far = tokenizer.decode(input_ids_wm[0], skip_special_tokens=False)
                retok_ids_ns = [t for t in tokenizer(gen_text_so_far)["input_ids"] if t not in special_ids]
                gen_ids_ns = [t for t in input_ids_wm[0].tolist() if t not in special_ids]

                if len(gen_ids_ns) != (len(retok_ids_ns) + _retok_offset):
                    if len(gen_ids_ns) > (len(retok_ids_ns) + _retok_offset):
                        if tid not in special_ids:
                            bit_index_to_token_id[bit_index] = tid
                        _retok_offset += 1
                    else:
                        if tid not in special_ids:
                            for i in range(bit_index, min(bit_index + 2, len(bit_index_to_token_id))):
                                if i < len(bit_index_to_token_id):
                                    bit_index_to_token_id[i] = tid
                        _retok_offset -= 1
                        bit_index += 2
                else:
                    if tid not in special_ids:
                        bit_index_to_token_id[bit_index] = tid
                    bit_index += 1
            except Exception:
                bit_index += 1
        else:
            if tid not in special_ids:
                bit_index_to_token_id[bit_index] = tid
            bit_index += 1

    return {
        "prompt_text": prompt,
        "generated_text_wm": tokenizer.decode(input_ids_wm[0, prompt_len:], skip_special_tokens=True),
        "input_ids_wm": input_ids_wm[:, prompt_len:], 
        "secret_bitstream": secret_bitstream,
        "bit_index_to_token_id": bit_index_to_token_id,
        "running_mean_p": running_mean_p,
        "p_count": p_count,
    }

def generate_baseline(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: str = "cpu",
) -> str:
    """
    Greedy unwatermarked generation (deterministic for fixed model+prompt) used to derive
    the attribute vector x. Match this to what verifiers will reproduce.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )
    gen_ids = out[0, prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)

def recover_bitstream(
    full_sequence_ids: List[int],
    vocab_size: int,
    device: str,
    special_ids: set,
    ground_truth_tokens: List[int] = None,
) -> Tuple[List[int], List[int]]:
    recovered_bits, recovered_tokens = [], []
    matches, total_checks = 0, 0
    filtered_ids = [tid for tid in full_sequence_ids if tid not in special_ids]
    # Ensure the mask covers all token ids present in the sequence. Some tokenizers
    # may produce ids >= `vocab_size` (e.g. added tokens); extend the effective
    # vocab size so mask indexing never goes out of bounds.
    if filtered_ids:
        max_token_id = max(filtered_ids)
        if max_token_id >= vocab_size:
            vocab_size = max_token_id + 1
    
    for bit_idx, actual_token_id in enumerate(filtered_ids):
        if ground_truth_tokens and bit_idx >= len(ground_truth_tokens): break

        mask_A = get_vectorized_partition(vocab_size, device, bit_idx)
        # Guard against any remaining out-of-bounds access just in case.
        if actual_token_id >= mask_A.shape[0]:
            # If token id is outside, treat it as belonging to the B set (bit=1)
            recovered_bits.append(1)
        else:
            recovered_bits.append(0 if mask_A[actual_token_id].item() else 1)
        recovered_tokens.append(actual_token_id)

        if ground_truth_tokens:
            expected_id = ground_truth_tokens[bit_idx]
            if expected_id != -1:
                total_checks += 1
                if actual_token_id == expected_id: matches += 1

    if ground_truth_tokens and total_checks > 0:
        print(f"Token Accuracy: {matches}/{total_checks} ({(matches/total_checks)*100:.2f}%)")

    return recovered_bits, recovered_tokens

def recover_bitstream_from_text(
    full_text: str,
    tokenizer: AutoTokenizer,
    device: str,
    ground_truth_tokens: List[int] | None = None,
) -> Tuple[List[int], List[int]]:
    enc = tokenizer(full_text, return_tensors="pt")
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    return recover_bitstream(
        full_sequence_ids=enc["input_ids"][0].tolist(),
        vocab_size=tokenizer.vocab_size,
        device=device,
        special_ids=special_ids,
        ground_truth_tokens=ground_truth_tokens,
    )

def log_generation_result(out: Dict) -> None:
    print(f"\n--- Generation Result ---\nPrompt: {out['prompt_text']}\nOutput: {out['generated_text_wm']}")

def log_recovery_evaluation(secret, extracted, label=""):
    if not secret: return
    # Handle length mismatches for BER calculation
    min_len = min(len(secret), len(extracted))
    errs = sum(1 for i in range(min_len) if secret[i] != extracted[i])
    errs += abs(len(secret) - len(extracted)) # Penalize length difference
    print(f"BER [{label}]: {(errs/len(secret))*100:.2f}%")

__all__ = [
    "get_vectorized_partition",
    "generate_with_watermark",
    "generate_baseline",
    "recover_bitstream",
    "recover_bitstream_from_text",
    "log_generation_result",
    "log_recovery_evaluation",
]