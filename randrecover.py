import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import hashlib
import random
from typing import Dict, List, Tuple

def get_vectorized_partition(vocab_size: int, key: str, device: str, seed_index: int, window_size: int = 2) -> torch.BoolTensor:
    """
    Generates a boolean mask for the entire vocabulary instantly.
    True = Set A, False = Set B.
    """

    seed_string = f"{key}:{seed_index}"

    # Convert the first 16 chars of the hex hash to a 64-bit integer
    seed = int(hashlib.md5(seed_string.encode()).hexdigest()[:16], 16)

    # Seed a PyTorch Generator on the correct device
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    # Vectorized Generation: Create a random tensor of shape [vocab_size]
    partition_mask = torch.rand(vocab_size, generator=g, device=device) > 0.5

    return partition_mask

def generate_with_watermark(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    secret_bitstream: List[int],
    secret_key: str = "secret",
    device: str = "cpu",
    generate_baseline: bool = False,
) -> Dict:
    """Generate text while embedding a secret bitstream.

    Returns a dict containing `generated_text`, `input_ids` (tensor),
    `secret_bitstream`, and `running_mean_p`.
    """
    special_ids = set(getattr(tokenizer, "all_special_ids", []))

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Maintain separate contexts for watermark and baseline generations
    input_ids_wm = inputs["input_ids"].clone()
    input_ids_base = inputs["input_ids"].clone()

    # Use externally supplied secret bitstream
    bit_index = 0
    max_new_tokens = len(secret_bitstream)

    running_mean_p = 0.0
    p_count = 0

    # Compute initial count of non-special tokens in the prompt to seed position counter
    non_special_pos = sum(1 for _id in input_ids_wm[0].tolist() if _id not in special_ids)

    def _sample_modified_token(probs: torch.Tensor, mask_A: torch.Tensor, p: float, q: float, random_bit: int) -> torch.Tensor:
        """Apply partition-based modification and sample one token."""
        modified = probs.clone()
        if random.random() < (2 * q):
            choose_set_A = random_bit == 0
        else:
            choose_set_A = (p > 0.5)

        if choose_set_A:
            modified[~mask_A] = 0
        else:
            modified[mask_A] = 0

        # normalize and sample
        modified = modified / modified.sum()
        return torch.multinomial(modified, num_samples=1)

    def _sample_baseline_token(probs: torch.Tensor) -> torch.Tensor:
        """Sample directly from the model distribution (baseline)."""
        return torch.multinomial(probs, num_samples=1)

    for _ in range(max_new_tokens):
        # forward for watermark context
        with torch.no_grad():
            outputs_wm = model(input_ids_wm)
            logits_wm = outputs_wm.logits[0, -1, :]
            probs_wm = torch.softmax(logits_wm, dim=-1)

            # forward for baseline context only if requested
            probs_base = None
            if generate_baseline:
                outputs_base = model(input_ids_base)
                logits_base = outputs_base.logits[0, -1, :]
                probs_base = torch.softmax(logits_base, dim=-1)

        vocab_size = probs_wm.shape[-1]

        # Use the non-special-token position as seed for watermark partitioning
        seed_index = non_special_pos
        mask_A = get_vectorized_partition(
            vocab_size=vocab_size,
            key=secret_key,
            device=device,
            seed_index=seed_index,
        )

        p = probs_wm[mask_A].sum().item()
        p_count += 1
        running_mean_p += (p - running_mean_p) / p_count
        q = min(p, 1 - p)
        random_bit = secret_bitstream[bit_index]

        # sample watermark token
        next_token_id_wm = _sample_modified_token(probs_wm, mask_A, p, q, random_bit)

        # sample baseline token only if requested
        next_token_id_base = None
        if generate_baseline and probs_base is not None:
            next_token_id_base = _sample_baseline_token(probs_base)

        # Append watermark token
        input_ids_wm = torch.cat([input_ids_wm, next_token_id_wm.unsqueeze(0)], dim=-1)
        # Append baseline token if generated
        if next_token_id_base is not None:
            input_ids_base = torch.cat([input_ids_base, next_token_id_base.unsqueeze(0)], dim=-1)

        # Update non-special position counter depending on whether the watermark token is special
        try:
            new_id = int(next_token_id_wm.item())
        except Exception:
            new_id = int(next_token_id_wm.tolist()[0])
        if new_id not in special_ids:
            non_special_pos += 1
        bit_index += 1

    generated_text_wm = tokenizer.decode(input_ids_wm[0], skip_special_tokens=True)

    if generate_baseline:
        generated_text_base = tokenizer.decode(input_ids_base[0], skip_special_tokens=True)
        input_ids_base_out = input_ids_base
    else:
        generated_text_base = None
        input_ids_base_out = None

    return {
        "generated_text_wm": generated_text_wm,
        "input_ids_wm": input_ids_wm,
        "generated_text_base": generated_text_base,
        "input_ids_base": input_ids_base_out,
        "secret_bitstream": secret_bitstream,
        "running_mean_p": running_mean_p,
        "p_count": p_count,
        "baseline_generated": generate_baseline,
    }


def recover_bitstream(
    full_sequence_ids: List[int],
    prompt_length: int,
    vocab_size: int,
    key: str,
    device: str,
    special_ids: set,
    window_size: int = 2,
) -> List[int]:
    recovered_bits: List[int] = []

    # Ensure the partition mask length is large enough to index any token ids
    if len(full_sequence_ids) > 0:
        max_id_in_sequence = max(full_sequence_ids)
    else:
        max_id_in_sequence = 0
    required_vocab = max(vocab_size, max_id_in_sequence + 1)

    # Start counter at number of non-special tokens in the prompt portion
    non_special_counter = sum(1 for _id in full_sequence_ids[:prompt_length] if _id not in special_ids)

    # Iterate through only the generated tokens
    for i in range(prompt_length, len(full_sequence_ids)):
        # Re-create the same partition mask using the non-special token index
        mask_A = get_vectorized_partition(
            vocab_size=required_vocab,
            key=key,
            device=device,
            seed_index=non_special_counter,
            window_size=window_size,
        )

        actual_token_id = full_sequence_ids[i]
        if mask_A[actual_token_id].item():
            recovered_bits.append(0)
        else:
            recovered_bits.append(1)

        # Update the non-special counter for the next position
        if actual_token_id not in special_ids:
            non_special_counter += 1

    return recovered_bits


def recover_bitstream_from_text(
    full_text: str,
    prompt_text: str,
    tokenizer: AutoTokenizer,
    vocab_size: int,
    key: str,
    device: str,
    special_ids: set,
    window_size: int = 2,
) -> List[int]:
    # Re-tokenize the complete text (prompt + generated continuation)
    enc = tokenizer(full_text, return_tensors="pt")
    full_ids = enc["input_ids"][0].tolist()

    # Determine prompt length by tokenizing the prompt text
    prompt_enc = tokenizer(prompt_text, return_tensors="pt")
    prompt_len = prompt_enc["input_ids"].shape[1]

    return recover_bitstream(
        full_sequence_ids=full_ids,
        prompt_length=prompt_len,
        vocab_size=vocab_size,
        key=key,
        device=device,
        special_ids=special_ids,
        window_size=window_size,
    )


def compare_tokenizations(input_ids_tensor: torch.Tensor, generated_text: str, tokenizer: AutoTokenizer) -> Dict:
    gen_ids = input_ids_tensor[0].tolist()
    retok_enc = tokenizer(generated_text, return_tensors="pt")
    retok_ids = retok_enc["input_ids"][0].tolist()

    len_gen = len(gen_ids)
    len_ret = len(retok_ids)

    min_len = min(len_gen, len_ret)
    mismatches = [(i, gen_ids[i], retok_ids[i]) for i in range(min_len) if gen_ids[i] != retok_ids[i]]

    return {
        "len_gen": len_gen,
        "len_ret": len_ret,
        "mismatches": mismatches,
        "gen_ids": gen_ids,
        "retok_ids": retok_ids,
    }


__all__ = [
    "get_vectorized_partition",
    "generate_with_watermark",
    "recover_bitstream",
    "recover_bitstream_from_text",
    "compare_tokenizations",
    "log_generation_result",
    "log_recovery_evaluation",
    "log_tokenization_comparison",
]


def log_generation_result(out: Dict) -> None:
    """Print concise generation results from `generate_with_watermark` output."""
    print("\n--- Generation Result ---")
    print("-- Watermarked Generation --")
    print(f"Generated Text (watermark): {out['generated_text_wm']}")
    print(f"Average p over {out['p_count']} steps: {out['running_mean_p']:.6f}")
    if out.get("baseline_generated", False):
        print("-- Baseline Generation (no tampering) --")
        print(f"Generated Text (baseline): {out['generated_text_base']}")
    else:
        print("-- Baseline Generation (no tampering) --")
        print("Baseline not generated")


def log_recovery_evaluation(secret_bitstream: List[int], extracted: List[int], label: str = "extraction") -> None:
    """Print recovery results and BER for a single extraction result.

    Args:
        secret_bitstream: The original secret bitstream (list of 0/1).
        extracted: The recovered bitstream to evaluate against the secret.
        label: Human-readable label for this extraction method (printed).
    """
    if len(secret_bitstream) > 0:
        errors = sum(1 for orig, ext in zip(secret_bitstream, extracted) if orig != ext)
        ber = errors / len(secret_bitstream)

        print(f"Bit Error Rate (BER) {label}:  {ber * 100:.2f}%")
    else:
        print("Secret bitstream is empty; cannot compute BER.")


def log_tokenization_comparison(tok_cmp: Dict, tokenizer: AutoTokenizer) -> None:
    """Print tokenization comparison diagnostics prepared by `compare_tokenizations`."""
    print("\n--- Tokenization Comparison ---")
    print(f"Generation token length: {tok_cmp['len_gen']}")
    print(f"Retokenized token length: {tok_cmp['len_ret']}")
    print(f"Total mismatches (within min length): {len(tok_cmp['mismatches'])}")
    if tok_cmp['mismatches']:
        print("First mismatches (index, gen_id, retok_id, gen_token, retok_token):")
        for idx, g_id, r_id in tok_cmp['mismatches'][:10]:
            g_tok = tokenizer.convert_ids_to_tokens(g_id)
            r_tok = tokenizer.convert_ids_to_tokens(r_id)
            print(f"  {idx}: {g_id} != {r_id}  |  {g_tok}  !=  {r_tok}")