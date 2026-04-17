import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from randrecover import (
    generate_with_watermark,
    recover_bitstream,
    recover_bitstream_from_text,
    compare_tokenizations,
    log_generation_result,
    log_recovery_evaluation,
    log_tokenization_comparison,
)
import random
from prc import key_gen, encode, detect


def main():
    # device setup
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading on {device}...")

    def _maybe_sync():
        if device == "cuda":
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    prompt = "What is the meaning of life?"

    # run generation
    # Create secret bitstream here and pass into generation/recovery
    # Use the PRC system to generate a key and encode it into a codeword
    key = key_gen()
    codeword = encode(key)
    # convert boolean codeword into 0/1 ints for the watermarking code
    secret_bitstream = [1 if b else 0 for b in codeword]
    print(f"Initial secret bitstream length: {len(secret_bitstream)}")
    secret_key = key

    gen_start = time.perf_counter()
    out = generate_with_watermark(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        secret_bitstream=secret_bitstream,
        secret_key=secret_key,
        device=device,
        generate_baseline=True,
    )
    _maybe_sync()
    gen_time = time.perf_counter() - gen_start
    print(f"Generation time: {gen_time:.3f}s")

    generated_text = out["generated_text_wm"]
    input_ids = out["input_ids_wm"]
    generated_text_base = out["generated_text_base"]
    input_ids_base = out["input_ids_base"]
    secret_bitstream = out["secret_bitstream"]

    # log generation
    log_generation_result(out)

    # prepare special ids set here
    special_ids = set(getattr(tokenizer, "all_special_ids", []))

    # recover from retokenized text
    rec_start = time.perf_counter()
    extracted_from_text = recover_bitstream_from_text(
        full_text=generated_text,
        prompt_text=prompt,
        tokenizer=tokenizer,
        vocab_size=tokenizer.vocab_size,
        key=secret_key,
        device=device,
        special_ids=special_ids,
    )

    # recover from generation context ids
    full_sequence_list = input_ids[0].tolist()
    prompt_length = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
    extracted_from_context = recover_bitstream(
        full_sequence_ids=full_sequence_list,
        prompt_length=prompt_length,
        vocab_size=tokenizer.vocab_size,
        key=secret_key,
        device=device,
        special_ids=special_ids,
    )

    _maybe_sync()

    # # Baseline extractions for the untampered generation
    # extracted_from_text_base = recover_bitstream_from_text(
    #     full_text=generated_text_base,
    #     prompt_text=prompt,
    #     tokenizer=tokenizer,
    #     vocab_size=tokenizer.vocab_size,
    #     key=secret_key,
    #     device=device,
    #     special_ids=special_ids,
    # )
    # full_sequence_list_base = input_ids_base[0].tolist()
    # extracted_from_context_base = recover_bitstream(
    #     full_sequence_ids=full_sequence_list_base,
    #     prompt_length=prompt_length,
    #     vocab_size=tokenizer.vocab_size,
    #     key=secret_key,
    #     device=device,
    #     special_ids=special_ids,
    # )

    # logging/evaluation for watermarked generation
    print("\n=== Watermarked generation recovery ===")
    log_recovery_evaluation(secret_bitstream, extracted_from_text, extracted_from_context)

    # Attempt detection on the recovered bitstreams
    # convert recovered int bitstreams back to boolean codewords for PRC detection
    try:
        recovered_codeword_context = [bool(b) for b in extracted_from_context]
        detected_context = detect(secret_key, recovered_codeword_context)
    except Exception:
        detected_context = False

    try:
        recovered_codeword_text = [bool(b) for b in extracted_from_text]
        detected_text = detect(secret_key, recovered_codeword_text)
    except Exception:
        detected_text = False

    print("\n--- PRC Detection Results ---")
    print(f"Detected from context extraction: {detected_context}")
    print(f"Detected from retokenized-text extraction: {detected_text}")

    # tokenization comparison diagnostics
    tok_cmp = compare_tokenizations(input_ids_tensor=input_ids, generated_text=generated_text, tokenizer=tokenizer)
    log_tokenization_comparison(tok_cmp, tokenizer)


if __name__ == "__main__":
    main()
