import torch
import random
import prc
from transformers import AutoModelForCausalLM, AutoTokenizer
from randrecover import (
    generate_with_watermark, recover_bitstream, recover_bitstream_from_text,
    log_generation_result, log_recovery_evaluation
)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    # 1. Setup PRC
    CODE_LEN = 300
    prc.set_code_length(CODE_LEN)
    seed = random.getrandbits(64)
    secret_key = prc.key_gen_from_seed(seed)
    codeword = prc.encode(secret_key)
    secret_bitstream = [1 if b else 0 for b in codeword]

    # 2. Watermarked Generation
    out = generate_with_watermark(model, tokenizer, "What is the meaning of life?", secret_bitstream, str(secret_key), device)
    log_generation_result(out)

    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    gen_bit_to_token = out["bit_index_to_token_id"]

    # 3. Recovery
    print("\n--- Evaluation ---")
    extracted_ctx, _ = recover_bitstream(
        out["input_ids_wm"][0].tolist(), tokenizer.vocab_size, str(secret_key), device, special_ids
    )
    
    extracted_txt, _ = recover_bitstream_from_text(
        out["generated_text_wm"], tokenizer, vocab_size=tokenizer.vocab_size, 
        key=str(secret_key), device=device, special_ids=special_ids, 
        ground_truth_tokens=gen_bit_to_token
    )

    random_control = [random.randint(0, 1) for _ in range(CODE_LEN)]

    # 4. Evaluation & Detection
    def safe_detect(bits, label):
        # Enforce exact length for PRC module
        bits = (bits + [0] * CODE_LEN)[:CODE_LEN]
        res = prc.detect(secret_key, [bool(b) for b in bits])
        print(f"PRC Detection [{label}]: {res}")

    log_recovery_evaluation(secret_bitstream, extracted_ctx, "Context")
    log_recovery_evaluation(secret_bitstream, extracted_txt, "Text")
    log_recovery_evaluation(secret_bitstream, random_control, "Random Control")

    print("\n--- PRC Detection Results ---")
    safe_detect(extracted_ctx, "Context")
    safe_detect(extracted_txt, "Text")
    safe_detect(random_control, "Random Control")

if __name__ == "__main__":
    main()