from hashlib import sha256
import torch
import random
import prc
import cprf
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

    CODE_LEN = 300

    # Setup
    sk = cprf.keygen(1024, CODE_LEN)

    # Issue
    f_accept_all = [0]*CODE_LEN
    dk_accept_all = sk.constrain(f_accept_all) # this is essentially f = 1

    f_reject = [1]*CODE_LEN
    dk_reject = sk.constrain(f_reject)

    f_alt_accept = [1 if i%2 == 0 else -1 for i in range(CODE_LEN)]
    if (CODE_LEN % 2 != 0):
        f_alt_accept[-1] = 0
    dk_alt_accept = sk.constrain(f_alt_accept)

    # Generate
    x = [1]*CODE_LEN # for now choose an arbitrary x to test succeeding

    # print inner product of f and x
    print(f"Inner product of f_alt_accept and x: {sum(f_alt_accept[i]*x[i] for i in range(CODE_LEN))}")
    print(f"Inner product of f_reject and x: {sum(f_reject[i]*x[i] for i in range(CODE_LEN))}")
    print(f"Inner product of f_accept_all and x: {sum(f_accept_all[i]*x[i] for i in range(CODE_LEN))}")

    r = sk.eval(x)
    prc.set_code_length(CODE_LEN)
    s = prc.key_gen_from_seed(sha256(r).digest())
    c = prc.encode(s)
    secret_bitstream = [1 if b else 0 for b in c]

    # Use r.hex() as the string-based key for watermarking because str(s) evaluates to the dynamic object memory address
    out = generate_with_watermark(model, tokenizer, "What is the meaning of life?", secret_bitstream, device)
    log_generation_result(out)

    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    gen_bit_to_token = out["bit_index_to_token_id"]

    # Detect
    recovered_r_accept_all = dk_accept_all.c_eval(x)
    recovered_s_accept_all = prc.key_gen_from_seed(sha256(recovered_r_accept_all).digest())

    recovered_r_reject = dk_reject.c_eval(x)
    recovered_s_reject = prc.key_gen_from_seed(sha256(recovered_r_reject).digest())

    recovered_r_alt_accept = dk_alt_accept.c_eval(x)
    recovered_s_alt_accept = prc.key_gen_from_seed(sha256(recovered_r_alt_accept).digest())

    print(f"CEval on x matches r_accept_all: {recovered_r_accept_all == r}")
    print(f"CEval on x matches r_reject: {recovered_r_reject == r}")
    print(f"CEval on x matches r_alt_accept: {recovered_r_alt_accept == r}")

    print("\n--- Evaluation ---")
    extracted_ctx, _ = recover_bitstream(
        out["input_ids_wm"][0].tolist(), tokenizer.vocab_size, device, special_ids
    )
    
    extracted_txt, _ = recover_bitstream_from_text(
        out["generated_text_wm"], tokenizer, vocab_size=tokenizer.vocab_size, 
        device=device, special_ids=special_ids, 
        ground_truth_tokens=gen_bit_to_token
    )

    random_control = [random.randint(0, 1) for _ in range(CODE_LEN)]

    def safe_detect(bits, sk, label):
        # Enforce exact length for PRC module
        bits = (bits + [0] * CODE_LEN)[:CODE_LEN]
        res = prc.detect(sk, [bool(b) for b in bits])
        print(f"PRC Detection [{label}]: {res}")

    log_recovery_evaluation(secret_bitstream, extracted_ctx, "Context")
    log_recovery_evaluation(secret_bitstream, extracted_txt, "Text")
    log_recovery_evaluation(secret_bitstream, random_control, "Random Control")

    print("\n--- PRC Detection Results ---")
    safe_detect(extracted_ctx, recovered_s_accept_all, "Context")
    safe_detect(extracted_txt, recovered_s_accept_all, "Text Accept All")
    safe_detect(extracted_txt, recovered_s_alt_accept, "Text Alt Accept")
    safe_detect(extracted_txt, recovered_s_reject, "Text Reject")
    safe_detect(random_control, recovered_s_accept_all, "Random Control")

if __name__ == "__main__":
    main()