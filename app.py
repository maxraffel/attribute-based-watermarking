from hashlib import sha256
import random
import prc
from randrecover import (
    recover_bitstream,
    recover_bitstream_from_text,
    log_generation_result,
    log_recovery_evaluation,
)
import watermarking as wm
from watermarking import issue, setup

# ANSI (Windows 10+ conhost / Windows Terminal)
_GREEN = "\033[92m"
_RED = "\033[91m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _dot_mod(f: list[int], x: list[int], modulus: int) -> int:
    return sum(f[i] * x[i] for i in range(len(x))) % modulus


def _expect_ceval_matches_r(f: list[int], x: list[int], modulus: int) -> bool:
    """CPRF c_eval recovers master eval iff f·x ≡ 0 (mod modulus) for this construction."""
    return _dot_mod(f, x, modulus) == 0


def _print_check(name: str, got: bool, expected: bool) -> bool:
    ok = got == expected
    tag = f"{_GREEN}PASS{_RESET}" if ok else f"{_RED}FAIL{_RESET}"
    exp_c = _GREEN if expected else _RED
    print(f"  {tag}  {name}  (got {got}, {exp_c}expected {expected}{_RESET})")
    return ok


def _print_metric_check(name: str, value: float, expected_high: bool) -> bool:
    high = value >= 0.99
    ok = high == expected_high
    tag = f"{_GREEN}PASS{_RESET}" if ok else f"{_RED}FAIL{_RESET}"
    want = "high (≥99%)" if expected_high else "low (<99%)"
    print(f"  {tag}  {name}: {value:.2%}  ({_DIM}expected {want}{_RESET})")
    return ok


def f_alt_constrained_to_x(x: list[int], code_len: int, modulus: int) -> list[int]:
    """Alternating +1 / -1 pattern with one free coordinate chosen so f·x ≡ 0 (mod modulus).

    Picks a pivot with odd x[i] (invertible mod 2^n when modulus=1024) and solves for that f[pivot].
    """
    f = [1 if j % 2 == 0 else -1 for j in range(code_len)]
    pivot = next((i for i in range(code_len - 1, -1, -1) if (x[i] & 1)), None)
    if pivot is None:
        raise ValueError(
            f"x has no odd coordinate; cannot find invertible pivot mod {modulus} to force f·x ≡ 0."
        )
    acc = 0
    for j in range(code_len):
        if j == pivot:
            continue
        acc = (acc + f[j] * (x[j] % modulus)) % modulus
    f[pivot] = (-acc * pow(x[pivot] % modulus, -1, modulus)) % modulus
    return f


def main():
    CODE_LEN = wm.SECURITY_PARAM
    print(f"Using device: {wm.DEVICE}")

    sk = setup(1024)

    f_accept_all = [0] * CODE_LEN
    dk_accept_all = issue(sk, f_accept_all)

    f_reject = [1] * CODE_LEN
    dk_reject = issue(sk, f_reject)

    prompt = "What is the meaning of life?"
    print("Running watermarking.generate (baseline → x → watermarked output) ...")
    out, x = wm.generate(sk, prompt)

    f_alt_accept = f_alt_constrained_to_x(x, CODE_LEN, sk.modulus)
    dk_alt_accept = issue(sk, f_alt_accept)

    ip_alt = sum(f_alt_accept[i] * x[i] for i in range(CODE_LEN)) % sk.modulus
    print(f"Inner product f_alt_accept·x (mod {sk.modulus}): {ip_alt}")
    print(f"Inner product of f_reject and x: {sum(f_reject[i] * x[i] for i in range(CODE_LEN))}")
    print(f"Inner product of f_accept_all and x: {sum(f_accept_all[i] * x[i] for i in range(CODE_LEN))}")

    r = sk.eval(x)
    prc.set_code_length(CODE_LEN)
    s = prc.key_gen_from_seed(sha256(r).digest())
    secret_bitstream = out["secret_bitstream"]

    log_generation_result(out)

    special_ids = set(getattr(wm.TOKENIZER, "all_special_ids", []))
    gen_bit_to_token = out["bit_index_to_token_id"]

    recovered_r_accept_all = dk_accept_all.c_eval(x)
    recovered_s_accept_all = prc.key_gen_from_seed(sha256(recovered_r_accept_all).digest())

    recovered_r_reject = dk_reject.c_eval(x)
    recovered_s_reject = prc.key_gen_from_seed(sha256(recovered_r_reject).digest())

    recovered_r_alt_accept = dk_alt_accept.c_eval(x)
    recovered_s_alt_accept = prc.key_gen_from_seed(sha256(recovered_r_alt_accept).digest())

    def s_similarity(s1, s2):
        pem1 = prc.key_to_pem(s1)
        pem2 = prc.key_to_pem(s2)
        if not pem1 or not pem2:
            return 0.0
        matches = sum(1 for a, b in zip(pem1, pem2) if a == b)
        return matches / max(len(pem1), len(pem2))

    sim_accept = s_similarity(s, recovered_s_accept_all)
    sim_reject = s_similarity(s, recovered_s_reject)
    sim_alt = s_similarity(s, recovered_s_alt_accept)

    match_accept = recovered_r_accept_all == r
    match_reject = recovered_r_reject == r
    match_alt = recovered_r_alt_accept == r

    checks: list[bool] = []

    print(f"\n{_DIM}--- CPRF c_eval vs master (expected = inner product f·x ≡ 0 mod modulus) ---{_RESET}")
    checks.append(
        _print_check(
            "CEval vs r (accept_all)",
            match_accept,
            _expect_ceval_matches_r(f_accept_all, x, sk.modulus),
        )
    )
    checks.append(
        _print_check(
            "CEval vs r (reject)",
            match_reject,
            _expect_ceval_matches_r(f_reject, x, sk.modulus),
        )
    )
    checks.append(
        _print_check(
            "CEval vs r (alt_accept)",
            match_alt,
            _expect_ceval_matches_r(f_alt_accept, x, sk.modulus),
        )
    )

    print(f"\n{_DIM}--- PRC key PEM similarity (high iff c_eval recovered r) ---{_RESET}")
    checks.append(_print_metric_check("Similarity s vs recovered_s_accept_all", sim_accept, match_accept))
    checks.append(_print_metric_check("Similarity s vs recovered_s_reject", sim_reject, match_reject))
    checks.append(_print_metric_check("Similarity s vs recovered_s_alt_accept", sim_alt, match_alt))

    print("\n--- Evaluation ---")
    extracted_ctx, _ = recover_bitstream(
        out["input_ids_wm"][0].tolist(),
        wm.TOKENIZER.vocab_size,
        wm.DEVICE,
        special_ids,
    )

    extracted_txt, _ = recover_bitstream_from_text(
        out["generated_text_wm"],
        wm.TOKENIZER,
        wm.DEVICE,
        ground_truth_tokens=gen_bit_to_token,
    )

    random_control = [random.randint(0, 1) for _ in range(CODE_LEN)]

    log_recovery_evaluation(secret_bitstream, extracted_ctx, "Context")
    log_recovery_evaluation(secret_bitstream, extracted_txt, "Text")
    log_recovery_evaluation(secret_bitstream, random_control, "Random Control")

    def prc_detect_bits(bits, s_key) -> bool:
        b = (bits + [0] * CODE_LEN)[:CODE_LEN]
        return prc.detect(s_key, [bool(t) for t in b])

    print(f"\n{_DIM}--- PRC detection (expected True only for valid key + plausible bitstream) ---{_RESET}")
    checks.append(
        _print_check(
            "PRC Context + accept_all key",
            prc_detect_bits(extracted_ctx, recovered_s_accept_all),
            True,
        )
    )
    checks.append(
        _print_check(
            "PRC Text + accept_all key",
            prc_detect_bits(extracted_txt, recovered_s_accept_all),
            True,
        )
    )
    checks.append(
        _print_check(
            "PRC Text + alt_accept key",
            prc_detect_bits(extracted_txt, recovered_s_alt_accept),
            True,
        )
    )
    checks.append(
        _print_check(
            "PRC Text + reject key",
            prc_detect_bits(extracted_txt, recovered_s_reject),
            False,
        )
    )
    checks.append(
        _print_check(
            "PRC Random + accept_all key",
            prc_detect_bits(random_control, recovered_s_accept_all),
            False,
        )
    )

    passed = sum(checks)
    total = len(checks)
    banner = f"{_GREEN}All checks passed ({passed}/{total}){_RESET}" if passed == total else f"{_RED}Some checks failed ({passed}/{total} passed){_RESET}"
    print(f"\n{banner}")


if __name__ == "__main__":
    main()
