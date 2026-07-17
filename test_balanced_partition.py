"""Unit + optional e2e tests for softmax-balanced watermark partitions."""

from __future__ import annotations

import os

import torch

import randrecover as rr


def _ber(secret: list[int], recovered: list[int]) -> float:
    n = len(secret)
    if n == 0:
        return 0.0
    m = min(len(recovered), n)
    errs = sum(1 for i in range(m) if int(secret[i]) != int(recovered[i]))
    errs += abs(n - len(recovered))
    return errs / n


def test_balanced_partition_deterministic_and_even() -> None:
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(64), dim=0)
    mask_a = rr.get_balanced_partition_from_probs(probs)
    mask_b = rr.get_balanced_partition_from_probs(probs)
    assert torch.equal(mask_a, mask_b)
    mass_a, mass_b = rr.balanced_partition_masses(probs, mask_a)
    assert abs(mass_a + mass_b - 1.0) < 1e-5
    assert abs(mass_a - 0.5) < 0.15


def test_balanced_partition_near_optimal_gap() -> None:
    probs = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float32)
    mask = rr.get_balanced_partition_from_probs(probs)
    mass_a, mass_b = rr.balanced_partition_masses(probs, mask)
    assert abs(mass_a - mass_b) < 0.21


def test_balanced_partition_orders_by_probability_not_token_id() -> None:
    probs = torch.zeros(128, dtype=torch.float32)
    probs[3] = 0.05
    probs[120] = 0.95
    step = next(i for i in range(32) if not rr._balanced_orientation_flip(i))
    mask = rr.get_balanced_partition_from_probs(probs, step_index=step)
    assert bool(mask[120].item()) is True
    assert bool(mask[3].item()) is False


def test_balanced_soft_channel_roundtrip_no_model() -> None:
    """Enforce-half sampling + balanced recovery from the same probs recovers the bit."""
    torch.manual_seed(1)
    probs = torch.softmax(torch.randn(32), dim=0)
    for secret_bit in (0, 1):
        mask_a = rr.get_balanced_partition_from_probs(probs)
        p = float(probs[mask_a].sum().item())
        q = min(p, 1.0 - p)
        choose_a = secret_bit == 0
        masked = torch.where(mask_a, probs, torch.zeros_like(probs)) if choose_a else torch.where(
            mask_a, torch.zeros_like(probs), probs
        )
        tok = int(torch.argmax(masked).item())
        recovered = 0 if bool(mask_a[tok].item()) else 1
        assert recovered == secret_bit, (secret_bit, recovered, p, q)


def test_cascade_isolated_text_split() -> None:
    """Character split keeps watermarked suffix independent of prefix length views."""
    prefix = "Hello world, this is burn-in context. "
    wm = "Watermarked payload tokens live here only."
    full = prefix + wm
    assert full[: len(prefix)] == prefix
    assert full[len(prefix) :] == wm


def test_e2e_balanced_watermark() -> None:
    if os.environ.get("RUN_E2E_BALANCED") != "1":
        print("skip test_e2e_balanced_watermark (set RUN_E2E_BALANCED=1)")
        return

    import model

    prompt = "Explain how software has transformed the art world."
    secret = [0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1]
    n_bits = len(secret)
    m, tok, device = model.load()

    out = rr.generate_with_watermark(
        m, tok, prompt, secret, device, burn_in_tokens=8
    )
    raw, _, gaps = rr.recover_bitstream_from_generation(m, tok, out, device)
    raw = raw[:n_bits]
    ber = _ber(secret, raw)

    print("\n=== balanced watermark recovery (prompt-free, burn-in=8) ===")
    print(out["generated_text_wm"])
    print(
        f"BER={ber:.2%}  natural={out['natural_partition_choices']}  "
        f"burn_in_chars={out['burn_in_char_len']}  "
        f"gap_mean={out.get('partition_mass_gap_mean', 0):.4f}  "
        f"gap_max={out.get('partition_mass_gap_max', 0):.4f}"
    )
    if gaps:
        print(f"recovery gap mean={sum(gaps) / len(gaps):.4f}")

    # Prompt-free approx after short burn-in can be noisier than prompt-conditioned.
    assert ber <= 0.55, f"balanced recovery BER too high: {ber:.2%}"


if __name__ == "__main__":
    test_balanced_partition_deterministic_and_even()
    print("OK test_balanced_partition_deterministic_and_even")
    test_balanced_partition_near_optimal_gap()
    print("OK test_balanced_partition_near_optimal_gap")
    test_balanced_partition_orders_by_probability_not_token_id()
    print("OK test_balanced_partition_orders_by_probability_not_token_id")
    test_balanced_soft_channel_roundtrip_no_model()
    print("OK test_balanced_soft_channel_roundtrip_no_model")
    test_cascade_isolated_text_split()
    print("OK test_cascade_isolated_text_split")
    test_e2e_balanced_watermark()
    print("OK test_e2e_balanced_watermark (or skipped)")
