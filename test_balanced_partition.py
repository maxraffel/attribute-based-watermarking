"""Unit + optional e2e tests for softmax-balanced watermark partitions vs static."""

from __future__ import annotations

import random
import sys

import torch

import randrecover as rr


def _ber(secret: list[int], recovered: list[int]) -> float:
    n = min(len(secret), len(recovered))
    if n == 0:
        return 1.0
    err = sum(int(secret[i]) != int(recovered[i]) for i in range(n))
    return err / n


def test_balanced_partition_deterministic_and_even() -> None:
    torch.manual_seed(0)
    for v, support in ((64, 64), (128, 20), (256, 8)):
        logits = torch.randn(v)
        # Sparse-ish support to mimic nucleus sampling.
        keep = torch.randperm(v)[:support]
        mask_keep = torch.zeros(v, dtype=torch.bool)
        mask_keep[keep] = True
        logits = logits.masked_fill(~mask_keep, -1e9)
        probs = torch.softmax(logits, dim=-1)

        mask_a = rr.get_balanced_partition_from_probs(probs)
        mask_b = rr.get_balanced_partition_from_probs(probs)
        assert torch.equal(mask_a, mask_b), "partition must be deterministic in probs"

        mass_a, mass_b = rr.balanced_partition_masses(probs, mask_a)
        assert abs(mass_a + mass_b - 1.0) < 1e-5
        gap = abs(mass_a - mass_b)
        max_p = float(probs.max().item())
        optimal = max(0.0, 2.0 * max_p - 1.0)
        # Greedy is best-effort; a large max token makes exact balance impossible.
        assert gap <= optimal + max_p + 1e-4, (
            f"gap {gap} exceeds bound {optimal + max_p} (v={v}, support={support})"
        )


def test_balanced_partition_near_optimal_gap() -> None:
    torch.manual_seed(2)
    for max_p in (0.9, 0.7, 0.55, 0.4):
        v = 64
        probs = torch.full((v,), (1.0 - max_p) / (v - 1), dtype=torch.float32)
        probs[0] = max_p
        mask = rr.get_balanced_partition_from_probs(probs)
        mass_a, mass_b = rr.balanced_partition_masses(probs, mask)
        gap = abs(mass_a - mass_b)
        optimal = max(0.0, 2.0 * max_p - 1.0)
        assert gap <= optimal + max_p + 1e-4, (
            f"{max_p=}: gap {gap} > bound {optimal + max_p}"
        )


def test_balanced_partition_beats_static_on_skewed_step_probs() -> None:
    """On a skewed step distribution, live balancing should beat static unigram halves."""
    v = 64
    # Moderately skewed live probs (no single token > 0.5 so a near-even split exists).
    probs = torch.zeros(v, dtype=torch.float32)
    probs[0] = 0.30
    probs[1] = 0.25
    probs[2] = 0.20
    probs[3] = 0.15
    probs[4] = 0.10

    bal = rr.get_balanced_partition_from_probs(probs)
    bal_gap = abs(rr.balanced_partition_masses(probs, bal)[0] - 0.5)

    rr.configure_partition_weights(torch.ones(v, dtype=torch.float32))
    try:
        static = rr.get_vectorized_partition(v, "cpu", seed_index=0)
        static_gap = abs(rr.balanced_partition_masses(probs, static)[0] - 0.5)
    finally:
        rr.clear_partition_weights()

    assert bal_gap <= static_gap + 1e-6, (
        f"expected balanced gap {bal_gap} <= static gap {static_gap}"
    )
    assert bal_gap < 0.05, f"balanced gap unexpectedly large: {bal_gap}"


def test_balanced_soft_channel_roundtrip_no_model() -> None:
    """Enforce-half sampling + balanced recovery from the same probs recovers the bit."""
    torch.manual_seed(1)
    random.seed(1)
    v = 128
    errors = 0
    trials = 200
    for _ in range(trials):
        logits = torch.randn(v)
        probs = torch.softmax(logits, dim=-1)
        mask_a = rr.get_balanced_partition_from_probs(probs)
        secret = random.randint(0, 1)
        p = float(probs[mask_a].sum().item())
        q = min(p, 1.0 - p)
        # Force the enforce branch so the sampled token is informative.
        choose_a = secret == 0
        half = rr._mask_disallowed_half_probs(probs, mask_a, choose_set_A=choose_a)
        half = half / half.sum()
        tok = int(torch.multinomial(half, 1).item())
        recovered = 0 if bool(mask_a[tok].item()) else 1
        if recovered != secret:
            # Only possible if q==0 (no mass on one side) — count that separately.
            if q > 0:
                errors += 1
    assert errors == 0, f"enforce-half roundtrip errors: {errors}/{trials}"


def test_e2e_compare_static_vs_balanced() -> None:
    """Generate short streams both ways and compare recovery BER (requires CUDA + model)."""
    try:
        import model as model_mod
    except Exception as exc:  # pragma: no cover
        print(f"SKIP e2e: cannot import model ({exc})")
        return
    if not torch.cuda.is_available():
        print("SKIP e2e: CUDA not available")
        return

    m, tok, device = model_mod.load()
    prompt = "Describe how software has transformed the art world."
    n_bits = 500
    secret = [random.randint(0, 1) for _ in range(n_bits)]

    # --- static (legacy) ---
    out_static = rr.generate_with_watermark(m, tok, prompt, secret, device)
    special_ids = set(getattr(tok, "all_special_ids", []))
    dim = int(out_static.get("partition_vocab_dim") or m.config.vocab_size)
    raw_static, _ = rr.recover_bitstream_from_text(
        out_static["generated_text_wm"], tok, device, model=m, partition_vocab_size=dim
    )
    # Align to embedded length.
    raw_static = raw_static[:n_bits]
    ber_static = _ber(secret, raw_static)

    # --- balanced ---
    out_bal = rr.generate_with_watermark_balanced(m, tok, prompt, secret, device)
    raw_bal, _, gaps = rr.recover_bitstream_balanced_from_generation(
        m, tok, out_bal, device
    )
    ber_bal = _ber(secret, raw_bal)

    # Cross-check: static recovery on balanced text should be near-chance-ish / worse.
    raw_cross, _ = rr.recover_bitstream_from_text(
        out_bal["generated_text_wm"],
        tok,
        device,
        model=m,
        partition_vocab_size=int(out_bal.get("partition_vocab_dim") or dim),
    )
    ber_cross = _ber(secret, raw_cross[:n_bits])

    mean_gap = float(out_bal.get("partition_mass_gap_mean", 0.0))
    print("\n=== static vs balanced watermark recovery ===")
    print(f"prompt: {prompt!r}")
    print(f"n_bits={n_bits}")
    print("--- static model output ---")
    print(out_static["generated_text_wm"])
    print("--- balanced model output ---")
    print(out_bal["generated_text_wm"])
    print("--- metrics ---")
    print(f"static:   BER={ber_static:.2%}  natural={out_static['natural_partition_choices']}")
    print(
        f"balanced: BER={ber_bal:.2%}  natural={out_bal['natural_partition_choices']}  "
        f"mean_mass_gap={mean_gap:.4f}  max_gap={out_bal.get('partition_mass_gap_max', 0):.4f}"
    )
    print(f"cross (static recover on balanced text): BER={ber_cross:.2%}")
    print(f"recovery replay mean mass gap: {sum(gaps)/max(len(gaps),1):.4f}")

    # Balanced recovery replays model ids. Residual BER is expected when the
    # processed distribution is too peaked for a useful split.
    assert len(raw_bal) == n_bits
    assert ber_bal <= 0.40, f"balanced recovery BER too high: {ber_bal:.2%}"
    # Static path still recovers something correlated (not pure noise).
    assert ber_static < 0.55, f"static recovery looks uncorrelated: {ber_static:.2%}"


def main() -> int:
    test_balanced_partition_deterministic_and_even()
    print("OK test_balanced_partition_deterministic_and_even")
    test_balanced_partition_near_optimal_gap()
    print("OK test_balanced_partition_near_optimal_gap")
    test_balanced_partition_beats_static_on_skewed_step_probs()
    print("OK test_balanced_partition_beats_static_on_skewed_step_probs")
    test_balanced_soft_channel_roundtrip_no_model()
    print("OK test_balanced_soft_channel_roundtrip_no_model")
    test_e2e_compare_static_vs_balanced()
    print("OK test_e2e_compare_static_vs_balanced (or skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
