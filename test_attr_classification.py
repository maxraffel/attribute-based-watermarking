"""
End-to-end attribute + CPRF checks: watermarked generation, explicit x, PRC detect, and f·x mod m.

Run: uv run python test_attr_classification.py
"""

from __future__ import annotations

import logging
import sys

# Before importing ``watermarking`` (HF downloads), avoid httpx INFO noise.
for _name in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
    logging.getLogger(_name).setLevel(logging.WARNING)

import watermarking as wm
from check_report import CheckReporter, expect_cprf_ceval_ok
from closed_vocab import CPRF_ATTR_DIM, VOCABULARY, f_for_required_keywords


def _quiet_hf_http_loggers() -> None:
    for name in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _f_dot_x_mod(f: list[int], x: list[int], modulus: int) -> int:
    return sum(f[i] * x[i] for i in range(len(x))) % modulus


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    _quiet_hf_http_loggers()

    rep = CheckReporter()
    modulus = 1024
    sk = wm.setup(modulus)
    prompt = (
        "Explain the benefits of penicillin during the 19th and 20th centuries."
    )

    rep.section("Generate (watermarked) and recover attribute x")
    out = wm.generate(sk, prompt)
    text = out["generated_text_wm"]
    x = wm.attr_x_for_prompt(sk, prompt)
    rep.console.print(f"  [dim]len(x)={len(x)} (expect {CPRF_ATTR_DIM}), CPRF modulus={sk.modulus}[/]")
    n = len(VOCABULARY)
    rep.console.print(f"  [dim]NLI prefix (absence bits, len {n}):[/] {x[:n]}")
    rep.console.print(f"  [dim]tail (first 8 of {len(x) - n}):[/] {x[n : n + 8]}...")

    dk_open = wm.issue_unconstrained(sk)
    dk_match = wm.issue_keyword_policy(sk, ["medicine"])
    dk_wrong = wm.issue_keyword_policy(sk, ["finance"])

    f_zero = [0] * CPRF_ATTR_DIM
    f_match = f_for_required_keywords(["medicine"])
    f_wrong = f_for_required_keywords(["finance"])

    rep.section("PRC + constrained keys (same expectations as app.py)")
    master_ok = wm.master_detect(sk, prompt, text)
    rep.add_boolean("master_detect(good)", master_ok, True)
    rep.add_boolean("detect(unconstrained)", wm.detect(dk_open, prompt, text), True)
    rep.add_boolean("detect(matching policy)", wm.detect(dk_match, prompt, text), True)
    rep.add_boolean("detect(wrong policy)", wm.detect(dk_wrong, prompt, text), False)
    rep.add_boolean(
        "master_detect(wrong prompt)",
        wm.master_detect(sk, "Say only the word: hello.", text),
        False,
    )

    rep.section("CPRF inner product f·x mod m on recovered x")
    rep.console.print(
        f"  [dim]⟨f_open,x⟩ mod m =[/] {_f_dot_x_mod(f_zero, x, sk.modulus)}  "
        f"(unconstrained f is all zeros)"
    )
    rep.console.print(
        f"  [dim]⟨f_match,x⟩ mod m =[/] {_f_dot_x_mod(f_match, x, sk.modulus)}  "
        f"(medicine policy)"
    )
    rep.console.print(
        f"  [dim]⟨f_wrong,x⟩ mod m =[/] {_f_dot_x_mod(f_wrong, x, sk.modulus)}  "
        f"(finance policy)"
    )

    rep.add_boolean(
        "cprf: unconstrained agrees (f·x ≡ 0)",
        expect_cprf_ceval_ok(f_zero, x, sk.modulus),
        True,
    )
    # When the watermark verifies, constrained detect matches iff f·x ≡ 0.
    if master_ok:
        rep.add_boolean(
            "cprf: matching policy ⟺ detect(dk_match)",
            expect_cprf_ceval_ok(f_match, x, sk.modulus) == wm.detect(dk_match, prompt, text),
            True,
        )
        rep.add_boolean(
            "cprf: wrong policy ⟺ detect(dk_wrong)",
            expect_cprf_ceval_ok(f_wrong, x, sk.modulus) == wm.detect(dk_wrong, prompt, text),
            True,
        )

    wrong_prompt = "Say only the word: hello."
    x_wrong = wm.attr_x_for_prompt(sk, wrong_prompt)
    rep.section("Wrong baseline x (different prompt) vs same watermark text")
    rep.console.print(f"  [dim]⟨f_zero,x_wrong⟩ mod m =[/] {_f_dot_x_mod(f_zero, x_wrong, sk.modulus)}")
    rep.add_boolean(
        "cprf: wrong-prompt x still satisfies unconstrained f",
        expect_cprf_ceval_ok(f_zero, x_wrong, sk.modulus),
        True,
    )

    passed, total = rep.summary()
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
