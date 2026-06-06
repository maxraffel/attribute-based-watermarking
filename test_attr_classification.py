"""
End-to-end checks: watermarked generation, attributes from watermarked text (same as ``detect``),
PRC + CPRF inner products.

Run: uv run python test_attr_classification.py
"""

from __future__ import annotations

import logging
import sys

for _name in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
    logging.getLogger(_name).setLevel(logging.WARNING)

import model
import randrecover
import watermarking as wm
from check_report import CheckReporter, expect_cprf_ceval_ok
from text_attributes import CPRF_ATTR_DIM, VOCABULARY, derive_attributes, f_for_required_keywords


def _quiet_hf_http_loggers() -> None:
    for name in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _f_dot_attributes_mod(f: list[int], attributes: list[int], modulus: int) -> int:
    return sum(f[i] * attributes[i] for i in range(len(attributes))) % modulus


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

    rep.section("Generate (watermarked) and attributes from transcript")
    out = wm.generate(sk, prompt)
    text = out["generated_text_wm"]
    attributes = derive_attributes(text, sk.modulus)
    rep.console.print(f"  [dim]len(attributes)={len(attributes)} (expect {CPRF_ATTR_DIM}), CPRF modulus={sk.modulus}[/]")
    n = len(VOCABULARY)
    rep.console.print(f"  [dim]prefix from watermarked text (len {n}):[/] {attributes[:n]}")
    rep.console.print(f"  [dim]tail (first 8 of {len(attributes) - n}):[/] {attributes[n : n + 8]}...")
    encode_attributes = out["attributes"]
    if encode_attributes != attributes:
        rep.console.print(
            "  [yellow]Note: encode-time attributes (from greedy baseline) differ from verify-time "
            "attributes (from watermarked text); detect uses the latter.[/]"
        )

    dk_open = wm.issue(sk, [])
    dk_match = wm.issue(sk, ["medicine"])
    dk_wrong = wm.issue(sk, ["finance"])

    f_zero = [0] * CPRF_ATTR_DIM
    f_match = f_for_required_keywords(["medicine"])
    f_wrong = f_for_required_keywords(["finance"])

    rep.section("PRC + constrained keys (same expectations as app.py)")
    master_ok, _ = wm.master_detect(sk, text)
    rep.add_boolean("master_detect(good)", master_ok, True)
    d0, _ = wm.detect(dk_open, text)
    rep.add_boolean("detect(unconstrained)", d0, True)
    dm, _ = wm.detect(dk_match, text)
    rep.add_boolean("detect(matching policy)", dm, True)
    dw, _ = wm.detect(dk_wrong, text)
    rep.add_boolean("detect(wrong policy)", dw, False)
    m, tok, device = model.load()
    wrong_tx = randrecover.negative_control_transcript_like(
        text,
        tok,
        device,
        n_bits=wm.SECURITY_PARAM * wm.WM_BIT_REDUNDANCY,
        model=m,
    )
    mw, _ = wm.master_detect(sk, wrong_tx)
    rep.add_boolean("master_detect(wrong transcript)", mw, False)

    rep.section("CPRF inner product ⟨f,attributes⟩ mod m on recovered attributes")
    rep.console.print(
        f"  [dim]⟨f_open,attributes⟩ mod m =[/] {_f_dot_attributes_mod(f_zero, attributes, sk.modulus)}  "
        f"(unconstrained f is all zeros)"
    )
    rep.console.print(
        f"  [dim]⟨f_match,attributes⟩ mod m =[/] {_f_dot_attributes_mod(f_match, attributes, sk.modulus)}  "
        f"(medicine policy)"
    )
    rep.console.print(
        f"  [dim]⟨f_wrong,attributes⟩ mod m =[/] {_f_dot_attributes_mod(f_wrong, attributes, sk.modulus)}  "
        f"(finance policy)"
    )

    rep.add_boolean(
        "cprf: unconstrained agrees (⟨f,attributes⟩ ≡ 0)",
        expect_cprf_ceval_ok(f_zero, attributes, sk.modulus),
        True,
    )
    if master_ok:
        dm2, _ = wm.detect(dk_match, text)
        rep.add_boolean(
            "cprf: matching policy ⟺ detect(dk_match)",
            expect_cprf_ceval_ok(f_match, attributes, sk.modulus) == dm2,
            True,
        )
        dw2, _ = wm.detect(dk_wrong, text)
        rep.add_boolean(
            "cprf: wrong policy ⟺ detect(dk_wrong)",
            expect_cprf_ceval_ok(f_wrong, attributes, sk.modulus) == dw2,
            True,
        )

    rep.section("Alternate transcript → different attributes (CPRF sanity)")
    other_attributes = derive_attributes("Say only the word: hello.", sk.modulus)
    rep.console.print(
        f"  [dim]⟨f_zero,other_attributes⟩ mod m =[/] "
        f"{_f_dot_attributes_mod(f_zero, other_attributes, sk.modulus)}"
    )
    rep.add_boolean(
        "cprf: unconstrained still agrees on unrelated short text",
        expect_cprf_ceval_ok(f_zero, other_attributes, sk.modulus),
        True,
    )

    passed, total = rep.summary()
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
