"""Smoke checks for the watermarking API; set ``WM_DETECT_VERBOSE=1`` for failure excerpts."""

import logging
import os

import watermarking as wm

from check_report import CheckReporter
from closed_vocab import VOCABULARY
from detect_diagnostics import log_detect_mismatch


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    verbose_diag = os.environ.get("WM_DETECT_VERBOSE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    sk = wm.setup(1024)
    prompt = (
        "In three short paragraphs, describe a common outpatient medical procedure "
        "and what the patient should expect before and after the visit."
    )
    out = wm.generate(sk, prompt)
    text = out["generated_text_wm"]
    x_encode = out["attr_x"]
    baseline_text = out["baseline_text"]
    modulus = sk.modulus

    dk_open = wm.issue_unconstrained(sk)
    dk_match = wm.issue_keyword_policy(sk, ["medicine"])
    dk_wrong = wm.issue_keyword_policy(sk, ["finance"])

    rep = CheckReporter()

    def check(name: str, got: object, expected: bool, wm_text: str) -> None:
        rep.add_boolean(name, got, expected)
        if bool(got) == bool(expected):
            return
        log_detect_mismatch(
            scenario="app",
            test_name=name,
            expect_true=expected,
            got=got,
            x_encode=x_encode,
            wm_text=wm_text,
            modulus=modulus,
            baseline_text=baseline_text,
            vocabulary=VOCABULARY,
            verbose=verbose_diag,
            rich_console=rep.console,
        )

    rep.section("Black-box (watermarking API only)")
    check("master_detect(good)", wm.master_detect(sk, text), True, text)
    check("detect(unconstrained)", wm.detect(dk_open, text), True, text)
    check("detect(matching policy)", wm.detect(dk_match, text), True, text)
    check("detect(wrong policy)", wm.detect(dk_wrong, text), False, text)
    wrong = "This is unrelated text used only as a negative control."
    check("master_detect(wrong transcript)", wm.master_detect(sk, wrong), False, wrong)
    rep.summary()


if __name__ == "__main__":
    main()
