import logging

import watermarking as wm

from check_report import CheckReporter


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    sk = wm.setup(1024)
    prompt = (
        "Explain the benefits of penicillin during the 19th and 20th centuries."
    )
    out = wm.generate(sk, prompt)
    text = out["generated_text_wm"]

    dk_open = wm.issue_unconstrained(sk)
    dk_match = wm.issue_keyword_policy(sk, ["medicine"])
    dk_wrong = wm.issue_keyword_policy(sk, ["finance"])

    rep = CheckReporter()
    rep.section("Black-box (watermarking API only)")
    rep.add_boolean("master_detect(good)", wm.master_detect(sk, prompt, text), True)
    rep.add_boolean("detect(unconstrained)", wm.detect(dk_open, prompt, text), True)
    rep.add_boolean("detect(matching policy)", wm.detect(dk_match, prompt, text), True)
    rep.add_boolean("detect(wrong policy)", wm.detect(dk_wrong, prompt, text), False)
    rep.add_boolean(
        "master_detect(wrong prompt)",
        wm.master_detect(sk, "Say only the word: hello.", text),
        False,
    )
    rep.summary()


if __name__ == "__main__":
    main()
