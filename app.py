import watermarking as wm

from check_report import CheckReporter


def main():
    sk = wm.setup(1024)
    prompt = (
        "In a few short sentences, explain the meaning of life and why humans ask this question."
    )
    out = wm.generate(sk, prompt)
    text = out["generated_text_wm"]

    dk_open = wm.issue_unconstrained(sk)
    dk_match = wm.issue_keyword_policy(sk, ["meaning", "life", "nosuchtoken"])
    dk_wrong = wm.issue_keyword_policy(sk, ["python", "water"])

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
