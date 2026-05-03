from hashlib import sha256
from typing import List

import torch
import cprf
import prc
import randrecover
from transformers import AutoModelForCausalLM, AutoTokenizer

from attr_x_nli import derive_x
from closed_vocab import CPRF_ATTR_DIM, issue_constrained_key_for_keywords

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading with device: ", DEVICE)

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
MODEL = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)

SECURITY_PARAM = 300


def _baseline(prompt: str) -> str:
    return randrecover.generate_baseline(
        MODEL, TOKENIZER, prompt, SECURITY_PARAM, DEVICE
    )


def setup(modulus: int) -> cprf.MasterKey:
    return cprf.keygen(modulus, CPRF_ATTR_DIM)


def issue(sk: cprf.MasterKey, f: List[int]) -> cprf.ConstrainedKey:
    return sk.constrain(f)


def issue_unconstrained(sk: cprf.MasterKey) -> cprf.ConstrainedKey:
    """Constrained key with f = 0 (always agrees with master eval for any x)."""
    return issue(sk, [0] * CPRF_ATTR_DIM)


def issue_keyword_policy(sk: cprf.MasterKey, required: List[str]) -> cprf.ConstrainedKey:
    """
    Constrained key: inner product uses only the label prefix of x.
    Detection agrees with the master key when every known required label has an NLI
    ``match'' (x[i]=0) on the greedy baseline for those indices.
    """
    return issue_constrained_key_for_keywords(sk, required)


def attr_x_for_prompt(sk: cprf.MasterKey, prompt: str) -> List[int]:
    return derive_x(_baseline(prompt), sk.modulus)


def generate(sk: cprf.MasterKey, prompt: str) -> dict:
    baseline = _baseline(prompt)
    x = derive_x(baseline, sk.modulus)
    r = sk.eval(x)
    prc.set_code_length(SECURITY_PARAM)
    c = prc.encode(prc.key_gen_from_seed(sha256(r).digest()))
    bits = [1 if b else 0 for b in c]
    return randrecover.generate_with_watermark(MODEL, TOKENIZER, prompt, bits, DEVICE)


def detect(dk: cprf.ConstrainedKey, prompt: str, watermarked_text: str) -> bool:
    """Recover PRC bits; x is derived from the greedy ``prompt`` baseline (NLI label bits + tail)."""
    x = derive_x(_baseline(prompt), dk.modulus)
    prc.set_code_length(SECURITY_PARAM)
    recovered_s = prc.key_gen_from_seed(sha256(dk.c_eval(x)).digest())
    bits, _ = randrecover.recover_bitstream_from_text(watermarked_text, TOKENIZER, DEVICE)
    bits = (bits + [0] * SECURITY_PARAM)[:SECURITY_PARAM]
    return prc.detect(recovered_s, [bool(b) for b in bits])


def master_detect(sk: cprf.MasterKey, prompt: str, watermarked_text: str) -> bool:
    """Same as ``detect`` but using the master key (oracle verifier)."""
    x = derive_x(_baseline(prompt), sk.modulus)
    prc.set_code_length(SECURITY_PARAM)
    s = prc.key_gen_from_seed(sha256(sk.eval(x)).digest())
    bits, _ = randrecover.recover_bitstream_from_text(watermarked_text, TOKENIZER, DEVICE)
    bits = (bits + [0] * SECURITY_PARAM)[:SECURITY_PARAM]
    return prc.detect(s, [bool(b) for b in bits])
