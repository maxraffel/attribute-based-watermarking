from hashlib import sha256
from typing import List, Tuple

import torch
import cprf
import prc
import randrecover
from transformers import AutoModelForCausalLM, AutoTokenizer

from x_derivation import derive_x, get_keybert

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
MODEL = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)

SECURITY_PARAM = 300


def setup(size: int) -> cprf.MasterKey:
    return cprf.keygen(size, SECURITY_PARAM)


def issue(sk: cprf.MasterKey, f: List[int]) -> cprf.ConstrainedKey:
    return sk.constrain(f)


def generate(
    sk: cprf.MasterKey,
    prompt: str,
) -> Tuple[dict, List[int]]:
    baseline = randrecover.generate_baseline(
        MODEL, TOKENIZER, prompt, SECURITY_PARAM, DEVICE
    )
    x, _ = derive_x(baseline, SECURITY_PARAM, sk.modulus, keybert=get_keybert())
    r = sk.eval(x)
    prc.set_code_length(SECURITY_PARAM)
    s = prc.key_gen_from_seed(sha256(r).digest())
    c = prc.encode(s)
    secret_bitstream = [1 if b else 0 for b in c]
    out = randrecover.generate_with_watermark(
        MODEL, TOKENIZER, prompt, secret_bitstream, DEVICE
    )
    return out, x


def detect(dk: cprf.ConstrainedKey, x: List[int], out: str) -> bool:
    n = len(x)
    prc.set_code_length(n)
    recovered_r = dk.c_eval(x)
    recovered_s = prc.key_gen_from_seed(sha256(recovered_r).digest())
    bits, _ = randrecover.recover_bitstream_from_text(out, TOKENIZER, DEVICE)
    bits = (bits + [0] * n)[:n]
    return prc.detect(recovered_s, [bool(b) for b in bits])


def master_detect(sk: cprf.MasterKey, x: List[int], out: str) -> bool:
    n = len(x)
    prc.set_code_length(n)
    r = sk.eval(x)
    s = prc.key_gen_from_seed(sha256(r).digest())
    bits, _ = randrecover.recover_bitstream_from_text(out, TOKENIZER, DEVICE)
    bits = (bits + [0] * n)[:n]
    return prc.detect(s, [bool(b) for b in bits])
