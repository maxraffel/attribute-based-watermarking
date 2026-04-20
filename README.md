# Attribute-based Watermarking for LLMs
This repository is an implementation for ............ by .............

Currently the entire base construction is implemented, apart from adverserially robust classification for attribute generation

Building PRC (make sure to run this at the start each time in case of update):
```sh
maturin develop --release -m prc/Cargo.toml
```

Run with
```sh
python -m venv .venv  # requirements are tested for Python 3.11
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Currently model and size of generation/randomness to recover is hardcoded

Uses PRC from here https://github.com/cloudflareresearch/poc-watermark/tree/main/prc
- Modified to enable deterministic key generation
Uses CPRF from here https://github.com/sachaservan/cprf.git
- Linked using ctypes

## TODO/Issues:
- Proper citations/references, maybe correctly fork/branch from utilized repos
- Make the length of output be separate from the length of string encoded
- Extensive test suite
- Instead of decoding entire output for each token generated, just decode a window of tokens, but to avoid causing more retokenization issues from cutting off the window arbitrarily, find a matching token to sync up.