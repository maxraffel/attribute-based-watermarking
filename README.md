# Attribute-based Watermarking for LLMs
This repository is an implementation for ............

Currently only the RandRecover functionality is implemented
Run with
```sh
python -m venv .venv  # requirements are tested for Python 3.11
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Building PRC:
```sh
maturin develop --release -m prc/Cargo.toml
python prc/example.py  # make sure we can run the code
```

Currently model and size of generation/randomness to recover is hardcoded

Uses PRC from here https://github.com/cloudflareresearch/poc-watermark/tree/main/prc
Uses CPRF from here https://github.com/sachaservan/cprf.git

## TODO/Issues:
- proper citations/references, maybe correctly fork/branch from utilized repos
- Add CPRF
- PRC package does not encode on a key and value, just a key, so it just detects, not decodes, this runs into further issues because it means the key length must match the length of the randomness recovered from the input