# Attribute-based Watermarking for LLMs
This repository is an implementation for ............ by .............

Currently the entire base construction is implemented, apart from adverserially robust classification for attribute generation

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). Python 3.11 or newer is required.

Install dependencies (including `maturin` for building PRC):

```sh
uv sync --extra dev
```

The first `uv sync` downloads a full PyTorch CUDA 12.6 wheel set, which is large and can take several minutes.

Building PRC (run after cloning or whenever `prc/` changes):

```sh
uv run maturin develop --release -m prc/Cargo.toml
```

## Run

```sh
uv run python app.py
```

You can also activate the project virtual environment (`.venv` after `uv sync`) and run `python app.py` normally.

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
- look into more operations replacing lists with bitwise etc
