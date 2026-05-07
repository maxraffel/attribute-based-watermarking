# Attribute-based Undetectable Watermarking for Generative AI Models

Submission for Neurips 2026

This project implements text watermarking using:

- **CPRF** (constrained pseudorandom function) to bind policy/attributes
- **PRC** (pseudorandom code) to encode and detect watermark bits in generated text
- A zero-shot attribute extractor (`attr_x_nli.py`) to derive an attribute vector from text

The fastest way to get started is:

1. Install dependencies
2. Build the PRC extension
3. Run `app.py`

---

## Quick Start

### 1) Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Rust toolchain (required by `maturin` to build `prc`)

### 2) Install dependencies

```sh
uv sync --extra dev
```

### 3) Build PRC extension

Run this after cloning, and again any time `prc/` changes:

```sh
uv run maturin develop --release -m prc/Cargo.toml
```

### 4) Run the main demo

```sh
uv run python app.py
```

This runs an end-to-end pass with Rich output: setup, generation, watermark checks, and policy checks.

---

## How to Run the Project

### Main command

```sh
uv run python app.py
```

### Useful optional environment variables

`app.py` supports these overrides:

- `APP_CODE_LENGTH` (or `WATERMARK_CODE_LENGTH`)
- `APP_WM_BIT_REDUNDANCY` (or `WATERMARK_WM_BIT_REDUNDANCY`)

Example:

```sh
APP_CODE_LENGTH=300 APP_WM_BIT_REDUNDANCY=3 uv run python app.py
```

### Other scripts

- Attribute + CPRF consistency checks:

  ```sh
  uv run python test_attr_classification.py
  ```

- Policy detection benchmark:

  ```sh
  uv run python benchmark_policy_detection.py --runs 50 --code-length 300
  ```

---

## What the Pipeline Does

At a high level:

1. Generate baseline text from a prompt.
2. Derive attribute vector `x` from text (`attr_x_nli.py`):
   - Prefix from zero-shot label scores over `VOCABULARY`
   - Fixed tail from project constants
3. Use CPRF to compute a seed input (`sk.eval(x)` or constrained `dk.c_eval(x)`).
4. Hash that output into a PRC key and embed/detect watermark bits.

Important behavior:

- During generation, `x` is derived from the baseline text.
- During detection, `x` is derived from the watermarked text.
- Detection is sensitive to whether those derived attributes match in the way the protocol expects.

---

## Project Layout

- `app.py` - Main end-to-end demo/check runner
- `watermarking.py` - Core API: setup/generate/detect/master_detect
- `attr_x_nli.py` - NLI-based attribute derivation
- `closed_vocab.py` - Vocabulary + attribute vector sizing helpers
- `randrecover.py` - Watermark embedding/recovery channel logic
- `benchmark_policy_detection.py` - Repeated benchmark runs
- `test_attr_classification.py` - Attribute/CPRF focused checks
- `cprf/` - CPRF shared library integration
- `prc/` - PRC Rust extension
- `colab.ipynb` - Notebook workflow

---

## Models Downloaded at Runtime

The first run downloads models from Hugging Face:

- LM: `meta-llama/Llama-3.2-1B-Instruct`
- Zero-shot model: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`

Make sure you have enough disk space (and VRAM if running on GPU).

---

## Colab

Use `colab.ipynb` for a notebook-based setup/run flow (including `%pip` installs and notebook login flow).

For private repos, provide a `GITHUB_TOKEN` secret in Colab so the notebook can clone/pull.

---

## Tuning Notes

- Change subject behavior by editing:
  - `NLI_HYPOTHESIS_TEMPLATE` in `attr_x_nli.py`
  - `VOCABULARY` in `closed_vocab.py`
- Change attribute size by editing `ATTR_TAIL_DIM` in `closed_vocab.py`.
- Change PRC code length with:
  - `APP_CODE_LENGTH` / `WATERMARK_CODE_LENGTH`, or
  - `set_prc_code_length(...)` in code.

---

## License and Upstream

Third-party CPRF/PRC components are used under their respective MIT licenses.

- PRC base: [cloudflare/poc-watermark `prc`](https://github.com/cloudflareresearch/poc-watermark/tree/main/prc)
- CPRF base: [sachaservan/cprf](https://github.com/sachaservan/cprf)
