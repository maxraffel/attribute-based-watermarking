# Attribute-based Undetectable Watermarking for Generative AI Models

Submission for Neurips 2026

This project implements text watermarking using:

- **CPRF** (constrained pseudorandom function) to bind policy/attributes
- **PRC** (pseudorandom code) to encode and detect watermark bits in generated text
- A label classifier (`text_attributes.py`) to derive an attribute vector from text

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

Edit constants at the top of `app.py`, or call `model.configure(...)` before running. For the LM hub id and sampling overrides (`temperature`, `top_p`, `top_k`), see `model.py`.

### Other scripts

- Attribute + CPRF consistency checks:

  ```sh
  uv run python test_attr_classification.py
  ```

  Classifies a preset paragraph and prints label scores in a Rich table.

- Policy detection benchmark:

  ```sh
  uv run python benchmark_policy_detection.py --runs 50 --code-length 300
  ```

---

## What the Pipeline Does

At a high level:

1. Generate baseline text from a prompt.
2. Derive attribute vector from text (`text_attributes.derive_attributes`):
   - Prefix from label classification scores over `VOCABULARY`
   - Fixed tail from project constants
3. Use CPRF to compute a seed input (`sk.eval(attributes)` or constrained `dk.c_eval(attributes)`).
4. Hash that output into a PRC key and embed/detect watermark bits.

Important behavior:

- During generation, attributes are derived from the baseline text.
- During detection, attributes are derived from the watermarked text.
- Detection is sensitive to whether those derived attributes match in the way the protocol expects.

---

## Project Layout

- `app.py` - Main end-to-end demo/check runner
- `model.py` - Hugging Face LM hub id, sampling overrides, lazy loading
- `watermarking.py` - Core API: setup/generate/detect/master_detect
- `text_attributes.py` - Closed vocabulary, label classification, and `derive_attributes`
- `randrecover.py` - Watermark embedding/recovery channel logic
- `benchmark_policy_detection.py` - Repeated benchmark runs
- `test_attr_classification.py` - Simple label-classification demo on sample text
- `cprf/` - CPRF shared library integration
- `prc/` - PRC Rust extension
- `colab.ipynb` - Notebook workflow

---

## Models Downloaded at Runtime

The first run downloads models from Hugging Face:

- LM: `meta-llama/Llama-3.2-1B-Instruct`
- Label classifier: `BAAI/bge-reranker-v2-m3` (GPU, bf16 weights)

Make sure you have enough disk space (and VRAM if running on GPU).

---

## Colab

Use `colab.ipynb` for a notebook-based setup/run flow (including `%pip` installs and notebook login flow).

For private repos, provide a `GITHUB_TOKEN` secret in Colab so the notebook can clone/pull.

---

## Tuning Notes

- Change subject behavior by editing:
  - `LABEL_QUERY_TEMPLATE` in `text_attributes.py`
  - `VOCABULARY` in `text_attributes.py`
- Change attribute size by editing `ATTR_TAIL_DIM` in `text_attributes.py`.
- Change PRC code length by setting `CODE_LENGTH` in `app.py` (or `wm.SECURITY_PARAM` + `prc.set_code_length(...)` in scripts).
- Change the watermark LM or sampling: `model.configure(model_id=..., temperature=..., top_p=..., top_k=...)`.

---

## License and Upstream

Third-party CPRF/PRC components are used under their respective MIT licenses.

- PRC base: [cloudflare/poc-watermark `prc`](https://github.com/cloudflareresearch/poc-watermark/tree/main/prc)
- CPRF base: [sachaservan/cprf](https://github.com/sachaservan/cprf)
