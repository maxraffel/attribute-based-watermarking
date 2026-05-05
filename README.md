# Attribute-based watermarking for LLMs

This repository ties a **CPRF** (constrained pseudorandom function) attribute vector `x` to a **PRC** (pseudorandom code) watermark on text from a causal LM. **Encoding** still sets `x = derive_x(baseline)` where the baseline is the greedy continuation of the prompt. **`detect` / `master_detect` only take `watermarked_text`**: they set `x = derive_x(watermarked_text)` (same zero-shot prefix + fixed tail as in `attr_x_nli.derive_x`). Verification succeeds only when that reconstructed `x` matches the `x` used at encode time (the watermarked output usually needs to preserve the same **argmax primary label** as the baseline). **`generate` still returns `attr_x`** (the encode-time `x`) for debugging or auditing.

## How it works

1. **Baseline text** — For a fixed prompt, the code runs sampling with the same new-token horizon as the watermarked path (``watermarking.wm_channel_bits_length()`` = logical ``SECURITY_PARAM`` × ``WM_BIT_REDUNDANCY``), set via ``set_prc_code_length`` / ``set_wm_bit_redundancy``.
2. **Attribute `x`** — `derive_x` in `attr_x_nli.py` maps that string to an integer vector of length `CPRF_ATTR_DIM` (see `closed_vocab.py`):
   - **Prefix** (`len(VOCABULARY)` entries): each closed-vocab label gets a score from a Hugging Face **`zero-shot-classification`** pipeline (`multi_label=False` so scores are a **softmax** over candidates—comparable for picking one primary subject; `hypothesis_template` from **`NLI_HYPOTHESIS_TEMPLATE`** in `attr_x_nli.py`). Coordinate `i` is **0** for the label with the **highest** score (ties: smallest index in `VOCABULARY`), and **1** for all others.
   - **Tail** (`ATTR_TAIL_DIM` entries): a **fixed**, predetermined sequence (expanded from a project constant with SHAKE256, then reduced mod the CPRF modulus)—the same for every baseline, independent of text or prefix. Keyword constraints **do not** depend on the tail (`f` is padded with zeros on the tail).
3. **CPRF** — A master key is generated with dimension `CPRF_ATTR_DIM`. The inner product **⟨f, x⟩ ≡ 0 (mod modulus)** is what makes a constrained key’s `c_eval(x)` agree with `eval(x)` for a given policy vector `f`. Unconstrained keys use **f = 0**. Keyword policies set **f** only on indices of known required labels in `VOCABULARY`.
4. **PRC** — `r = sk.eval(x)` (or `dk.c_eval(x)` under a policy). The PRC secret is keyed from **SHA256(r)**; bits are embedded with `randrecover` during generation and recovered for detection.

Because the **prefix** of `x` comes from zero-shot scores on whichever string is passed to `derive_x`, encode-time `x` (baseline) and verify-time `x` (watermarked transcript) can differ; only the **tail** is shared and fixed. If they differ, `master_detect` fails even for a valid watermark string.

## Layout

| Piece | Role |
|--------|------|
| `watermarking.py` | Llama load, `set_prc_code_length`, `setup` / `generate` / `detect` / `master_detect`, `issue_*` |
| `closed_vocab.py` | `VOCABULARY`, `ATTR_TAIL_DIM`, `CPRF_ATTR_DIM`, `f_for_required_keywords` |
| `attr_x_nli.py` | Zero-shot scores → prefix bits; fixed tail; one INFO log line with final scores per label |
| `randrecover.py` | Baseline gen, watermark injection, bit recovery |
| `cprf/` | CPRF shared library (ctypes) |
| `prc/` | PRC Rust extension (maturin) |
| `x_derivation.py` | Optional KeyBERT-based experiment (not used by `watermarking.py`) |

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). Python 3.11 or newer is required.

Install dependencies (including `maturin` for building PRC):

```sh
uv sync --extra dev
```

The first `uv sync` pulls a full PyTorch CUDA 12.6 wheel set from the configured index, which is large and can take several minutes.

Build PRC after cloning or whenever `prc/` changes:

```sh
uv run maturin develop --release -m prc/Cargo.toml
```

**Models** — At runtime the project downloads from Hugging Face:

- **Causal LM:** `meta-llama/Llama-3.2-1B-Instruct` (see `watermarking.py`).
- **Zero-shot NLI:** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (see `attr_x_nli.py`).

You need enough disk space and (for GPU) VRAM for both; the DeBERTa checkpoint is large relative to the 1B instruct model.

## Run

Demo checks (Rich pass/fail):

```sh
uv run python app.py
```

**Google Colab:** open [`colab.ipynb`](colab.ipynb) in Colab (or upload it with the repo). It uses `%pip`, `notebook_login()`, and `runpy` instead of a shell-only workflow. For a **private** GitHub repo, add a Colab secret **`GITHUB_TOKEN`** (classic PAT with `repo`, or fine-grained read on the repo) and enable notebook access; the notebook clones via `https://oauth2:<token>@github.com/...`. Section **§6** in the notebook pulls the latest commit (`git fetch` / `git pull --ff-only`) and re-sets `origin` with the token when needed; then run **§7** for `app.py`.

Attribute + CPRF consistency checks (includes the same watermark checks plus explicit `x` and **f·x** reporting):

```sh
uv run python test_attr_classification.py
```

Policy + PRC benchmark (one table per prompt: success counts, BER, timings, prefix ``x`` agreement):

```sh
uv run python benchmark_policy_detection.py --runs 50 --code-length 300
```

Repeatable ``--prompt-case id:prompt`` adds cases (default: two built-ins). Use ``--reuse-key`` to reuse one CPRF key per prompt id across runs. ``generate`` / ``detect`` / ``master_detect`` expose timings and recovered bits for this script (see ``watermarking.py``).

You can also activate `.venv` and run `python app.py` as usual.

## Tuning

- **Primary label** — The prefix uses **argmax** over zero-shot scores (see `attr_x_nli.py`). To change what “subject” means for the model, edit **`NLI_HYPOTHESIS_TEMPLATE`** or **`VOCABULARY`**.
- **Vocabulary and CPRF size** — Edit **`VOCABULARY`** and **`ATTR_TAIL_DIM`** in `closed_vocab.py` (changing them changes `CPRF_ATTR_DIM` and invalidates old keys relative to new `x`).

PRC length / generation horizon: call ``watermarking.set_prc_code_length(n)`` before ``generate`` (default at import is 300), or pass ``--code-length`` to the benchmark script.

## Upstream components

- **PRC** — Based on [cloudflare/poc-watermark `prc`](https://github.com/cloudflareresearch/poc-watermark/tree/main/prc), with deterministic key generation as in this tree.
- **CPRF** — From [sachaservan/cprf](https://github.com/sachaservan/cprf), loaded via ctypes (`cprf/cprf.so`).

## TODO / issues

- Proper citations and forks for upstream repos.
- Decouple output length from encode horizon where possible.
- Broader automated tests (beyond `test_attr_classification.py` and `cprf/test_cprf.py`).
- Windowed decoding in `randrecover` instead of full-sequence decode per step.
- Optional: drop unused `keybert` dependency if `x_derivation.py` is removed or moved behind extras.
