# Packaged partition unigram weights

Static token-frequency weights used by `randrecover.get_vectorized_partition` to
build weight-balanced watermark masks.

## Runtime

`ensure_partition_weights` **only loads** these files (or a matching legacy
`.cache/` copy). It does **not** recount a corpus during generation or detection.

Expected path pattern:

```text
data/partition_weights/<tokenizer_id_safe>_v<vocab_size>.pt
```

Example for the default LM:

```text
data/partition_weights/meta-llama_Llama-3.2-1B-Instruct_v128256.pt
```

## Rebuild (one-time / when changing tokenizer)

```sh
uv sync --extra weights
uv run python scripts/build_partition_weights.py --model-id meta-llama/Llama-3.2-1B-Instruct --max-tokens 5000000
```

Then commit the new `.pt` under this directory so clones and Colab do not recount.
