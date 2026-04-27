# Pseudorandom codes implemented in rust

This folder contains a rust crate, `prc`, implementing a few zero-bit,
pseudorandom codes (ia.cr/2024/235) from the literature.

**WARNING:** This code is not production ready and is intended for experimental
evaluation only. No effort has been made to resist side channels. Users should
also expect bugs that impact security.

The key generation algorithm for LDPC and LDPC2 is rather slow, so you'll want
to compile for release:

```
➜  prc git:(cpatton/update-prc) ✗ cargo run --example test --release
   Compiling pyo3-build-config v0.23.3
   Compiling pyo3-macros-backend v0.23.3
   Compiling pyo3-ffi v0.23.3
   Compiling pyo3 v0.23.3
   Compiling pyo3-macros v0.23.3
   Compiling prc v0.1.0 (/Users/chris/bitbucket.cfdata.org/tbrooks-mejia/generative_image_watermarking/prc)
    Finished `release` profile [optimized] target(s) in 5.93s
     Running `target/release/examples/test`
LDCP2 w/ plausibly pseudorandom parameters
Generating key (this may take a few seconds) ...
Generated key in 3.066867666s
Generated codeword in 187.041µs
Decoded codeword in 692.416µs
Let's try some non-codewords ...
Let's try tweaking 1638 bits of a bunch of codewords ...

LDCP w/ plausibly pseudorandom parameters
Generating key (this may take a few seconds) ...
Generated key in 990.917417ms
Generated codeword in 180.375µs
Decoded codeword in 169.75µs
Let's try some non-codewords ...
Let's try tweaking 163 bits of a bunch of codewords ...

LDCP cranked up as much as possible
Generating key (this may take a few seconds) ...
Generated key in 408.684625ms
Generated codeword in 189.208µs
Decoded codeword in 16.666µs
Let's try some non-codewords ...
Let's try tweaking 5406 bits of a bunch of codewords ...

PRF w/ somewhat pseudorandom parameters
Generating key (this may take a few seconds) ...
Generated key in 256.667µs
Generated codeword in 334.875µs
Decoded codeword in 193µs
Let's try some non-codewords ...
Let's try tweaking 163 bits of a bunch of codewords ...

PRF cranked up as much as possible
Generating key (this may take a few seconds) ...
Generated key in 268.083µs
Generated codeword in 1.302542ms
Decoded codeword in 617.791µs
Let's try some non-codewords ...
Let's try tweaking 819 bits of a bunch of codewords ...
```

## Python wrapper

A wrapper for Python via maturin/PyO3 is provided.

From the **repository root** of this project (recommended):

```sh
uv sync --extra dev
uv run maturin develop --release -m prc/Cargo.toml
uv run python prc/example.py
```

Alternatively, create a virtual environment, install `maturin`, run `maturin develop --release` from this `prc` directory, then `python example.py`.
