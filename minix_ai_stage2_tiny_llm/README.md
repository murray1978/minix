# MINIX AI Stage 2: tiny autoregressive LLM

This suite is the next step after the MINIX Stage 1 platform and tensor-kernel
probes. It loads a complete deterministic transformer checkpoint, runs
multi-layer causal inference with a KV cache, and generates byte tokens.

The code remains deliberately small and conservative:

- plain single-threaded FP32 C;
- ordinary unprivileged MINIX process;
- buffered regular-file checkpoint loading;
- no Python dependency on MINIX;
- no BLAS, pthreads, SIMD requirement, GPU, CUDA, or file-backed `mmap`;
- checked size arithmetic before every variable-sized allocation;
- model and reference payload checksums;
- heap-backed runtime state and KV caches.

This is a numerical and systems test model, not a trained language model. Its
bundled deterministic output for the prompt `MINIX` is `XXXXXXXX`.

## Model architecture

The bundled checkpoint uses:

```text
Vocabulary:          256 direct byte tokens
Context length:       16 tokens
Model width:           16
Attention heads:        4
Head width:             4
Transformer layers:     2
SwiGLU width:           32
Position encoding:    RoPE
Normalisation:        RMSNorm
LM output:            tied token embedding
Parameters:           9,296 FP32 values
Weight payload:       37,184 bytes
```

The forward path is:

```text
byte token
  -> token embedding
  -> 2 x [RMSNorm, causal attention/KV cache, residual,
          RMSNorm, SwiGLU feed-forward, residual]
  -> final RMSNorm
  -> tied embedding projection
  -> logits
  -> argmax next byte token
```

## Test numbering

Tests 18-20 remain reserved for the separate deliberate process-failure and
post-failure health checks from the Stage 1 audit. This suite starts at 21.

21. Checkpoint header, dimension, length, weight-count and checksum validation
22. Full multi-layer forward pass against Python-generated reference logits
23. KV-cache reset and deterministic replay
24. Deterministic autoregressive byte-token generation
25. Rejection of bad magic, bad weight count and truncated payloads
26. Repeated checkpoint load, generation and unload with allocation accounting
27. End-to-end token throughput benchmark

## Files

- `ai_stage2_tiny_llm.c` — loader, transformer runtime and tests 21-27
- `tiny_minix_llm.bin` — bundled deterministic FP32 checkpoint
- `tiny_minix_llm.ref` — prompt, expected generated tokens and per-step logits
- `generate_tiny_model.py` — host-side NumPy generator for both binary files
- `Makefile` — MINIX native build (`MAN=` avoids a man-page dependency)
- `Makefile.host` — optional Unix host comparison build
- `run-stage2-tiny-llm.sh` — logged MINIX build and test run
- `SHA256SUMS` — artifact hashes

## Install in the MINIX source workspace

Extract under `/usr/src`:

```sh
cd /usr/src
tar -xzf minix_ai_stage2_tiny_llm.tar.gz
cd minix_ai_stage2_tiny_llm
```

## Build on MINIX

```sh
make CC=clang
```

The executable is statically linked by the normal MINIX build rules unless the
local build configuration says otherwise.

List the tests:

```sh
./ai_stage2_tiny_llm --list
```

Run the complete suite:

```sh
./ai_stage2_tiny_llm --verbose
```

Expected summary:

```text
Summary: 7 passed, 0 failed, 0 informational
```

Run one test:

```sh
./ai_stage2_tiny_llm --test 24 --verbose
```

Increase repeated lifecycle testing and benchmark duration:

```sh
./ai_stage2_tiny_llm \
    --iterations 1000 \
    --bench-tokens 65536 \
    --verbose
```

The MINIX clock has approximately 16.67 ms resolution in the current VM. Use a
large `--bench-tokens` value so the benchmark spans many clock ticks.

## Logged run

```sh
chmod +x run-stage2-tiny-llm.sh
ITERATIONS=1000 BENCH_TOKENS=65536 ./run-stage2-tiny-llm.sh
```

Logs are stored under:

```text
results/stage2-tiny-llm-YYYYMMDD-HHMMSS.log
results/stage2-tiny-llm-latest.log
```

## Checkpoint validation

The loader rejects a checkpoint before inference when any of these checks fail:

- magic, format version or endian tag;
- zero, unreasonable or inconsistent dimensions;
- model width not divisible by the number of heads;
- odd head width, which this RoPE implementation does not support;
- calculated parameter count not matching the header;
- file length not exactly matching the declared FP32 payload;
- FNV-1a payload checksum mismatch;
- allocation-size overflow or allocation failure.

The reference file has independent dimension, length and checksum validation.

## Regenerating the checkpoint on Windows or Linux

The bundled files are ready to use on MINIX. Regeneration is optional and
requires Python 3 plus NumPy:

```sh
python3 generate_tiny_model.py --output .
```

Then rebuild and rerun the C suite on a host:

```sh
make -f Makefile.host clean all
./ai_stage2_tiny_llm --verbose
```

The generator uses explicit float32 accumulation order to keep the C and Python
reference traces close. The C comparison tolerance is still nonzero to allow
normal `libm` and compiler variation.

## What passing means

A clean MINIX run demonstrates that the OS can:

- load and validate a complete transformer checkpoint;
- allocate model weights, working buffers and a multi-layer KV cache;
- reproduce trusted multi-step logits;
- perform deterministic autoregressive generation;
- reject malformed model files safely;
- repeatedly load, infer and unload without suite-tracked leaks;
- benchmark end-to-end CPU token inference.

It does not yet demonstrate useful language quality, model quantisation,
large-model performance, sampling, a production tokenizer, GPU acceleration,
or an AI system service. Those belong to later stages.
