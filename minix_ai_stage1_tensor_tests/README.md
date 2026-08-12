# MINIX AI Stage 1 tensor-kernel tests

This is the second Stage 1 suite. It continues from the platform-readiness
probe and implements tests 12-17 from the audit.

The code is deliberately plain single-threaded FP32 C. It does not require
BLAS, pthreads, SIMD, a GPU, CUDA, file-backed `mmap`, or MINIX source changes.

## Coverage

12. Row-major FP32 matrix multiplication and a coarse 64x64 benchmark
13. Numerically stable max-subtracted softmax
14. RMSNorm
15. One scaled causal attention head
16. One tiny Llama-style pre-norm transformer block
17. Repeated allocate/infer/free stability with internal allocation accounting

The integrated block uses:

- sequence length 3
- model width 4
- one attention head
- RMSNorm
- causal scaled dot-product attention
- residual connections
- SwiGLU feed-forward width 6

These dimensions are intentionally tiny. The goal is numerical and runtime
conformance, not useful language generation yet.

## Files

- `ai_stage1_tensor.c` — tests 12-17 and the tensor kernels
- `Makefile` — native MINIX build; `MAN=` prevents a missing-man-page failure
- `Makefile.host` — optional Unix host comparison build
- `run-stage1-tensor.sh` — build, run, and retain a timestamped result log
- `reference_generator.py` — optional host-side regeneration of trusted vectors

## Build on MINIX

```sh
cd /usr/src/minix_ai_stage1_tensor_tests
make CC=clang
```

List tests:

```sh
./ai_stage1_tensor --list
```

Run the full suite:

```sh
./ai_stage1_tensor --verbose
```

Run an individual test:

```sh
./ai_stage1_tensor --test 15 --verbose
```

Adjust the repeat and benchmark counts:

```sh
./ai_stage1_tensor --iterations 1000 --bench-reps 64 --verbose
```

The MINIX timer is coarse, so larger `--bench-reps` values give a more useful
matmul estimate. Benchmark speed is informational and does not affect pass or
fail status.

## Logged run

```sh
chmod +x run-stage1-tensor.sh
ITERATIONS=1000 BENCH_REPS=64 ./run-stage1-tensor.sh
```

Results are retained under:

```text
results/stage1-tensor-YYYYMMDD-HHMMSS.log
results/stage1-tensor-latest.log
```

## Expected result

A clean run ends with:

```text
Summary: 6 passed, 0 failed, 0 informational
```

## Safety and portability

- Tensor-size multiplication and allocation-size addition are checked before
  every tracked allocation.
- Large tensor buffers are heap-backed rather than stack-backed.
- The suite uses deterministic hardcoded inputs and trusted reference outputs.
- Tolerances allow normal FP32/libm variation while remaining tight enough to
  expose indexing, masking, normalization, or projection mistakes.
- Test 17 verifies that all suite-managed allocations are released after every
  transformer iteration.
- The model weights are deterministic patterns generated from small integers;
  `reference_generator.py` documents and regenerates the reference output.

Passing this suite means MINIX can execute the basic mathematical dataflow of a
small causal transformer on one CPU. The next step is a tiny checkpoint loader,
token embeddings, a vocabulary, and autoregressive token generation.
