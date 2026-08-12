# MINIX AI Stage 1 platform readiness tests

This directory implements the audit's platform-readiness gate: tests 1-11.
It does not yet implement matrix multiplication, softmax, attention, or a
transformer block. Those become the Stage 1 functional implementation after
this gate passes.

## Files

- `ai_stage1_probe.c` — one unprivileged test program containing tests 1-11.
- `Makefile` — native MINIX/NetBSD-style build.
- `Makefile.host` — optional comparison build on another Unix-like host.
- `run-stage1-gate.sh` — builds, runs and logs the full gate.

## Copy into MINIX

A suitable location is:

```sh
mkdir -p /usr/src/minix/tests/ai_stage1
cp ai_stage1_probe.c Makefile run-stage1-gate.sh README.md \
    /usr/src/minix/tests/ai_stage1/
cd /usr/src/minix/tests/ai_stage1
```

The tests can also be kept outside the system source tree while they are being
developed.

## Build

```sh
make
```

If compiler selection is needed:

```sh
make CC=clang
```

## Run safely

Take a VirtualBox snapshot before the allocation test, then start with 64 MiB:

```sh
./ai_stage1_probe --max-mib 64
```

Run the planned 128 MiB ceiling after the 64 MiB run passes:

```sh
./ai_stage1_probe --max-mib 128
```

Run and preserve a timestamped log:

```sh
./run-stage1-gate.sh
```

or:

```sh
MAX_MIB=64 ./run-stage1-gate.sh
```

## Individual tests

```sh
./ai_stage1_probe --list
./ai_stage1_probe --test 3
./ai_stage1_probe --test 5 --max-mib 64
```

## Exit status

- `0`: all selected pass/fail tests passed. Informational tests do not fail the run.
- `1`: at least one selected test failed.
- `2`: command-line usage error.

## Test coverage

1. ABI and type sizes
2. Basic FP32 arithmetic
3. `libm` conformance
4. NaN and infinity handling
5. Progressive heap allocation and page touching
6. `calloc` multiplication overflow and allocation failure
7. Exact binary FP32 round trip
8. Truncated-file detection
9. `fseeko`, `ftello`, `fstat` and loader-side bounds checks
10. `fseek(-1L, SEEK_SET)` behaviour characterisation
11. `CLOCK_MONOTONIC` resolution and ordering

## Safety notes

- The allocation test holds only one test allocation at a time and frees it
  before moving to the next size.
- Memory is touched every 4096 bytes so a successful `malloc` is not accepted
  as proof of usable memory without backing pages.
- The near-`SIZE_MAX` request is never written to even if it unexpectedly
  succeeds.
- Test 10 is informational. The eventual tensor loader must reject negative
  offsets and malformed dimensions before calling any seek function.
- Keep tensor/model buffers on the heap; the audited default stack limit is
  only 4 MiB.

## Recommended gate

Passing tests 1-11 means the platform is ready to begin the tensor-kernel code.
The next code set should add matrix multiplication, stable softmax, RMSNorm,
one attention head, and then one tiny transformer block with trusted reference
vectors.
