# MINIX 3.3.0 Audit: Priority Summary

## Executive conclusion
Stage 1 is ready to begin without MINIX source changes, provided that runtime conformance tests are executed first and the initial scope remains a small unprivileged single-CPU FP32 workload using buffered regular-file I/O.

## Blocking findings
There are zero confirmed blockers that must be fixed before writing and running a small unprivileged FP32 tensor program.

## Runtime validation items
1. Compiler and ABI sanity.
2. Required libm operations.
3. Heap allocation and allocation failure.
4. Regular-file binary I/O.
5. Matrix multiplication.
6. Numerically stable softmax.
7. Normalisation.
8. Attention.
9. Repeat-run stability.
10. Process-failure isolation.

## Non-blocking operating-system limitations
- 32-bit address-space constraints limit practical model and buffer sizing.
- Default stack limit is 4 MiB and must be respected by implementation choices.
- Single-CPU execution is slower but acceptable for Stage 1 conformance and functional validation.
- CLOCK_MONOTONIC tick wrap on i386 is a strongly supported long-uptime limitation.
- Optional file-backed mmap is not required for Stage 1 and can be avoided.
- VM region low-end shrink and split limitations are not required by ordinary malloc-based Stage 1 paths.
- Unrelated disabled features remain outside Stage 1 scope unless they are explicitly exercised.

## Rejected findings
- VFS concurrent character-device filp operations were excluded because Stage 1 uses regular-file I/O and does not require this trigger path.
- Controlling-TTY select cleanup was excluded because Stage 1 does not depend on TTY select teardown semantics.
- VM low-end shrink and split operations were excluded as non-blocking for ordinary malloc/fread/libm workflows.
- ISO 9660 findings were excluded as unrelated to Stage 1 regular-file execution.
- X11 findings were excluded as unrelated to Stage 1 command-line FP32 runtime work.
- USB findings were excluded as unrelated to Stage 1 compute-path requirements.
- Networking findings were excluded as unrelated to Stage 1 local regular-file inference path.
- SMP-only issues were excluded because Stage 1 accepts single-CPU execution.
- fseek SEEK_SET zero-extension was excluded as a confirmed defect because it is explicitly documented compatibility behavior intended to allow 4 GiB seek values via signed long input patterns.

## Project priority
Immediate priority is runtime conformance testing, not operating-system modification.
