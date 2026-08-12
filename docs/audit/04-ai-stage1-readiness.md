# MINIX 3.3.0 AI Stage 1 Readiness

Verdict: suitable out of the box.

This verdict is based on source-supported capability for the Stage 1 scope, with remaining uncertainty handled through the dedicated runtime validation plan.

## Required packages
Source-supported facts:
- The base tree includes libc, libm, PM, VM, VFS, and the standard user-space build infrastructure required for a small C program linked with libm.
- No source evidence indicates a mandatory extra package for a minimal FP32 tensor test program.

Estimates and operational cautions:
- If a specific compiler binary is missing in a particular VM image, package installation may be needed operationally, but this is environment-specific rather than a source-level blocker.

## Compiler and linker
Source-supported facts:
- The build system and library layout provide C toolchain integration and libc/libm link paths for user programs.
- Stage 1 does not depend on unusual ABI features beyond ordinary 32-bit userland C and libm usage.

Estimates:
- Minor makefile wiring may be needed for a new test command integration step, but this is expected project work, not an OS readiness defect.

## FP32 and libm
Source-supported facts:
- Required single-precision operations are present in libm source and wrappers.
- No source evidence in the Stage 1 subset shows required FP32 primitives being stubbed out.

Estimates:
- Numeric quality and edge-case behavior must still be confirmed by runtime conformance tests.

## Heap and address space
Source-supported facts:
- MINIX defines a largest legal file offset constant as MAX_FILE_POS at 0x7fffffff.
- The system is 32-bit userland in this target context, constraining contiguous allocation headroom and total practical working set.
- VM and allocator paths required for ordinary malloc-based growth are present.

Estimates:
- Practical safe upper bounds depend on runtime fragmentation and co-resident service pressure.
- Recommended maximum initial allocation test target is 128 MiB contiguous, then increase in steps only after passing stability checks.

## Stack limit
Source-supported facts:
- The default stack limit for user processes is 4 MiB in this environment and has to be respected.

Estimates:
- Deep recursion and large stack buffers should be avoided; heap-backed buffers are preferred.

## Binary regular-file I/O
Source-supported facts:
- Regular-file buffered stdio paths are present and active.
- fseek SEEK_SET behavior intentionally zero-extends signed long input to support 4 GiB-style compatibility behavior.
- Regular-file operations may still be constrained by MAX_FILE_POS handling in the underlying path.

Interpretation for Stage 1:
- The fseek conversion is treated as a compatibility quirk, not a confirmed defect.
- Runtime characterization with fseek(fp, -1L, SEEK_SET) remains required.
- Stage 1 loader should use fseeko and explicit validation of offsets, tensor dimensions, and file lengths.
- libc modification is not recommended until runtime behavior and intended compatibility are established.

## Timing
Source-supported facts:
- PM do_gettime uses getuptime data and adds boottime for both CLOCK_REALTIME and CLOCK_MONOTONIC return values.
- On i386, clock_t is unsigned long and the kernel monotonic counter is stored as clock_t and increments per tick.
- Therefore CLOCK_MONOTONIC wrap after 2^32/system_hz seconds is a strongly supported long-uptime limitation.
- Privileged time-setting paths can mutate boottime, which can shift PM CLOCK_MONOTONIC reported absolute timestamp values.

Interpretation for Stage 1:
- This is low severity, high confidence, and non-blocking for short-lived unprivileged Stage 1 runs.
- It may matter for a future continuously running AI service.

## Single-CPU operation
Source-supported facts:
- Stage 1 scope does not require SMP correctness or high throughput.
- Single-CPU operation is acceptable for functional validation.

Estimates:
- Performance will be limited; latency and throughput are expected to be modest.

## Process isolation
Source-supported facts:
- PM/VFS/VM process lifecycle paths for ordinary crash handling and process exit are present.
- Unprivileged users cannot set system time directly through PM paths that require superuser privileges.

Estimates:
- Isolation behavior still needs empirical confirmation with deliberate fault-injection tests.

## Model-size guidance
Source-supported facts:
- 32-bit addressing, 4 MiB stack, and Stage 1 local-file workflow favor small model artifacts and bounded intermediate buffers.

Estimates:
- Keep initial model and activations in the low tens of MiB.
- Keep single-allocation requests initially at or below 128 MiB until allocator and repeat-run tests pass.

## Remaining uncertainties
- Runtime floating-point edge conformance under this specific VM image.
- Exact behavior of regular-file seek edge cases across filesystems and stdio wrappers.
- Long-uptime timing behavior and boottime-adjustment effects on monotonic timestamp interpretation.
- Real-world memory fragmentation impact under repeated allocate/infer/free loops.

## Final conclusion
- Stage 1 can begin without MINIX source changes: yes.
- Single-CPU execution acceptable for Stage 1: yes.
- Recommended maximum initial allocation test: 128 MiB contiguous.
- Preferred file-loading strategy: buffered regular-file I/O with explicit length and bounds checks, instead of file-backed mmap.
- Next task: runtime validation, not source repair.
