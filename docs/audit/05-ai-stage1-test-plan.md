# MINIX 3.3.0 AI Stage 1 Test Plan

Execution scope:
- Ordinary unprivileged user process.
- Single-CPU acceptable.
- Regular-file binary I/O path.
- No MINIX source modifications during this plan.

## Test 1: ABI and type-size report
- Objective: confirm fundamental ABI assumptions used by Stage 1 data structures and file formats.
- Source APIs exercised: printf, sizeof, limits.h macros, stdint types.
- Input: no external data; compile-time and runtime type report.
- Expected result: reported sizes and alignments are internally consistent and match assumptions used by loader and tensor code.
- Pass criteria: all required type/size assertions pass and report is captured.
- Failure interpretation: ABI mismatch requires adapting data structures and parser logic before further tests.
- Approximate memory requirement: less than 1 MiB.
- VM snapshot recommended: no.

## Test 2: basic FP32 arithmetic
- Objective: verify baseline float arithmetic and accumulation behavior.
- Source APIs exercised: C float operators and basic loops.
- Input: deterministic small vectors with known sums and products.
- Expected result: output within predefined tolerance for each operation.
- Pass criteria: all checks within tolerance and no floating-point exception trap.
- Failure interpretation: compiler flags, runtime environment, or numeric assumptions are incorrect.
- Approximate memory requirement: less than 1 MiB.
- VM snapshot recommended: no.

## Test 3: libm conformance
- Objective: validate required libm operations used by softmax and normalization.
- Source APIs exercised: expf, logf, sqrtf, powf, tanhf.
- Input: deterministic argument table including small, moderate, and large magnitudes.
- Expected result: finite outputs where mathematically expected and predictable domain/range behavior otherwise.
- Pass criteria: relative or absolute error within tolerance against precomputed references.
- Failure interpretation: library behavior or tolerance policy requires adjustment before model math validation.
- Approximate memory requirement: less than 2 MiB.
- VM snapshot recommended: no.

## Test 4: NaN and infinity handling
- Objective: characterize IEEE-style edge propagation relevant to robust inference.
- Source APIs exercised: isnan, isinf, signbit, expf, logf, sqrtf, division and multiplication.
- Input: NaN, positive infinity, negative infinity, signed zero, and representative finite values.
- Expected result: behavior matches documented C/libm semantics for each operation class.
- Pass criteria: all edge-case checks match expected classification and error signaling behavior.
- Failure interpretation: runtime guardrails must be strengthened to avoid undefined numeric states.
- Approximate memory requirement: less than 2 MiB.
- VM snapshot recommended: no.

## Test 5: progressively larger heap allocations
- Objective: establish practical contiguous allocation limits and fragmentation sensitivity.
- Source APIs exercised: malloc, free, memset.
- Input: stepped allocation sizes from small blocks up to target ceiling.
- Expected result: allocations succeed up to a stable threshold and fail cleanly beyond it.
- Pass criteria: no crash or corruption; failures return null and are handled.
- Failure interpretation: initial model-size targets must be reduced and allocation strategy adjusted.
- Approximate memory requirement: up to 128 MiB initial ceiling, then optional higher exploratory steps.
- VM snapshot recommended: yes.

## Test 6: calloc overflow and allocation failure
- Objective: confirm safe handling of overflow-sized requests and low-memory failures.
- Source APIs exercised: calloc, malloc, errno checks, free.
- Input: crafted element-count and size pairs that overflow multiplication, plus stress-induced low-memory scenarios.
- Expected result: overflow and exhaustion are rejected cleanly without corruption.
- Pass criteria: allocator returns null for invalid requests and program continues safely.
- Failure interpretation: memory-safety risk in loader or tensor workspace management.
- Approximate memory requirement: up to 16 MiB plus stress overhead.
- VM snapshot recommended: yes.

## Test 7: binary FP32 write/read round trip
- Objective: verify end-to-end buffered regular-file binary I/O fidelity.
- Source APIs exercised: fopen, fwrite, fflush, fseek or fseeko, fread, fclose.
- Input: deterministic FP32 array including signed, zero, denormal-like small, and large finite values.
- Expected result: readback bytes and decoded floats match written data.
- Pass criteria: exact byte match and per-element equality under raw-bit comparison.
- Failure interpretation: file mode, buffering, or conversion assumptions are incorrect.
- Approximate memory requirement: 4 to 32 MiB depending on array size.
- VM snapshot recommended: no.

## Test 8: truncated-file handling
- Objective: ensure loader detects and reports short read conditions safely.
- Source APIs exercised: fopen, fread, feof, ferror, fclose.
- Input: intentionally truncated binary tensor file relative to declared dimensions.
- Expected result: loader reports truncation and exits gracefully without out-of-bounds use.
- Pass criteria: deterministic error path, nonzero exit code, no crash.
- Failure interpretation: unsafe parser or missing file-length validation.
- Approximate memory requirement: less than 8 MiB.
- VM snapshot recommended: no.

## Test 9: fseeko and file-length validation
- Objective: validate robust offset and bounds handling using explicit 64-bit-capable seek API.
- Source APIs exercised: fseeko, ftello, fstat, fread.
- Input: files with known lengths and test offsets around boundaries.
- Expected result: valid offsets succeed, invalid offsets are rejected by loader checks before unsafe reads.
- Pass criteria: all boundary checks behave as designed; no silent wraparound acceptance.
- Failure interpretation: loader offset policy must be corrected before model loading.
- Approximate memory requirement: less than 8 MiB.
- VM snapshot recommended: no.

## Test 10: fseek negative SEEK_SET characterization
- Objective: characterize compatibility behavior and underlying regular-file limit handling.
- Source APIs exercised: fseek, ftell, fread, errno.
- Input: call pattern fseek(fp, -1L, SEEK_SET) on a regular file.
- Expected result: behavior is documented and reproducible, either explicit failure or large resulting offset handling.
- Pass criteria: observed result is captured and incorporated into loader policy.
- Failure interpretation: if behavior is ambiguous across filesystems, Stage 1 must avoid fseek-based signed-long offset logic.
- Approximate memory requirement: less than 2 MiB.
- VM snapshot recommended: no.

## Test 11: CLOCK_MONOTONIC resolution and ordering
- Objective: verify monotonic ordering and effective timing resolution for short-run benchmarks.
- Source APIs exercised: clock_gettime with CLOCK_MONOTONIC and clock_getres.
- Input: repeated timestamp sampling around known short delays and loop intervals.
- Expected result: nondecreasing sequence and resolution consistent with system_hz-derived granularity.
- Pass criteria: no backward steps in short-run unprivileged execution; measured granularity stable.
- Failure interpretation: timing utility layer must add defensive filtering or alternate measurement strategy.
- Approximate memory requirement: less than 2 MiB.
- VM snapshot recommended: no.

## Test 12: matrix multiplication
- Objective: validate core dense linear algebra kernel correctness.
- Source APIs exercised: malloc, free, deterministic loops, optional timing calls.
- Input: small and medium matrices with known reference outputs.
- Expected result: output numerically matches reference within FP32 tolerance.
- Pass criteria: error metrics remain below thresholds across tested dimensions.
- Failure interpretation: arithmetic kernel or indexing logic defect.
- Approximate memory requirement: 4 to 32 MiB.
- VM snapshot recommended: no.

## Test 13: numerically stable softmax
- Objective: validate max-subtracted stable softmax implementation.
- Source APIs exercised: expf, sum accumulation, normalization divide.
- Input: vectors containing both moderate and large-magnitude logits.
- Expected result: finite probabilities, each in range [0,1], summing near 1.
- Pass criteria: no overflow-induced infinities and sum error below tolerance.
- Failure interpretation: stability strategy or math-path handling is insufficient.
- Approximate memory requirement: less than 4 MiB.
- VM snapshot recommended: no.

## Test 14: RMSNorm or LayerNorm
- Objective: validate normalization kernel stability and correctness.
- Source APIs exercised: mean or RMS accumulation, sqrtf, scaling operations.
- Input: deterministic vectors spanning low and high variance.
- Expected result: normalized output satisfies expected invariants within tolerance.
- Pass criteria: invariant checks pass and no divide-by-zero instability.
- Failure interpretation: normalization math or epsilon handling defect.
- Approximate memory requirement: less than 8 MiB.
- VM snapshot recommended: no.

## Test 15: one attention head
- Objective: validate single-head attention dataflow and tensor-shape correctness.
- Source APIs exercised: matmul kernel, softmax kernel, normalization helpers, heap allocation.
- Input: tiny deterministic Q, K, V tensors and expected reference output.
- Expected result: output matches reference within tolerance and dimensions are consistent.
- Pass criteria: all shape checks pass and numeric error remains within threshold.
- Failure interpretation: indexing, scaling, or softmax integration defect.
- Approximate memory requirement: 8 to 32 MiB.
- VM snapshot recommended: yes.

## Test 16: one tiny transformer block
- Objective: validate integrated Stage 1 compute block with realistic call sequence.
- Source APIs exercised: attention, normalization, elementwise ops, residual add, allocator lifecycle.
- Input: tiny deterministic block parameters and input tensor.
- Expected result: deterministic output and stable memory behavior across runs.
- Pass criteria: output reproducibility under repeated invocation and no leaks.
- Failure interpretation: integration-level defect across kernels or memory ownership.
- Approximate memory requirement: 16 to 64 MiB.
- VM snapshot recommended: yes.

## Test 17: repeated allocation inference free loop
- Objective: detect leaks, fragmentation growth, and long-run instability in normal cycle.
- Source APIs exercised: malloc, calloc, free, full inference path calls, timing APIs.
- Input: fixed workload repeated for many iterations.
- Expected result: stable success rate, bounded runtime variation, no progressive failure trend.
- Pass criteria: no monotonic memory-loss signal and no crash over target iteration count.
- Failure interpretation: leak, fragmentation sensitivity, or latent state corruption.
- Approximate memory requirement: 32 to 128 MiB depending on workload size.
- VM snapshot recommended: yes.

## Test 18: deliberate SIGSEGV
- Objective: validate process-failure isolation for illegal memory access.
- Source APIs exercised: signal default handling, process exit status, parent waitpid checks.
- Input: controlled invalid dereference in a dedicated test binary.
- Expected result: faulting process terminates, system services remain healthy.
- Pass criteria: expected signal termination and clean ability to launch subsequent programs.
- Failure interpretation: isolation or cleanup path issue requiring deeper PM or VFS investigation.
- Approximate memory requirement: less than 2 MiB.
- VM snapshot recommended: yes.

## Test 19: deliberate SIGFPE or abort
- Objective: validate process-failure containment for arithmetic trap or explicit abort path.
- Source APIs exercised: raise or abort, signal handling, waitpid status decoding.
- Input: dedicated program that triggers SIGFPE or calls abort.
- Expected result: process terminates as expected without destabilizing unrelated services.
- Pass criteria: correct termination status and successful subsequent command execution.
- Failure interpretation: process cleanup or signal-path regression.
- Approximate memory requirement: less than 2 MiB.
- VM snapshot recommended: yes.

## Test 20: post-failure system health check
- Objective: verify operational continuity after deliberate crash tests.
- Source APIs exercised: standard command execution, file I/O sanity checks, process creation and wait paths.
- Input: sequence of basic shell and file operations after tests 18 and 19.
- Expected result: system remains responsive and baseline operations continue to work.
- Pass criteria: all health-check commands succeed without reboot.
- Failure interpretation: failure isolation is incomplete and requires OS-level investigation.
- Approximate memory requirement: less than 8 MiB.
- VM snapshot recommended: yes.

## Test execution order recommendation
1. Run tests 1 through 4 to establish ABI and numeric baseline.
2. Run tests 5 through 11 to validate memory, file I/O, and timing behavior.
3. Run tests 12 through 17 for tensor-kernel and integration stability.
4. Run tests 18 through 20 last, with snapshots enabled.

## Expected decision gate
- If tests 1 through 17 pass, Stage 1 implementation can proceed without MINIX source changes.
- If tests 18 through 20 pass, process-failure isolation is adequate for continued iterative development.
