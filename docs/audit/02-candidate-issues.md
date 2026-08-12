# MINIX 3.3.0 Audit Phase 3: Stage 1 Candidate Issues

Scope constrained for this phase:
- Stage 1 only: compile, run, heap allocation, regular-file FP32 input, libm math calls, CLOCK_MONOTONIC timing, cleanup/exit/fault confinement for ordinary unprivileged user programs.
- Baseline: MINIX 3.3.0 at commit 588a35b.
- No source changes were made in this phase.

Method used:
- Reassessed Phase 2 unfinished-work items against concrete Stage 1 reachability.
- Kept only issues with a direct caller chain from Stage 1 operations.
- Checked later history per affected file for evidence of post-baseline fixes.

## Reassessment of required Phase 2 carryovers

### P2-S1-001 (minix/servers/vfs/read.c read_write filp serialization FIXME)
- Reassessment result: rejected for Stage 1 candidate set.
- Reason: this path requires character-device I/O and/or concurrent operations on a shared filp. Stage 1 workload here is regular-file FP32 input in a single-process path, which does not require this trigger.
- Status: rejected.

### P2-S1-002 (minix/servers/vfs/misc.c controlling-tty select FIXME)
- Reassessment result: rejected for Stage 1 candidate set.
- Reason: this path is specific to controlling TTY teardown on session-leader exit. Stage 1 workload can execute through regular files and does not require select-on-tty behavior.
- Later history note: commit 27d0ecdb6 exists but only changes the FIXME comment text and does not implement missing select handling.
- Status: rejected.

### P2-S1-004 (minix/servers/vm/region.c low-shrink/split limitations)
- Reassessment result: downgraded to non-blocking limitation for Stage 1.
- Reason: Stage 1 normal malloc/fread/libm path does not require low-end region shrink or memtype split behavior to function correctly.
- Later history note: commit 48e74378c adds diagnostics in split failure path, not functional implementation for unsupported memtypes.
- Status: rejected from candidate defect list.

## Candidate findings (maximum 10)

### Finding P3-S1-001
- Finding ID: P3-S1-001
- Classification: likely defect
- Severity: Medium
- Confidence: Medium
- Component: libc stdio regular-file seek behavior
- Source path and symbol: lib/libc/stdio/fseek.c, fseek
- Evidence:
	- fseek casts SEEK_SET offsets through an unsigned conversion:
		- off_t offset;
		- if (whence == SEEK_SET)
			offset = (unsigned long)l_offset;
	- This is active code in the regular-file stdio path.
- Trigger or precondition:
	- Program calls fseek(fp, negative_value, SEEK_SET), including accidental negative offsets from arithmetic underflow in model/data parsing logic.
- Expected behaviour:
	- Negative SEEK_SET offsets should be rejected by failing the operation with an error (for example EINVAL) rather than being accepted as a very large positive absolute offset.
- Actual or likely behaviour:
	- Negative long value is zero-extended to unsigned long, then passed to fseeko as a large positive off_t.
	- On regular files this may succeed (seek beyond EOF), masking caller bugs.
- User-visible consequence:
	- Silent reposition to an unintended large file offset can cause unexpected EOF reads, malformed model/data interpretation, and hard-to-debug run-to-run discrepancies.
- Reproduction or validation method:
	- Build a small user program that opens a regular file and calls fseek(fp, -1L, SEEK_SET), then checks return code and ftell/fread behavior.
	- Validate whether failure is reported or a large positive position is accepted.
- Existing test coverage:
	- No dedicated Stage 1-focused regression test located in minix/tests for negative SEEK_SET stdio behavior.
- Relevant later commit, if found:
	- No post-588a35b fix commit found for this file behavior in the inspected history (recent file history is mainly build/toolchain/layout changes).
- Minimal compatible fix:
	- In fseek, when whence == SEEK_SET and l_offset < 0, set errno to EINVAL and return -1 before conversion.
- Regression risks:
	- Low; code that unintentionally relied on current wrap/zero-extend behavior would start failing explicitly.
- Status: candidate

### Finding P3-S1-002
- Finding ID: P3-S1-002
- Classification: portability limitation
- Severity: Low
- Confidence: Medium
- Component: PM CLOCK_MONOTONIC time accounting
- Source path and symbol:
	- minix/servers/pm/time.c, do_clock_gettime
	- sys/arch/i386/include/ansi.h, _BSD_CLOCK_T_
- Evidence:
	- do_clock_gettime stores uptime ticks in clock_t and derives timespec from that value.
	- On i386 in this tree, _BSD_CLOCK_T_ is defined as unsigned long.
	- Therefore monotonic time precision/range depends on 32-bit tick counter behavior.
- Trigger or precondition:
	- Very long uptime where the underlying clock_t tick count wraps.
- Expected behaviour:
	- CLOCK_MONOTONIC should continue increasing monotonically across long uptimes.
- Actual or likely behaviour:
	- If tick counter wraps at clock_t width, returned monotonic time can jump backward or reset modulo wrap period.
- User-visible consequence:
	- Long-running timing measurements can become invalid after wrap; elapsed-time computations in user programs may underflow or produce negative-like deltas.
- Reproduction or validation method:
	- Use a controlled kernel/PM test that seeds uptime near wrap boundary (or accelerated tick simulation), call clock_gettime(CLOCK_MONOTONIC) across wrap, and verify monotonic ordering.
- Existing test coverage:
	- No Stage 1-focused long-uptime monotonic-wrap test identified in minix/tests.
- Relevant later commit, if found:
	- No direct post-588a35b fix identified in inspected pm/time.c history.
- Minimal compatible fix:
	- Preserve a widened monotonic accumulator in PM (or consume a 64-bit monotonic source) before converting to timespec.
- Regression risks:
	- Medium; timekeeping changes can affect ABI expectations, timeout behavior, and interaction with other PM/kernel timing paths.
- Status: candidate

## Stage 1 readiness conclusion

- Candidate count retained after strict reassessment: 2 (<= 10 limit satisfied).
- The three required reassessed Phase 2 items (P2-S1-001, P2-S1-002, P2-S1-004) were not retained as Stage 1 defect candidates because their triggers are outside the regular-file single-process Stage 1 path or are non-blocking limitations.
- Readiness judgement for Stage 1:
	- Provisionally ready for ordinary short-to-medium duration Stage 1 workloads.
	- Residual risk remains for edge-case file-seek error handling and very long-uptime monotonic timing behavior.
