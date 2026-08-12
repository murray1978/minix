# MINIX 3.3.0 Audit Phase 2: Unfinished-Work Inventory (Stage 1 Focus)

Inputs preserved:
- docs/audit/00-component-map.md (read, not modified)

Audit scope used for this phase:
- minix/servers/pm
- minix/servers/vm
- minix/servers/vfs
- minix/kernel
- lib/libc
- lib/libm
- minix/lib/libc
- include
- sys/sys
- share/mk
- minix/tests
- top-level build files (only where directly relevant)

Selection rule applied:
- Included only unfinished/disabled/limited evidence with plausible impact on Stage 1 concerns: compile/link, exec/load, FP32/libm, heap/allocation, binary file I/O, timing, termination/isolation, repeat-run stability.
- Excluded unrelated subsystem gaps (for example networking features unless they intersect Stage 1 core path, GUI/media/driver features outside Stage 1 workload).

## Findings

### Item ID: P2-S1-001
**Classification:** likely defect  
**Stage 1 relevance:** significant  
**Confidence:** high  
**Source path:** minix/servers/vfs/read.c  
**Function or symbol:** read_write  
**Exact line range:** 180-196

## Source evidence
> /* FIXME: multiple read/write operations on a single filp should be serialized... */  
> ... "so we settle for this hack for now."

Meaning: concurrent char-device operations on one filp are not serialized; VFS advances file position optimistically as a workaround.

## Reachability
- Ordinary user read/write on character devices enters VFS read_write.
- Path is active in S_ISCHR(vp->v_mode) branch before/while asynchronous device I/O is in flight.

## Stage 1 impact
- Not a normal regular-file tensor-data path, but can affect logging/stdio behavior if redirected to character devices and can affect repeatability/order assumptions.

## Disposition
- carry forward to Phase 3

---

### Item ID: P2-S1-002
**Classification:** likely defect  
**Stage 1 relevance:** significant  
**Confidence:** medium  
**Source path:** minix/servers/vfs/misc.c  
**Function or symbol:** free_proc (controlling-tty cleanup path)  
**Exact line range:** 657-671

## Source evidence
> if ((fp->fp_flags & FP_SESLDR) && fp->fp_tty != 0) { ...  
> (void) cdev_close(dev); /* Ignore any errors. */  
> /* FIXME: missing select check */

Meaning: VFS acknowledges missing select-state handling when revoking controlling TTY file descriptors during session-leader exit.

## Reachability
- pm_exit calls free_proc(FP_EXITING) in VFS (misc.c:687, misc.c:693), which reaches this branch when the exiting process is a session leader with controlling TTY.

## Stage 1 impact
- Can affect cleanup correctness after abnormal termination of interactive test runs and may produce wakeup/poll/select inconsistencies.

## Disposition
- requires a runtime test

---

### Item ID: P2-S1-003
**Classification:** incomplete feature  
**Stage 1 relevance:** minor  
**Confidence:** high  
**Source path:** minix/servers/vm/mem_anon.c  
**Function or symbol:** anon_resize  
**Exact line range:** 115-121

## Source evidence
> /* Shrinking not implemented; silently ignored.  
>  * (Which is ok for brk().) */

Meaning: anonymous-region shrink requests are intentionally ignored in this handler.

## Reachability
- VM region extension/shrink path uses memtype callback ev_resize (mem_anon.c:38, region.c:1053).

## Stage 1 impact
- Does not block ordinary heap growth; may reduce effectiveness of shrink/reclaim behavior and could influence long repeat-run memory behavior.

## Disposition
- non-blocking limitation

---

### Item ID: P2-S1-004
**Classification:** incomplete feature  
**Stage 1 relevance:** minor  
**Confidence:** high  
**Source path:** minix/servers/vm/region.c  
**Function or symbol:** map_unmap_region / split_region  
**Exact line range:** 1093-1094, 1161-1163

## Source evidence
> "VM: low-shrinking not implemented for %s"  
> "VM: split region not implemented for %s"

Meaning: certain region operations are explicitly not implemented for memtypes lacking those callbacks.

## Reachability
- Called from active unmap paths (region.c:1278, region.c:1282) and split path checks in split_region.

## Stage 1 impact
- Potentially relevant to advanced mmap/unmap patterns; ordinary malloc/free tensor loops are usually unaffected.

## Disposition
- requires a runtime test

---

### Item ID: P2-S1-005
**Classification:** intentional limitation  
**Stage 1 relevance:** minor  
**Confidence:** high  
**Source path:** minix/servers/vm/main.c, minix/servers/vm/mmap.c  
**Function or symbol:** enable_filemap toggle; do_vfs_mmap/do_mmap guards  
**Exact line range:** main.c 384-385; mmap.c 143-144, 257

## Source evidence
> enable_filemap=1; /* yes by default */  
> env_parse("filemap", "d", 0, &enable_filemap, 0, 1);  
> if(!enable_filemap) return ENXIO;

Meaning: file-backed mmap support is runtime-toggleable and explicitly returns ENXIO when disabled.

## Reachability
- Triggered by mmap/file-mapping requests through VM.

## Stage 1 impact
- Stage 1 can avoid file-backed mmap and use read/fread plus heap buffers. Therefore this is usually non-blocking for the described workload.

## Disposition
- non-blocking limitation

---

### Item ID: P2-S1-006
**Classification:** test gap  
**Stage 1 relevance:** minor  
**Confidence:** high  
**Source path:** minix/tests/test42.c  
**Function or symbol:** ptrace attach test block  
**Exact line range:** 905-910

## Source evidence
> #if 0  
> /* FIXME: disabled until we can reliably determine PM's pid */

Meaning: one ptrace-related assertion is intentionally disabled.

## Reachability
- Test-suite only; not runtime production path.

## Stage 1 impact
- Indicates coverage gap in process-control test area, but does not directly block tensor program execution.

## Disposition
- non-blocking limitation

---

### Item ID: P2-S1-007
**Classification:** test gap  
**Stage 1 relevance:** minor  
**Confidence:** high  
**Source path:** minix/tests/Makefile  
**Function or symbol:** architecture-conditional test compile flags  
**Exact line range:** 25-29

## Source evidence
> # LSC FIXME: Compilation error for now on ARM with that!  
> COPTS.test51.c= -mhard-float  
> COPTS.test52.c= -mhard-float

Meaning: known ARM-side test compilation limitation.

## Reachability
- Build/test-time only; only for ARM-conditional block.

## Stage 1 impact
- Irrelevant to x86 single-CPU Stage 1 target in this repository context; does not affect core Stage 1 runtime path.

## Disposition
- irrelevant to Stage 1

---

### Item ID: P2-S1-008
**Classification:** intentional limitation  
**Stage 1 relevance:** none  
**Confidence:** high  
**Source path:** minix/lib/libc/sys/bind.c  
**Function or symbol:** bind fallback branch  
**Exact line range:** 78-80

## Source evidence
> "bind: not implemented for fd %d"  
> errno = ENOSYS;

Meaning: some socket cases in MINIX-specific libc wrappers are explicitly unimplemented.

## Reachability
- Networking/socket path only.

## Stage 1 impact
- Not part of Stage 1 tensor inference requirements (compile/exec/fp32/heap/file/timing).

## Disposition
- irrelevant to Stage 1

---

### Item ID: P2-S1-009
**Classification:** incomplete feature  
**Stage 1 relevance:** none  
**Confidence:** high  
**Source path:** minix/kernel/system.c, minix/kernel/clock.c  
**Function or symbol:** do_schedule / clock interrupt watchdog comment  
**Exact line range:** system.c 645-652; clock.c 87

## Source evidence
> /* FIXME this preempts the process... */  
> /* FIXME this is a problem for SMP... */  
> /* FIXME watchdog for slave cpus! */

Meaning: explicit SMP-related unfinished concerns.

## Reachability
- Kernel scheduling/clock code under SMP concerns.

## Stage 1 impact
- Stage 1 explicitly allows single-CPU operation; these are SMP-specific and therefore non-blocking for single-CPU runs.

## Disposition
- irrelevant to Stage 1

---

### Item ID: P2-S1-010
**Classification:** dead code  
**Stage 1 relevance:** minor  
**Confidence:** high  
**Source path:** minix/servers/vm/mmap.c  
**Function or symbol:** mmap_file bounds check block  
**Exact line range:** 111-118

## Source evidence
> #if 0  
> /* XXX ld.so relies on longer-than-file mapping */  
> ... return EINVAL;

Meaning: a stricter bounds check is disabled in source comments for compatibility with loader behavior.

## Reachability
- Currently dead (#if 0), so not active behavior.

## Stage 1 impact
- No direct active impact shown for ordinary fread-based model loading; potential relevance only for mmap-heavy loaders.

## Disposition
- dead or unreachable

---

### Item ID: P2-S1-011
**Classification:** insufficient evidence  
**Stage 1 relevance:** minor  
**Confidence:** medium  
**Source path:** minix/servers/vfs/read.c  
**Function or symbol:** read_write  
**Exact line range:** 168, 202

## Source evidence
> panic("VFS: read_write tries to access char dev NO_DEV");  
> panic("VFS: read_write tries to access block dev NO_DEV");

Meaning: VFS has hard panic paths if vnode/device metadata is internally inconsistent.

## Reachability
- The checks are in active read/write code, but require an invalid `v_sdev` state (`NO_DEV`) on a character/block vnode.
- This phase found no direct ordinary-user path proving a normal unprivileged tensor program can induce this state without prior subsystem corruption.

## Stage 1 impact
- If triggered, impact would be system-wide instability; however current evidence does not show routine Stage 1 operations reaching these panics.

## Disposition
- insufficient evidence

---

## Focused questions

1. Is any required libm single-precision function absent, stubbed, disabled, or marked inaccurate?
- No direct evidence of absence/stub in required functions from this phase.
- lib/libm/Makefile includes required single-precision entries such as e_expf, e_sqrtf, s_tanhf, wrapper variants (w_expf, w_sqrtf) at lines 90, 92, 158, 163, 180, 184, 191.
- XXX notes found in the Makefile are architecture/compat notes and not evidence that required i386 Stage 1 functions are stubbed out.

2. Is the active allocator implementation complete for ordinary heap allocations?
- Source evidence supports yes for ordinary usage, with caveats.
- Allocator has active growth path via sbrk (malloc.c:365, 375) and control of break (malloc.c:400, 1042), and uses MMAP for allocator metadata/page directory (malloc.c:556-557).
- No Stage-1-blocking TODO/FIXME was found inside the ordinary allocation path in this phase.

3. Is file-backed mmap required for malloc, or can Stage 1 safely avoid it?
- Stage 1 can safely avoid file-backed mmap.
- Allocator uses anonymous mapping macro (MMAP using MAP_ANON|MAP_PRIVATE, malloc.c:314-315) and sbrk-based growth.
- VM file-mapping toggle (enable_filemap) affects file-backed mmap requests (mmap.c:143-144, 257) but is not required for fread+heap workflows.

4. Are there source-visible default limits likely to prevent allocations of 1 MB, 16 MB, 64 MB, or 128 MB?
- No explicit numeric default caps for those sizes were identified in scanned Stage-1 paths.
- Limit classes are defined (RLIMIT_DATA, RLIMIT_STACK, RLIMIT_AS) in sys/sys/resource.h:83-84,91; behavior semantics documented in lib/libc/sys/getrlimit.2:70,105,111-122,153.
- This phase found no source-visible default hardcoded value proving those sizes are blocked by default.

5. Are binary fread, fwrite, and lseek paths marked incomplete?
- No unfinished markers found in these core paths.
- fread partial-read handling is explicit (fread.c:82) and zero-size behavior is defined (fread.c:65).
- fwrite behavior and error accounting are explicit (fwrite.c:76,81).
- lseek wrapper path is implemented (lib/libc/sys/lseek.c:57-60).

6. Is CLOCK_MONOTONIC implemented and available to ordinary processes?
- Yes.
- PM dispatch includes clock calls (pm/table.c:47-49) and do_gettime handles CLOCK_MONOTONIC (pm/time.c:34-35), with do_getres support (pm/time.c:55).
- Only clock_settime(CLOCK_MONOTONIC) is rejected as expected (pm/time.c:81-83).

7. Are user-process crashes expected to remain confined to that process?
- Source evidence indicates process-level confinement intent.
- PM signal handling limits user-initiated lethal signals to privileged processes (pm/signal.c:607-610) and excludes privileged processes during broadcast kill (pm/signal.c:599-601); RS-only srv_kill controls service cleanup (pm/signal.c:211,216-219).
- A runtime stress test is still advisable for full confirmation.

8. Are any relevant tests disabled or expected to fail?
- Yes, but non-blocking for core Stage 1 path:
  - minix/tests/test42.c:905-910 disabled ptrace block.
  - minix/tests/Makefile:27-29 ARM compile limitation note.

9. Is any issue specific to SMP, and therefore irrelevant when Stage 1 runs on one CPU?
- Yes.
- Kernel comments in kernel/system.c:645-652 and kernel/clock.c:87 are explicitly SMP/slave-CPU oriented and are non-blocking for single-CPU Stage 1 operation.

10. Does any finding presently change the preliminary verdict of suitable after package installation?
- No.
- Current findings are mostly non-blocking limitations or test gaps, with two runtime-behavior candidates in VFS/VM that merit Phase 3 follow-up but do not yet justify changing the preliminary verdict.

## Phase 2 conclusion

- The highest-value Stage-1 follow-ups for Phase 3 are:
  - P2-S1-001 (VFS char-device concurrent I/O serialization workaround)
  - P2-S1-002 (VFS controlling-tty cleanup missing select check)
  - P2-S1-004 (VM low-shrink/split not implemented paths)
- No evidence in this phase alone overturns the working assessment that Stage 1 is likely suitable after package installation.
