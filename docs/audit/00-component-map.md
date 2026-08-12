# MINIX 3.3.0 Audit Phase 1: Component Map

Scope: Stage 1 requirements for an ordinary unprivileged computational user program (compile, run, allocate memory, do FP32 math, load files, report errors, repeat runs, and terminate without destabilizing services).

Evidence sources were taken from MINIX-specific paths first, then directly relevant libc/libm/build infrastructure files.

## Stage 1 dependency summary

- Process creation/execution path: required.
- Executable loading: required.
- VM/heap growth/allocation: required.
- Filesystem and VFS I/O path: required.
- Native filesystem servers: required.
- libc/libm/stdio/time/resource interfaces: required.
- Toolchain/build/install path for user command: required.
- Signals and abnormal termination behavior: required.
- Service restart behavior around user failures: relevant for system stability.

## 1) Process creation and execution (PM)

- Purpose:
  - Process lifecycle management for user processes, including fork/exec/wait, credentials, signals, and process state transitions.
- Source directories:
  - minix/servers/pm
- Important entry points:
  - main loop: minix/servers/pm/main.c
  - syscall dispatch table: minix/servers/pm/table.c
  - exec handling: minix/servers/pm/exec.c
  - signal handling: minix/servers/pm/signal.c
- Relevant system calls / IPC paths:
  - PM call dispatch includes fork/exec/wait/signal/time/resource operations (see minix/servers/pm/table.c).
  - do_exec forwards execution setup to VFS with VFS_PM_EXEC message (minix/servers/pm/exec.c), then PM completes process state updates and invokes sys_exec.
- Tests:
  - Broad PM behavior is exercised by minix/tests test programs and run scripts in minix/tests.
- Known limitations visible in source:
  - PM startup explicitly notes no live update support for now (minix/servers/pm/main.c).
- Stage 1 dependency:
  - Yes. Stage 1 program launch and repeat execution rely on this path.

## 2) Executable loading path (PM + VFS + VM + kernel)

- Purpose:
  - Validate executable, prepare argv/env, map program image, finalize process context, and transfer control to new image.
- Source directories:
  - minix/servers/pm
  - minix/servers/vfs
  - minix/servers/vm
  - minix/kernel
  - lib/libc/sys (exec interfaces)
- Important entry points:
  - PM side: do_exec/do_newexec/exec_restart in minix/servers/pm/exec.c
  - Kernel-side completion call: sys_exec invocation from PM (minix/servers/pm/exec.c)
  - User API contract: lib/libc/sys/execve.2
- Relevant system calls / IPC paths:
  - PM <-> VFS through VFS_PM_EXEC and related PM/VFS exec messages.
  - PM -> kernel via sys_exec for final register/entry setup.
- Tests:
  - General process tests in minix/tests (including repeated spawn/exec scenarios in test suite).
- Known limitations visible in source:
  - None recorded here as defects; execution path includes explicit partial-exec kill handling for failure cleanup in PM.
- Stage 1 dependency:
  - Yes. The test binary itself must exec reliably.

## 3) Virtual memory management (VM server)

- Purpose:
  - Address-space management, page fault handling, mapping, and process VM policy enforcement.
- Source directories:
  - minix/servers/vm
- Important entry points:
  - VM main loop and message dispatch: minix/servers/vm/main.c
  - mapping logic and mmap handling: minix/servers/vm/mmap.c
- Relevant system calls / IPC paths:
  - VM receives VM_* calls and VM_PAGEFAULT messages.
  - VFS/RS interactions include VM file mapping requests and service initialization mapping.
- Tests:
  - vm-focused coverage exists in minix/tests (including testvm target and support config).
- Known limitations visible in source:
  - File-system backed mapping support is runtime-gated by enable_filemap in VM mmap path.
- Stage 1 dependency:
  - Yes. Dynamic allocations and mappings ultimately depend on VM correctness.

## 4) Heap growth and allocation

- Purpose:
  - Provide dynamic memory allocation for tensors and intermediate buffers.
- Source directories:
  - lib/libc/stdlib (allocator core)
  - lib/libc/sys (brk/sbrk wrappers)
  - minix/servers/vm (backing VM behavior)
- Important entry points:
  - allocator implementation: lib/libc/stdlib/malloc.c
  - sbrk/brk wrappers: lib/libc/sys/_sbrk.c and lib/libc/sys/_brk.c
- Relevant system calls / IPC paths:
  - User code calls malloc/calloc/realloc/free; allocator can use VM-backed primitives and brk/sbrk interface path depending on internal policy.
- Tests:
  - Indirectly covered across minix/tests programs with allocation-heavy behavior.
- Known limitations visible in source:
  - No defect claims in this phase; allocator uses established NetBSD-derived code with MINIX integration hooks.
- Stage 1 dependency:
  - Yes. Core requirement for tensor buffers.

## 5) Filesystem and VFS access

- Purpose:
  - POSIX file operations, pathname resolution, descriptor operations, and device/filesystem request routing.
- Source directories:
  - minix/servers/vfs
  - lib/libc/stdio
  - lib/libc/sys
- Important entry points:
  - VFS main request loop and worker model: minix/servers/vfs/main.c
  - stdio open path: lib/libc/stdio/fopen.c
- Relevant system calls / IPC paths:
  - User syscalls/stdio -> VFS.
  - VFS routes to mounted FS servers and device endpoints.
- Tests:
  - minix/tests includes file and FS-related scripts and binaries.
- Known limitations visible in source:
  - None asserted here as defects; asynchronous worker/reply design is explicit in VFS.
- Stage 1 dependency:
  - Yes. Required for loading binary test vectors and model files.

## 6) Native filesystem support relevant to Stage 1

- Purpose:
  - Actual filesystem implementations used beneath VFS for local files.
- Source directories:
  - minix/fs
- Important entry points:
  - per-FS servers (mfs, pfs, ext2, procfs and conditional hgfs/vbfs)
  - FS build selection: minix/fs/Makefile
- Relevant system calls / IPC paths:
  - VFS <-> FS request/reply protocol.
- Tests:
  - minix/tests includes FS scripts such as testmfs.sh and testisofs.sh.
- Known limitations visible in source:
  - iso9660fs subdir is commented out in minix/fs/Makefile with note that fixed version is pending merge.
- Stage 1 dependency:
  - Yes. Stage 1 requires dependable sequential file reads for binary inputs.

## 7) libc interfaces required by Stage 1

- Purpose:
  - C runtime APIs for process control, memory, I/O, math linkage, timing, and error handling.
- Source directories:
  - lib/libc
  - minix/lib/libc/sys (MINIX-specific syscall layer included via libc makefiles)
- Important entry points:
  - libc build and composition: lib/libc/Makefile
  - syscall wrappers and interfaces in lib/libc/sys
- Relevant system calls / IPC paths:
  - libc wrappers invoke PM/VFS/VM/kernel paths as appropriate.
- Tests:
  - minix/tests links against libc and in some cases additional low-level libs.
- Known limitations visible in source:
  - MINIX libc build excludes some subsystems (for example RPC path in this tree) that are not required for Stage 1 tensor execution.
- Stage 1 dependency:
  - Yes.

## 8) libm / floating-point math support

- Purpose:
  - Mathematical routines used by softmax, norm, and attention operations.
- Source directories:
  - lib/libm
- Important entry points:
  - libm build list and architecture-specific assembly/C sources: lib/libm/Makefile
- Relevant system calls / IPC paths:
  - Pure user-space math library execution; no special privileged IPC path required for basic operations.
- Tests:
  - Math is indirectly exercised by minix/tests linking with -lm.
- Known limitations visible in source:
  - Architecture-specific source substitutions are present; this phase records mapping only, not defect claims.
- Stage 1 dependency:
  - Yes. Required for exp/log/sqrt/tanh-class operations.

## 9) Standard I/O and binary file operations

- Purpose:
  - fopen/fread/fwrite/fseek/ftell and buffered stream behavior for model/test vector ingestion.
- Source directories:
  - lib/libc/stdio
- Important entry points:
  - fopen path: lib/libc/stdio/fopen.c
  - broader stdio implementation assembled through lib/libc stdio make include.
- Relevant system calls / IPC paths:
  - stdio functions route to open/read/write/lseek and therefore through VFS.
- Tests:
  - minix/tests and many userland tools rely on stdio routines.
- Known limitations visible in source:
  - No phase-1 defect classification; only component mapping.
- Stage 1 dependency:
  - Yes.

## 10) Timing and clock interfaces

- Purpose:
  - Repeatability and elapsed-time measurement for benchmark and stability loops.
- Source directories:
  - include/time.h
  - lib/libc/sys (clock/gettimeofday interfaces)
  - minix/servers/pm (clock call dispatch)
- Important entry points:
  - public API declarations including clock_gettime/clock_getres: include/time.h
  - PM handlers mapped in minix/servers/pm/table.c (PM_CLOCK_GETTIME/GETRES/SETTIME)
  - clock_settime wrapper behavior: lib/libc/sys/clock_settime.c
- Relevant system calls / IPC paths:
  - user time APIs -> PM clock handlers; some privileged time-setting paths can use clockctl device policy.
- Tests:
  - Time-related behavior is exercised in existing test corpus and runtime utilities.
- Known limitations visible in source:
  - clock_settime wrapper explicitly handles EPERM and optional clockctl fallback.
- Stage 1 dependency:
  - Yes (at least read-only timing for measurement loops).

## 11) Compiler and linker toolchain for user programs

- Purpose:
  - Build a new user command with native system toolchain and libraries.
- Source directories:
  - top-level Makefile
  - share/mk/bsd.prog.mk
  - usr.bin/Makefile
  - bin/Makefile
- Important entry points:
  - system build orchestration and subdir ordering: Makefile
  - per-program build/install rules and library linkage defaults: share/mk/bsd.prog.mk
  - user command directories registered in bin/Makefile and usr.bin/Makefile
- Relevant system calls / IPC paths:
  - Not a runtime IPC component; this is build-time infrastructure.
- Tests:
  - Build/test framework targets and minix/tests integration exist in tree.
- Known limitations visible in source:
  - None required for mapping; component availability depends on normal build/package state.
- Stage 1 dependency:
  - Yes. Stage 1 binary must compile and link natively.

## 12) Process resource limits

- Purpose:
  - Bound CPU/data/stack/open-files/address-space behavior and define failure modes.
- Source directories:
  - sys/sys/resource.h
  - lib/libc/sys/getrlimit.2
  - minix/servers/pm
- Important entry points:
  - resource constants and rlimit ABI: sys/sys/resource.h
  - API semantics and RLIMIT_DATA/RLIMIT_STACK interaction with brk/stack growth: lib/libc/sys/getrlimit.2
- Relevant system calls / IPC paths:
  - getrlimit/setrlimit libc interface to PM-managed process policy.
- Tests:
  - Covered indirectly by process-management and shell behavior tests.
- Known limitations visible in source:
  - None classified here; mapping captures explicit documented limit behavior.
- Stage 1 dependency:
  - Yes. Important for large buffers and repeated stress loops.

## 13) Signals and abnormal termination

- Purpose:
  - Deliver/handle signals, terminate failed processes cleanly, preserve system service stability.
- Source directories:
  - minix/servers/pm
- Important entry points:
  - signal operations and kill paths: minix/servers/pm/signal.c
  - signal syscalls mapped in minix/servers/pm/table.c
- Relevant system calls / IPC paths:
  - sigaction/sigprocmask/sigreturn/sigsuspend/kill through PM.
  - PM coordinates kernel signal operations via sys_* calls.
- Tests:
  - Signal behavior exercised by dedicated and broad minix/tests cases.
- Known limitations visible in source:
  - No defect claims in phase 1; normal delayed-stop and cleanup paths are explicit.
- Stage 1 dependency:
  - Yes. Needed for robust handling of crashes or forced termination.

## 14) Service restart behavior when a user process fails (RS + PM)

- Purpose:
  - Ensure system services remain supervised and recoverable independent of ordinary user process failure.
- Source directories:
  - minix/servers/rs
  - minix/servers/pm
- Important entry points:
  - RS main loop and request handling: minix/servers/rs/main.c
  - RS control calls (up/down/restart/refresh/update) in RS dispatch.
  - PM srv_kill path restricted to RS in minix/servers/pm/signal.c.
- Relevant system calls / IPC paths:
  - Services send heartbeat notifications to RS.
  - RS performs lifecycle operations and can request cleanup signaling via PM srv_kill path.
- Tests:
  - Service behavior is covered indirectly through system integration tests; some service-specific tests exist under minix/tests.
- Known limitations visible in source:
  - No phase-1 defect statement; mapping only.
- Stage 1 dependency:
  - Indirect but relevant. Stage 1 should fail without destabilizing service supervision.

## 15) Build and installation path for a new user command

- Purpose:
  - Add a user-space Stage 1 executable as a regular command in system build.
- Source directories:
  - bin/ (for base commands)
  - usr.bin/ (for larger userland commands)
  - share/mk/bsd.prog.mk (standard program rules)
- Important entry points:
  - Example command pattern: bin/echo/Makefile (PROG plus bsd.prog.mk include)
  - parent directory registration: bin/Makefile and usr.bin/Makefile SUBDIR lists
- Relevant system calls / IPC paths:
  - Build/install infrastructure, not runtime IPC.
- Tests:
  - New command can be validated via minix/tests harness scripts or custom test entry.
- Known limitations visible in source:
  - None intrinsic to adding a normal command through existing make infrastructure.
- Stage 1 dependency:
  - Yes. This is the straightforward integration path for Stage 1 program delivery.

## Cross-component flow for Stage 1

1. Build-time:
- Source in bin or usr.bin subdir, built by bsd.prog.mk rules, linked with libc and libm.

2. Launch-time:
- Shell invokes exec path; PM coordinates VFS/kernel for executable setup.

3. Runtime compute loop:
- malloc and VM mapping provide tensor memory.
- stdio/VFS/FS stack loads binary test vectors and model data.
- libm and FP32 code execute numerics.
- clock APIs collect timing if requested.
- resource limits bound process behavior.

4. Failure/termination:
- PM signal/exit paths handle abnormal termination.
- RS continues supervising critical services independently.

This concludes Phase 1 component mapping only.
