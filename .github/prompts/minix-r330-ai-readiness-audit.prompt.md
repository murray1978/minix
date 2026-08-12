Audit the MINIX 3.3.0 source tree in this workspace and determine whether it is suitable for Stage 1 of a native AI/LLM runtime.

The repository corresponds to MINIX R3.3.0 commit `588a35b`, matching the ISO:

`minix_R3.3.0-588a35b.iso`

The source is edited through VS Code and compiled inside a MINIX 3.3.0 virtual machine. Do not assume Linux, glibc, GCC, CMake, systemd, CUDA, POSIX threads, or modern C-library behaviour.

This is an investigation and documentation task. Do not modify MINIX source files during this audit.

# Project context

The long-term project is to explore a fusion of:

* the MINIX microkernel and service architecture;
* small, understandable transformer implementations inspired by `llm.c` and `llama2.c`;
* isolated and restartable AI services using typed MINIX IPC.

Stage 1 does not require an AI system service or kernel integration.

Stage 1 consists of an ordinary unprivileged user-space C program that performs and validates basic transformer inference operations.

Single-CPU operation is acceptable. Performance is secondary to correctness and reliability.

# Stage 1 workload

The proposed Stage 1 program should be able to:

* compile with the native MINIX Clang toolchain;
* run as an ordinary unprivileged process;
* allocate tensor buffers dynamically;
* load binary test vectors and small model data from the filesystem;
* perform FP32 tensor operations;
* compare results against reference outputs;
* report numeric errors;
* repeat tests to expose memory leaks or instability;
* exit without crashing or disturbing other MINIX services.

Required mathematical operations include:

* vector addition and multiplication;
* FP32 matrix multiplication;
* embedding lookup;
* RMSNorm or LayerNorm;
* softmax;
* attention score calculation;
* weighted attention output;
* feed-forward network operations;
* optionally one complete transformer block.

A representative initial tensor size is:

* batch size: 1;
* sequence length: 16;
* embedding width: 32;
* attention heads: 4;
* feed-forward width: 64;
* transformer layers: 1.

These dimensions are illustrative and may be changed if source-supported limits require it.

# Primary question

Determine which of these conclusions is best supported:

1. **Suitable out of the box**
   MINIX R3.3.0 can run Stage 1 using components included in the normal installation.

2. **Suitable after package installation**
   Stage 1 should work after installing development packages such as Clang, build tools, headers, or libraries, without changing MINIX source.

3. **Suitable after minor operating-system fixes**
   Stage 1 requires bounded fixes to libc, libm, VFS, VM, process limits, build files, or another specific component.

4. **Unsuitable without substantial work**
   Stage 1 is blocked by major reliability, ABI, floating-point, memory-management, filesystem, or compiler problems.

Do not select a conclusion until the source evidence and available tests have been examined.

# Scope

Inspect MINIX-specific code first:

* `minix/kernel`
* `minix/servers`
* `minix/drivers`
* `minix/fs`
* `minix/lib`
* `minix/tests`
* `lib`
* `include`
* `usr.bin`
* `usr.sbin`
* `etc`
* `releasetools`
* top-level build files

Inspect imported NetBSD or third-party code only when it directly supplies functionality required by Stage 1, such as:

* libc;
* libm;
* stdio;
* memory allocation;
* compiler runtime support;
* floating-point helpers;
* file I/O;
* time measurement.

Do not perform a general security audit of all imported userland code.

# Audit output

Create the following files:

* `docs/audit/00-component-map.md`
* `docs/audit/01-unfinished-work.md`
* `docs/audit/02-candidate-issues.md`
* `docs/audit/03-priority-summary.md`
* `docs/audit/04-ai-stage1-readiness.md`
* `docs/audit/05-ai-stage1-test-plan.md`

Create missing directories as required.

Do not alter existing audit files without first reading and preserving useful existing content.

# Phase 1: component map

Create or update:

`docs/audit/00-component-map.md`

Map the MINIX components relevant to an ordinary computational user program.

Include:

* process creation and execution;
* executable loading;
* virtual-memory management;
* heap growth and allocation;
* filesystem and VFS access;
* native filesystem support;
* libc;
* libm;
* standard I/O;
* timing and clock interfaces;
* compiler and linker toolchain;
* process resource limits;
* signals and abnormal termination;
* service restart behaviour when a user process fails;
* build and installation paths for a new user command.

For every component, record:

* purpose;
* source directories;
* important entry points;
* relevant system calls or IPC paths;
* tests;
* known limitations visible in the source;
* whether Stage 1 depends upon it.

Do not speculate about defects in this phase.

# Phase 2: unfinished-work inventory

Create or update:

`docs/audit/01-unfinished-work.md`

Search relevant code for:

* `TODO`;
* `FIXME`;
* `XXX`;
* `HACK`;
* `#if 0`;
* `ENOSYS`;
* “not implemented”;
* “unsupported”;
* “temporary”;
* “workaround”;
* disabled tests;
* expected-failure tests;
* comments describing known races;
* comments describing inaccurate floating-point behaviour;
* incomplete error handling;
* disabled architecture support;
* hard-coded resource limits;
* panic or assertion paths reachable from user operations.

For each item, record:

* exact source path;
* function or symbol;
* line range;
* source evidence;
* reachability;
* relevance to Stage 1;
* classification as intentional limitation, dead code, documentation debt, incomplete feature, likely defect, or unrelated issue;
* confidence level.

Do not treat old code style alone as a defect.

# Phase 3: candidate defects and limitations

Create or update:

`docs/audit/02-candidate-issues.md`

Limit the first report to no more than 15 strong candidates.

Concentrate on issues that could affect Stage 1:

## Compiler and executable support

* missing compiler components;
* missing linker tools;
* incorrect target assumptions;
* generated-header dependencies;
* ABI mismatches;
* unsupported C features used by small transformer runtimes;
* build failures caused by incomplete package installation.

## Floating-point and mathematics

* incorrect or missing `expf`, `sqrtf`, `logf`, `powf`, `tanhf`, or related functions;
* float-to-integer conversion problems;
* FP exception handling;
* compiler optimisation assumptions;
* denormal, NaN, infinity, or overflow handling;
* inconsistent single-precision implementations;
* unexpected promotion to double;
* inaccurate softmax-relevant functions.

## Memory

* large contiguous allocation failures;
* integer overflow in allocation-size calculations;
* process address-space limits;
* heap-growth limitations;
* stack-size limitations;
* `malloc`, `calloc`, `realloc`, and `free` error paths;
* alignment guarantees;
* mapping or page-fault defects;
* memory accounting;
* leaks after process termination;
* behaviour under allocation failure.

## Filesystem and binary data

* binary `fopen`, `fread`, `fwrite`, `fseek`, and `ftell` behaviour;
* short reads and writes;
* file-offset limits;
* large-file limitations;
* data truncation;
* error reporting;
* filesystem consistency;
* sequential-read reliability;
* loading model weights into memory;
* native filesystem limits relevant to model files.

## Timing and repeatability

* monotonic-clock availability;
* timer resolution;
* APIC or virtual-machine timer defects;
* elapsed-time measurement;
* repeat-run stability;
* process cleanup after normal or abnormal termination.

## User-process isolation

* malformed executable or input-file handling;
* effects of process crashes;
* whether an unprivileged numerical process can affect system services;
* resource exhaustion;
* missing limits that could destabilise the system.

For every candidate, use this format:

## Finding ID

**Title:**
**Classification:** confirmed defect, likely defect, incomplete feature, packaging limitation, hardware limitation, portability limitation, insufficient error handling, missing test coverage, or maintainability issue
**Severity:** critical, high, medium, low, or informational
**Confidence:** high, medium, or low
**Stage 1 relevance:** blocking, significant, minor, or none
**Component:**
**Source path:**
**Function or symbol:**
**Line range:**

### Evidence

Describe exactly what the source does.

### Trigger or precondition

Identify the input, allocation size, file state, compiler option, hardware state, timing condition, or execution path required.

### Expected behaviour

Describe the required behaviour for Stage 1.

### Actual or likely behaviour

Describe the source-supported result.

### User-visible consequence

Explain the effect on a Stage 1 tensor program.

### Reproduction or validation method

Provide a small test that can run inside the existing MINIX virtual machine.

### Existing tests

Identify relevant tests and determine whether they exercise the path.

### Later-history evidence

Inspect later repository history for applicable fixes.

Include:

* commit identifier;
* commit message;
* affected files;
* whether the R3.3.0 code contains the same issue;
* whether the later fix can be backported without newer dependencies.

Do not claim that a later rewrite proves an R3.3.0 defect.

### Minimal compatible fix

Describe the smallest R3.3.0-compatible fix.

### Regression risks

Describe possible effects on ABI, IPC, libc, VM, VFS, filesystem behaviour, or existing programs.

### Status

Use one of:

* candidate;
* reproduced;
* confirmed;
* rejected;
* fixed.

Reject findings that lack a concrete execution path or source evidence.

# Phase 4: historical verification

Use Git history after commit `588a35b` as forensic evidence.

For relevant files and subsystems:

* inspect later bug-fix commits;
* search for fixes involving libc, libm, VM, VFS, memory allocation, floating point, executable loading, clocks, and file I/O;
* compare later changes with R3.3.0;
* distinguish bug fixes from feature additions and broad redesigns;
* identify safe backport candidates;
* do not merge, cherry-pick, or modify the working tree.

Suggested Git searches include concepts such as:

* fix;
* overflow;
* float;
* libm;
* malloc;
* mmap;
* VM;
* VFS;
* short read;
* file offset;
* timer;
* clock;
* assertion;
* panic;
* leak;
* alignment;
* Clang.

Do not rely solely on commit messages. Inspect the actual diffs.

# Phase 5: priority summary

Create or update:

`docs/audit/03-priority-summary.md`

Group findings into:

* confirmed or strongly supported defects;
* candidates requiring runtime tests;
* package or installation omissions;
* compiler and build-tool limitations;
* floating-point risks;
* memory-management risks;
* filesystem and binary-I/O risks;
* virtual-machine-specific risks;
* physical-hardware risks;
* test-infrastructure gaps;
* low-risk fixes;
* high-risk architectural changes.

Rank findings using:

1. strength of evidence;
2. likelihood of affecting Stage 1;
3. reproducibility;
4. system reliability impact;
5. fix complexity;
6. ability to test inside VirtualBox;
7. relevance to later AI-service work.

Do not recommend unrelated modernisation work merely because the code is old.

# Phase 6: AI Stage 1 readiness decision

Create:

`docs/audit/04-ai-stage1-readiness.md`

Use this structure:

# MINIX 3.3.0 AI Stage 1 Readiness

## Overall verdict

Select exactly one:

* suitable out of the box;
* suitable after package installation;
* suitable after minor operating-system fixes;
* unsuitable without substantial work;
* insufficient evidence.

Explain the verdict using source evidence and existing tests.

## Required packages and tools

List:

* compiler;
* linker;
* headers;
* libc components;
* libm components;
* build tools;
* optional diagnostic tools.

Clearly distinguish packages included in the normal installation from packages installed separately.

## Compiler and ABI readiness

Assess:

* Clang availability;
* C-language compatibility;
* 32-bit assumptions;
* `sizeof` assumptions;
* floating-point ABI;
* alignment;
* linker behaviour;
* generated build dependencies.

## Floating-point readiness

Assess:

* FP32 arithmetic;
* required libm functions;
* numeric consistency;
* softmax stability;
* normalisation operations;
* NaN and infinity handling;
* likely optimisation restrictions.

## Memory readiness

Assess:

* practical process address space;
* safe initial model size;
* heap allocation;
* stack limits;
* allocation failure behaviour;
* recommended memory budget.

Estimate a conservative Stage 1 memory ceiling from source-supported limits. Clearly mark estimates as estimates.

## Filesystem readiness

Assess:

* native filesystem suitability;
* binary weight loading;
* test-vector loading;
* short-read handling;
* safe initial file-size limit;
* whether `mmap` is required.

Stage 1 should prefer buffered file I/O unless evidence supports using `mmap`.

## Timing readiness

Assess:

* available timing APIs;
* timer accuracy;
* VM-specific limitations;
* whether timing is sufficient for coarse benchmarking.

## Isolation and failure behaviour

Assess what happens if the Stage 1 process:

* divides by zero;
* accesses invalid memory;
* exhausts its heap;
* reads a corrupt model file;
* terminates unexpectedly;
* runs indefinitely.

Determine whether these failures remain confined to the user process.

## Single-CPU assessment

Explain why single-CPU execution is or is not sufficient for Stage 1.

Do not classify low performance alone as a blocker.

## Blocking findings

List only findings that must be resolved before writing the Stage 1 program.

## Non-blocking findings

List findings that may affect performance, convenience, or later stages but do not prevent Stage 1.

## Recommended first implementation

Recommend:

* source location;
* build integration method;
* initial tensor dimensions;
* memory budget;
* required test-vector format;
* compiler flags;
* operations to implement first.

Do not write the implementation during this audit.

# Phase 7: executable validation plan

Create:

`docs/audit/05-ai-stage1-test-plan.md`

Design a sequence of small C tests.

Each test must include:

* objective;
* source APIs exercised;
* input;
* expected result;
* pass criteria;
* failure interpretation;
* required memory;
* expected runtime class;
* whether it can destabilise the VM;
* whether a snapshot is advised.

Include at least these tests:

## Test 1: compiler sanity

Compile and run a small C program using:

* integer types;
* `float`;
* standard I/O;
* heap allocation.

## Test 2: libm conformance

Compare selected results from:

* `expf`;
* `sqrtf`;
* `logf`;
* `tanhf`, if available;
* `powf`, if required.

Use fixed reference values generated on a trusted reference platform.

## Test 3: heap allocation

Test progressively larger allocations, safe failure handling, writes across allocated pages, and cleanup.

Do not intentionally exhaust the complete system without warning.

## Test 4: binary file I/O

Write and read deterministic FP32 data and verify exact byte reproduction.

## Test 5: matrix multiplication

Run a small known matrix multiplication and compare maximum absolute and relative error.

## Test 6: softmax

Test ordinary values, large positive values, large negative values, and numerically stabilised softmax.

Verify that probabilities are finite and sum approximately to one.

## Test 7: normalisation

Test RMSNorm or LayerNorm against trusted reference vectors.

## Test 8: attention

Test one small attention head with deterministic inputs.

## Test 9: transformer block

Run one complete tiny transformer block and compare its output with reference data.

## Test 10: repetition and cleanup

Repeat the block enough times to reveal corruption, leaks, or timing instability.

## Test 11: malformed input

Test:

* truncated tensor data;
* invalid dimensions;
* multiplication overflow in element counts;
* unsupported model version;
* allocation failure.

The program must reject malformed inputs without crashing MINIX.

## Test 12: process failure isolation

Deliberately terminate or fault the test program and verify that:

* the shell remains usable;
* VFS remains usable;
* networking remains usable;
* system services remain running;
* the test can be launched again.

# Audit quality requirements

* Do not modify MINIX source during this audit.
* Do not implement the tensor runtime.
* Do not invent source paths, line numbers, symbols, tests, packages, or limits.
* State when evidence is unavailable.
* Distinguish facts, inferences, estimates, and hypotheses.
* Cite exact paths and symbols for source-derived conclusions.
* Inspect callers and error paths before reporting a defect.
* Inspect existing tests before claiming coverage is missing.
* Do not assume Linux behaviour applies to MINIX.
* Do not report old style or missing modern features as defects by themselves.
* Prefer a small number of well-supported findings.
* Treat Stage 1 correctness as more important than performance.
* Treat single-CPU operation as acceptable unless the source reveals a correctness problem.
* Stop after creating or updating the six audit documents.
* Finish with a concise summary of:

  * documents created;
  * overall readiness verdict;
  * blocking issues;
  * tests that should run before implementation.
