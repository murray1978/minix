Audit the MINIX 3.3.0 source at commit `588a35b`.

This is an analysis task. Do not change source code.

## First-pass scope

Inspect only:

* `minix/kernel`
* `minix/servers`
* `minix/drivers`
* `minix/fs`
* `minix/lib`
* `minix/tests`
* `releasetools`

Exclude third-party and broadly imported userland code unless a MINIX-specific component directly depends on it.

## Phase 1: component map

Create `docs/audit/00-component-map.md`.

For each major component, record:

* purpose;
* main executable or service;
* principal source directory;
* important entry points;
* IPC relationships;
* kernel calls used;
* configuration files;
* build targets;
* existing tests;
* architecture-specific portions;
* obvious disabled or unfinished functionality.

Do not report defects during this phase unless they are explicit in the source.

## Phase 2: unfinished-work inventory

Search the scoped source for evidence such as:

* TODO;
* FIXME;
* XXX;
* HACK;
* `#if 0`;
* `ENOSYS`;
* “not implemented”;
* “unsupported”;
* unconditional panic paths;
* ignored return values;
* comments describing races or missing locking;
* disabled drivers or filesystems;
* partial architecture implementations;
* test cases that are disabled or expected to fail.

Create `docs/audit/01-unfinished-work.md`.

For every item, determine whether it is active code, dead code, documentation debt, an intentional limitation, or a plausible defect.

## Phase 3: high-confidence issue candidates

Perform control-flow and data-flow inspection of the scoped source.

Concentrate on:

* unchecked lengths and integer conversions;
* buffer and descriptor ownership;
* IPC message validation;
* grant and memory-copy validation;
* missing error propagation;
* resource leaks on error paths;
* interrupt and DMA state transitions;
* device reset and recovery paths;
* service restart state reconstruction;
* concurrency and SMP assumptions;
* timeout handling;
* malformed-device-input handling;
* filesystem consistency;
* assumptions made only by emulators;
* code paths with assertions but no runtime recovery;
* tests that do not cover corresponding failure paths.

Create `docs/audit/02-candidate-issues.md`.

Limit the first report to at most 15 high-confidence candidates. Do not pad the report with weak findings.

Every candidate must use the finding format specified in `.github/copilot-instructions.md`.

## Phase 4: historical verification

For each candidate, inspect later Git history affecting the same file or subsystem.

Look for commits after `588a35b` that:

* explicitly fix a bug;
* add missing validation;
* correct locking or SMP behaviour;
* repair hardware compatibility;
* restore a disabled feature;
* add a regression test;
* change the same suspicious code.

Do not assume later code is compatible with R3.3.0.

Add applicable commit IDs and backport notes to the candidate report.

## Final summary

Create `docs/audit/03-priority-summary.md` containing:

* confirmed or strongly supported defects;
* candidates needing a runtime reproduction;
* incomplete planned features;
* hardware-compatibility gaps;
* test-infrastructure gaps;
* low-risk fixes;
* high-risk architectural changes;
* suggested order of investigation.

Rank items by:

1. reproducibility;
2. user impact;
3. architectural value;
4. ability to test in VirtualBox;
5. ability to test on physical hardware;
6. implementation risk.

Do not implement fixes during this audit.
