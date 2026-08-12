# MINIX 3.3.0 repository instructions

This repository is the MINIX 3.3.0 source tree corresponding to Git commit `588a35b`, used by the ISO `minix_R3.3.0-588a35b.iso`.

## Platform and build environment

* The primary target is 32-bit x86 MINIX 3.3.0.
* The source is edited from VS Code on Windows through SSHFS.
* Compilation and installation occur inside the MINIX virtual machine.
* Do not assume that the Windows host can compile or test this source.
* The reference full build command is `make build`.
* The development build command is `make build MKUPDATE=yes`.
* Builds must be invoked through the existing VS Code MINIX SSH build tasks.
* Do not introduce Linux-specific, glibc-specific, systemd-specific, GNU-only, or modern POSIX APIs without first verifying that MINIX 3.3.0 provides them.
* Do not assume a modern C standard. Preserve the conventions and compiler limitations of the existing source unless a modernization task explicitly says otherwise.

## Audit rules

When asked to audit code:

1. Do not modify source files unless explicitly instructed.
2. Treat the existing implementation as the primary evidence.
3. Give every finding an exact file path, function or symbol, and relevant line range.
4. Explain the concrete execution path that could trigger the issue.
5. Distinguish among:

   * confirmed defect;
   * likely defect;
   * incomplete or disabled feature;
   * hardware-compatibility limitation;
   * portability limitation;
   * insufficient error handling;
   * missing test coverage;
   * maintainability issue;
   * optional modernization.
6. Do not describe an old style, obsolete API, missing modern feature, or unusual microkernel design as a defect by itself.
7. Reject findings that cannot be tied to specific source evidence.
8. State uncertainty clearly and assign a confidence level.
9. Prefer small reproducible tests over speculative fixes.
10. Check existing tests and callers before recommending a change.
11. Identify whether a proposed fix affects IPC protocols, ABI structures, boot-image composition, service restart behaviour, or driver privileges.
12. Do not recommend broad rewrites when a minimal compatible fix is possible.

## Historical comparison

Later repository history may be examined as forensic evidence of bugs fixed after commit `588a35b`, but later code must not automatically be copied into R3.3.0.

For every relevant later commit:

* state the commit identifier;
* describe what changed;
* verify that the affected R3.3.0 code is substantially the same;
* identify dependencies introduced after R3.3.0;
* decide whether the fix can be safely backported;
* distinguish a true fix from later architectural development.

## Required finding format

Each candidate issue must include:

* Finding ID
* Classification
* Severity
* Confidence
* Component
* Source path and symbol
* Evidence
* Trigger or precondition
* Expected behaviour
* Actual or likely behaviour
* User-visible consequence
* Reproduction or validation method
* Existing test coverage
* Relevant later commit, if found
* Minimal compatible fix
* Regression risks
* Status: candidate, reproduced, confirmed, rejected, or fixed

Do not present a candidate as confirmed until it has been reproduced, demonstrated through a definite control-flow argument, or supported by an applicable later fix.
