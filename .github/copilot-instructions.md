# MINIX 3.3.0 repository instructions

This repository is the MINIX 3.3.0 source tree corresponding to Git commit `588a35b`, used by the ISO `minix_R3.3.0-588a35b.iso`.

## Platform and build environment

* The primary target is 32-bit x86 MINIX 3.3.0.
* The source is edited from VS Code on Windows through the existing SSHFS-mounted source tree.
* Compilation, installation, tests, diagnostics, hashes, and runtime execution occur inside the MINIX virtual machine.
* The user performs all MINIX shell interaction manually in an existing MINIX shell.
* Copilot must not attempt to execute MINIX commands through VS Code, SSH, PowerShell, WSL, MSYS, Git Bash, or another local shell.
* Do not assume that the Windows host can compile or test this source.
* The reference full build command is `make build`.
* The development build command is `make build MKUPDATE=yes`.
* These build commands are reference commands for the user to run manually in MINIX. Copilot must not invoke them.
* Do not introduce Linux-specific, glibc-specific, systemd-specific, GNU-only, or modern POSIX APIs without first verifying that MINIX 3.3.0 provides them.
* Do not assume a modern C standard. Preserve the conventions and compiler limitations of the existing source unless a modernization task explicitly says otherwise.

## Critical Copilot execution rule

**Copilot edits files only. The user runs all MINIX shell commands manually.**

The VS Code / Copilot terminal integration is not reliable for this MINIX VM, so terminal execution is explicitly outside Copilot's role.

Copilot may:

* read existing project files;
* inspect code and reason about changes;
* edit existing project files;
* create new C/source files, headers, shell runners, reports, Makefile entries, and other project files;
* prepare exact commands for the user to run manually in MINIX.

Copilot must not:

* open or configure a terminal;
* use VS Code terminal integration;
* invoke VS Code build tasks;
* invoke SSH or SSHFS;
* use PowerShell to connect to MINIX;
* use WSL, MSYS, Git Bash, or another Windows/local shell to run MINIX commands;
* run `make`;
* invoke the compiler;
* run generated shell scripts;
* run compiled MINIX programs;
* run tests or diagnostics;
* calculate hashes by executing commands;
* attempt to reconnect, repair, restart, remount, or diagnose the MINIX shell/session or SSHFS mapping unless the user explicitly requests a task about that infrastructure;
* assume that a command succeeded because a source or runner file was created;
* invent runtime output;
* proceed to a later experimental stage without the user's returned MINIX results when the current stage requires runtime validation.

If a task requires runtime evidence, Copilot must create or edit the necessary files, give the exact MINIX commands to the user, and stop.

## Filesystem workflow

The existing SSHFS-mounted source tree is the editing interface.

Copilot may read, edit, and create project files through that mapped filesystem.

Do not attempt to discover, recreate, remount, repair, or reconfigure the SSHFS connection unless the user explicitly asks for that work.

Treat files visible through the mapped tree as source-editing access only, not as permission to execute commands on the MINIX VM.

## Command handoff

When work is ready to build, test, validate, hash, or run, stop editing and provide the exact commands the user should run manually in the already-open MINIX shell.

Use this handoff format unless the task specifies another exact format:

```text
CREATED:
<files created or modified>

RUN IN MINIX MANUALLY:

cd <required-directory>
<command>
<command if needed>
```

Then stop.

The user will run the commands in MINIX and paste or upload the results.

Do not execute the commands yourself.

## MINIX userland and shell compatibility

Assume MINIX / NetBSD-style userland, not GNU/Linux userland.

Generated shell scripts must:

* use `#!/bin/sh`;
* use POSIX shell syntax;
* avoid Bash-only syntax unless explicitly required;
* use LF line endings;
* contain no UTF-8 BOM.

Do not assume GNU command-line options or GNU utilities are installed.

In particular, do not use:

```text
sha256sum
```

Use:

```text
openssl dgst -sha256 <file>
```

unless a task explicitly establishes another available hashing command.

When proposing a command, prefer utilities and options already demonstrated to work on this MINIX system.

## Experimental workflow

Work in small, measurable stages.

For each requested task:

1. Do only the requested stage or sub-stage.
2. Do not bundle future stages into the current change.
3. Do not silently modify unrelated files.
4. Preserve historical artifacts unless explicitly instructed otherwise.
5. Do not overwrite an accepted report merely to reuse its filename when the task requires a new historical comparison.
6. Do not change acceptance criteria or numeric tolerances automatically.
7. If a test, identity check, or acceptance gate fails, report the failure and stop for review.
8. Do not proceed to training, cache regeneration, dataset changes, evaluator changes, service changes, or another stage unless the current prompt explicitly authorizes it.
9. When the prompt says `STOP`, stop after completing that task.
10. Prefer one narrowly scoped diagnostic or repair at a time.

## Task identities and current values

Prompts may contain exact project identities and acceptance values, including:

* dataset SHA-256 values;
* checkpoint SHA-256 values;
* tokenizer SHA-256 values;
* cache/report SHA-256 values;
* expected record counts;
* expected target-token counts;
* split counts;
* acceptance thresholds;
* known regression values.

Treat values explicitly supplied in the current task as binding inputs for that task.

Do not replace current prompt-supplied values with older values remembered from a previous Copilot conversation, older report, historical artifact, or stale context.

If current files disagree with a prompt-supplied identity, report the disagreement and stop rather than silently choosing one.

Historical values must remain clearly labelled as historical and must not silently become current acceptance values.

## Reports and runtime evidence

Do not invent runtime results.

Source inspection may support conclusions about code structure and control flow, but claims about actual MINIX execution must come from user-run MINIX output unless the task explicitly asks only for static analysis.

When creating a runner or diagnostic:

* make it deterministic where practical;
* make failures explicit;
* print concise identity and gate results;
* print a final summary suitable for the user to paste back into the conversation;
* do not silently continue after an identity or prerequisite failure unless the task explicitly requires continued diagnostics.

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

## Final workflow reminder

**COPILOT EDITS FILES ONLY.**

**THE USER RUNS ALL MINIX COMMANDS MANUALLY.**
