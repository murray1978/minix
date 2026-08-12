The VS Code workspace root is the mapped MINIX source directory:

    M:\

This corresponds to `/usr/src` inside the MINIX VM.

Use workspace-relative paths when creating or modifying files. The MINIX
drivers directory is:

    minix/drivers/

Therefore, create this driver under:

    minix/drivers/ai_driver/

Do not create an additional `usr/src`, `M:`, or `minix` directory level.

Create a tutorial-quality character-driver template for a future AI inference
service. Place it under:

    minix/drivers/ai_driver/

Do not put the language model or transformer runtime inside the driver yet.
This first version must be a small, functional mock driver that demonstrates
the correct MINIX character-driver structure and can later be connected to a
separate AI model service using MINIX IPC.

Before writing code, inspect this exact source tree and follow its existing
conventions. In particular, locate and study:

- the existing hello/example character driver;
- other small drivers that use libchardriver;
- minix/include/minix/chardriver.h;
- minix/lib/libchardriver/chardriver.c;
- the SEF startup and chardriver_announce patterns;
- the driver Makefiles and parent build-system integration.

Do not invent API names or callback signatures. Use the signatures and
conventions from this checked-out MINIX source.

Goal
====

Create a device that can eventually be exposed as:

    /dev/ai_driver

The prototype should support shell usage resembling:

    echo '-v on' > /dev/ai_driver
    echo '-v off' > /dev/ai_driver
    echo 'status' > /dev/ai_driver
    echo 'reset' > /dev/ai_driver
    echo 'i robots need sleep' > /dev/ai_driver
    echo 'i "robots need sleep"' > /dev/ai_driver
    cat /dev/ai_driver

For now, an input command beginning with `i ` should produce a mock response,
for example:

    prompt=robots need sleep
    response=[mock AI response] robots need sleep

This is a character driver, not a block driver.

Required files
==============

Create at least:

    minix/drivers/ai_driver/ai_driver.c
    minix/drivers/ai_driver/Makefile
    minix/drivers/ai_driver/README.md
    minix/drivers/ai_driver/test-ai-driver.sh

Add the directory to the appropriate parent MINIX build file, following the
same pattern as nearby drivers. Keep build-system changes minimal.

Driver behaviour
================

Use libchardriver and the standard MINIX character-driver task loop.

Implement the callbacks required by this source tree, including the relevant
open, close, read, write, ioctl if needed, and cancellation callbacks. Only
include callbacks that are actually supported by this MINIX revision.

Use SEF startup and announce the driver using the same pattern as the existing
example driver.

Use safe-copy grants correctly:

- copy write data from the calling process with the appropriate
  sys_safecopyfrom-style API;
- copy read data back with the corresponding sys_safecopyto-style API;
- validate all grant lengths and offsets;
- never dereference a user-process pointer directly.

Initial implementation limits:

    maximum command length: 1024 bytes
    maximum prompt length:  768 bytes
    maximum response size:  4096 bytes

Store state in one global driver-state structure for this first prototype:

    verbose enabled/disabled
    state: idle, done, or error
    last command
    last prompt
    current response
    response length
    monotonically increasing request number
    last error code or message

Command parser
==============

Support these newline-terminated commands:

    -v on
    -v off
    status
    reset
    i <prompt>

Also accept matching outer double quotes around a prompt:

    i "robots need sleep"

Do not implement general shell quoting or escaping. Only strip one matching
pair of outer double quotes.

Command semantics:

1. `-v on`

   Enable verbose mode and set the readable response to something such as:

       verbose=on

2. `-v off`

   Disable verbose mode and set the readable response to:

       verbose=off

3. `status`

   Set the readable response to a multi-line status block:

       state=idle
       request=3
       verbose=on
       prompt_length=0
       response_length=...

4. `reset`

   Clear the prompt, response, errors, and state. Preserve the request counter.
   Return a short acknowledgement.

5. `i <prompt>`

   Validate the prompt, increment the request counter, and create a mock
   response. The response should include the request number and prompt.

Errors
======

Return appropriate MINIX/POSIX errors:

    EINVAL   malformed or unknown command
    E2BIG    command, prompt, or response exceeds its configured limit
    ENXIO    unsupported minor device
    EBUSY    reserved for a later asynchronous implementation
    EFAULT   grant-copy failure where appropriate

Do not silently truncate commands or prompts.

Read semantics
==============

Implement normal character-device read behaviour using the supplied file
position:

- return bytes from the current response beginning at the requested position;
- return only as many bytes as fit in the caller's buffer;
- return zero at end-of-file;
- support repeated partial reads correctly;
- do not clear the response merely because one reader reaches EOF;
- a later command replaces the previous response.

This first version does not need to block while waiting for inference because
the mock response is produced synchronously in the write callback.

However, add a clearly marked tutorial section in both the source comments and
README explaining how the next version can support asynchronous inference:

- the write callback sends a request to a separate `ai_model` service;
- a read may be left pending using the actual EDONTREPLY mechanism supported
  by this MINIX tree;
- save endpoint, grant, size, request ID, and position;
- complete the read later using the actual chardriver reply function;
- implement cancellation correctly;
- allow only one pending request initially;
- return EBUSY for a second request.

Do not implement fake asynchronous code with guessed API names. Base the
tutorial description on the actual headers and libchardriver implementation in
this tree.

Minor devices
=============

For the first version, support only minor 0:

    /dev/ai_driver

Structure the code so these could later be added:

    minor 1: /dev/ai_status
    minor 2: /dev/ai_log

Do not permanently allocate or modify a global device major number without
first explaining the implications.

For the tutorial, follow the same temporary/prototype major-number strategy as
the existing hello driver, if that is what this source tree uses. Clearly warn
that this is only valid when the conflicting example driver is not running.

README
======

Write a detailed README covering:

1. Why this is a character driver rather than a block driver.
2. Directory and file layout.
3. The MINIX message flow:

       shell
         -> VFS
         -> ai_driver
         -> mock backend
         -> response buffer
         -> read/cat

4. Explanation of each callback.
5. Safe-copy grant handling.
6. Response-buffer and file-position behaviour.
7. Building the driver.
8. Installing or starting it with the actual service-management commands
   supported by MINIX 3.3.0.
9. Creating `/dev/ai_driver` with `mknod`, using a clearly identified
   prototype major and minor 0.
10. Stopping and updating the service.
11. Running the supplied test script.
12. Troubleshooting:
    - ENXIO;
    - service not started;
    - wrong major number;
    - stale device node;
    - safe-copy failure;
    - driver restart;
    - build not including the new directory.
13. How this prototype will later be split into:

       ai_driver character service
              |
              | typed MINIX IPC
              v
       ai_model inference service

14. Why the model should remain outside the driver:
    - responsiveness;
    - independent restart;
    - least authority;
    - easier fault isolation;
    - later CPU/GPU backend replacement.

Test script
===========

Create `test-ai-driver.sh` that:

- verifies `/dev/ai_driver` exists and is a character device;
- writes `-v on`;
- reads and checks the response;
- writes `status`;
- checks that verbose mode is reported;
- writes `i robots need sleep`;
- checks that the mock response contains the prompt;
- tests a quoted prompt;
- tests `reset`;
- sends an invalid command and confirms that the write fails;
- sends an oversized command and confirms that it fails;
- exits nonzero on failure;
- prints a concise PASS/FAIL summary.

The script must use tools normally available in a base MINIX installation.
Avoid Bash-specific syntax unless `/bin/bash` is verified to exist. Prefer
portable `/bin/sh`.

Code quality
============

Use C compatible with the native MINIX Clang and the MINIX build system:

    -std=c99
    WARNS=4 compatible
    no variable-length arrays
    no threads
    no mmap
    no external libraries
    no GNU-only functions unless already used by nearby MINIX drivers

Mark functions `static` where appropriate.

Check every size calculation and avoid signed/unsigned conversion warnings.
Initialize variables where the older Clang data-flow analysis may otherwise
produce `-Wsometimes-uninitialized`.

Use bounded parsing and formatting. If `snprintf()` is used, check its result
for truncation.

Include tutorial comments, but keep the driver implementation readable rather
than filling every line with commentary.

Do not modify kernel code, VFS, libchardriver, dmap, or global headers for this
prototype unless compilation proves a change is absolutely required. If such a
change appears necessary, stop and explain it instead of making a broad
unreviewed edit.

Verification
============

After creating the files:

1. Show every file created or modified.
2. Show the exact build command.
3. Build the driver from this MINIX source tree.
4. Fix all compiler warnings and errors.
5. Show the final successful build output.
6. Show the service startup and `mknod` commands, but do not run destructive or
   system-wide installation commands without explicit confirmation.
7. Summarize the command protocol and future IPC integration points.
8. Call out any source-tree-specific API differences discovered while
   comparing the current MINIX code with more recent driver documentation.

VS Code and Copilot are operating through the mapped Windows workspace rooted
at M:\. Do not attempt to execute Windows build commands against this source.

After creating the files, provide the MINIX-side build commands using paths
rooted at /usr/src, for example:

    cd /usr/src/minix/drivers/ai_driver
    make CC=clang

The actual build will be run inside the MINIX VM over SSH or its console.

The deliverable is a working mock character driver plus a tutorial that can be
used as the foundation for `/dev/ai_driver`.

When inspecting files, search relative to the workspace root. For example,
inspect:

    minix/drivers/examples/
    minix/drivers/
    minix/include/minix/chardriver.h
    minix/lib/libchardriver/

Do not search for these under an additional `/usr/src` directory because the
workspace root already represents `/usr/src`.