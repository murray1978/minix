# ai_driver: MINIX 3.3.0 tutorial character driver

## 1. Why this is a character driver
This prototype accepts small command strings and returns text responses.
That access pattern is message-oriented and sequential, which matches a
character driver. It is not a block storage interface and does not expose
fixed-size sectors.

## 2. Directory and file layout
- minix/drivers/ai_driver/ai_driver.c: character driver front-end.
- minix/drivers/ai_driver/Makefile: service build file.
- minix/drivers/ai_driver/test-ai-driver.sh: shell validation script.
- minix/drivers/ai_driver/README.md: tutorial and operations guide.

## 3. MINIX message flow
```text
shell command
  -> VFS write/read request
  -> ai_driver (libchardriver dispatch)
  -> typed IPC + grants into ai_model
  -> response buffer update
  -> VFS read path (cat) returns response bytes
```

Future split:
```text
ai_driver (character service)
  -> typed MINIX IPC
ai_model (separate inference service)
```

## 4. Callback overview
Implemented callbacks match this tree's minix/include/minix/chardriver.h.
- cdr_open: validates minor device.
- cdr_close: validates minor device.
- cdr_read: copies response bytes to caller with sys_safecopyto.
- cdr_write: copies command bytes from caller with sys_safecopyfrom,
  parses command, and performs synchronous request/reply with ai_model.
- cdr_ioctl: validates minor and returns ENOTTY for unsupported control
  operations in this prototype.
- cdr_cancel: reserved for asynchronous follow-up; currently no pending I/O,
  returns EDONTREPLY unless a saved pending request is matched.

Not implemented in this prototype:
- cdr_select, cdr_intr, cdr_alarm, cdr_other: not needed for synchronous mock
  behavior.

## 5. Safe-copy grant handling
The driver never dereferences user pointers.
- Write path: sys_safecopyfrom(endpoint, grant, 0, local_buffer, size).
- Read path: sys_safecopyto(endpoint, grant, 0, response+offset, chunk).
- Copy failures map to EFAULT.
- Lengths and offsets are validated before copying.

## 6. Response buffer and file-position behavior
- Maximum response size is 4096 bytes.
- Reads honor the supplied file position.
- Partial reads are supported.
- EOF returns 0 when position is at or past response length.
- Reaching EOF does not clear the response.
- A later accepted command replaces the previous response.

## 7. Command protocol
Commands must be newline-terminated.
Supported commands:
- -v on
- -v off
- -t <temperature>
- -p <top-p>
- -n <steps>
- -s <seed>
- status
- reset
- i <prompt>
- i "<prompt>"

Limits:
- maximum command length: 1024 bytes
- maximum prompt length: 768 bytes
- maximum response size: 4096 bytes
- maximum generation steps: 256

Errors:
- EINVAL: malformed or unknown command.
- E2BIG: command, prompt, or generated response exceeds limits.
- ENXIO: unsupported minor (anything other than minor 0).
- EBUSY: reserved for asynchronous follow-up implementation.
- EFAULT: safe-copy grant transfer failure.

## 8. Building the driver
Inside MINIX VM:
```sh
cd /usr/src/minix/drivers/ai_driver
make CC=clang
```

## 9. Starting service and creating /dev/ai_driver
This tree already reserves major 17 for the hello example in
minix/include/minix/dmap.h. For this tutorial prototype, use the same
temporary strategy only when hello is not running.

Before first start, add service stanzas for both ai_model and ai_driver in
/etc/system.conf.

Important: editing `etc/system.conf` in the source tree does not change the
live VM file. You must update `/etc/system.conf` in the running MINIX system
and then restart/update the services.

Minimal example for ai_model:
```text
service ai_model
{
  uid 0;
  ipc
    SYSTEM pm rs ds vm vfs ai_driver
  ;
  system ALL;
  vm BASIC;
  io NONE;
  irq NONE;
};
```

Minimal example for ai_driver:
```text
service ai_driver
{
  uid 0;
  ipc
    SYSTEM pm rs ds vm vfs ai_model
  ;
  system ALL;
  vm BASIC;
  io NONE;
  irq NONE;
};
```

Start ai_model first:
```sh
service up /service/ai_model -label ai_model
```

Start service:
```sh
service up /service/ai_driver -label ai_driver -major 17
```

Create device node:
```sh
mknod /dev/ai_driver c 17 0
chmod 666 /dev/ai_driver
```

Important: major 17 is a prototype choice and may conflict with hello.
Do not treat it as a permanent allocation.

## 10. Stopping and updating service
Stop:
```sh
service down ai_driver
```

Refresh with rebuilt binary:
```sh
service update /service/ai_driver -label ai_driver
```

## 11. Running the supplied test script
```sh
cd /usr/src/minix/drivers/ai_driver
sh ./test-ai-driver.sh
```

Denied-IPC hardening check:
```sh
AI_DRIVER_EXPECT_IPC_DENIED=yes sh ./test-ai-driver.sh
```
This mode verifies that an inference write fails cleanly when RS IPC policy
blocks ai_driver -> ai_model traffic, while ai_driver remains alive and
reports error state via `status`.

## 12. Troubleshooting
- ENXIO:
  - The node may have the wrong minor, or driver checks rejected nonzero minor.
- Service not started:
  - Verify service list and restart with service up.
- ipc mask denied SENDREC:
  - Ensure `ai_driver` IPC list includes `ai_model` label.
  - Ensure `ai_model` IPC list includes `ai_driver` label.
  - Confirm services were started with labels `ai_driver` and `ai_model`.
- service 'ai_driver' not found in '/etc/system.conf':
  - Add the ai_driver stanza shown above to /etc/system.conf, then retry
    service up.
- Wrong major number:
  - Recreate node with the same major passed to service up.
- Stale device node:
  - Remove and recreate /dev/ai_driver after major changes.
- Safe-copy failure:
  - Verify caller endpoint is valid and request sizes are sane.
- Driver restart:
  - Re-run service up or service update, then retest I/O.
- Build not including new directory:
  - Confirm minix/drivers/Makefile contains SUBDIR+= ai_driver.
- Backend unavailable:
  - Ensure ai_model is up with label ai_model.
  - Confirm checkpoint/tokenizer paths exist under
    /usr/src/minix_ai_stage3_stories15m.
 - Driver crash after IPC denial:
  - Use the hardened driver that checks sendrec return values before touching
    reply fields, revokes grants on all paths, and records bounded error text.

## 13. Planned split for real inference
Planned architecture:
```text
ai_driver character service
      |
      | typed MINIX IPC
      v
ai_model inference service
```

The driver remains focused on command/stream mediation. The model service
owns heavyweight inference logic and model lifecycle.

## 14. Why model runtime stays outside driver
Keeping the model out of the driver improves:
- responsiveness: driver path remains short and predictable.
- independent restart: ai_model can crash/restart without replacing device API.
- least authority: narrower privileges in each service.
- fault isolation: compute faults stay away from device front-end.
- backend replacement: later CPU/GPU backends can be swapped without changing
  the device contract.

## Asynchronous follow-up tutorial notes
The existing libchardriver interface in this tree supports real deferred
replies through EDONTREPLY and chardriver_reply_task.

A next version can:
- accept i <prompt> in write callback.
- send typed IPC request to ai_model.
- if read arrives before model response, store one pending read
  (endpoint, grant, size, id, position) and return EDONTREPLY.
- when ai_model reply arrives, fill response buffer and complete pending read
  with chardriver_reply_task(saved_endpt, saved_id, bytes_or_error).
- support cancellation in cdr_cancel by matching endpoint and id and returning
  EINTR.
- return EBUSY when a second request is issued while one is pending.

No guessed API names are required for this pattern; it uses symbols that are
present in minix/include/minix/chardriver.h and minix/lib/libchardriver/chardriver.c.

## Source-tree-specific notes compared to newer docs
- This MINIX revision uses cdr_read/cdr_write callbacks with an explicit
  cdev_id_t id and grant-based copying.
- Deferred replies are done with EDONTREPLY plus chardriver_reply_task,
  not older SUSPEND flows.
- cdr_cancel is part of the modern asynchronous cancellation path and should
  be implemented if reads or writes may be deferred.

  ## After Server is up and talking to driver an RC
  # Local startup for ai_model + ai_driver
case "$1" in
start|autoboot)
    # Start backend first
    service fi ai_model >/dev/null 2>&1 || \
        service up /service/ai_model -label ai_model

    # Start driver front-end
    service fi ai_driver >/dev/null 2>&1 || \
        service up /service/ai_driver -label ai_driver -major 17

    # Ensure device node exists
    if [ ! -c /dev/ai_driver ]; then
        rm -f /dev/ai_driver
        mknod /dev/ai_driver c 17 0
        chmod 666 /dev/ai_driver
    fi
    ;;
esac
