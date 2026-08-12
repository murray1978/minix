# ai_model service

## Purpose
`ai_model` hosts model inference outside the character driver and exposes
an internal typed IPC API used by `ai_driver`.

## IPC protocol
Shared protocol definitions live in `minix/include/minix/ai_model.h`.

Requests:
- `AI_MODEL_GENERATE`: infer text for a prompt copied through grants.
- `AI_MODEL_STATUS`: return backend and last-request telemetry.
- `AI_MODEL_RESET`: clear per-request runtime state.

Replies:
- `AI_MODEL_REPLY` with `AI_M_RESULT` and telemetry fields.

## Runtime model
This service embeds the current stories15M runtime implementation from:
- `/usr/src/minix_ai_stage3_stories15m/minix_llama.c`

The model is loaded once at service startup and kept resident in memory until
service shutdown/restart.

## Startup
Add a label in `/etc/system.conf`:

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

The `ai_driver` stanza must allow `ai_model` in its `ipc` list as well.
Editing the source-tree copy `etc/system.conf` does not modify the running
system file `/etc/system.conf`; update the live file in the VM before running
`service up`.

Start:

```sh
service up /service/ai_model -label ai_model
```

Optional startup arguments:
- `-m <checkpoint.bin>` set checkpoint path
- `-z <tokenizer.bin>` set tokenizer path
- `-D <diag.txt>` write compact forward-pass diagnostics
- `-X <positions>` log only first N positions (default 4)

Defaults:
- checkpoint: `/usr/src/minix_ai_stage3_stories15m/stories15M.bin`
- tokenizer: `/usr/src/minix_ai_stage3_stories15m/tokenizer.bin`

## Deterministic divergence checks
Use the same inputs across all engines:
- checkpoint: `stories15M.bin`
- tokenizer: `tokenizer.bin`
- prompt: `Once upon a time`
- temperature: `0`
- steps: `64`

### 1) Upstream run.c on host
Build and run upstream `karpathy/llama2.c`:
```sh
./run stories15M.bin -z tokenizer.bin -t 0 -n 64 -i "Once upon a time"
```

### 2) Host minix_llama build
```sh
./minix_llama stories15M.bin -z tokenizer.bin -t 0 -n 64 -i "Once upon a time"
```

### 3) MINIX ai_model through ai_driver
```sh
echo "-t 0" > /dev/ai_driver
echo "-n 64" > /dev/ai_driver
echo 'i "Once upon a time"' > /dev/ai_driver
cat /dev/ai_driver
```

### Optional diagnostics
Start ai_model with diagnostics enabled:
```sh
service down ai_model
service up /service/ai_model -label ai_model -args "-D /tmp/ai_diag.txt -X 4"
```

The diagnostics file contains per-position checkpoints:
- input token id
- embedding checksum
- layer-0 RMSNorm checksum
- layer-0 q/k/v pre-RoPE checksums
- layer-0 q/k post-RoPE checksums
- layer-0 attention output checksum
- layer-0 FFN output checksum
- final logits checksum
- top-10 token ids and logits

Compare these logs against a host run from `minix_llama` using the same
`-D` and `-X` flags to isolate the first diverging operation.

## Known limitation
This integration routes requests through the real stories15M backend, but it
does not guarantee semantically coherent or numerically correct output quality.
That evaluation remains a separate model-quality task.
