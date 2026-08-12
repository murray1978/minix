# MINIX AI Stage 3: `stories15M` inference

This package is the next step after the Stage 1 platform/kernel tests and the
Stage 2 synthetic autoregressive model.

It implements a native FP32 Llama 2 inference engine for MINIX 3.3.0 and is
compatible with the version-0 checkpoint and tokenizer formats used by
Andrej Karpathy's `llama2.c`.

The acceptance target is the pretrained **TinyStories `stories15M` model**:

- model width: 288
- layers: 6
- attention heads: 6
- KV heads: 6
- context: 256
- parameters: approximately 15 million
- checkpoint: 60.8 MB FP32
- tokenizer vocabulary: 32,000

The original project documents `stories15M.bin` as its first pretrained model
and provides a simple pure-C inference engine. This MINIX version retains the
same architecture and model/tokenizer formats but deliberately differs in its
OS-facing implementation:

- buffered `fread()` checkpoint loading rather than `mmap()`;
- exact checkpoint-length validation;
- checked `size_t` arithmetic suitable for a 32-bit process;
- strict tokenizer validation;
- `CLOCK_MONOTONIC` timing;
- deterministic seeds;
- optional full-logit binary traces;
- no threads, OpenMP, CUDA, or external runtime dependency.

## Files

```text
minix_llama.c             Native inference engine
Makefile                  MINIX bsd.prog.mk build
Makefile.host             Host build for reference traces
compare_trace.py          Compare host and MINIX logit traces
make_smoke_fixture.py     Generate a tiny compatible smoke model/tokenizer
download-model.ps1        Download the real model/tokenizer on Windows
download-model.sh         Download the real model/tokenizer on Unix
run-stories15m.sh         Standard MINIX execution wrapper
THIRD_PARTY_NOTICE.md     llama2.c attribution and MIT terms
```

The pretrained model is not included in the archive.

## 1. Download the model on Windows

From PowerShell in this directory:

```powershell
powershell -ExecutionPolicy Bypass -File .\download-model.ps1
```

The script downloads:

```text
stories15M.bin
tokenizer.bin
```

and verifies the published SHA256 for `stories15M.bin`:

```text
cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a
```

Copy the entire directory onto the mapped MINIX source drive, for example:

```text
M:\minix_ai_stage3_stories15m
```

MINIX itself does not need working HTTPS download tools.

## 2. Build on MINIX

```sh
cd /usr/src/minix_ai_stage3_stories15m
make CC=clang
```

The Makefile includes `MAN=` so the experimental command does not require a
manual page.

## 3. Validate without inference

```sh
./minix_llama stories15M.bin \
    -z tokenizer.bin \
    -c \
    -i "Once upon a time"
```

This validates:

- little-endian version-0 checkpoint header;
- all dimensions and divisibility constraints;
- overflow-safe expected parameter count;
- exact checkpoint length;
- tokenizer header and every vocabulary entry;
- prompt tokenization and decoding.

No transformer forward pass is performed.

## 4. First real pretrained generation

Use greedy sampling first so the result is deterministic:

```sh
./minix_llama stories15M.bin \
    -z tokenizer.bin \
    -t 0 \
    -n 128 \
    -s 1 \
    -i "Once upon a time" \
    -v
```

Or:

```sh
chmod +x run-stories15m.sh
./run-stories15m.sh
```

For more varied output after deterministic validation:

```sh
./minix_llama stories15M.bin \
    -z tokenizer.bin \
    -t 1.0 \
    -p 0.9 \
    -s 12345 \
    -n 256 \
    -i "One day, a little robot"
```

The total step count includes prompt tokens and cannot exceed the model context.

## 5. Host/MINIX reference-logit comparison

Build the same source on Windows through WSL/Linux, or on another Unix host:

```sh
make -f Makefile.host
```

Produce a deterministic greedy host trace:

```sh
./minix_llama stories15M.bin \
    -z tokenizer.bin \
    -t 0 \
    -n 48 \
    -s 1 \
    -i "Once upon a time" \
    -r host.trace
```

Run the same command on MINIX, changing only the trace name:

```sh
./minix_llama stories15M.bin \
    -z tokenizer.bin \
    -t 0 \
    -n 48 \
    -s 1 \
    -i "Once upon a time" \
    -r minix.trace
```

Copy `minix.trace` to the host and compare:

```sh
python3 compare_trace.py host.trace minix.trace --tolerance 1e-5
```

The comparator checks:

- identical forward-pass count;
- identical input token at every position;
- every vocabulary logit;
- maximum absolute error.

A greedy token-path difference causes an immediate failure even when logits are
otherwise close.

A 48-step trace with a 32,000-token vocabulary is approximately 6.1 MB.

## 6. Smoke fixture

The source can be compiled and exercised without downloading the 60.8 MB model:

```sh
python3 make_smoke_fixture.py --output-dir smoke
make -f Makefile.host
./minix_llama smoke/smoke_model.bin \
    -z smoke/smoke_tokenizer.bin \
    -t 0 \
    -n 16 \
    -i "Hi" \
    -v
```

The fixture is not trained and its output is meaningless. It only validates
the version-0 loader, tokenizer, forward path, KV cache, and cleanup.

## Acceptance gates

### Gate A — loader

```text
checkpoint and tokenizer validation: PASS
```

### Gate B — real learned generation

The program loads `stories15M.bin` and emits decoded text without a service
failure, VM/VFS error, or process crash.

### Gate C — numerical comparison

Host and MINIX traces follow the same greedy token path and remain inside the
selected logit tolerance.

### Gate D — lifecycle

Run repeated separate process invocations and check the MINIX logs:

```sh
i=0
while [ "$i" -lt 20 ]; do
    ./minix_llama stories15M.bin \
        -z tokenizer.bin -t 0 -n 64 \
        -i "Once upon a time" >/dev/null || exit 1
    i=$((i + 1))
done
```

Then inspect `/var/log/messages` for PM, VM, VFS, RS, allocation, or service
restart errors.

## Expected resource use

Approximate values for `stories15M`:

- checkpoint weights: about 60.8 MB;
- KV caches: about 3.5 MB;
- logits: about 128 KB;
- tokenizer strings and tables: several hundred KB to a few MB;
- temporary activations: below 1 MB.

The engine prints measured allocation categories with `-v`.

## Current limitations

- FP32 only;
- one CPU;
- no SIMD-specific kernel;
- no OpenMP;
- no CUDA;
- version-0 `llama2.c` checkpoints only;
- generation only, not training;
- context cannot roll over after the model's configured sequence length;
- no MINIX system service or IPC interface yet.

After this stage passes, the logical choices are:

1. optimize FP32 matrix multiplication;
2. add the `runq.c` Q8_0 format;
3. separate model loading, tokenization, and inference into restartable MINIX
   services;
4. investigate a GPU service and CUDA-like accelerator boundary later.


## MINIX 3.3.0 compiler compatibility

The source declares terminating helper functions with a portable `noreturn`
attribute because the native build uses `WARNS=4` and promotes
`-Wmissing-noreturn` to an error.

The Makefile defines `_LARGEFILE_SOURCE`. In the MINIX 3.3.0 `stdio.h`,
`fseeko()` and `ftello()` are exposed under `_LARGEFILE_SOURCE`,
`_NETBSD_SOURCE`, or a suitable X/Open feature level; `_POSIX_C_SOURCE` alone
does not expose these declarations in this header revision.

Variables populated by paired short-circuit read/size checks are initialized
explicitly for compatibility with the older native Clang
`-Wsometimes-uninitialized` analysis.

## Stage 4 location

Stage 4 training utilities are kept in a separate tree:

`/usr/src/minix_ai_stage3_stories15m/minix_ai_stage4_train`

This Stage 3 folder remains dedicated to inference/runtime artifacts.
