# AI Stage 3 Baseline

This baseline freezes the known-good Stage 3 behavior before Stage 4 training work.

## Artifact identity

- Checkpoint path: /usr/src/minix_ai_stage3_stories15m/stories15M.bin
- Checkpoint bytes: 60816028
- Checkpoint SHA-256: cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a
- Tokenizer path: /usr/src/minix_ai_stage3_stories15m/tokenizer.bin
- Tokenizer bytes: capture with `wc -c tokenizer.bin` or `stat -f %z tokenizer.bin`
- Tokenizer hash: capture with `sha256 tokenizer.bin` (or `sha256sum`/`openssl dgst -sha256`)

## Model dimensions

From the current Stage 3 runtime configuration:

- dim: 288
- hidden_dim: 768
- n_layers: 6
- n_heads: 6
- n_kv_heads: 6
- vocab_size: 32000
- seq_len: 256

## Prompt and deterministic test

Prompt:

Once upon a time

Deterministic command:

./minix_llama stories15M.bin -z tokenizer.bin -t 0 -n 64 -i "Once upon a time"

Prompt token IDs:

- Record from `./minix_llama stories15M.bin -z tokenizer.bin -c -i "Once upon a time"`

Greedy generated text baseline:

- Record exact output from deterministic command above.

Standalone token rate baseline:

- Record stderr line: generated N token(s) in S seconds (R token/s)
- Previously observed reference: approximately 15 token/s

## ai_model and ai_driver integration baseline

Driver command sequence:

- echo "-t 0" > /dev/ai_driver
- echo "-s 1" > /dev/ai_driver
- echo "-v on" > /dev/ai_driver
- echo "-n 64" > /dev/ai_driver
- echo "-p 0.9" > /dev/ai_driver
- echo 'i "Once upon a time"' > /dev/ai_driver
- cat /dev/ai_driver

Observed successful response pattern:

- request=...
- backend=ai_model
- tokens=...
- elapsed_ms=...
- response=...

Record exact current `/dev/ai_driver` generated text in this section after each verified release build.

## Binary sizes

Record with:

- ls -l /service/ai_model /service/ai_driver /usr/src/minix_ai_stage3_stories15m/minix_llama

Previously observed:

- /service/ai_model: 259168 bytes
- /service/ai_driver: 107394 bytes

Executable file sizes
=====================

/service/ai_driver:
    107394 bytes

/service/ai_model:
    259168 bytes

/usr/src/minix_ai_stage3_stories15m/minix_llama:
    247733 bytes

## Source identity
===============

MINIX source commit:
    588a35b

minix_llama.c bytes:
    51691

minix_llama.c lines:
    1774

minix_llama.c SHA-256:
    <capture>

ai_model source SHA-256:
    <capture tree or principal-source hashes>

ai_driver source SHA-256:
    <capture tree or principal-source hashes>

Compiler:
    <clang --version>

System:
    <uname -a>

## size(1) output snapshots

Record with:

- size /service/ai_model
- size /service/ai_driver
- size /usr/src/minix_ai_stage3_stories15m/minix_llama


   text    data     bss     dec     hex filename
  49676     252   51044  100972   18a6c /service/ai_driver
 221296    1324   15028  237648   3a050 /service/ai_model
 217274    1264   13172  231710   3891e /usr/src/minix_ai_stage3_stories15m/minix_llama

## Compiler and build commands

MINIX native:

- cd /usr/src/minix_ai_stage3_stories15m
- make clean
- make CC=clang

- cd /usr/src/minix/servers/ai_model
- make clean
- make CC=clang
- make install

- cd /usr/src/minix/drivers/ai_driver
- make clean
- make CC=clang
- make install

## Stage 3 regression guard

Run:

- cd /usr/src/minix_ai_stage3_stories15m
- sh ./test-stage3-regression.sh

Expected:

- checkpoint byte-length check passes
- SHA-256 check passes
- checkpoint/tokenizer validation passes
- deterministic greedy generation passes
- no repeated <unk> run
- optional /dev/ai_driver check passes when services are running

## Notes

- Stage 4 must not overwrite or mutate stories15M.bin.
- Stage 4 adapter work is additive and external to base checkpoint bytes.
