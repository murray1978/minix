# MINIX AI Stage 4: native training bootstrap

This folder contains Stage 4-only training artifacts and utilities.

## Stage 4E A4 accepted validation chain

The accepted A4 dataset is `stage4e-approved.records`. The B/C baseline and
cache-validation stages established the frozen-base metrics and validated the
target cache. The accepted training validation chain then established:

- D1: zero-update validation;
- D2: one-step training validation;
- D3: two-step training validation;
- D4: deterministic 10-step trajectory;
- D5: independent trajectory repeat with zero total mismatches;
- D6: step-5 checkpoint round-trip with zero state-byte mismatches; and
- D7: deterministic checkpoint resume, with resumed steps 6--10 matching the
  continuous path and final adapter and Adam state bitwise identical.

F1 is a fixed-horizon observation that starts at optimizer step 5 and ends at
step 50. The validation threshold is first observed at step 15. The lowest
observed validation loss is `8.328697455428` at step 20. By step 50, train
loss has continued to decrease to `3.888226861341` while validation loss has
regressed to `11.624778713258`. No single-token collapse was observed, and
the final validation top-1 set contains 15 unique tokens.

The test split remains sealed through F1. Step 20 is only the lowest observed
validation loss in F1; no production adapter selection or export has occurred.

Target cache binaries, checkpoint binaries, and native executables are
generated artifacts and are normally excluded from Git. The accepted D6
step-5 checkpoint SHA-256 is
`ecb75d52881df04a0c47fcd4b852909a865c62cb54929ab33b230c7cf1e5ec89`; it can
be reproduced by the accepted D6 procedure.

Current utility:

- stage4_train_adapter.c: initializes a standalone adapter sidecar file
  from Stage 3 checkpoint metadata.
- stage4_eval.c: teacher-forced evaluation-only causal loss with log-sum-exp.
- stage4_gradient_check.c: synthetic analytic gradient and clipping checks.
- stage4_one_token_train.c: real-model one-token frozen-base update check.
- stage4_train.c: Stage 4D full-record frozen-base adapter overfit trainer.
- stage4e_dataset_check.c: Stage 4E-A multi-record dataset validator/reporter.

Build on MINIX:

```sh
cd /usr/src/minix_ai_stage3_stories15m/minix_ai_stage4_train
make CC=clang
```

Run:

```sh
./stage4_train_adapter \
  /usr/src/minix_ai_stage3_stories15m/stories15M.bin \
  /usr/src/minix_ai_stage3_stories15m/minix_ai_stage4_train/stories15M.adapter.bin \
  -r 8
```

Single-line equivalent (safer when pasting in interactive shells):

```sh
./stage4_train_adapter /usr/src/minix_ai_stage3_stories15m/stories15M.bin /usr/src/minix_ai_stage3_stories15m/minix_ai_stage4_train/stories15M.adapter.bin -r 8
```

Inspect/validate an existing adapter without modifying files:

```sh
./stage4_train_adapter --inspect \
  /usr/src/minix_ai_stage3_stories15m/stories15M.bin \
  /usr/src/minix_ai_stage3_stories15m/minix_ai_stage4_train/stories15M.adapter.bin \
  -r 8
```

Adapter accounting layout (rank=8, dim=288, vocab=32000):

- A: rank x dim = 8 x 288 = 2304 params = 9216 bytes
- B: vocab x rank = 32000 x 8 = 256000 params = 1024000 bytes
- total: 258304 params = 1033216 bytes

Stage 4B evaluation-only loss:

```sh
make stage4Eval CC=clang
./stage4_eval \
  /usr/src/minix_ai_stage3_stories15m/stories15M.bin \
  -z /usr/src/minix_ai_stage3_stories15m/tokenizer.bin \
  -d ./eval-small.dataset
```

Zero-adapter deterministic regression:

```sh
sh ./test-zero-adapter-regression.sh
```

Stage 4C synthetic gradient and clipping checks:

```sh
make stage4GradientCheck CC=clang
./stage4_gradient_check
```

The synthetic check now includes both:

- no-op clipping check (`threshold=0.5`), and
- active clipping check (`threshold=0.005`) with direction-preservation.

Stage 4D full-record overfit run:

```sh
make stage4Train CC=clang
./stage4_train \
  /usr/src/minix_ai_stage3_stories15m/stories15M.bin \
  -z /usr/src/minix_ai_stage3_stories15m/tokenizer.bin \
  -a stories15M.adapter-overfit.bin \
  -d overfit-one.records \
  -r 8 \
  --optimizer adam \
  --learning-rate 0.001 \
  --steps 500 \
  --eval-every 10 \
  --seed 1
```

Stage 4D trainer properties:

- computes loss/gradients on target tokens only (prompt masked out);
- frozen base checkpoint (no transformer weight updates);
- supports SGD and Adam (`--optimizer adam|sgd`);
- saves inference adapter (`-a`) and separate train checkpoint
  (`--train-checkpoint`, default `stories15M.adapter-overfit.train.ckpt`);
- validates save/reload loss consistency;
- includes deterministic continuous-vs-resumed comparison by default
  (`--resume-check 1`).

Stage 4E-A dataset validation and split report:

```sh
make stage4eDatasetCheck CC=clang
./stage4e_dataset_check \
  -d stage4e-approved.records \
  -z /usr/src/minix_ai_stage3_stories15m/tokenizer.bin
```

One-record fixture regression:

```sh
make stage4eOneRecordTest CC=clang
```

Candidate and approved report generation:

```sh
make stage4eCheckCandidate CC=clang
make stage4eCheckApproved CC=clang
```

Stage 4E-B unadapted baseline evaluation (train/validation/test split metrics):

```sh
make stage4eBaselineUnadapted CC=clang
```

This writes:

- `stage4e-baseline-unadapted-report.txt`

The Stage 4E-B evaluator uses the frozen base checkpoint only (no adapter
sidecar), computes target-window loss/perplexity, and emits totals plus
per-split metrics for `train`, `validation`, `test` (and `regression` if
present).

Equivalent one-shot script inside MINIX:

```sh
sh ./run-stage4e-baseline-unadapted.sh
```

Outputs:

- parser and validation diagnostics with line numbers;
- per-record target-window metadata in verbose mode (`-v`);
- machine-readable summary report:
  `stage4e-dataset-report.txt`.

Validation includes:

- required metadata fields (`id`, `task`, `source`, `split`, `weight`);
- optional `source_unit` metadata (defaults to `source` when omitted);
- split/task/source allow-lists;
- duplicate-field detection;
- single-line visible target convention;
- tokenized target-boundary and context-length checks;
- duplicate and cross-split leakage checks.

Stage 4E-A report fields include:

- split/task/source/source-unit counts;
- duplicate ids/prompts/targets;
- duplicate target counts bucketed by same split vs cross split;
- repeated source units across splits;
- possible cross-split near-duplicate count.

Properties:

- reads only checkpoint header metadata;
- writes a separate adapter file;
- does not modify /usr/src/minix_ai_stage3_stories15m/stories15M.bin.

Current expected result for `stories15M.bin` with `-r 8`:

```text
mode: write
adapter tensor layout
  A: rank x dim = 8 x 288, params=2304, bytes=9216
  B: vocab x rank = 32000 x 8, params=256000, bytes=1024000
  total params=258304 bytes=1033216
base checkpoint SHA-256 before: cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a
base checkpoint bytes: 60816028
stage4 adapter init complete
checkpoint: /usr/src/minix_ai_stage3_stories15m/stories15M.bin
adapter: /usr/src/minix_ai_stage3_stories15m/minix_ai_stage4_train/stories15M.adapter.bin
rank: 8
config: dim=288 hidden=768 layers=6 heads=6 kv_heads=6
trainable_params: 258304
adapter_bytes: 1033216
adapter_file_bytes: 1033356
memory_accounting adapter_weight_bytes=1033216
memory_accounting gradient_bytes=1033216
memory_accounting adam_first_moment_bytes=1033216
memory_accounting adam_second_moment_bytes=1033216
memory_accounting total_trainable_storage_bytes=4132864
adapter file SHA-256: <computed at runtime>
base checkpoint SHA-256 after: cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a
round-trip reload: PASS
note: stories15M.bin was not modified
```
