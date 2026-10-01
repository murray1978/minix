# Stage 4E Repository Audit Proposal

## 1. Milestone Summary

This is a read-only classification proposal for the Stage 4E-F1-A4 milestone. No Git, build, hash, runtime, or filesystem-cleanup command was run for this audit.

The accepted F1 report records `f1_fixed_horizon_pass=yes`, 45 updates from optimizer step 5 through 50, and zero test gradient/evaluation rows. It records first validation acceptance at step 15, lowest observed validation loss `8.328697455428` at step 20, final validation loss `11.624778713258`, regression after the lowest point, no single-token collapse, and 15 final validation top-1 tokens.

Suggested tag after a separately reviewed local commit: `stage4e-training-validation-2026-10-01`.

## 2. TRACK Candidates

### Core source and build definition

Propose tracking the Stage 4 source-of-truth implementation:

- `Makefile`
- `README.md` (update separately before relying on it as the current Stage 4E index; its visible content predates the accepted D/F chain)
- `stage4_adapter_train.c`, `stage4_adapter_train.h`
- all `stage4*.c` sources, including `stage4e_d1_a4_zero_update_trainer.c` through `stage4e_f1_a4_fixed_horizon.c`
- `stage4e_dataset_check.c`, `stage4e_cache_build_a4.c`, `stage4e_cache_validate_a4.c`, `stage4e_cache_equivalence_a4.c`, and `stage4e_c2c_hidden_diag.c`

### Reproducible runners and datasets

Propose tracking runners that document or reproduce accepted A4 work, particularly:

- `run-stage4e-a4-dataset-repair.sh`
- `run-stage4e-b1r-a4-full-context.sh`
- `run-stage4e-b2r-a4-full-context-top1.sh`
- `run-stage4e-b3r-a4-generation.sh`
- `run-stage4e-b4ar-a4-repeat.sh`
- `run-stage4e-b4br-a4-generation-repeat.sh`
- `run-stage4e-c2a-a4-cache.sh`
- `run-stage4e-c2b-a4-cache-validation.sh`
- `run-stage4e-c2c-a4-full-context-equivalence.sh`
- `run-stage4e-d1-a4-zero-update-trainer.sh` through `run-stage4e-d7-a4-resume.sh`
- `run-stage4e-f1-a4-fixed-horizon.sh`
- `stage4e-approved.records`

The older non-A4 runners are useful only if maintaining the full exploratory history is intentional. Otherwise, put them in `HUMAN_REVIEW` rather than automatically tracking them in this milestone.

### Accepted compact reports and trajectories

Propose tracking these compact acceptance records:

- `stage4e-dataset-report-a4.txt`
- `stage4e-baseline-unadapted-full-context-a4-report.txt` (B1R)
- `stage4e-baseline-unadapted-full-context-a4-top1-report.txt` (B2R)
- `stage4e-baseline-unadapted-generation-a4-report.txt` (B3)
- `stage4e-c2a-a4-cache-report.txt`
- `stage4e-c2b-a4-cache-validation-report.txt`
- `stage4e-c2c-a4-full-context-equivalence-report.txt`
- `stage4e-d1-a4-zero-update-trainer-report.txt`
- `stage4e-d2-a4-one-step-report.txt`
- `stage4e-d3-a4-two-step-report.txt`
- `stage4e-d4-a4-trajectory-report.txt`, `stage4e-d4-a4-trajectory.tsv`
- `stage4e-d5-a4-repeat-report.txt`, `stage4e-d5-a4-repeat-trajectory.tsv`
- `stage4e-d6-a4-checkpoint-report.txt`
- `stage4e-d7-a4-resume-report.txt`, `stage4e-d7-a4-resume-trajectory.tsv`
- `stage4e-f1-a4-fixed-horizon-report.txt`, `stage4e-f1-a4-fixed-horizon-trajectory.tsv`

The F1 report pins the supplied D1/D2/D3/D4/D6/D7 identities and the D6 checkpoint SHA, so it is the compact evidence tying the final continuation back to the accepted chain.

## 3. IGNORE_GENERATED Candidates

Propose ignoring generated native executables that have a same-base-name source file, including:

- `stage4_eval`, `stage4_gradient_check`, `stage4_one_token_train`, `stage4_train`, `stage4_train_adapter`
- `stage4e_baseline_eval`, `stage4e_baseline_full_context_eval`, `stage4e_baseline_full_context_top1_eval`, `stage4e_baseline_top1_eval`, `stage4e_baseline_unadapted_generation`
- `stage4e_cache_build`, `stage4e_cache_build_a4`, `stage4e_cache_equivalence`, `stage4e_cache_equivalence_a4`, `stage4e_cache_validate`, `stage4e_cache_validate_a4`
- `stage4e_d1_a4_zero_update_trainer` through `stage4e_d7_a4_resume`
- `stage4e_dataset_check`, `stage4e_f1_a4_fixed_horizon`

Also propose ignoring:

- `*.o`, including `stage4_train_adapter.o`
- generated runtime adapter/checkpoint products: `stories15M.adapter*.bin`, `stories15M.adapter*.ckpt`
- generated cache binaries: `stage4e-target-cache*.bin`
- generated checkpoint binaries: `stage4e-d6-a4-step5-checkpoint.bin`
- transient console captures: `stage4e-*-console.txt`

These recommendations do not classify accepted report/TSV artifacts as generated noise. Those remain visible and are proposed as track candidates above.

## 4. SAFE_TO_DELETE Candidates

No deletion is proposed for any accepted A4 report, trajectory, dataset, source, or runner.

The following are reasonable `SAFE_TO_DELETE` candidates only after a human confirms that historical A/B/C exploration is not being retained in this repository:

- `stage4e-a4-dataset-repair-console.txt` and other `stage4e-*-console.txt` files when their corresponding compact report/TSV is retained and they contain no unique failure evidence.
- duplicated transient executable and object outputs covered by `IGNORE_GENERATED`.
- `git-checkpoint-status.txt`, because it is a stale captured diagnostic and visibly reports failure of `git branch --show-current` under the stated Git 1.9.0.

Do not delete failed-run diagnostics automatically. In particular, preserve diagnostic-only console/log material until its content has been reviewed for unique evidence not captured in a report.

## 5. HUMAN_REVIEW Candidates

- All pre-A4 historical reports, console files, and repeat-run text files not explicitly named in the accepted chain, including B1/B2 refreshes, B4 repeat inputs, and C2 hidden-state mismatch diagnostics.
- `stage4e-candidate.records`, `stage4e-candidate-report.txt`, and `stage4e-candidate-to-approved.diff`: these may document the A4 dataset decision, but their retention value is a project-history choice.
- `stage4e-a2-split-repair-proposal.txt`, `stage4e-a3-validation-explanation-proposal.txt`, `stage4e-approval-log.txt`, `stage4e-duplicate-target-review.txt`, and `stage4e-pre-c-task-label-audit.txt`: likely decision records, but not proved necessary for F1 reproduction.
- `stage4e-cache-format-design.txt`: likely useful design documentation; track only after confirming it matches the accepted A4 format.
- `stage4e-target-cache.bin`: see the cache recommendation below.
- `.gdbinit`: debugger preferences may be useful, but are not established as portable project configuration.
- `eval-small.dataset`, `overfit-one.records`, `stage4e-one-record-test.records`, `test-stage4e-one-record.sh`, and `test-zero-adapter-regression.sh`: test fixtures/scripts that may be useful, but are outside the accepted A4 continuation chain and should be reviewed as a group.
- `README.md`: source documentation should be tracked, but its current visible content predates the D/F chain; review/update separately before including it as milestone documentation.

## 6. HUMAN_REVIEW_SENSITIVE Candidates

- `chat.json`: the editor could not synchronize it because it exceeds 50 MB. Its purpose and contents were not inspected. Treat it as personal conversation/application state; it may contain prompts, paths, or sensitive material. Do not commit without a manual sensitive-content review.

No credential or token value was exposed during this audit.

## 7. Unrelated Tracked Modifications

The following were explicitly identified as unrelated to the AI milestone unless independently proven otherwise:

| Path | AI_RELEVANCE | Classification | Evidence / recommendation |
| --- | --- | --- | --- |
| `etc/etc-release` | unknown | HUMAN_REVIEW | Text release metadata; no visible Stage 4 relationship. Do not stage with the AI milestone by default. |
| `minix/drivers/storage/memory/memory` | unknown | HUMAN_REVIEW | Appears to be a built driver binary from its path/name. Do not stage with the AI milestone by default. |
| `minix/drivers/storage/ramdisk/image` | unknown | HUMAN_REVIEW | Appears to be a generated boot/driver image from its path/name. Do not stage with the AI milestone by default. |

## 8. Proposed `.gitignore` Patterns

Proposal only; do not apply during this audit.

```gitignore
# Stage 4 native build outputs
minix_ai_stage3_stories15m/minix_ai_stage4_train/stage4_eval
minix_ai_stage3_stories15m/minix_ai_stage4_train/stage4_gradient_check
minix_ai_stage3_stories15m/minix_ai_stage4_train/stage4_one_token_train
minix_ai_stage3_stories15m/minix_ai_stage4_train/stage4_train
minix_ai_stage3_stories15m/minix_ai_stage4_train/stage4_train_adapter
minix_ai_stage3_stories15m/minix_ai_stage4_train/stage4e_*
!minix_ai_stage3_stories15m/minix_ai_stage4_train/stage4e_*.c

# Object files and transient runtime products
minix_ai_stage3_stories15m/minix_ai_stage4_train/*.o
minix_ai_stage3_stories15m/minix_ai_stage4_train/stories15M.adapter*.bin
minix_ai_stage3_stories15m/minix_ai_stage4_train/stories15M.adapter*.ckpt
minix_ai_stage3_stories15m/minix_ai_stage4_train/stage4e-target-cache*.bin
minix_ai_stage3_stories15m/minix_ai_stage4_train/stage4e-d6-a4-step5-checkpoint.bin

# Transient runner captures; reports and trajectories remain visible
minix_ai_stage3_stories15m/minix_ai_stage4_train/stage4e-*-console.txt
```

The `stage4e_*` executable pattern relies on the explicit `!stage4e_*.c` exception and should be manually checked before adoption. A more exact list of executable names is safer if this repository expects future non-executable `stage4e_*` files.

## 9. D6 Checkpoint Recommendation

**Recommendation: `IGNORE_BUT_DOCUMENT_SHA`.**

`stage4e-d6-a4-step5-checkpoint.bin` is exactly reproducible, approximately 3.1 MB, and its checkpoint/resume semantics are already documented by the D6 report, D7 deterministic-resume evidence, and F1’s pinned SHA `ecb75d52881df04a0c47fcd4b852909a865c62cb54929ab33b230c7cf1e5ec89`.

Avoid storing this binary in the milestone commit when the source, runner, report, format validation, and identity are all retained. Rebuild/restore it using the documented D6 process and verify against the recorded SHA. Escalate to `TRACK_BINARY` only if future F2/G work requires an immutable binary handoff unavailable from the accepted generation procedure.

## 10. Cache Binary Recommendation

| Path | Recommendation | Rationale |
| --- | --- | --- |
| `stage4e-target-cache-a4.bin` | IGNORE_GENERATED | This is the authoritative A4 cache: the A4 cache report records SHA `d6288ad3d128c7182c20088aa57a1bca8bcaa1e13314fb0cdbd60eba14278def`, and the C2c A4 report records bitwise hidden/logit equivalence. Retain its generator, validator, compact reports, and SHA rather than the binary. |
| `stage4e-target-cache.bin` | HUMAN_REVIEW | It is not the accepted A4 cache and no inspected report establishes its identity or continuing use. Do not delete it until an owner confirms it is superseded and not needed for earlier history. |

## 11. Accepted Reports/Trajectories Recommended for Preservation

Preserve the A4 dataset, B1R/B2R/B3 baseline reports, C2a/C2b/C2c cache reports, D1 through D7 reports, D4/D5/D7 trajectories, and the F1 report/trajectory listed in Section 2.

The minimum evidence set for reproducing the final F1 interpretation is:

1. `stage4e-approved.records`
2. A4 B1R/B2R/B3 and C2a/C2b/C2c compact reports
3. D1/D2/D3/D4/D5/D6/D7 compact reports and their D4/D5/D7 trajectories
4. F1 report and trajectory
5. source/build/runners that generated them

## 12. Historical Diagnostics Recommended for Removal or Archival

Archive or review before deletion rather than removing immediately:

- hidden-state mismatch diagnostics (`stage4e-c2c-hidden-*`, `stage4e-c2c-b1-b2-forward-audit.txt`)
- repeated baseline refresh/repeat captures (`stage4e-b4*`, older B1/B2 reports and consoles)
- pre-A4 candidate/repair diagnostics and temporary comparison outputs
- console captures whose compact acceptance report/TSV has been retained

The likely archival boundary is “pre-A4 exploratory diagnostics” versus “A4 accepted chain.” Do not classify an item as disposable merely because it is old; retain it if it explains a design correction not captured in a final design/report document.

## 13. Repository-Root Configuration Review

| Path | Purpose if evident | Machine-specific / sensitive status | Proposed classification |
| --- | --- | --- | --- |
| `.github/copilot-instructions.md` | Repository workflow and MINIX execution constraints | Project-specific, not credential-bearing in inspected content | TRACK |
| `.clinerules` | Cline tool-use instructions | Editor-agent policy; not clearly project build source | HUMAN_REVIEW |
| `.continuerc.json` | Continue extension setting (`disableIndexing`) | Personal/editor configuration | HUMAN_REVIEW |
| `.vscode/settings.json` | VS Code AI feature setting | Personal/editor configuration | HUMAN_REVIEW |
| `chat.json` | Unknown; unavailable for inspection because larger than synchronization limit | Potential personal state and potentially sensitive | HUMAN_REVIEW_SENSITIVE |
| `git-checkpoint-status-u.sh` | Writes a Git status snapshot | Uses `git branch --show-current`, unsupported by stated Git 1.9.0; hard-coded `/usr/src` path | SAFE_TO_DELETE or HUMAN_REVIEW if it is retained as a repair target |
| `git-checkpoint-status.sh` | Same visible purpose/content as `git-checkpoint-status-u.sh` | Same compatibility issue and hard-coded path | SAFE_TO_DELETE or HUMAN_REVIEW if it is retained as a repair target |
| `git-checkpoint-status.txt` | Captured output from the above | Contains stale failed `--show-current` invocation and machine/repository status | SAFE_TO_DELETE |

## 14. Suggested Future Local Commit Boundary

Suggested local milestone commit boundary, subject to human review:

- Stage 4 source, headers, Makefile, selected accepted runners
- A4 dataset source records
- accepted compact A4/B/C/D/F reports and trajectories
- repository workflow instructions intentionally maintained with this project
- this audit proposal after it is reviewed and accepted

Keep out of that commit:

- compiled executables, object files, generated cache/checkpoint/adapter binaries, transient console logs
- personal/editor configuration and `chat.json`
- unrelated MINIX files in `etc/` and driver image/binary paths
- ambiguous historical diagnostics until separately classified

## 15. Proposed Tag Name

`stage4e-training-validation-2026-10-01`

This document is a proposal only. It does not stage, ignore, delete, rename, commit, tag, regenerate, or otherwise alter any project artifact.
