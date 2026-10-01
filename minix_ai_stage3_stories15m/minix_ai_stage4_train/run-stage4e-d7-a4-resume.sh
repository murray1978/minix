#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" 2>/dev/null && pwd)
cd "$SCRIPT_DIR" || exit 1

BASE_CHECKPOINT="/usr/src/minix_ai_stage3_stories15m/stories15M.bin"
TOKENIZER="/usr/src/minix_ai_stage3_stories15m/tokenizer.bin"
DATASET="stage4e-approved.records"
CACHE="stage4e-target-cache-a4.bin"
D1_REPORT="stage4e-d1-a4-zero-update-trainer-report.txt"
D2_REPORT="stage4e-d2-a4-one-step-report.txt"
D3_REPORT="stage4e-d3-a4-two-step-report.txt"
D4_REPORT="stage4e-d4-a4-trajectory-report.txt"
D4_TSV="stage4e-d4-a4-trajectory.tsv"
D6_REPORT="stage4e-d6-a4-checkpoint-report.txt"
D6_CHECKPOINT="stage4e-d6-a4-step5-checkpoint.bin"
D7_SOURCE="stage4e_d7_a4_resume.c"
D7_EXECUTABLE="stage4e_d7_a4_resume"
D7_TARGET="stage4eD7A4Resume"
EXPECTED_PROVENANCE="checkpoint-resume-equivalence-2026-10-01-a"
REPORT="stage4e-d7-a4-resume-report.txt"
TRAJECTORY="stage4e-d7-a4-resume-trajectory.tsv"
CONSOLE_LOG="stage4e-d7-a4-resume-console.txt"
BUILD_LOG=".stage4e-d7-a4-build.tmp"
RUN_LOG=".stage4e-d7-a4-run.tmp"

EXPECTED_BASE="cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a"
EXPECTED_TOKENIZER="50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361"
EXPECTED_DATASET="71744cea4de049e2e192cb2a4494c8a4a68e769b3a7f3b8ed4880433e4583b84"
EXPECTED_CACHE="d6288ad3d128c7182c20088aa57a1bca8bcaa1e13314fb0cdbd60eba14278def"
EXPECTED_D1="45ebaa750f17d02d80feed332b16c9c29c3782a544aed7544da72a62161c76a6"
EXPECTED_D2="c27aad7474df093f5a3ff781bb3619b71b3e284edda46290f5814c350e99d97e"
EXPECTED_D3="f418305f3fb9263480b8a80f34b02a966ebbb94f305f2aa5897d5c2f4356b94e"
EXPECTED_D4="d57d81ed4608df0c4a662404911c16a72fa475f0aea1459e68533e522cebace1"
EXPECTED_D6_CHECKPOINT="ecb75d52881df04a0c47fcd4b852909a865c62cb54929ab33b230c7cf1e5ec89"

log() {
    printf '%s\n' "$*"
    printf '%s\n' "$*" >> "$CONSOLE_LOG"
}

cleanup() {
    rm -f "$BUILD_LOG" "$RUN_LOG"
}

fail() {
    log "$*"
    log "Stage4E-D7-A4=FAIL"
    cleanup
    exit 1
}

check_file() {
    if [ ! -f "$1" ]; then fail "FAIL: missing file: $1"; fi
}

sha256_hex() {
    openssl dgst -sha256 "$1" 2>/dev/null | awk '{print $NF}'
}

report_value() {
    awk -F= -v wanted="$2" '$1 == wanted { print substr($0, index($0, "=") + 1); exit }' "$1"
}

: > "$CONSOLE_LOG" || exit 1
log "=============================================="
log "Stage 4E-D7-A4 checkpoint resume equivalence"
log "=============================================="

check_file "$BASE_CHECKPOINT"
check_file "$TOKENIZER"
check_file "$DATASET"
check_file "$CACHE"
check_file "$D1_REPORT"
check_file "$D2_REPORT"
check_file "$D3_REPORT"
check_file "$D4_REPORT"
check_file "$D4_TSV"
check_file "$D6_REPORT"
check_file "$D6_CHECKPOINT"
check_file "$D7_SOURCE"
check_file "Makefile"

BASE_ACTUAL=$(sha256_hex "$BASE_CHECKPOINT")
TOKENIZER_ACTUAL=$(sha256_hex "$TOKENIZER")
DATASET_ACTUAL=$(sha256_hex "$DATASET")
CACHE_ACTUAL=$(sha256_hex "$CACHE")
D1_ACTUAL=$(sha256_hex "$D1_REPORT")
D2_ACTUAL=$(sha256_hex "$D2_REPORT")
D3_ACTUAL=$(sha256_hex "$D3_REPORT")
D4_ACTUAL=$(sha256_hex "$D4_REPORT")
D4_TSV_ACTUAL=$(sha256_hex "$D4_TSV")
D6_CHECKPOINT_ACTUAL=$(sha256_hex "$D6_CHECKPOINT")
D6_REPORT_ACTUAL=$(sha256_hex "$D6_REPORT")
if [ "$BASE_ACTUAL" != "$EXPECTED_BASE" ] ||
    [ "$TOKENIZER_ACTUAL" != "$EXPECTED_TOKENIZER" ] ||
    [ "$DATASET_ACTUAL" != "$EXPECTED_DATASET" ] ||
    [ "$CACHE_ACTUAL" != "$EXPECTED_CACHE" ] ||
    [ "$D1_ACTUAL" != "$EXPECTED_D1" ] ||
    [ "$D2_ACTUAL" != "$EXPECTED_D2" ] ||
    [ "$D3_ACTUAL" != "$EXPECTED_D3" ] ||
    [ "$D4_ACTUAL" != "$EXPECTED_D4" ] ||
    [ "$D6_CHECKPOINT_ACTUAL" != "$EXPECTED_D6_CHECKPOINT" ]; then
    log "base_checkpoint_sha256=$BASE_ACTUAL"
    log "tokenizer_sha256=$TOKENIZER_ACTUAL"
    log "dataset_sha256=$DATASET_ACTUAL"
    log "cache_sha256=$CACHE_ACTUAL"
    log "d1_report_sha256=$D1_ACTUAL"
    log "d2_report_sha256=$D2_ACTUAL"
    log "d3_report_sha256=$D3_ACTUAL"
    log "d4_report_sha256=$D4_ACTUAL"
    log "d6_report_sha256=$D6_REPORT_ACTUAL"
    log "d6_checkpoint_sha256=$D6_CHECKPOINT_ACTUAL"
    fail "FAIL: a pinned D7 input identity differs"
fi
if [ "$(report_value "$D6_REPORT" checkpoint_roundtrip_pass)" != yes ] ||
    [ "$(report_value "$D6_REPORT" checkpoint_sha256)" != "$EXPECTED_D6_CHECKPOINT" ] ||
    [ "$(report_value "$D4_REPORT" d4_trajectory_pass)" != yes ]; then
    fail "FAIL: D4/D6 reference artifact acceptance fields mismatch"
fi
log "base_checkpoint_identity=PASS"
log "tokenizer_identity=PASS"
log "dataset_identity=PASS"
log "cache_identity=PASS"
log "d1_report_identity=PASS"
log "d2_report_identity=PASS"
log "d3_report_identity=PASS"
log "d4_report_identity=PASS"
log "d6_report_identity=PASS"
log "d6_checkpoint_identity=PASS"

D7_SOURCE_SHA=$(sha256_hex "$D7_SOURCE")
if [ -z "$D7_SOURCE_SHA" ]; then fail "FAIL: unable to hash D7 source"; fi
rm -f "$D7_EXECUTABLE"
make "$D7_TARGET" CC=clang > "$BUILD_LOG" 2>&1
BUILD_STATUS=$?
cat "$BUILD_LOG" | tee -a "$CONSOLE_LOG"
if [ "$BUILD_STATUS" -ne 0 ]; then fail "FAIL: D7 executable build failed"; fi
check_file "$D7_EXECUTABLE"
if [ "$(sha256_hex "$D7_SOURCE")" != "$D7_SOURCE_SHA" ]; then fail "FAIL: D7 source changed during build"; fi
D7_EXECUTABLE_SHA=$(sha256_hex "$D7_EXECUTABLE")
if [ -z "$D7_EXECUTABLE_SHA" ]; then fail "FAIL: unable to hash D7 executable"; fi
log "d7_source_sha256=$D7_SOURCE_SHA"
log "d7_executable_sha256=$D7_EXECUTABLE_SHA"

"./$D7_EXECUTABLE" "$BASE_CHECKPOINT" -z "$TOKENIZER" \
    -d "$DATASET" -c "$CACHE" -1 "$D1_REPORT" -2 "$D2_REPORT" \
    -3 "$D3_REPORT" -4 "$D4_REPORT" -6 "$D6_REPORT" \
    -7 "$D6_CHECKPOINT" > "$RUN_LOG" 2>&1
RUN_STATUS=$?
cat "$RUN_LOG" | tee -a "$CONSOLE_LOG"
if [ "$(report_value "$RUN_LOG" d7_build_provenance)" != "$EXPECTED_PROVENANCE" ]; then
    fail "FAIL: D7 executable provenance mismatch"
fi
if [ "$RUN_STATUS" -ne 0 ]; then fail "FAIL: D7 resume equivalence run failed"; fi
check_file "$REPORT"
check_file "$TRAJECTORY"

if [ "$(sha256_hex "$D6_REPORT")" != "$D6_REPORT_ACTUAL" ] ||
    [ "$(sha256_hex "$D4_TSV")" != "$D4_TSV_ACTUAL" ]; then
    fail "FAIL: D4/D6 reference artifacts changed during D7"
fi
REPORT_D6=$(report_value "$REPORT" d6_report_sha256)
REPORT_D6_CHECKPOINT=$(report_value "$REPORT" d6_checkpoint_sha256)
if [ "$REPORT_D6" != "$D6_REPORT_ACTUAL" ] ||
    [ "$REPORT_D6_CHECKPOINT" != "$EXPECTED_D6_CHECKPOINT" ]; then
    fail "FAIL: D7 report reference identities mismatch"
fi

for KEY in step_index_mismatches gradient_row_count_mismatches evaluation_row_count_mismatches \
    grad_A_nonzero_count_mismatches grad_A_max_mismatches \
    grad_B_nonzero_count_mismatches grad_B_max_mismatches \
    norm_before_clip_mismatches norm_after_clip_mismatches \
    clip_applied_mismatches clip_consistency_mismatches \
    train_total_loss_mismatches train_average_loss_mismatches train_top1_mismatches \
    validation_total_loss_mismatches validation_average_loss_mismatches \
    validation_top1_mismatches validation_acceptance_mismatches; do
    VALUE=$(report_value "$REPORT" "$KEY")
    if [ "$VALUE" != 0 ]; then fail "FAIL: resume mismatch counter $KEY=$VALUE"; fi
done

PASS=$(report_value "$REPORT" d7_resume_pass)
D4_MATCH=$(report_value "$REPORT" continuous_d4_trajectory_match)
STEP5=$(report_value "$REPORT" step5_continuous_vs_checkpoint_total_state_byte_mismatches)
ROWS=$(report_value "$REPORT" compared_resume_steps)
TOTAL=$(report_value "$REPORT" total_resume_trajectory_field_mismatches)
FINAL_STATE=$(report_value "$REPORT" final_state_bitwise_match)
if [ "$PASS" != yes ] || [ "$D4_MATCH" != yes ] || [ "$STEP5" != 0 ] ||
    [ "$ROWS" != 5 ] || [ "$TOTAL" != 0 ] || [ "$FINAL_STATE" != yes ] ||
    [ "$(report_value "$REPORT" final_total_state_byte_mismatches)" != 0 ] ||
    [ "$(report_value "$REPORT" continuous_final_train_average_loss)" != 9.577087323852 ] ||
    [ "$(report_value "$REPORT" resumed_final_train_average_loss)" != 9.577087323852 ] ||
    [ "$(report_value "$REPORT" continuous_final_validation_average_loss)" != 8.829185963434 ] ||
    [ "$(report_value "$REPORT" resumed_final_validation_average_loss)" != 8.829185963434 ]; then
    fail "FAIL: D7 report failed deterministic resume acceptance"
fi
log "continuous_d4_trajectory_match=$D4_MATCH"
log "step5_continuous_vs_checkpoint_total_state_byte_mismatches=$STEP5"
log "continuous_optimizer_step=10"
log "resumed_optimizer_step=10"
log "compared_resume_steps=$ROWS"
log "total_resume_trajectory_field_mismatches=$TOTAL"
log "deterministic_resume_trajectory_match=yes"
log "final_A_byte_mismatches=$(report_value "$REPORT" final_A_byte_mismatches)"
log "final_B_byte_mismatches=$(report_value "$REPORT" final_B_byte_mismatches)"
log "final_m1_A_byte_mismatches=$(report_value "$REPORT" final_m1_A_byte_mismatches)"
log "final_m2_A_byte_mismatches=$(report_value "$REPORT" final_m2_A_byte_mismatches)"
log "final_m1_B_byte_mismatches=$(report_value "$REPORT" final_m1_B_byte_mismatches)"
log "final_m2_B_byte_mismatches=$(report_value "$REPORT" final_m2_B_byte_mismatches)"
log "final_total_state_byte_mismatches=$(report_value "$REPORT" final_total_state_byte_mismatches)"
log "final_state_bitwise_match=$FINAL_STATE"
log "continuous_final_train_average_loss=$(report_value "$REPORT" continuous_final_train_average_loss)"
log "resumed_final_train_average_loss=$(report_value "$REPORT" resumed_final_train_average_loss)"
log "continuous_final_validation_average_loss=$(report_value "$REPORT" continuous_final_validation_average_loss)"
log "resumed_final_validation_average_loss=$(report_value "$REPORT" resumed_final_validation_average_loss)"
log "Stage4E-D7-A4=PASS"
cleanup
exit 0
