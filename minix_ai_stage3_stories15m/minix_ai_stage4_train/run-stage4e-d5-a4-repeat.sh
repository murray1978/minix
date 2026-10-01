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
D5_SOURCE="stage4e_d5_a4_repeat.c"
D5_EXECUTABLE="stage4e_d5_a4_repeat"
D5_TARGET="stage4eD5A4Repeat"
EXPECTED_PROVENANCE="ten-step-repeatability-2026-10-01-a"
REPORT="stage4e-d5-a4-repeat-report.txt"
TRAJECTORY="stage4e-d5-a4-repeat-trajectory.tsv"
CONSOLE_LOG="stage4e-d5-a4-repeat-console.txt"
BUILD_LOG=".stage4e-d5-a4-build.tmp"
RUN_LOG=".stage4e-d5-a4-run.tmp"

EXPECTED_BASE="cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a"
EXPECTED_TOKENIZER="50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361"
EXPECTED_DATASET="71744cea4de049e2e192cb2a4494c8a4a68e769b3a7f3b8ed4880433e4583b84"
EXPECTED_CACHE="d6288ad3d128c7182c20088aa57a1bca8bcaa1e13314fb0cdbd60eba14278def"
EXPECTED_D1="45ebaa750f17d02d80feed332b16c9c29c3782a544aed7544da72a62161c76a6"
EXPECTED_D2="c27aad7474df093f5a3ff781bb3619b71b3e284edda46290f5814c350e99d97e"
EXPECTED_D3="f418305f3fb9263480b8a80f34b02a966ebbb94f305f2aa5897d5c2f4356b94e"

log() {
    printf '%s\n' "$*"
    printf '%s\n' "$*" >> "$CONSOLE_LOG"
}

cleanup() {
    rm -f "$BUILD_LOG" "$RUN_LOG"
}

fail() {
    log "$*"
    log "Stage4E-D5-A4=FAIL"
    cleanup
    exit 1
}

check_file() {
    if [ ! -f "$1" ]; then
        fail "FAIL: missing file: $1"
    fi
}

sha256_hex() {
    openssl dgst -sha256 "$1" 2>/dev/null | awk '{print $NF}'
}

report_value() {
    awk -F= -v wanted="$2" '$1 == wanted { print substr($0, index($0, "=") + 1); exit }' "$1"
}

positive() {
    awk -v value="$1" 'BEGIN { exit !((value + 0) > 0) }'
}

train_below() {
    awk -v value="$1" -v base="$2" 'BEGIN { exit !((value + 0) < (base + 0)) }'
}

: > "$CONSOLE_LOG" || exit 1
log "=============================================="
log "Stage 4E-D5-A4 independent trajectory repeat"
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
check_file "$D5_SOURCE"
check_file "Makefile"

BASE_ACTUAL=$(sha256_hex "$BASE_CHECKPOINT")
TOKENIZER_ACTUAL=$(sha256_hex "$TOKENIZER")
DATASET_ACTUAL=$(sha256_hex "$DATASET")
CACHE_ACTUAL=$(sha256_hex "$CACHE")
D1_ACTUAL=$(sha256_hex "$D1_REPORT")
D2_ACTUAL=$(sha256_hex "$D2_REPORT")
D3_ACTUAL=$(sha256_hex "$D3_REPORT")
D4_REPORT_ACTUAL=$(sha256_hex "$D4_REPORT")
D4_TSV_ACTUAL=$(sha256_hex "$D4_TSV")
if [ "$BASE_ACTUAL" != "$EXPECTED_BASE" ] ||
    [ "$TOKENIZER_ACTUAL" != "$EXPECTED_TOKENIZER" ] ||
    [ "$DATASET_ACTUAL" != "$EXPECTED_DATASET" ] ||
    [ "$CACHE_ACTUAL" != "$EXPECTED_CACHE" ] ||
    [ "$D1_ACTUAL" != "$EXPECTED_D1" ] ||
    [ "$D2_ACTUAL" != "$EXPECTED_D2" ] ||
    [ "$D3_ACTUAL" != "$EXPECTED_D3" ]; then
    log "base_checkpoint_sha256=$BASE_ACTUAL"
    log "tokenizer_sha256=$TOKENIZER_ACTUAL"
    log "dataset_sha256=$DATASET_ACTUAL"
    log "cache_sha256=$CACHE_ACTUAL"
    log "d1_report_sha256=$D1_ACTUAL"
    log "d2_report_sha256=$D2_ACTUAL"
    log "d3_report_sha256=$D3_ACTUAL"
    fail "FAIL: one or more pinned D5 identities differ"
fi
if [ "$(report_value "$D4_REPORT" d4_trajectory_pass)" != yes ] ||
    [ "$(report_value "$D4_REPORT" optimizer_steps_after)" != 10 ] ||
    [ "$(report_value "$D4_REPORT" test_gradient_rows_seen)" != 0 ] ||
    [ "$(report_value "$D4_REPORT" test_evaluation_rows_seen)" != 0 ]; then
    fail "FAIL: D4 reference report is not an accepted sealed trajectory"
fi
log "base_checkpoint_identity=PASS"
log "tokenizer_identity=PASS"
log "dataset_identity=PASS"
log "cache_identity=PASS"
log "d1_report_identity=PASS"
log "d2_report_identity=PASS"
log "d3_report_identity=PASS"
log "d4_report_identity=PASS"
log "d4_trajectory_identity=PASS"

D5_SOURCE_SHA=$(sha256_hex "$D5_SOURCE")
if [ -z "$D5_SOURCE_SHA" ]; then fail "FAIL: cannot hash D5 source"; fi
rm -f "$D5_EXECUTABLE"
make "$D5_TARGET" CC=clang > "$BUILD_LOG" 2>&1
BUILD_STATUS=$?
cat "$BUILD_LOG" | tee -a "$CONSOLE_LOG"
if [ "$BUILD_STATUS" -ne 0 ]; then fail "FAIL: D5 build failed"; fi
check_file "$D5_EXECUTABLE"
if [ "$(sha256_hex "$D5_SOURCE")" != "$D5_SOURCE_SHA" ]; then fail "FAIL: D5 source changed during build"; fi
D5_EXECUTABLE_SHA=$(sha256_hex "$D5_EXECUTABLE")
if [ -z "$D5_EXECUTABLE_SHA" ]; then fail "FAIL: cannot hash D5 executable"; fi
log "d5_source_sha256=$D5_SOURCE_SHA"
log "d5_executable_sha256=$D5_EXECUTABLE_SHA"

"./$D5_EXECUTABLE" "$BASE_CHECKPOINT" -z "$TOKENIZER" \
    -d "$DATASET" -c "$CACHE" -1 "$D1_REPORT" -2 "$D2_REPORT" \
    -3 "$D3_REPORT" -4 "$D4_REPORT" -5 "$D4_TSV" \
    > "$RUN_LOG" 2>&1
RUN_STATUS=$?
cat "$RUN_LOG" | tee -a "$CONSOLE_LOG"
if [ "$(report_value "$RUN_LOG" d5_build_provenance)" != "$EXPECTED_PROVENANCE" ]; then
    fail "FAIL: D5 executable provenance mismatch"
fi
if [ "$RUN_STATUS" -ne 0 ]; then fail "FAIL: D5 repeat trajectory execution failed"; fi
check_file "$REPORT"
check_file "$TRAJECTORY"

D4_REPORT_AFTER=$(sha256_hex "$D4_REPORT")
D4_TSV_AFTER=$(sha256_hex "$D4_TSV")
if [ "$D4_REPORT_AFTER" != "$D4_REPORT_ACTUAL" ] || [ "$D4_TSV_AFTER" != "$D4_TSV_ACTUAL" ]; then
    fail "FAIL: D4 reference report/trajectory changed during D5"
fi

REPORT_D4=$(report_value "$REPORT" d4_report_sha256)
REPORT_D4_TSV=$(report_value "$REPORT" d4_trajectory_sha256)
if [ "$REPORT_D4" != "$D4_REPORT_ACTUAL" ] || [ "$REPORT_D4_TSV" != "$D4_TSV_ACTUAL" ]; then
    fail "FAIL: D5 report does not identify the pinned D4 reference artifacts"
fi

STEP0=$(report_value "$REPORT" step0_d1_metrics_match)
STEP1=$(report_value "$REPORT" step1_d2_metrics_match)
STEP2=$(report_value "$REPORT" step2_d3_metrics_match)
ROWS=$(report_value "$REPORT" compared_trajectory_rows)
TOTAL=$(report_value "$REPORT" total_trajectory_field_mismatches)
MATCH=$(report_value "$REPORT" deterministic_trajectory_match)
OPT_BEFORE=$(report_value "$REPORT" optimizer_steps_before)
OPT_AFTER=$(report_value "$REPORT" optimizer_steps_after)
TEST_GRAD=$(report_value "$REPORT" test_gradient_rows_seen)
TEST_EVAL=$(report_value "$REPORT" test_evaluation_rows_seen)
TRAIN_FINAL=$(report_value "$REPORT" final_train_average_loss)
VAL_FINAL=$(report_value "$REPORT" final_validation_average_loss)
ALL_GRAD=$(report_value "$REPORT" all_gradients_finite)
ALL_PARAMS=$(report_value "$REPORT" all_parameters_finite)
ALL_MOMENTS=$(report_value "$REPORT" all_optimizer_moments_finite)
ALL_LOSSES=$(report_value "$REPORT" all_losses_finite)
PASS=$(report_value "$REPORT" d5_repeatability_pass)

for KEY in step_index_mismatches gradient_row_count_mismatches evaluation_row_count_mismatches \
    grad_A_nonzero_count_mismatches \
    grad_A_max_mismatches grad_B_nonzero_count_mismatches grad_B_max_mismatches \
    norm_before_clip_mismatches norm_after_clip_mismatches clip_applied_mismatches \
    clip_consistency_mismatches train_total_loss_mismatches train_average_loss_mismatches \
    train_top1_mismatches validation_total_loss_mismatches validation_average_loss_mismatches \
    validation_top1_mismatches validation_acceptance_mismatches; do
    VALUE=$(report_value "$REPORT" "$KEY")
    if [ "$VALUE" != 0 ]; then fail "FAIL: trajectory mismatch counter $KEY=$VALUE"; fi
done

if [ "$(report_value "$REPORT" fresh_initialization_used)" != yes ] ||
    [ "$STEP0" != yes ] || [ "$STEP1" != yes ] || [ "$STEP2" != yes ] ||
    [ "$ROWS" != 11 ] || [ "$TOTAL" != 0 ] || [ "$MATCH" != yes ] ||
    [ "$OPT_BEFORE" != 0 ] || [ "$OPT_AFTER" != 10 ] ||
    [ "$TEST_GRAD" != 0 ] || [ "$TEST_EVAL" != 0 ] ||
    [ "$ALL_GRAD" != yes ] || [ "$ALL_PARAMS" != yes ] ||
    [ "$ALL_MOMENTS" != yes ] || [ "$ALL_LOSSES" != yes ] ||
    [ "$PASS" != yes ] || ! train_below "$TRAIN_FINAL" 10.439190954456 ||
    [ "$TRAIN_FINAL" != 9.577087323852 ] || [ "$VAL_FINAL" != 8.829185963434 ]; then
    fail "FAIL: D5 report failed repeatability acceptance"
fi

log "step0_d1_metrics_match=$STEP0"
log "step1_d2_metrics_match=$STEP1"
log "step2_d3_metrics_match=$STEP2"
log "optimizer_steps_before=$OPT_BEFORE"
log "optimizer_steps_after=$OPT_AFTER"
log "test_gradient_rows_seen=$TEST_GRAD"
log "test_evaluation_rows_seen=$TEST_EVAL"
log "compared_trajectory_rows=$ROWS"
log "total_trajectory_field_mismatches=$TOTAL"
log "deterministic_trajectory_match=$MATCH"
log "final_train_average_loss=$TRAIN_FINAL"
log "final_validation_average_loss=$VAL_FINAL"
log "all_gradients_finite=$ALL_GRAD"
log "all_parameters_finite=$ALL_PARAMS"
log "all_optimizer_moments_finite=$ALL_MOMENTS"
log "all_losses_finite=$ALL_LOSSES"
log "Stage4E-D5-A4=PASS"
cleanup
exit 0
