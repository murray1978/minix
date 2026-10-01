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
D6_SOURCE="stage4e_d6_a4_checkpoint.c"
D6_EXECUTABLE="stage4e_d6_a4_checkpoint"
D6_TARGET="stage4eD6A4Checkpoint"
EXPECTED_PROVENANCE="step5-checkpoint-roundtrip-2026-10-01-a"
CHECKPOINT="stage4e-d6-a4-step5-checkpoint.bin"
REPORT="stage4e-d6-a4-checkpoint-report.txt"
CONSOLE_LOG="stage4e-d6-a4-checkpoint-console.txt"
BUILD_LOG=".stage4e-d6-a4-build.tmp"
RUN_LOG=".stage4e-d6-a4-run.tmp"

EXPECTED_BASE="cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a"
EXPECTED_TOKENIZER="50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361"
EXPECTED_DATASET="71744cea4de049e2e192cb2a4494c8a4a68e769b3a7f3b8ed4880433e4583b84"
EXPECTED_CACHE="d6288ad3d128c7182c20088aa57a1bca8bcaa1e13314fb0cdbd60eba14278def"
EXPECTED_D1="45ebaa750f17d02d80feed332b16c9c29c3782a544aed7544da72a62161c76a6"
EXPECTED_D2="c27aad7474df093f5a3ff781bb3619b71b3e284edda46290f5814c350e99d97e"
EXPECTED_D3="f418305f3fb9263480b8a80f34b02a966ebbb94f305f2aa5897d5c2f4356b94e"
EXPECTED_D4="d57d81ed4608df0c4a662404911c16a72fa475f0aea1459e68533e522cebace1"

log() {
    printf '%s\n' "$*"
    printf '%s\n' "$*" >> "$CONSOLE_LOG"
}

cleanup() {
    rm -f "$BUILD_LOG" "$RUN_LOG"
}

fail() {
    log "$*"
    log "Stage4E-D6-A4=FAIL"
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
log "Stage 4E-D6-A4 step-5 checkpoint round-trip"
log "=============================================="

check_file "$BASE_CHECKPOINT"
check_file "$TOKENIZER"
check_file "$DATASET"
check_file "$CACHE"
check_file "$D1_REPORT"
check_file "$D2_REPORT"
check_file "$D3_REPORT"
check_file "$D4_REPORT"
check_file "$D6_SOURCE"
check_file "Makefile"

BASE_ACTUAL=$(sha256_hex "$BASE_CHECKPOINT")
TOKENIZER_ACTUAL=$(sha256_hex "$TOKENIZER")
DATASET_ACTUAL=$(sha256_hex "$DATASET")
CACHE_ACTUAL=$(sha256_hex "$CACHE")
D1_ACTUAL=$(sha256_hex "$D1_REPORT")
D2_ACTUAL=$(sha256_hex "$D2_REPORT")
D3_ACTUAL=$(sha256_hex "$D3_REPORT")
D4_ACTUAL=$(sha256_hex "$D4_REPORT")
if [ "$BASE_ACTUAL" != "$EXPECTED_BASE" ] ||
    [ "$TOKENIZER_ACTUAL" != "$EXPECTED_TOKENIZER" ] ||
    [ "$DATASET_ACTUAL" != "$EXPECTED_DATASET" ] ||
    [ "$CACHE_ACTUAL" != "$EXPECTED_CACHE" ] ||
    [ "$D1_ACTUAL" != "$EXPECTED_D1" ] || [ "$D2_ACTUAL" != "$EXPECTED_D2" ] ||
    [ "$D3_ACTUAL" != "$EXPECTED_D3" ] || [ "$D4_ACTUAL" != "$EXPECTED_D4" ]; then
    log "base_checkpoint_sha256=$BASE_ACTUAL"
    log "tokenizer_sha256=$TOKENIZER_ACTUAL"
    log "dataset_sha256=$DATASET_ACTUAL"
    log "cache_sha256=$CACHE_ACTUAL"
    log "d1_report_sha256=$D1_ACTUAL"
    log "d2_report_sha256=$D2_ACTUAL"
    log "d3_report_sha256=$D3_ACTUAL"
    log "d4_report_sha256=$D4_ACTUAL"
    fail "FAIL: one or more pinned D6 input identities differ"
fi
log "base_checkpoint_identity=PASS"
log "tokenizer_identity=PASS"
log "dataset_identity=PASS"
log "cache_identity=PASS"
log "d1_report_identity=PASS"
log "d2_report_identity=PASS"
log "d3_report_identity=PASS"
log "d4_report_identity=PASS"

D6_SOURCE_SHA=$(sha256_hex "$D6_SOURCE")
if [ -z "$D6_SOURCE_SHA" ]; then fail "FAIL: cannot hash D6 source"; fi
rm -f "$D6_EXECUTABLE" "$CHECKPOINT"
make "$D6_TARGET" CC=clang > "$BUILD_LOG" 2>&1
BUILD_STATUS=$?
cat "$BUILD_LOG" | tee -a "$CONSOLE_LOG"
if [ "$BUILD_STATUS" -ne 0 ]; then fail "FAIL: D6 checkpoint executable build failed"; fi
check_file "$D6_EXECUTABLE"
if [ "$(sha256_hex "$D6_SOURCE")" != "$D6_SOURCE_SHA" ]; then fail "FAIL: D6 source changed during build"; fi
D6_EXECUTABLE_SHA=$(sha256_hex "$D6_EXECUTABLE")
if [ -z "$D6_EXECUTABLE_SHA" ]; then fail "FAIL: cannot hash D6 executable"; fi
log "d6_source_sha256=$D6_SOURCE_SHA"
log "d6_executable_sha256=$D6_EXECUTABLE_SHA"

"./$D6_EXECUTABLE" "$BASE_CHECKPOINT" -z "$TOKENIZER" \
    -d "$DATASET" -c "$CACHE" -1 "$D1_REPORT" -2 "$D2_REPORT" \
    -3 "$D3_REPORT" -4 "$D4_REPORT" > "$RUN_LOG" 2>&1
RUN_STATUS=$?
cat "$RUN_LOG" | tee -a "$CONSOLE_LOG"
if [ "$(report_value "$RUN_LOG" d6_build_provenance)" != "$EXPECTED_PROVENANCE" ]; then
    fail "FAIL: D6 executable provenance mismatch"
fi
if [ "$RUN_STATUS" -ne 0 ]; then fail "FAIL: D6 checkpoint round-trip execution failed"; fi
check_file "$CHECKPOINT"
check_file "$REPORT"

CHECKPOINT_SHA=$(sha256_hex "$CHECKPOINT")
if [ -z "$CHECKPOINT_SHA" ]; then fail "FAIL: unable to hash D6 checkpoint"; fi
REPORT_CHECKPOINT_SHA=$(report_value "$REPORT" checkpoint_sha256)
EXPECTED_BYTES=$(report_value "$REPORT" checkpoint_expected_total_bytes)
ACTUAL_BYTES=$(report_value "$REPORT" checkpoint_actual_total_bytes)
STEP5_MATCH=$(report_value "$REPORT" step5_d4_metrics_match)
RELOADED_STEP=$(report_value "$REPORT" reloaded_optimizer_step)
TOTAL_MISMATCHES=$(report_value "$REPORT" total_state_byte_mismatches)
TRAIN_MATCH=$(report_value "$REPORT" pre_save_vs_post_load_train_metrics_match)
VALIDATION_MATCH=$(report_value "$REPORT" pre_save_vs_post_load_validation_metrics_match)
TEST_GRAD=$(report_value "$REPORT" test_gradient_rows_seen)
TEST_EVAL=$(report_value "$REPORT" test_evaluation_rows_seen)
PARAMS_FINITE=$(report_value "$REPORT" loaded_parameters_finite)
MOMENTS_FINITE=$(report_value "$REPORT" loaded_optimizer_moments_finite)
D6_PASS=$(report_value "$REPORT" checkpoint_roundtrip_pass)
if [ "$CHECKPOINT_SHA" != "$REPORT_CHECKPOINT_SHA" ] ||
    [ "$EXPECTED_BYTES" != 3099856 ] || [ "$ACTUAL_BYTES" != "$EXPECTED_BYTES" ] ||
    [ "$STEP5_MATCH" != yes ] || [ "$RELOADED_STEP" != 5 ] ||
    [ "$TOTAL_MISMATCHES" != 0 ] || [ "$TRAIN_MATCH" != yes ] ||
    [ "$VALIDATION_MATCH" != yes ] || [ "$TEST_GRAD" != 0 ] ||
    [ "$TEST_EVAL" != 0 ] || [ "$PARAMS_FINITE" != yes ] ||
    [ "$MOMENTS_FINITE" != yes ] || [ "$D6_PASS" != yes ]; then
    fail "FAIL: D6 checkpoint round-trip acceptance failed"
fi

log "optimizer_step_before_training=0"
log "optimizer_step_at_save=5"
log "reloaded_optimizer_step=$RELOADED_STEP"
log "step5_d4_metrics_match=$STEP5_MATCH"
log "checkpoint_expected_total_bytes=$EXPECTED_BYTES"
log "checkpoint_actual_total_bytes=$ACTUAL_BYTES"
log "checkpoint_sha256=$CHECKPOINT_SHA"
log "total_state_byte_mismatches=$TOTAL_MISMATCHES"
log "pre_save_vs_post_load_train_metrics_match=$TRAIN_MATCH"
log "pre_save_vs_post_load_validation_metrics_match=$VALIDATION_MATCH"
log "test_gradient_rows_seen=$TEST_GRAD"
log "test_evaluation_rows_seen=$TEST_EVAL"
log "loaded_parameters_finite=$PARAMS_FINITE"
log "loaded_optimizer_moments_finite=$MOMENTS_FINITE"
log "Stage4E-D6-A4=PASS"
cleanup
exit 0
