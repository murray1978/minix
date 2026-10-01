#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" 2>/dev/null && pwd)
cd "$SCRIPT_DIR" || exit 1

BASE_CHECKPOINT="/usr/src/minix_ai_stage3_stories15m/stories15M.bin"
TOKENIZER="/usr/src/minix_ai_stage3_stories15m/tokenizer.bin"
DATASET="stage4e-approved.records"
CACHE="stage4e-target-cache-a4.bin"
B1R_REPORT="stage4e-baseline-unadapted-full-context-a4-report.txt"
B2R_REPORT="stage4e-baseline-unadapted-full-context-a4-top1-report.txt"
C2C_REPORT="stage4e-c2c-a4-full-context-equivalence-report.txt"
REPORT="stage4e-d1-a4-zero-update-trainer-report.txt"
CONSOLE_LOG="stage4e-d1-a4-zero-update-trainer-console.txt"
BUILD_LOG=".stage4e-d1-a4-build.tmp"
RUN_LOG=".stage4e-d1-a4-run.tmp"

EXPECTED_BASE="cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a"
EXPECTED_TOKENIZER="50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361"
EXPECTED_DATASET="71744cea4de049e2e192cb2a4494c8a4a68e769b3a7f3b8ed4880433e4583b84"
EXPECTED_CACHE="d6288ad3d128c7182c20088aa57a1bca8bcaa1e13314fb0cdbd60eba14278def"
EXPECTED_B1R="b92264d62bfa603d98b7f00580f1e7e0bf6b2c6cefb8e5ec5db7753e17249b2a"
EXPECTED_B2R="1c50c0868358349977d627529ae9f9b04ee155580cbf1e3740d276a5fd968c40"
EXPECTED_C2C="e62a40d4fe6dabc3edad4a065e89c78aaadca5496dacaee6da31255e807c35ad"

log() {
    printf '%s\n' "$*"
    printf '%s\n' "$*" >> "$CONSOLE_LOG"
}

cleanup() {
    rm -f "$BUILD_LOG" "$RUN_LOG"
}

fail() {
    log "$*"
    log "Stage4E-D1-A4=FAIL"
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

: > "$CONSOLE_LOG" || exit 1
log "=============================================="
log "Stage 4E-D1-A4 cached zero-update validation"
log "=============================================="

check_file "$BASE_CHECKPOINT"
check_file "$TOKENIZER"
check_file "$DATASET"
check_file "$CACHE"
check_file "$B1R_REPORT"
check_file "$B2R_REPORT"
check_file "$C2C_REPORT"
check_file "Makefile"

BASE_ACTUAL=$(sha256_hex "$BASE_CHECKPOINT")
TOKENIZER_ACTUAL=$(sha256_hex "$TOKENIZER")
DATASET_ACTUAL=$(sha256_hex "$DATASET")
CACHE_ACTUAL=$(sha256_hex "$CACHE")
B1R_ACTUAL=$(sha256_hex "$B1R_REPORT")
B2R_ACTUAL=$(sha256_hex "$B2R_REPORT")
C2C_ACTUAL=$(sha256_hex "$C2C_REPORT")

if [ "$BASE_ACTUAL" != "$EXPECTED_BASE" ] ||
    [ "$TOKENIZER_ACTUAL" != "$EXPECTED_TOKENIZER" ] ||
    [ "$DATASET_ACTUAL" != "$EXPECTED_DATASET" ] ||
    [ "$CACHE_ACTUAL" != "$EXPECTED_CACHE" ] ||
    [ "$B1R_ACTUAL" != "$EXPECTED_B1R" ] ||
    [ "$B2R_ACTUAL" != "$EXPECTED_B2R" ] ||
    [ "$C2C_ACTUAL" != "$EXPECTED_C2C" ]; then
    log "base_checkpoint_sha256=$BASE_ACTUAL"
    log "tokenizer_sha256=$TOKENIZER_ACTUAL"
    log "dataset_sha256=$DATASET_ACTUAL"
    log "cache_sha256=$CACHE_ACTUAL"
    log "b1r_a4_report_sha256=$B1R_ACTUAL"
    log "b2r_a4_report_sha256=$B2R_ACTUAL"
    log "c2c_a4_report_sha256=$C2C_ACTUAL"
    fail "FAIL: one or more pinned D1 input identities differ"
fi

log "base_checkpoint_identity=PASS"
log "tokenizer_identity=PASS"
log "dataset_identity=PASS"
log "cache_identity=PASS"
log "b1r_a4_report_identity=PASS"
log "b2r_a4_report_identity=PASS"
log "c2c_a4_report_identity=PASS"

make stage4eD1A4ZeroUpdateTrainer CC=clang > "$BUILD_LOG" 2>&1
BUILD_STATUS=$?
cat "$BUILD_LOG" | tee -a "$CONSOLE_LOG"
if [ "$BUILD_STATUS" -ne 0 ]; then
    fail "FAIL: D1 A4 zero-update trainer build failed"
fi

./stage4e_d1_a4_zero_update_trainer "$BASE_CHECKPOINT" \
    -z "$TOKENIZER" -d "$DATASET" -c "$CACHE" \
    -1 "$B1R_REPORT" -2 "$B2R_REPORT" -3 "$C2C_REPORT" \
    -r "$REPORT" > "$RUN_LOG" 2>&1
RUN_STATUS=$?
cat "$RUN_LOG" | tee -a "$CONSOLE_LOG"
if [ "$RUN_STATUS" -ne 0 ]; then
    fail "FAIL: D1 cached zero-update evaluation failed"
fi
check_file "$REPORT"

REPORT_DATASET=$(report_value "$REPORT" dataset_sha256)
REPORT_CACHE=$(report_value "$REPORT" cache_sha256)
REPORT_BASE=$(report_value "$REPORT" base_checkpoint_sha256)
REPORT_TOKENIZER=$(report_value "$REPORT" tokenizer_sha256)
REPORT_B1R=$(report_value "$REPORT" b1r_a4_report_sha256)
REPORT_B2R=$(report_value "$REPORT" b2r_a4_report_sha256)
REPORT_C2C=$(report_value "$REPORT" c2c_a4_report_sha256)
RANK=$(report_value "$REPORT" rank)
SCALE=$(report_value "$REPORT" scale)
PARAMETERS=$(report_value "$REPORT" adapter_parameter_count)
A_CHANGED=$(report_value "$REPORT" adapter_A_changed)
B_CHANGED=$(report_value "$REPORT" adapter_B_changed)
OPTIMIZER_BEFORE=$(report_value "$REPORT" optimizer_steps_before)
OPTIMIZER_AFTER=$(report_value "$REPORT" optimizer_steps_after)
OPTIMIZER_STATE_CHANGED=$(report_value "$REPORT" optimizer_state_changed)
EVALUATED_ROWS=$(report_value "$REPORT" evaluated_rows)
TRAIN_ROWS=$(report_value "$REPORT" train_rows_seen)
VALIDATION_ROWS=$(report_value "$REPORT" validation_rows_seen)
TEST_ROWS=$(report_value "$REPORT" test_rows_seen)
ZERO_CORRECTION_ROWS=$(report_value "$REPORT" zero_correction_rows)
NONZERO_CORRECTION_ROWS=$(report_value "$REPORT" nonzero_correction_rows)
LOGIT_MISMATCHES=$(report_value "$REPORT" final_vs_base_logit_bit_mismatches)
OVERALL_COUNT=$(report_value "$REPORT" overall_target_count)
OVERALL_TOTAL=$(report_value "$REPORT" overall_total_loss)
OVERALL_AVERAGE=$(report_value "$REPORT" overall_average_loss)
TRAIN_COUNT=$(report_value "$REPORT" train_target_count)
TRAIN_TOTAL=$(report_value "$REPORT" train_total_loss)
TRAIN_AVERAGE=$(report_value "$REPORT" train_average_loss)
VALIDATION_COUNT=$(report_value "$REPORT" validation_target_count)
VALIDATION_TOTAL=$(report_value "$REPORT" validation_total_loss)
VALIDATION_AVERAGE=$(report_value "$REPORT" validation_average_loss)
TEST_COUNT=$(report_value "$REPORT" test_target_count)
TEST_TOTAL=$(report_value "$REPORT" test_total_loss)
TEST_AVERAGE=$(report_value "$REPORT" test_average_loss)
OVERALL_TOP1=$(report_value "$REPORT" overall_top1_correct)
TRAIN_TOP1=$(report_value "$REPORT" train_top1_correct)
VALIDATION_TOP1=$(report_value "$REPORT" validation_top1_correct)
TEST_TOP1=$(report_value "$REPORT" test_top1_correct)
VALIDATION_BASELINE=$(report_value "$REPORT" validation_baseline_loss)
VALIDATION_IMPROVEMENT=$(report_value "$REPORT" validation_required_relative_improvement)
VALIDATION_MAX=$(report_value "$REPORT" validation_acceptance_loss_max)
VALIDATION_THRESHOLD_APPLIED=$(report_value "$REPORT" validation_threshold_applied)
B1R_MATCH=$(report_value "$REPORT" b1r_a4_loss_metrics_match)
B2R_MATCH=$(report_value "$REPORT" b2r_a4_top1_metrics_match)
D1_PASS=$(report_value "$REPORT" zero_update_validation_pass)

if [ "$REPORT_DATASET" != "$EXPECTED_DATASET" ] ||
    [ "$REPORT_CACHE" != "$EXPECTED_CACHE" ] ||
    [ "$REPORT_BASE" != "$EXPECTED_BASE" ] ||
    [ "$REPORT_TOKENIZER" != "$EXPECTED_TOKENIZER" ] ||
    [ "$REPORT_B1R" != "$EXPECTED_B1R" ] ||
    [ "$REPORT_B2R" != "$EXPECTED_B2R" ] ||
    [ "$REPORT_C2C" != "$EXPECTED_C2C" ]; then
    fail "FAIL: D1 report identities differ from pinned input identities"
fi

BASE_AFTER=$(sha256_hex "$BASE_CHECKPOINT")
TOKENIZER_AFTER=$(sha256_hex "$TOKENIZER")
DATASET_AFTER=$(sha256_hex "$DATASET")
CACHE_AFTER=$(sha256_hex "$CACHE")
B1R_AFTER=$(sha256_hex "$B1R_REPORT")
B2R_AFTER=$(sha256_hex "$B2R_REPORT")
C2C_AFTER=$(sha256_hex "$C2C_REPORT")
if [ "$BASE_AFTER" != "$EXPECTED_BASE" ] ||
    [ "$TOKENIZER_AFTER" != "$EXPECTED_TOKENIZER" ] ||
    [ "$DATASET_AFTER" != "$EXPECTED_DATASET" ] ||
    [ "$CACHE_AFTER" != "$EXPECTED_CACHE" ] ||
    [ "$B1R_AFTER" != "$EXPECTED_B1R" ] ||
    [ "$B2R_AFTER" != "$EXPECTED_B2R" ] ||
    [ "$C2C_AFTER" != "$EXPECTED_C2C" ]; then
    fail "FAIL: a pinned D1 input identity changed during evaluation"
fi

REPORT_SHA=$(sha256_hex "$REPORT")
if [ -z "$REPORT_SHA" ]; then
    fail "FAIL: unable to obtain D1 report SHA-256"
fi

if [ "$RANK" != 8 ] || [ "$SCALE" != 1.0 ] ||
    [ "$PARAMETERS" != 258304 ] || [ "$A_CHANGED" != no ] ||
    [ "$B_CHANGED" != no ] || [ "$OPTIMIZER_BEFORE" != 0 ] ||
    [ "$OPTIMIZER_AFTER" != 0 ] || [ "$OPTIMIZER_STATE_CHANGED" != no ] ||
    [ "$EVALUATED_ROWS" != 165 ] || [ "$TRAIN_ROWS" != 111 ] ||
    [ "$VALIDATION_ROWS" != 33 ] || [ "$TEST_ROWS" != 21 ] ||
    [ "$ZERO_CORRECTION_ROWS" != 165 ] || [ "$NONZERO_CORRECTION_ROWS" != 0 ] ||
    [ "$LOGIT_MISMATCHES" != 0 ] ||
    [ "$OVERALL_COUNT" != 165 ] || [ "$OVERALL_TOTAL" != 1683.218059401455 ] ||
    [ "$OVERALL_AVERAGE" != 10.201321572130 ] ||
    [ "$TRAIN_COUNT" != 111 ] || [ "$TRAIN_TOTAL" != 1158.750195944566 ] ||
    [ "$TRAIN_AVERAGE" != 10.439190954456 ] ||
    [ "$VALIDATION_COUNT" != 33 ] || [ "$VALIDATION_TOTAL" != 300.506654007981 ] ||
    [ "$VALIDATION_AVERAGE" != 9.106262242666 ] ||
    [ "$TEST_COUNT" != 21 ] || [ "$TEST_TOTAL" != 223.961209448908 ] ||
    [ "$TEST_AVERAGE" != 10.664819497567 ] ||
    [ "$OVERALL_TOP1" != 2 ] || [ "$TRAIN_TOP1" != 0 ] ||
    [ "$VALIDATION_TOP1" != 2 ] || [ "$TEST_TOP1" != 0 ] ||
    [ "$VALIDATION_BASELINE" != 9.106262242666 ] ||
    [ "$VALIDATION_IMPROVEMENT" != 0.05 ] ||
    [ "$VALIDATION_MAX" != 8.650949130533 ] ||
    [ "$VALIDATION_THRESHOLD_APPLIED" != no ] ||
    [ "$B1R_MATCH" != yes ] || [ "$B2R_MATCH" != yes ] ||
    [ "$D1_PASS" != yes ]; then
    log "D1 report acceptance fields do not match the frozen A4 zero-update requirements"
    log "rank=$RANK scale=$SCALE adapter_parameter_count=$PARAMETERS"
    log "evaluated_rows=$EVALUATED_ROWS train_rows_seen=$TRAIN_ROWS validation_rows_seen=$VALIDATION_ROWS test_rows_seen=$TEST_ROWS"
    log "zero_correction_rows=$ZERO_CORRECTION_ROWS nonzero_correction_rows=$NONZERO_CORRECTION_ROWS final_vs_base_logit_bit_mismatches=$LOGIT_MISMATCHES"
    log "adapter_A_changed=$A_CHANGED adapter_B_changed=$B_CHANGED optimizer_steps_before=$OPTIMIZER_BEFORE optimizer_steps_after=$OPTIMIZER_AFTER optimizer_state_changed=$OPTIMIZER_STATE_CHANGED"
    log "overall_average_loss=$OVERALL_AVERAGE train_average_loss=$TRAIN_AVERAGE validation_average_loss=$VALIDATION_AVERAGE test_average_loss=$TEST_AVERAGE"
    log "overall_top1_correct=$OVERALL_TOP1 train_top1_correct=$TRAIN_TOP1 validation_top1_correct=$VALIDATION_TOP1 test_top1_correct=$TEST_TOP1"
    log "b1r_a4_loss_metrics_match=$B1R_MATCH b2r_a4_top1_metrics_match=$B2R_MATCH"
    fail "FAIL: one or more D1 zero-update acceptance gates failed"
fi

log "base_checkpoint_identity=PASS"
log "tokenizer_identity=PASS"
log "dataset_identity=PASS"
log "cache_identity=PASS"
log "b1r_a4_report_identity=PASS"
log "b2r_a4_report_identity=PASS"
log "c2c_a4_report_identity=PASS"
log "rank=$RANK"
log "scale=$SCALE"
log "adapter_parameter_count=$PARAMETERS"
log "evaluated_rows=$EVALUATED_ROWS"
log "train_rows_seen=$TRAIN_ROWS"
log "validation_rows_seen=$VALIDATION_ROWS"
log "test_rows_seen=$TEST_ROWS"
log "zero_correction_rows=$ZERO_CORRECTION_ROWS"
log "nonzero_correction_rows=$NONZERO_CORRECTION_ROWS"
log "final_vs_base_logit_bit_mismatches=$LOGIT_MISMATCHES"
log "adapter_A_changed=$A_CHANGED"
log "adapter_B_changed=$B_CHANGED"
log "optimizer_steps_before=$OPTIMIZER_BEFORE"
log "optimizer_steps_after=$OPTIMIZER_AFTER"
log "overall_average_loss=$OVERALL_AVERAGE"
log "train_average_loss=$TRAIN_AVERAGE"
log "validation_average_loss=$VALIDATION_AVERAGE"
log "test_average_loss=$TEST_AVERAGE"
log "overall_top1_correct=$OVERALL_TOP1"
log "train_top1_correct=$TRAIN_TOP1"
log "validation_top1_correct=$VALIDATION_TOP1"
log "test_top1_correct=$TEST_TOP1"
log "b1r_a4_loss_metrics_match=$B1R_MATCH"
log "b2r_a4_top1_metrics_match=$B2R_MATCH"
log "validation_baseline_loss=$VALIDATION_BASELINE"
log "validation_required_relative_improvement=$VALIDATION_IMPROVEMENT"
log "validation_acceptance_loss_max=$VALIDATION_MAX"
log "validation_threshold_applied=$VALIDATION_THRESHOLD_APPLIED"
log "zero_update_trainer_report_sha256=$REPORT_SHA"
log "Stage4E-D1-A4=PASS"

cleanup
exit 0
