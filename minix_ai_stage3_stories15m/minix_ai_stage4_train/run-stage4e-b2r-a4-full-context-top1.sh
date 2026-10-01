#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" 2>/dev/null && pwd)
cd "$SCRIPT_DIR" || exit 1

BASE_CHECKPOINT="/usr/src/minix_ai_stage3_stories15m/stories15M.bin"
TOKENIZER="/usr/src/minix_ai_stage3_stories15m/tokenizer.bin"
DATASET="stage4e-approved.records"
B1R_REPORT="stage4e-baseline-unadapted-full-context-a4-report.txt"
REPORT="stage4e-baseline-unadapted-full-context-a4-top1-report.txt"
CONSOLE_LOG="stage4e-b2r-a4-full-context-top1-console.txt"
BUILD_LOG=".stage4e-b2r-a4-build.tmp"
CHECK_LOG=".stage4e-b2r-a4-record0-check.tmp"
RUN_LOG=".stage4e-b2r-a4-run.tmp"

EXPECTED_BASE="cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a"
EXPECTED_TOKENIZER="50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361"
EXPECTED_DATASET="71744cea4de049e2e192cb2a4494c8a4a68e769b3a7f3b8ed4880433e4583b84"
EXPECTED_B1R_REPORT="b92264d62bfa603d98b7f00580f1e7e0bf6b2c6cefb8e5ec5db7753e17249b2a"
EXPECTED_KV_CHECKSUM="abb568d3ab418288"
EXPECTED_TOTAL_LOSS="1683.218059401455"
EXPECTED_AVERAGE_LOSS="10.201321572130"
EXPECTED_TRAIN_AVERAGE="10.439190954456"
EXPECTED_VALIDATION_AVERAGE="9.106262242666"
EXPECTED_TEST_AVERAGE="10.664819497567"

log() {
    printf '%s\n' "$*"
    printf '%s\n' "$*" >> "$CONSOLE_LOG"
}

cleanup() {
    rm -f "$BUILD_LOG" "$CHECK_LOG" "$RUN_LOG"
}

fail() {
    log "$*"
    log "Stage4E-B2R-A4=FAIL"
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
log "========================================"
log "Stage 4E-B2R-A4 full-context top-1 baseline"
log "========================================"

check_file "$BASE_CHECKPOINT"
check_file "$TOKENIZER"
check_file "$DATASET"
check_file "$B1R_REPORT"
check_file "Makefile"

BASE_ACTUAL=$(sha256_hex "$BASE_CHECKPOINT")
TOKENIZER_ACTUAL=$(sha256_hex "$TOKENIZER")
DATASET_ACTUAL=$(sha256_hex "$DATASET")
B1R_REPORT_ACTUAL=$(sha256_hex "$B1R_REPORT")

if [ "$BASE_ACTUAL" != "$EXPECTED_BASE" ]; then
    log "base_checkpoint_identity=FAIL"
    log "expected=$EXPECTED_BASE"
    log "actual=$BASE_ACTUAL"
    fail "FAIL: checkpoint identity mismatch"
fi
if [ "$TOKENIZER_ACTUAL" != "$EXPECTED_TOKENIZER" ]; then
    log "tokenizer_identity=FAIL"
    log "expected=$EXPECTED_TOKENIZER"
    log "actual=$TOKENIZER_ACTUAL"
    fail "FAIL: tokenizer identity mismatch"
fi
if [ "$DATASET_ACTUAL" != "$EXPECTED_DATASET" ]; then
    log "dataset_identity=FAIL"
    log "expected=$EXPECTED_DATASET"
    log "actual=$DATASET_ACTUAL"
    fail "FAIL: dataset identity mismatch"
fi
if [ "$B1R_REPORT_ACTUAL" != "$EXPECTED_B1R_REPORT" ]; then
    log "b1r_a4_report_identity=FAIL"
    log "expected=$EXPECTED_B1R_REPORT"
    log "actual=$B1R_REPORT_ACTUAL"
    fail "FAIL: B1R-A4 report identity mismatch"
fi
log "base_checkpoint_identity=PASS"
log "tokenizer_identity=PASS"
log "dataset_identity=PASS"
log "b1r_a4_report_identity=PASS"

make stage4eBaselineFullContextTop1Eval CC=clang > "$BUILD_LOG" 2>&1
BUILD_STATUS=$?
cat "$BUILD_LOG" | tee -a "$CONSOLE_LOG"
if [ "$BUILD_STATUS" -ne 0 ]; then
    fail "FAIL: B2R-A4 evaluator build failed"
fi

./stage4e_baseline_full_context_top1_eval "$BASE_CHECKPOINT" \
    -d "$DATASET" -z "$TOKENIZER" --record0-context-check > "$CHECK_LOG" 2>&1
CHECK_STATUS=$?
cat "$CHECK_LOG" | tee -a "$CONSOLE_LOG"
if [ "$CHECK_STATUS" -ne 0 ]; then
    fail "FAIL: record-0 context check failed; full evaluation not run"
fi
RECORD0_COUNT=$(report_value "$CHECK_LOG" record0.sequence_count)
RECORD0_BOUNDARY=$(report_value "$CHECK_LOG" record0.first_target_index)
RECORD0_CALLS=$(report_value "$CHECK_LOG" record0.forward_calls_before_first_target)
RECORD0_POSITION=$(report_value "$CHECK_LOG" record0.first_supervised_input_position)
RECORD0_TARGET=$(report_value "$CHECK_LOG" record0.first_supervised_target_token_id)
RECORD0_CHECKSUM=$(report_value "$CHECK_LOG" record0.kv_checksum_before_position_15)
RECORD0_GATE=$(report_value "$CHECK_LOG" record0_prompt_context_check)
if [ "$RECORD0_COUNT" != 19 ] || [ "$RECORD0_BOUNDARY" != 16 ] ||
    [ "$RECORD0_CALLS" != 16 ] || [ "$RECORD0_POSITION" != 15 ] ||
    [ "$RECORD0_TARGET" != 2669 ] || [ "$RECORD0_CHECKSUM" != "$EXPECTED_KV_CHECKSUM" ] ||
    [ "$RECORD0_GATE" != PASS ]; then
    log "record0_full_context_check=FAIL"
    log "record0_kv_checksum_before_position_15=$RECORD0_CHECKSUM"
    fail "FAIL: record-0 full-context regression mismatch; evaluator not run"
fi
log "record0_full_context_check=PASS"
log "record0_kv_checksum_before_position_15=$RECORD0_CHECKSUM"

./stage4e_baseline_full_context_top1_eval "$BASE_CHECKPOINT" \
    -d "$DATASET" -z "$TOKENIZER" -o "$REPORT" -v > "$RUN_LOG" 2>&1
RUN_STATUS=$?
cat "$RUN_LOG" | tee -a "$CONSOLE_LOG"
if [ "$RUN_STATUS" -ne 0 ]; then
    fail "FAIL: B2R-A4 full-context evaluation failed"
fi
check_file "$REPORT"

TOTAL_RECORDS=$(report_value "$REPORT" total_records)
TARGET_COUNT=$(report_value "$REPORT" target_prediction_count)
TOTAL_LOSS=$(report_value "$REPORT" total_target_loss)
AVERAGE_LOSS=$(report_value "$REPORT" average_target_loss)
TOP1_CORRECT=$(report_value "$REPORT" target_top1_correct)
TOP1_ACCURACY=$(report_value "$REPORT" target_top1_accuracy)
TRAIN_COUNT=$(report_value "$REPORT" split.train.target_prediction_count)
TRAIN_AVERAGE=$(report_value "$REPORT" split.train.average_target_loss)
TRAIN_TOP1=$(report_value "$REPORT" split.train.target_top1_correct)
VALIDATION_COUNT=$(report_value "$REPORT" split.validation.target_prediction_count)
VALIDATION_AVERAGE=$(report_value "$REPORT" split.validation.average_target_loss)
VALIDATION_TOP1=$(report_value "$REPORT" split.validation.target_top1_correct)
TEST_COUNT=$(report_value "$REPORT" split.test.target_prediction_count)
TEST_AVERAGE=$(report_value "$REPORT" split.test.average_target_loss)
TEST_TOP1=$(report_value "$REPORT" split.test.target_top1_correct)

B1R_TOTAL_LOSS=$(report_value "$B1R_REPORT" total_target_loss)
B1R_AVERAGE_LOSS=$(report_value "$B1R_REPORT" average_target_loss)
B1R_TRAIN_AVERAGE=$(report_value "$B1R_REPORT" split.train.average_target_loss)
B1R_VALIDATION_AVERAGE=$(report_value "$B1R_REPORT" split.validation.average_target_loss)
B1R_TEST_AVERAGE=$(report_value "$B1R_REPORT" split.test.average_target_loss)
DIFFERING_FIELDS=""
if [ "$TOTAL_LOSS" != "$B1R_TOTAL_LOSS" ] || [ "$TOTAL_LOSS" != "$EXPECTED_TOTAL_LOSS" ]; then DIFFERING_FIELDS="${DIFFERING_FIELDS}total_target_loss,"; fi
if [ "$AVERAGE_LOSS" != "$B1R_AVERAGE_LOSS" ] || [ "$AVERAGE_LOSS" != "$EXPECTED_AVERAGE_LOSS" ]; then DIFFERING_FIELDS="${DIFFERING_FIELDS}average_target_loss,"; fi
if [ "$TRAIN_AVERAGE" != "$B1R_TRAIN_AVERAGE" ] || [ "$TRAIN_AVERAGE" != "$EXPECTED_TRAIN_AVERAGE" ]; then DIFFERING_FIELDS="${DIFFERING_FIELDS}train_average_target_loss,"; fi
if [ "$VALIDATION_AVERAGE" != "$B1R_VALIDATION_AVERAGE" ] || [ "$VALIDATION_AVERAGE" != "$EXPECTED_VALIDATION_AVERAGE" ]; then DIFFERING_FIELDS="${DIFFERING_FIELDS}validation_average_target_loss,"; fi
if [ "$TEST_AVERAGE" != "$B1R_TEST_AVERAGE" ] || [ "$TEST_AVERAGE" != "$EXPECTED_TEST_AVERAGE" ]; then DIFFERING_FIELDS="${DIFFERING_FIELDS}test_average_target_loss,"; fi
if [ -n "$DIFFERING_FIELDS" ]; then
    log "b1r_a4_loss_metrics_match=no"
    log "differing_fields=${DIFFERING_FIELDS%,}"
    fail "FAIL: B2R-A4 loss metrics differ from B1R-A4"
fi

if [ "$TOTAL_RECORDS" != 30 ] || [ "$TARGET_COUNT" != 165 ] ||
    [ "$TRAIN_COUNT" != 111 ] || [ "$VALIDATION_COUNT" != 33 ] ||
    [ "$TEST_COUNT" != 21 ] || [ -z "$TOP1_CORRECT" ] ||
    [ -z "$TOP1_ACCURACY" ] || [ -z "$TRAIN_TOP1" ] ||
    [ -z "$VALIDATION_TOP1" ] || [ -z "$TEST_TOP1" ]; then
    fail "FAIL: corrected target counts or top-1 metrics mismatch"
fi

RECORD021_LINE=$(awk '/^record id=sysconf-ai-model-priority-val-021 / { print; exit }' "$RUN_LOG")
RECORD022_LINE=$(awk '/^record id=sysconf-ai-model-quantum-val-022 / { print; exit }' "$RUN_LOG")
RECORD031_LINE=$(awk '/^record id=sample-zero-temperature-val-031 / { print; exit }' "$RUN_LOG")
case "$RECORD021_LINE" in *"task=source_navigation"*) ;; *) fail "FAIL: record 021 semantic task check" ;; esac
if [ -n "$RECORD022_LINE" ]; then fail "FAIL: removed record 022 is still present"; fi
case "$RECORD031_LINE" in *"split=validation"*"task=explanation"*) ;; *) fail "FAIL: new record 031 semantic check" ;; esac

REPORT_SHA=$(sha256_hex "$REPORT")
if [ -z "$REPORT_SHA" ]; then
    fail "FAIL: unable to hash B2R-A4 report"
fi

log "total_records=$TOTAL_RECORDS"
log "target_prediction_count=$TARGET_COUNT"
log "train_target_prediction_count=$TRAIN_COUNT"
log "validation_target_prediction_count=$VALIDATION_COUNT"
log "test_target_prediction_count=$TEST_COUNT"
log "average_target_loss=$AVERAGE_LOSS"
log "validation_average_target_loss=$VALIDATION_AVERAGE"
log "target_top1_correct=$TOP1_CORRECT"
log "target_top1_accuracy=$TOP1_ACCURACY"
log "train_target_top1_correct=$TRAIN_TOP1"
log "validation_target_top1_correct=$VALIDATION_TOP1"
log "test_target_top1_correct=$TEST_TOP1"
log "b1r_a4_loss_metrics_match=yes"
log "report_sha256=$REPORT_SHA"
log "Stage4E-B2R-A4=PASS"

cleanup
exit 0
