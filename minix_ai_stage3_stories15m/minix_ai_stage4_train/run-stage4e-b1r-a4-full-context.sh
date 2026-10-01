#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" 2>/dev/null && pwd)
cd "$SCRIPT_DIR" || exit 1

BASE_CHECKPOINT="/usr/src/minix_ai_stage3_stories15m/stories15M.bin"
TOKENIZER="/usr/src/minix_ai_stage3_stories15m/tokenizer.bin"
DATASET="stage4e-approved.records"
EVALUATOR="./stage4e_baseline_full_context_eval"
REPORT="stage4e-baseline-unadapted-full-context-a4-report.txt"
CONSOLE_LOG="stage4e-b1r-a4-full-context-console.txt"
BUILD_LOG=".stage4e-b1r-a4-build.tmp"
CHECK_LOG=".stage4e-b1r-a4-record0-check.tmp"
RUN_LOG=".stage4e-b1r-a4-run.tmp"

EXPECTED_BASE="cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a"
EXPECTED_TOKENIZER="50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361"
EXPECTED_DATASET="71744cea4de049e2e192cb2a4494c8a4a68e769b3a7f3b8ed4880433e4583b84"
EXPECTED_KV_CHECKSUM="abb568d3ab418288"
EXPECTED_TRAIN_AVERAGE="10.439190954456"
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
    log "Stage4E-B1R-A4=FAIL"
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
log "Stage 4E-B1R-A4 corrected full-context baseline"
log "========================================"

check_file "$BASE_CHECKPOINT"
check_file "$TOKENIZER"
check_file "$DATASET"
check_file "Makefile"

BASE_ACTUAL=$(sha256_hex "$BASE_CHECKPOINT")
TOKENIZER_ACTUAL=$(sha256_hex "$TOKENIZER")
DATASET_ACTUAL=$(sha256_hex "$DATASET")

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
    fail "FAIL: repaired dataset identity mismatch"
fi

log "base_checkpoint_identity=PASS"
log "tokenizer_identity=PASS"
log "dataset_identity=PASS"

make stage4eBaselineFullContextEval CC=clang > "$BUILD_LOG" 2>&1
BUILD_STATUS=$?
cat "$BUILD_LOG" | tee -a "$CONSOLE_LOG"
if [ "$BUILD_STATUS" -ne 0 ]; then
    fail "FAIL: B1R evaluator build failed"
fi

"$EVALUATOR" "$BASE_CHECKPOINT" -d "$DATASET" -z "$TOKENIZER" \
    --record0-context-check > "$CHECK_LOG" 2>&1
CHECK_STATUS=$?
cat "$CHECK_LOG" | tee -a "$CONSOLE_LOG"
if [ "$CHECK_STATUS" -ne 0 ]; then
    fail "FAIL: record-0 prompt-context gate failed; full baseline not run"
fi

RECORD0_COUNT=$(report_value "$CHECK_LOG" record0.sequence_count)
RECORD0_BOUNDARY=$(report_value "$CHECK_LOG" record0.first_target_index)
RECORD0_CALLS=$(report_value "$CHECK_LOG" record0.forward_calls_before_first_target)
RECORD0_POSITION=$(report_value "$CHECK_LOG" record0.first_supervised_input_position)
RECORD0_TARGET=$(report_value "$CHECK_LOG" record0.first_supervised_target_token_id)
RECORD0_CHECKSUM=$(report_value "$CHECK_LOG" record0.kv_checksum_before_position_15)
RECORD0_GATE=$(report_value "$CHECK_LOG" record0_prompt_context_check)
if [ "$RECORD0_COUNT" != "19" ] || [ "$RECORD0_BOUNDARY" != "16" ] ||
    [ "$RECORD0_CALLS" != "16" ] || [ "$RECORD0_POSITION" != "15" ] ||
    [ "$RECORD0_TARGET" != "2669" ] ||
    [ "$RECORD0_CHECKSUM" != "$EXPECTED_KV_CHECKSUM" ] ||
    [ "$RECORD0_GATE" != "PASS" ]; then
    log "record0_full_context_check=FAIL"
    log "record0_kv_checksum_before_position_15=$RECORD0_CHECKSUM"
    fail "FAIL: record-0 full-context regression mismatch; evaluator not run"
fi
log "record0_full_context_check=PASS"
log "record0_kv_checksum_before_position_15=$RECORD0_CHECKSUM"

"$EVALUATOR" "$BASE_CHECKPOINT" -d "$DATASET" -z "$TOKENIZER" \
    -o "$REPORT" > "$RUN_LOG" 2>&1
RUN_STATUS=$?
cat "$RUN_LOG" | tee -a "$CONSOLE_LOG"
if [ "$RUN_STATUS" -ne 0 ]; then
    fail "FAIL: corrected A4 full-context baseline evaluation failed"
fi
check_file "$REPORT"

REPORT_DATASET_SHA=$(report_value "$REPORT" dataset_sha256)
TOTAL_RECORDS=$(report_value "$REPORT" total_records)
TOTAL_TARGETS=$(report_value "$REPORT" total_target_prediction_tokens)
TOTAL_LOSS=$(report_value "$REPORT" total_target_loss)
AVERAGE_LOSS=$(report_value "$REPORT" average_target_loss)
TRAIN_COUNT=$(report_value "$REPORT" split.train.target_prediction_tokens)
TRAIN_AVERAGE=$(report_value "$REPORT" split.train.average_target_loss)
VALIDATION_COUNT=$(report_value "$REPORT" split.validation.target_prediction_tokens)
VALIDATION_AVERAGE=$(report_value "$REPORT" split.validation.average_target_loss)
TEST_COUNT=$(report_value "$REPORT" split.test.target_prediction_tokens)
TEST_AVERAGE=$(report_value "$REPORT" split.test.average_target_loss)
ACCEPTANCE_MAX=$(report_value "$REPORT" validation_acceptance_loss_max)
REPORT_SHA=$(sha256_hex "$REPORT")

if [ "$REPORT_DATASET_SHA" != "$EXPECTED_DATASET" ]; then
    fail "FAIL: report dataset identity does not match repaired dataset"
fi
if [ "$TOTAL_RECORDS" != "30" ] || [ "$TOTAL_TARGETS" != "165" ] ||
    [ "$TRAIN_COUNT" != "111" ] || [ "$VALIDATION_COUNT" != "33" ] ||
    [ "$TEST_COUNT" != "21" ]; then
    fail "FAIL: target counts mismatch; expected 165/111/33/21"
fi
if [ -z "$TOTAL_LOSS" ] || [ -z "$AVERAGE_LOSS" ] ||
    [ -z "$TRAIN_AVERAGE" ] || [ -z "$VALIDATION_AVERAGE" ] ||
    [ -z "$TEST_AVERAGE" ] || [ -z "$ACCEPTANCE_MAX" ] ||
    [ -z "$REPORT_SHA" ]; then
    fail "FAIL: corrected report missing required metrics or hash"
fi

if [ "$TRAIN_AVERAGE" = "$EXPECTED_TRAIN_AVERAGE" ]; then
    TRAIN_MATCH=yes
else
    TRAIN_MATCH=no
fi
if [ "$TEST_AVERAGE" = "$EXPECTED_TEST_AVERAGE" ]; then
    TEST_MATCH=yes
else
    TEST_MATCH=no
fi
if [ "$TRAIN_MATCH" != yes ] || [ "$TEST_MATCH" != yes ]; then
    fail "FAIL: unchanged train/test averages differ from B1R historical cross-check"
fi

log "total_records=$TOTAL_RECORDS"
log "total_target_prediction_tokens=$TOTAL_TARGETS"
log "train_target_prediction_tokens=$TRAIN_COUNT"
log "validation_target_prediction_tokens=$VALIDATION_COUNT"
log "test_target_prediction_tokens=$TEST_COUNT"
log "total_target_loss=$TOTAL_LOSS"
log "average_target_loss=$AVERAGE_LOSS"
log "train_average_target_loss=$TRAIN_AVERAGE"
log "validation_average_target_loss=$VALIDATION_AVERAGE"
log "test_average_target_loss=$TEST_AVERAGE"
log "validation_acceptance_loss_max=$ACCEPTANCE_MAX"
log "unchanged_train_metrics_match=$TRAIN_MATCH"
log "unchanged_test_metrics_match=$TEST_MATCH"
log "report_sha256=$REPORT_SHA"
log "Stage4E-B1R-A4=PASS"

cleanup
exit 0
