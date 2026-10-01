#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" 2>/dev/null && pwd)
cd "$SCRIPT_DIR" || exit 1

BASE_CHECKPOINT="/usr/src/minix_ai_stage3_stories15m/stories15M.bin"
TOKENIZER="/usr/src/minix_ai_stage3_stories15m/tokenizer.bin"
DATASET="stage4e-approved.records"
DATASET_REPORT="stage4e-dataset-report-a4.txt"
CACHE="stage4e-target-cache-a4.bin"
REPORT="stage4e-c2a-a4-cache-report.txt"
CONSOLE_LOG="stage4e-c2a-a4-cache-console.txt"
BUILD_LOG=".stage4e-c2a-a4-build.tmp"
CHECK_LOG=".stage4e-c2a-a4-record0-check.tmp"
RUN_LOG=".stage4e-c2a-a4-build-run.tmp"

EXPECTED_BASE="cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a"
EXPECTED_TOKENIZER="50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361"
EXPECTED_DATASET="71744cea4de049e2e192cb2a4494c8a4a68e769b3a7f3b8ed4880433e4583b84"
EXPECTED_DATASET_REPORT="1fe9b1803637578945a725931d6f0c02230c88941e0060abae6e11b8e4d76ddb"
EXPECTED_KV_CHECKSUM="abb568d3ab418288"
EXPECTED_FILE_SIZE="192892"

log() {
    printf '%s\n' "$*"
    printf '%s\n' "$*" >> "$CONSOLE_LOG"
}

cleanup() {
    rm -f "$BUILD_LOG" "$CHECK_LOG" "$RUN_LOG"
}

fail() {
    log "$*"
    log "Stage4E-C2a-A4=FAIL"
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
log "Stage 4E-C2a-A4 frozen full-context cache build"
log "========================================"

check_file "$BASE_CHECKPOINT"
check_file "$TOKENIZER"
check_file "$DATASET"
check_file "$DATASET_REPORT"
check_file "Makefile"

BASE_ACTUAL=$(sha256_hex "$BASE_CHECKPOINT")
TOKENIZER_ACTUAL=$(sha256_hex "$TOKENIZER")
DATASET_ACTUAL=$(sha256_hex "$DATASET")
DATASET_REPORT_ACTUAL=$(sha256_hex "$DATASET_REPORT")

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
if [ "$DATASET_REPORT_ACTUAL" != "$EXPECTED_DATASET_REPORT" ]; then
    log "dataset_report_identity=FAIL"
    log "expected=$EXPECTED_DATASET_REPORT"
    log "actual=$DATASET_REPORT_ACTUAL"
    fail "FAIL: A4 dataset report identity mismatch"
fi
log "base_checkpoint_identity=PASS"
log "tokenizer_identity=PASS"
log "dataset_identity=PASS"
log "dataset_report_identity=PASS"

make stage4eCacheBuildA4 CC=clang > "$BUILD_LOG" 2>&1
BUILD_STATUS=$?
cat "$BUILD_LOG" | tee -a "$CONSOLE_LOG"
if [ "$BUILD_STATUS" -ne 0 ]; then
    fail "FAIL: A4 cache writer build failed"
fi

./stage4e_cache_build_a4 "$BASE_CHECKPOINT" -d "$DATASET" \
    -z "$TOKENIZER" --record0-context-check > "$CHECK_LOG" 2>&1
CHECK_STATUS=$?
cat "$CHECK_LOG" | tee -a "$CONSOLE_LOG"
if [ "$CHECK_STATUS" -ne 0 ]; then
    fail "FAIL: record-0 context regression failed; cache was not built"
fi
RECORD0_COUNT=$(report_value "$CHECK_LOG" record0.sequence_count)
RECORD0_BOUNDARY=$(report_value "$CHECK_LOG" record0.first_target_index)
FORWARDS_BEFORE_POSITION=$(report_value "$CHECK_LOG" record0.forward_calls_before_position_15)
FORWARDS_BEFORE_TARGET=$(report_value "$CHECK_LOG" record0.forward_calls_before_first_target)
RECORD0_TARGET=$(report_value "$CHECK_LOG" record0.first_supervised_target_token_id)
RECORD0_CHECKSUM=$(report_value "$CHECK_LOG" record0.kv_checksum_before_position_15)
RECORD0_GATE=$(report_value "$CHECK_LOG" record0_full_context_check)
if [ "$RECORD0_COUNT" != 19 ] || [ "$RECORD0_BOUNDARY" != 16 ] ||
    [ "$FORWARDS_BEFORE_POSITION" != 15 ] || [ "$FORWARDS_BEFORE_TARGET" != 16 ] ||
    [ "$RECORD0_TARGET" != 2669 ] || [ "$RECORD0_CHECKSUM" != "$EXPECTED_KV_CHECKSUM" ] ||
    [ "$RECORD0_GATE" != PASS ]; then
    log "record0_full_context_check=FAIL"
    log "record0_kv_checksum_before_position_15=$RECORD0_CHECKSUM"
    fail "FAIL: record-0 full-context preflight mismatch; cache was not built"
fi
log "record0_full_context_check=PASS"
log "record0_kv_checksum_before_position_15=$RECORD0_CHECKSUM"

./stage4e_cache_build_a4 "$BASE_CHECKPOINT" -d "$DATASET" \
    -z "$TOKENIZER" -o "$CACHE" -r "$REPORT" > "$RUN_LOG" 2>&1
RUN_STATUS=$?
cat "$RUN_LOG" | tee -a "$CONSOLE_LOG"
if [ "$RUN_STATUS" -ne 0 ]; then
    fail "FAIL: A4 cache generation failed"
fi
check_file "$CACHE"
check_file "$REPORT"

FORMAT_VERSION=$(report_value "$REPORT" cache_format_version)
HEADER_SIZE=$(report_value "$REPORT" header_size)
ENTRY_SIZE=$(report_value "$REPORT" entry_size)
REPORT_DATASET=$(report_value "$REPORT" dataset_sha256)
REPORT_BASE=$(report_value "$REPORT" base_checkpoint_sha256)
REPORT_TOKENIZER=$(report_value "$REPORT" tokenizer_sha256)
MODEL_DIM=$(report_value "$REPORT" model_dim)
VOCAB_SIZE=$(report_value "$REPORT" vocab_size)
RECORD_COUNT=$(report_value "$REPORT" record_count)
TARGET_ROWS=$(report_value "$REPORT" target_row_count)
TRAIN_ROWS=$(report_value "$REPORT" train_target_rows)
VALIDATION_ROWS=$(report_value "$REPORT" validation_target_rows)
TEST_ROWS=$(report_value "$REPORT" test_target_rows)
REGRESSION_ROWS=$(report_value "$REPORT" regression_target_rows)
CACHE_SIZE=$(report_value "$REPORT" cache_file_size)
FIRST_INDEX=$(report_value "$REPORT" first_row_record_index)
FIRST_POSITION=$(report_value "$REPORT" first_row_input_position)
FIRST_TARGET=$(report_value "$REPORT" first_row_target_token_id)
FIRST_SPLIT=$(report_value "$REPORT" first_row_split_id)
LAST_INDEX=$(report_value "$REPORT" last_row_record_index)
LAST_POSITION=$(report_value "$REPORT" last_row_input_position)
LAST_TARGET=$(report_value "$REPORT" last_row_target_token_id)
LAST_SPLIT=$(report_value "$REPORT" last_row_split_id)
NONFINITE=$(report_value "$REPORT" nonfinite_hidden_values)

if [ "$FORMAT_VERSION" != 1 ] || [ "$HEADER_SIZE" != 172 ] ||
    [ "$ENTRY_SIZE" != 1168 ] || [ "$REPORT_DATASET" != "$EXPECTED_DATASET" ] ||
    [ "$REPORT_BASE" != "$EXPECTED_BASE" ] || [ "$REPORT_TOKENIZER" != "$EXPECTED_TOKENIZER" ] ||
    [ "$MODEL_DIM" != 288 ] || [ "$VOCAB_SIZE" != 32000 ] ||
    [ "$RECORD_COUNT" != 30 ] || [ "$TARGET_ROWS" != 165 ] ||
    [ "$TRAIN_ROWS" != 111 ] || [ "$VALIDATION_ROWS" != 33 ] ||
    [ "$TEST_ROWS" != 21 ] || [ "$REGRESSION_ROWS" != 0 ] ||
    [ "$CACHE_SIZE" != "$EXPECTED_FILE_SIZE" ] ||
    [ "$FIRST_INDEX" != 0 ] || [ "$FIRST_POSITION" != 15 ] ||
    [ "$FIRST_TARGET" != 2669 ] || [ "$FIRST_SPLIT" != 0 ] ||
    [ "$LAST_INDEX" != 29 ] || [ "$LAST_POSITION" != 28 ] ||
    [ "$LAST_TARGET" != 13 ] || [ "$LAST_SPLIT" != 2 ] ||
    [ "$NONFINITE" != 0 ]; then
    fail "FAIL: A4 cache report does not match required format/count/row values"
fi

ACTUAL_FILE_SIZE=$(wc -c < "$CACHE" | tr -d ' ')
if [ "$ACTUAL_FILE_SIZE" != "$EXPECTED_FILE_SIZE" ]; then
    fail "FAIL: cache file size mismatch: expected=$EXPECTED_FILE_SIZE actual=$ACTUAL_FILE_SIZE"
fi
CACHE_SHA=$(sha256_hex "$CACHE")
if [ -z "$CACHE_SHA" ]; then
    fail "FAIL: could not calculate cache SHA-256"
fi
printf 'cache_sha256=%s\n' "$CACHE_SHA" >> "$REPORT" ||
    fail "FAIL: could not append cache SHA-256 to report"

log "cache_format_version=$FORMAT_VERSION"
log "header_size=$HEADER_SIZE"
log "entry_size=$ENTRY_SIZE"
log "record_count=$RECORD_COUNT"
log "target_row_count=$TARGET_ROWS"
log "train_target_rows=$TRAIN_ROWS"
log "validation_target_rows=$VALIDATION_ROWS"
log "test_target_rows=$TEST_ROWS"
log "regression_target_rows=$REGRESSION_ROWS"
log "cache_file_size=$ACTUAL_FILE_SIZE"
log "first_row_record_index=$FIRST_INDEX"
log "first_row_input_position=$FIRST_POSITION"
log "first_row_target_token_id=$FIRST_TARGET"
log "first_row_split_id=$FIRST_SPLIT"
log "last_row_record_index=$LAST_INDEX"
log "last_row_input_position=$LAST_POSITION"
log "last_row_target_token_id=$LAST_TARGET"
log "last_row_split_id=$LAST_SPLIT"
log "nonfinite_hidden_values=$NONFINITE"
log "cache_sha256=$CACHE_SHA"
log "Stage4E-C2a-A4=PASS"

cleanup
exit 0
