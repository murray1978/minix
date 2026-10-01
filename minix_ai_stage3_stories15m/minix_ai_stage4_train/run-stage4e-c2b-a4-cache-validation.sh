#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" 2>/dev/null && pwd)
cd "$SCRIPT_DIR" || exit 1

BASE_CHECKPOINT="/usr/src/minix_ai_stage3_stories15m/stories15M.bin"
TOKENIZER="/usr/src/minix_ai_stage3_stories15m/tokenizer.bin"
DATASET="stage4e-approved.records"
CACHE="stage4e-target-cache-a4.bin"
REPORT="stage4e-c2b-a4-cache-validation-report.txt"
CONSOLE_LOG="stage4e-c2b-a4-cache-validation-console.txt"
BUILD_LOG=".stage4e-c2b-a4-build.tmp"
VALIDATE_LOG=".stage4e-c2b-a4-validator.tmp"

EXPECTED_BASE="cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a"
EXPECTED_TOKENIZER="50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361"
EXPECTED_DATASET="71744cea4de049e2e192cb2a4494c8a4a68e769b3a7f3b8ed4880433e4583b84"
EXPECTED_CACHE="d6288ad3d128c7182c20088aa57a1bca8bcaa1e13314fb0cdbd60eba14278def"
EXPECTED_FILE_SIZE="192892"

log() {
    printf '%s\n' "$*"
    printf '%s\n' "$*" >> "$CONSOLE_LOG"
}

cleanup() {
    rm -f "$BUILD_LOG" "$VALIDATE_LOG"
}

fail() {
    log "$*"
    log "Stage4E-C2b-A4=FAIL"
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
log "Stage 4E-C2b-A4 structural cache validation"
log "========================================"

check_file "$BASE_CHECKPOINT"
check_file "$TOKENIZER"
check_file "$DATASET"
check_file "$CACHE"
check_file "Makefile"

BASE_ACTUAL=$(sha256_hex "$BASE_CHECKPOINT")
TOKENIZER_ACTUAL=$(sha256_hex "$TOKENIZER")
DATASET_ACTUAL=$(sha256_hex "$DATASET")
CACHE_ACTUAL=$(sha256_hex "$CACHE")

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
if [ "$CACHE_ACTUAL" != "$EXPECTED_CACHE" ]; then
    log "cache_identity=FAIL"
    log "expected=$EXPECTED_CACHE"
    log "actual=$CACHE_ACTUAL"
    fail "FAIL: A4 cache identity mismatch"
fi
log "base_checkpoint_identity=PASS"
log "tokenizer_identity=PASS"
log "dataset_identity=PASS"
log "cache_identity=PASS"

make stage4eCacheValidateA4 CC=clang > "$BUILD_LOG" 2>&1
BUILD_STATUS=$?
cat "$BUILD_LOG" | tee -a "$CONSOLE_LOG"
if [ "$BUILD_STATUS" -ne 0 ]; then
    fail "FAIL: A4 validator build failed"
fi

./stage4e_cache_validate_a4 "$BASE_CHECKPOINT" \
    -d "$DATASET" -z "$TOKENIZER" -c "$CACHE" -r "$REPORT" \
    > "$VALIDATE_LOG" 2>&1
VALIDATE_STATUS=$?
cat "$VALIDATE_LOG" | tee -a "$CONSOLE_LOG"
check_file "$REPORT"
if [ "$VALIDATE_STATUS" -ne 0 ]; then
    fail "FAIL: structural cache validation failed"
fi

REPORT_CACHE=$(report_value "$REPORT" cache_sha256)
REPORT_DATASET=$(report_value "$REPORT" dataset_sha256)
REPORT_BASE=$(report_value "$REPORT" base_checkpoint_sha256)
REPORT_TOKENIZER=$(report_value "$REPORT" tokenizer_sha256)
FORMAT_VERSION=$(report_value "$REPORT" cache_format_version)
HEADER_SIZE=$(report_value "$REPORT" header_size)
ENTRY_SIZE=$(report_value "$REPORT" entry_size)
MODEL_DIM=$(report_value "$REPORT" model_dim)
VOCAB_SIZE=$(report_value "$REPORT" vocab_size)
RECORD_COUNT=$(report_value "$REPORT" record_count)
EXPECTED_ROWS=$(report_value "$REPORT" expected_rows)
VALIDATED_ROWS=$(report_value "$REPORT" validated_rows)
TRAIN_ROWS=$(report_value "$REPORT" train_target_rows)
VALIDATION_ROWS=$(report_value "$REPORT" validation_target_rows)
TEST_ROWS=$(report_value "$REPORT" test_target_rows)
REGRESSION_ROWS=$(report_value "$REPORT" regression_target_rows)
CACHE_FILE_SIZE=$(report_value "$REPORT" cache_file_size)
DUPLICATES=$(report_value "$REPORT" duplicate_cache_rows)
MISSING=$(report_value "$REPORT" missing_cache_rows)
UNEXPECTED=$(report_value "$REPORT" unexpected_cache_rows)
RECORD_MISMATCHES=$(report_value "$REPORT" record_index_mismatches)
POSITION_MISMATCHES=$(report_value "$REPORT" input_position_mismatches)
TARGET_MISMATCHES=$(report_value "$REPORT" target_token_id_mismatches)
SPLIT_MISMATCHES=$(report_value "$REPORT" split_id_mismatches)
ORDER_MISMATCHES=$(report_value "$REPORT" row_order_mismatches)
NONFINITE=$(report_value "$REPORT" nonfinite_hidden_values)
HEADER_ERRORS=$(report_value "$REPORT" header_errors)
IDENTITY_ERRORS=$(report_value "$REPORT" identity_errors)
FILE_SIZE_ERRORS=$(report_value "$REPORT" file_size_errors)
TRAILING_ERRORS=$(report_value "$REPORT" trailing_byte_errors)
FIRST_INDEX=$(report_value "$REPORT" first_row_record_index)
FIRST_POSITION=$(report_value "$REPORT" first_row_input_position)
FIRST_TARGET=$(report_value "$REPORT" first_row_target_token_id)
FIRST_SPLIT=$(report_value "$REPORT" first_row_split_id)
LAST_INDEX=$(report_value "$REPORT" last_row_record_index)
LAST_POSITION=$(report_value "$REPORT" last_row_input_position)
LAST_TARGET=$(report_value "$REPORT" last_row_target_token_id)
LAST_SPLIT=$(report_value "$REPORT" last_row_split_id)

if [ "$REPORT_CACHE" != "$EXPECTED_CACHE" ] ||
    [ "$REPORT_DATASET" != "$EXPECTED_DATASET" ] ||
    [ "$REPORT_BASE" != "$EXPECTED_BASE" ] ||
    [ "$REPORT_TOKENIZER" != "$EXPECTED_TOKENIZER" ]; then
    fail "FAIL: report identities differ from authoritative identities"
fi
if [ "$FORMAT_VERSION" != 1 ] || [ "$HEADER_SIZE" != 172 ] ||
    [ "$ENTRY_SIZE" != 1168 ] || [ "$MODEL_DIM" != 288 ] ||
    [ "$VOCAB_SIZE" != 32000 ] || [ "$RECORD_COUNT" != 30 ] ||
    [ "$EXPECTED_ROWS" != 165 ] || [ "$VALIDATED_ROWS" != 165 ] ||
    [ "$TRAIN_ROWS" != 111 ] || [ "$VALIDATION_ROWS" != 33 ] ||
    [ "$TEST_ROWS" != 21 ] || [ "$REGRESSION_ROWS" != 0 ] ||
    [ "$CACHE_FILE_SIZE" != "$EXPECTED_FILE_SIZE" ]; then
    fail "FAIL: cache header/count/size fields do not match A4 requirements"
fi
if [ "$DUPLICATES" != 0 ] || [ "$MISSING" != 0 ] || [ "$UNEXPECTED" != 0 ] ||
    [ "$RECORD_MISMATCHES" != 0 ] || [ "$POSITION_MISMATCHES" != 0 ] ||
    [ "$TARGET_MISMATCHES" != 0 ] || [ "$SPLIT_MISMATCHES" != 0 ] ||
    [ "$ORDER_MISMATCHES" != 0 ] || [ "$NONFINITE" != 0 ] ||
    [ "$HEADER_ERRORS" != 0 ] || [ "$IDENTITY_ERRORS" != 0 ] ||
    [ "$FILE_SIZE_ERRORS" != 0 ] || [ "$TRAILING_ERRORS" != 0 ]; then
    fail "FAIL: one or more structural validation counters are nonzero"
fi
if [ "$FIRST_INDEX" != 0 ] || [ "$FIRST_POSITION" != 15 ] ||
    [ "$FIRST_TARGET" != 2669 ] || [ "$FIRST_SPLIT" != 0 ] ||
    [ "$LAST_INDEX" != 29 ] || [ "$LAST_POSITION" != 28 ] ||
    [ "$LAST_TARGET" != 13 ] || [ "$LAST_SPLIT" != 2 ]; then
    fail "FAIL: first/last cache row boundary metadata mismatch"
fi

log "cache_format_version=$FORMAT_VERSION"
log "header_size=$HEADER_SIZE"
log "entry_size=$ENTRY_SIZE"
log "record_count=$RECORD_COUNT"
log "expected_rows=$EXPECTED_ROWS"
log "validated_rows=$VALIDATED_ROWS"
log "train_target_rows=$TRAIN_ROWS"
log "validation_target_rows=$VALIDATION_ROWS"
log "test_target_rows=$TEST_ROWS"
log "regression_target_rows=$REGRESSION_ROWS"
log "cache_file_size=$CACHE_FILE_SIZE"
log "duplicate_cache_rows=$DUPLICATES"
log "missing_cache_rows=$MISSING"
log "unexpected_cache_rows=$UNEXPECTED"
log "record_index_mismatches=$RECORD_MISMATCHES"
log "input_position_mismatches=$POSITION_MISMATCHES"
log "target_token_id_mismatches=$TARGET_MISMATCHES"
log "split_id_mismatches=$SPLIT_MISMATCHES"
log "row_order_mismatches=$ORDER_MISMATCHES"
log "nonfinite_hidden_values=$NONFINITE"
log "header_errors=$HEADER_ERRORS"
log "identity_errors=$IDENTITY_ERRORS"
log "file_size_errors=$FILE_SIZE_ERRORS"
log "trailing_byte_errors=$TRAILING_ERRORS"
log "first_row_check=PASS"
log "last_row_check=PASS"
log "Stage4E-C2b-A4=PASS"

cleanup
exit 0
