#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" 2>/dev/null && pwd)
cd "$SCRIPT_DIR" || exit 1

BASE_CHECKPOINT="/usr/src/minix_ai_stage3_stories15m/stories15M.bin"
TOKENIZER="/usr/src/minix_ai_stage3_stories15m/tokenizer.bin"
DATASET="stage4e-approved.records"
DATASET_REPORT="stage4e-dataset-report-a4.txt"
REPORT="stage4e-baseline-unadapted-generation-a4-report.txt"
CONSOLE_LOG="stage4e-b3r-a4-generation-console.txt"
BUILD_LOG=".stage4e-b3r-a4-build.tmp"
RUN_LOG=".stage4e-b3r-a4-run.tmp"
SEMANTIC_LOG=".stage4e-b3r-a4-semantic.tmp"

EXPECTED_BASE="cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a"
EXPECTED_TOKENIZER="50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361"
EXPECTED_DATASET="71744cea4de049e2e192cb2a4494c8a4a68e769b3a7f3b8ed4880433e4583b84"
EXPECTED_DATASET_REPORT="1fe9b1803637578945a725931d6f0c02230c88941e0060abae6e11b8e4d76ddb"

log() {
    printf '%s\n' "$*"
    printf '%s\n' "$*" >> "$CONSOLE_LOG"
}

cleanup() {
    rm -f "$BUILD_LOG" "$RUN_LOG" "$SEMANTIC_LOG"
}

fail() {
    log "$*"
    log "Stage4E-B3R-A4=FAIL"
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
log "Stage 4E-B3R-A4 unadapted generation baseline"
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
    fail "FAIL: dataset identity mismatch"
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

make stage4eBaselineGenerationEval CC=clang > "$BUILD_LOG" 2>&1
BUILD_STATUS=$?
cat "$BUILD_LOG" | tee -a "$CONSOLE_LOG"
if [ "$BUILD_STATUS" -ne 0 ]; then
    fail "FAIL: generation evaluator build failed"
fi

./stage4e_baseline_unadapted_generation "$BASE_CHECKPOINT" \
    -d "$DATASET" -z "$TOKENIZER" -o "$REPORT" \
    --max-visible-output-tokens 64 -v > "$RUN_LOG" 2>&1
RUN_STATUS=$?
cat "$RUN_LOG" | tee -a "$CONSOLE_LOG"
if [ "$RUN_STATUS" -ne 0 ]; then
    fail "FAIL: A4 generation evaluation failed"
fi
check_file "$REPORT"

awk -F= '
/^record.id=/ {
    current = substr($0, index($0, "=") + 1)
    seen[current] = 1
    records++
    next
}
/^record.split=/ && current != "" {
    splits[current] = substr($0, index($0, "=") + 1)
    next
}
/^record.task=/ && current != "" {
    tasks[current] = substr($0, index($0, "=") + 1)
    next
}
/^record.index=/ { record_index_fields++; next }
/^record.record_index=/ { record_index_alias_fields++; next }
/^record.record_id=/ { record_id_fields++; next }
/^record.expected_visible_target=/ { expected_target_fields++; next }
/^record.expected_visible_target_token_ids=/ { expected_ids_fields++; next }
/^record.generated_text=/ { generated_text_fields++; next }
/^record.generated_token_ids=/ { generated_ids_fields++; next }
/^record.first_generated_token_id=/ { first_token_fields++; next }
/^record.target_window_match=/ { window_match_fields++; next }
/^record.terminator_generated=/ { terminator_fields++; next }
/^record.exact_response_match=/ { exact_match_fields++; next }
END {
    printf "records_evaluated=%d\n", records
    printf "record_index_fields=%d\n", record_index_fields
    printf "record_index_alias_fields=%d\n", record_index_alias_fields
    printf "record_id_fields=%d\n", record_id_fields
    printf "expected_target_fields=%d\n", expected_target_fields
    printf "expected_target_id_fields=%d\n", expected_ids_fields
    printf "generated_text_fields=%d\n", generated_text_fields
    printf "generated_token_id_fields=%d\n", generated_ids_fields
    printf "first_generated_token_fields=%d\n", first_token_fields
    printf "target_window_match_fields=%d\n", window_match_fields
    printf "terminator_generated_fields=%d\n", terminator_fields
    printf "exact_response_match_fields=%d\n", exact_match_fields
    printf "record021_task=%s\n", tasks["sysconf-ai-model-priority-val-021"]
    printf "record022_present=%s\n", ("sysconf-ai-model-quantum-val-022" in seen) ? "yes" : "no"
    printf "record031_present=%s\n", ("sample-zero-temperature-val-031" in seen) ? "yes" : "no"
    printf "record031_split=%s\n", splits["sample-zero-temperature-val-031"]
    printf "record031_task=%s\n", tasks["sample-zero-temperature-val-031"]
}
' "$REPORT" > "$SEMANTIC_LOG"
cat "$SEMANTIC_LOG" | tee -a "$CONSOLE_LOG"

EVALUATED_RECORDS=$(report_value "$SEMANTIC_LOG" records_evaluated)
RECORD_INDEX_FIELDS=$(report_value "$SEMANTIC_LOG" record_index_fields)
RECORD_INDEX_ALIAS_FIELDS=$(report_value "$SEMANTIC_LOG" record_index_alias_fields)
RECORD_ID_FIELDS=$(report_value "$SEMANTIC_LOG" record_id_fields)
EXPECTED_TARGET_FIELDS=$(report_value "$SEMANTIC_LOG" expected_target_fields)
EXPECTED_TARGET_ID_FIELDS=$(report_value "$SEMANTIC_LOG" expected_target_id_fields)
GENERATED_TEXT_FIELDS=$(report_value "$SEMANTIC_LOG" generated_text_fields)
GENERATED_TOKEN_ID_FIELDS=$(report_value "$SEMANTIC_LOG" generated_token_id_fields)
FIRST_GENERATED_TOKEN_FIELDS=$(report_value "$SEMANTIC_LOG" first_generated_token_fields)
TARGET_WINDOW_MATCH_FIELDS=$(report_value "$SEMANTIC_LOG" target_window_match_fields)
TERMINATOR_GENERATED_FIELDS=$(report_value "$SEMANTIC_LOG" terminator_generated_fields)
EXACT_RESPONSE_MATCH_FIELDS=$(report_value "$SEMANTIC_LOG" exact_response_match_fields)
RECORD021_TASK=$(report_value "$SEMANTIC_LOG" record021_task)
RECORD022_PRESENT=$(report_value "$SEMANTIC_LOG" record022_present)
RECORD031_PRESENT=$(report_value "$SEMANTIC_LOG" record031_present)
RECORD031_SPLIT=$(report_value "$SEMANTIC_LOG" record031_split)
RECORD031_TASK=$(report_value "$SEMANTIC_LOG" record031_task)

if [ "$EVALUATED_RECORDS" != 30 ] || [ "$RECORD_INDEX_FIELDS" != 30 ] ||
    [ "$RECORD_INDEX_ALIAS_FIELDS" != 30 ] || [ "$RECORD_ID_FIELDS" != 30 ] ||
    [ "$EXPECTED_TARGET_FIELDS" != 30 ] || [ "$EXPECTED_TARGET_ID_FIELDS" != 30 ] ||
    [ "$GENERATED_TEXT_FIELDS" != 30 ] || [ "$GENERATED_TOKEN_ID_FIELDS" != 30 ] ||
    [ "$FIRST_GENERATED_TOKEN_FIELDS" != 30 ] ||
    [ "$TARGET_WINDOW_MATCH_FIELDS" != 30 ] ||
    [ "$TERMINATOR_GENERATED_FIELDS" != 30 ] || [ "$EXACT_RESPONSE_MATCH_FIELDS" != 30 ] ||
    [ "$RECORD021_TASK" != source_navigation ] ||
    [ "$RECORD022_PRESENT" != no ] ||
    [ "$RECORD031_PRESENT" != yes ] ||
    [ "$RECORD031_SPLIT" != validation ] ||
    [ "$RECORD031_TASK" != explanation ]; then
    fail "FAIL: A4 generation semantic record checks failed"
fi

CONFIG_ADAPTER=$(report_value "$REPORT" adapter_loaded)
CONFIG_TEMPERATURE=$(report_value "$REPORT" generation_temperature)
CONFIG_TERMINATOR=$(report_value "$REPORT" newline_terminator_token_id)
CONFIG_MAX_TOKENS=$(report_value "$REPORT" max_visible_output_tokens)
CONFIG_CONTEXT=$(report_value "$REPORT" prompt_context)
if [ "$CONFIG_ADAPTER" != no ] || [ "$CONFIG_TEMPERATURE" != 0 ] ||
    [ "$CONFIG_TERMINATOR" != 13 ] || [ "$CONFIG_MAX_TOKENS" != 64 ] ||
    [ "$CONFIG_CONTEXT" != full_causal_prefill ]; then
    fail "FAIL: generation configuration mismatch"
fi

TOTAL_RECORDS=$(report_value "$REPORT" overall.record_count)
OVERALL_WINDOW_MATCH=$(report_value "$REPORT" overall.target_window_match_count)
OVERALL_TERMINATORS=$(report_value "$REPORT" overall.terminator_generated_count)
OVERALL_EXACT=$(report_value "$REPORT" overall.exact_response_match_count)
TRAIN_WINDOW_MATCH=$(report_value "$REPORT" train.target_window_match_count)
VALIDATION_WINDOW_MATCH=$(report_value "$REPORT" validation.target_window_match_count)
TEST_WINDOW_MATCH=$(report_value "$REPORT" test.target_window_match_count)
DISTINCT_WINDOWS=$(report_value "$REPORT" overall.distinct_generated_windows)
DISTINCT_FIRST_TOKENS=$(report_value "$REPORT" overall.distinct_first_generated_tokens)
OVERALL_MODE_ID=$(report_value "$REPORT" overall.most_common_first_generated_token_id)
OVERALL_MODE_COUNT=$(report_value "$REPORT" overall.most_common_first_generated_token_count)
OVERALL_MODE_RATE=$(report_value "$REPORT" overall.most_common_first_generated_token_rate)
REPORT_SHA=$(sha256_hex "$REPORT")

TRAIN_RECORDS=$(report_value "$REPORT" train.record_count)
VALIDATION_RECORDS=$(report_value "$REPORT" validation.record_count)
TEST_RECORDS=$(report_value "$REPORT" test.record_count)
if [ "$TRAIN_RECORDS" != 20 ] || [ "$VALIDATION_RECORDS" != 5 ] ||
    [ "$TEST_RECORDS" != 5 ]; then
    fail "FAIL: report split record counts mismatch"
fi

for scope in overall train validation test; do
    for metric in record_count target_window_match_count target_window_match_rate \
        terminator_generated_count terminator_generated_rate \
        exact_response_match_count exact_response_match_rate \
        distinct_generated_windows distinct_first_generated_tokens \
        most_common_first_generated_token_id most_common_first_generated_token_count \
        most_common_first_generated_token_rate; do
        METRIC_VALUE=$(report_value "$REPORT" "$scope.$metric")
        if [ -z "$METRIC_VALUE" ]; then
            fail "FAIL: missing aggregate field $scope.$metric"
        fi
    done
done

if [ "$TOTAL_RECORDS" != 30 ] ||
    [ -z "$OVERALL_WINDOW_MATCH" ] || [ -z "$OVERALL_TERMINATORS" ] ||
    [ -z "$OVERALL_EXACT" ] || [ -z "$TRAIN_WINDOW_MATCH" ] ||
    [ -z "$VALIDATION_WINDOW_MATCH" ] || [ -z "$TEST_WINDOW_MATCH" ] ||
    [ -z "$DISTINCT_WINDOWS" ] || [ -z "$DISTINCT_FIRST_TOKENS" ] ||
    [ -z "$OVERALL_MODE_ID" ] || [ -z "$OVERALL_MODE_COUNT" ] ||
    [ -z "$OVERALL_MODE_RATE" ] ||
    [ -z "$REPORT_SHA" ]; then
    fail "FAIL: generation report is missing required fields"
fi

log "adapter_loaded=$CONFIG_ADAPTER"
log "temperature=$CONFIG_TEMPERATURE"
log "terminator_token_id=$CONFIG_TERMINATOR"
log "max_visible_generated_tokens=$CONFIG_MAX_TOKENS"
log "prompt_context=$CONFIG_CONTEXT"
log "record021_task=$RECORD021_TASK"
log "record022_present=$RECORD022_PRESENT"
log "record031_present=$RECORD031_PRESENT"
log "record031_split=$RECORD031_SPLIT"
log "record031_task=$RECORD031_TASK"
log "total_records=$TOTAL_RECORDS"
log "overall_target_window_match_count=$OVERALL_WINDOW_MATCH"
log "overall_terminator_generated_count=$OVERALL_TERMINATORS"
log "overall_exact_response_match_count=$OVERALL_EXACT"
log "train_target_window_match_count=$TRAIN_WINDOW_MATCH"
log "validation_target_window_match_count=$VALIDATION_WINDOW_MATCH"
log "test_target_window_match_count=$TEST_WINDOW_MATCH"
log "distinct_generated_windows=$DISTINCT_WINDOWS"
log "distinct_first_generated_tokens=$DISTINCT_FIRST_TOKENS"
log "most_common_first_generated_token_id=$OVERALL_MODE_ID"
log "most_common_first_generated_token_count=$OVERALL_MODE_COUNT"
log "most_common_first_generated_token_rate=$OVERALL_MODE_RATE"
log "generation_report_sha256=$REPORT_SHA"
log "Stage4E-B3R-A4=PASS"

cleanup
exit 0
