#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" 2>/dev/null && pwd)
cd "$SCRIPT_DIR" || exit 1

BASE_CHECKPOINT="/usr/src/minix_ai_stage3_stories15m/stories15M.bin"
TOKENIZER="/usr/src/minix_ai_stage3_stories15m/tokenizer.bin"
DATASET="stage4e-approved.records"
REPORT="stage4e-dataset-report-a4.txt"
CONSOLE_LOG="stage4e-a4-dataset-repair-console.txt"
BUILD_LOG=".stage4e-a4-build.tmp"
CHECK_LOG=".stage4e-a4-check.tmp"

OLD_DATASET_SHA="b8aa6dc1dbfad7ed144e19a351850189956f79571b05cb70324d21876db0ea78"
EXPECTED_BASE_SHA="cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a"
EXPECTED_TOKENIZER_SHA="50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361"

log() {
    printf '%s\n' "$*"
    printf '%s\n' "$*" >> "$CONSOLE_LOG"
}

cleanup() {
    rm -f "$BUILD_LOG" "$CHECK_LOG"
}

fail() {
    log "$*"
    log "Stage4E-A4=FAIL"
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
log "Stage 4E-A4 semantic dataset repair check"
log "========================================"

check_file "$BASE_CHECKPOINT"
check_file "$TOKENIZER"
check_file "$DATASET"
check_file "Makefile"

BASE_ACTUAL=$(sha256_hex "$BASE_CHECKPOINT")
TOKENIZER_ACTUAL=$(sha256_hex "$TOKENIZER")
DATASET_ACTUAL=$(sha256_hex "$DATASET")

if [ "$BASE_ACTUAL" != "$EXPECTED_BASE_SHA" ]; then
    log "base_checkpoint_identity=FAIL"
    log "expected=$EXPECTED_BASE_SHA"
    log "actual=$BASE_ACTUAL"
    fail "FAIL: checkpoint identity mismatch"
fi
if [ "$TOKENIZER_ACTUAL" != "$EXPECTED_TOKENIZER_SHA" ]; then
    log "tokenizer_identity=FAIL"
    log "expected=$EXPECTED_TOKENIZER_SHA"
    log "actual=$TOKENIZER_ACTUAL"
    fail "FAIL: tokenizer identity mismatch"
fi
if [ "$DATASET_ACTUAL" = "$OLD_DATASET_SHA" ]; then
    fail "FAIL: dataset hash is unchanged; A4 repair was not applied"
fi

log "base_checkpoint_identity=PASS"
log "tokenizer_identity=PASS"
log "dataset_identity=PASS"
log "old_dataset_sha256=$OLD_DATASET_SHA"
log "new_dataset_sha256=$DATASET_ACTUAL"

make stage4eDatasetCheck CC=clang > "$BUILD_LOG" 2>&1
BUILD_STATUS=$?
cat "$BUILD_LOG" | tee -a "$CONSOLE_LOG"
if [ "$BUILD_STATUS" -ne 0 ]; then
    fail "FAIL: Stage 4E dataset checker build failed"
fi

./stage4e_dataset_check -d "$DATASET" -z "$TOKENIZER" \
    -c "$BASE_CHECKPOINT" -o "$REPORT" -v > "$CHECK_LOG" 2>&1
CHECK_STATUS=$?
cat "$CHECK_LOG" | tee -a "$CONSOLE_LOG"
if [ "$CHECK_STATUS" -ne 0 ]; then
    fail "FAIL: Stage 4E dataset checker failed"
fi
check_file "$REPORT"

TOTAL_RECORDS=$(report_value "$REPORT" total_records)
TRAIN_RECORDS=$(report_value "$REPORT" records_by_split.train)
VALIDATION_RECORDS=$(report_value "$REPORT" records_by_split.validation)
TEST_RECORDS=$(report_value "$REPORT" records_by_split.test)
TRAIN_CATEGORIES=$(report_value "$REPORT" distinct_task_categories.train)
VALIDATION_CATEGORIES=$(report_value "$REPORT" distinct_task_categories.validation)
TEST_CATEGORIES=$(report_value "$REPORT" distinct_task_categories.test)
COMMAND_COUNT=$(report_value "$REPORT" records_by_task.command)
COMMAND_REVIEW_COUNT=$(report_value "$REPORT" records_by_task.command_review)
SOURCE_NAV_COUNT=$(report_value "$REPORT" records_by_task.source_navigation)
EXPLANATION_COUNT=$(report_value "$REPORT" records_by_task.explanation)
DIAGNOSIS_COUNT=$(report_value "$REPORT" records_by_task.diagnosis)
CONTINUATION_COUNT=$(report_value "$REPORT" records_by_task.continuation)
VALIDATION_EXPLANATION=$(report_value "$REPORT" records_by_split_task.validation.explanation)
VALIDATION_SOURCE_NAV=$(report_value "$REPORT" records_by_split_task.validation.source_navigation)
VALIDATION_DIAGNOSIS=$(report_value "$REPORT" records_by_split_task.validation.diagnosis)
DUPLICATE_IDS=$(report_value "$REPORT" duplicate_ids)
DUPLICATE_PROMPTS=$(report_value "$REPORT" duplicate_prompts)
DUP_TARGET_SAME=$(report_value "$REPORT" duplicate_target_same_split)
DUP_TARGET_CROSS=$(report_value "$REPORT" duplicate_target_cross_split)
REPEATED_SOURCE_UNITS=$(report_value "$REPORT" repeated_source_units_cross_split)
NEAR_DUPLICATES=$(report_value "$REPORT" possible_cross_split_near_duplicates)
MALFORMED=$(report_value "$REPORT" malformed_records)
CONTEXT_OVERFLOW=$(report_value "$REPORT" context_overflow_records)
NEWLINE_RECORDS=$(report_value "$REPORT" records_with_newline_terminator)
WITHOUT_NEWLINE=$(report_value "$REPORT" records_without_terminator)
TERMINATOR_MISMATCHES=$(report_value "$REPORT" terminator_token_mismatches)
REPORT_SHA=$(sha256_hex "$REPORT")

RECORD021_TASK=$(awk '
/^record id=sysconf-ai-model-priority-val-021 / {
    for (i = 1; i <= NF; i++) if ($i ~ /^task=/) { sub(/^task=/, "", $i); print $i; exit }
}' "$CHECK_LOG")
RECORD022_PRESENT=$(awk '
/^record id=sysconf-ai-model-quantum-val-022 / { found = 1 }
END { print found ? "yes" : "no" }
' "$CHECK_LOG")
RECORD031_TASK=$(awk '
/^record id=sample-zero-temperature-val-031 / {
    for (i = 1; i <= NF; i++) {
        if ($i ~ /^task=/) { sub(/^task=/, "", $i); task = $i }
        if ($i ~ /^split=/) { sub(/^split=/, "", $i); rec_split = $i }
    }
    print "present=yes"
    print "split=" rec_split
    print "task=" task
    found = 1
}
END { if (!found) print "present=no" }
' "$CHECK_LOG")
RECORD031_PRESENT=$(printf '%s\n' "$RECORD031_TASK" | awk -F= '$1 == "present" { print $2 }')
RECORD031_SPLIT=$(printf '%s\n' "$RECORD031_TASK" | awk -F= '$1 == "split" { print $2 }')
RECORD031_TASK_VALUE=$(printf '%s\n' "$RECORD031_TASK" | awk -F= '$1 == "task" { print $2 }')
NEW_VISIBLE_TARGET_COUNT=$(awk '
/^record id=sample-zero-temperature-val-031 / { isnew = 1; next }
isnew && /visible_target_token_count=/ {
    for (i = 1; i <= NF; i++) {
        if ($i ~ /^visible_target_token_count=/) { sub(/^visible_target_token_count=/, "", $i); print $i; exit }
    }
}
' "$CHECK_LOG")
NEW_TRAINING_TARGET_COUNT=$(awk '
/^record id=sample-zero-temperature-val-031 / { isnew = 1; next }
isnew && /training_target_token_count=/ {
    for (i = 1; i <= NF; i++) {
        if ($i ~ /^training_target_token_count=/) { sub(/^training_target_token_count=/, "", $i); print $i; exit }
    }
}
' "$CHECK_LOG")
NEW_SEQUENCE_COUNT=$(awk '
/^record id=sample-zero-temperature-val-031 / { isnew = 1; next }
isnew && /target_prediction_positions:/ {
    line = $0
    sub(/^.*target_prediction_positions:[ ]*/, "", line)
    n = split(line, items, /[ ]+/)
    if (n > 0) {
        last = items[n]
        sub(/^.*target=/, "", last)
        sub(/[^0-9].*$/, "", last)
        print last + 1
    }
    exit
}
' "$CHECK_LOG")

if [ "$RECORD021_TASK" != source_navigation ] ||
    [ "$RECORD022_PRESENT" != no ] ||
    [ "$RECORD031_PRESENT" != yes ] || [ "$RECORD031_SPLIT" != validation ] ||
    [ "$RECORD031_TASK_VALUE" != explanation ]; then
    fail "FAIL: semantic spot check failed for records 021/022/031"
fi
if [ "$NEW_VISIBLE_TARGET_COUNT" != 14 ] ||
    [ "$NEW_TRAINING_TARGET_COUNT" != 15 ] ||
    [ "$NEW_SEQUENCE_COUNT" != 38 ]; then
    fail "FAIL: new record tokenizer count mismatch: visible=$NEW_VISIBLE_TARGET_COUNT training=$NEW_TRAINING_TARGET_COUNT sequence=$NEW_SEQUENCE_COUNT"
fi
if [ "$TOTAL_RECORDS" != 30 ] || [ "$TRAIN_RECORDS" != 20 ] ||
    [ "$VALIDATION_RECORDS" != 5 ] || [ "$TEST_RECORDS" != 5 ] ||
    [ "$COMMAND_COUNT" != 6 ] || [ "$COMMAND_REVIEW_COUNT" != 4 ] ||
    [ "$SOURCE_NAV_COUNT" != 9 ] || [ "$EXPLANATION_COUNT" != 5 ] ||
    [ "$DIAGNOSIS_COUNT" != 4 ] || [ "$CONTINUATION_COUNT" != 2 ] ||
    [ "$TRAIN_CATEGORIES" != 6 ] || [ "$VALIDATION_CATEGORIES" != 3 ] ||
    [ "$TEST_CATEGORIES" != 4 ] ||
    [ "$VALIDATION_EXPLANATION" != 1 ] ||
    [ "$VALIDATION_SOURCE_NAV" != 3 ] ||
    [ "$VALIDATION_DIAGNOSIS" != 1 ]; then
    fail "FAIL: split or task counts do not match A4 expectations"
fi
if [ "$DUPLICATE_IDS" != 0 ] || [ "$DUPLICATE_PROMPTS" != 0 ] ||
    [ "$DUP_TARGET_SAME" != 2 ] || [ "$DUP_TARGET_CROSS" != 2 ] ||
    [ "$REPEATED_SOURCE_UNITS" != 0 ] || [ "$NEAR_DUPLICATES" != 0 ] ||
    [ "$MALFORMED" != 0 ] || [ "$CONTEXT_OVERFLOW" != 0 ] ||
    [ "$NEWLINE_RECORDS" != 30 ] || [ "$WITHOUT_NEWLINE" != 0 ] ||
    [ "$TERMINATOR_MISMATCHES" != 0 ] || [ -z "$REPORT_SHA" ]; then
    fail "FAIL: integrity checks do not match A4 expectations"
fi

log "record021_task=$RECORD021_TASK"
log "record022_present=$RECORD022_PRESENT"
log "record031_present=$RECORD031_PRESENT"
log "record031_split=$RECORD031_SPLIT"
log "record031_task=$RECORD031_TASK_VALUE"
log "record031_visible_target_token_count=$NEW_VISIBLE_TARGET_COUNT"
log "record031_training_target_token_count=$NEW_TRAINING_TARGET_COUNT"
log "record031_sequence_token_count=$NEW_SEQUENCE_COUNT"
log "total_records=$TOTAL_RECORDS"
log "train_records=$TRAIN_RECORDS"
log "validation_records=$VALIDATION_RECORDS"
log "test_records=$TEST_RECORDS"
log "train_task_categories=$TRAIN_CATEGORIES"
log "validation_task_categories=$VALIDATION_CATEGORIES"
log "test_task_categories=$TEST_CATEGORIES"
log "duplicate_ids=$DUPLICATE_IDS"
log "duplicate_prompts=$DUPLICATE_PROMPTS"
log "duplicate_target_same_split=$DUP_TARGET_SAME"
log "duplicate_target_cross_split=$DUP_TARGET_CROSS"
log "repeated_source_units_cross_split=$REPEATED_SOURCE_UNITS"
log "possible_cross_split_near_duplicates=$NEAR_DUPLICATES"
log "malformed_records=$MALFORMED"
log "context_overflow_records=$CONTEXT_OVERFLOW"
log "new_dataset_sha256=$DATASET_ACTUAL"
log "dataset_report_sha256=$REPORT_SHA"
log "Stage4E-A4=PASS"

cleanup
exit 0
