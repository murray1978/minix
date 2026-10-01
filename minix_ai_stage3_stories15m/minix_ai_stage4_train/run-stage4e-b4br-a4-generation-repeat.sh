#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" 2>/dev/null && pwd)
cd "$SCRIPT_DIR" || exit 1

BASE_CHECKPOINT="/usr/src/minix_ai_stage3_stories15m/stories15M.bin"
TOKENIZER="/usr/src/minix_ai_stage3_stories15m/tokenizer.bin"
DATASET="stage4e-approved.records"
ACCEPTED_REPORT="stage4e-baseline-unadapted-generation-a4-report.txt"
RUN1="stage4e-b4br-a4-generation-run1.txt"
RUN2="stage4e-b4br-a4-generation-run2.txt"
RESULT_REPORT="stage4e-b4br-a4-generation-repeatability-report.txt"
CONSOLE_LOG="stage4e-b4br-a4-generation-repeat-console.txt"
BUILD_LOG=".stage4e-b4br-build.tmp"
RUN_LOG=".stage4e-b4br-run.tmp"
DIFF_12=".stage4e-b4br-diff-12.tmp"
DIFF_1A=".stage4e-b4br-diff-1a.tmp"
DIFF_2A=".stage4e-b4br-diff-2a.tmp"
ALL_DIFFS=".stage4e-b4br-all-diffs.tmp"
TOKEN_DIFF_IDS=".stage4e-b4br-token-diff-ids.tmp"
RECORD_DIFF_IDS=".stage4e-b4br-record-diff-ids.tmp"
FIELD_DIFFS=".stage4e-b4br-field-diffs.tmp"

EXPECTED_BASE="cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a"
EXPECTED_TOKENIZER="50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361"
EXPECTED_DATASET="71744cea4de049e2e192cb2a4494c8a4a68e769b3a7f3b8ed4880433e4583b84"
EXPECTED_ACCEPTED_REPORT="289c7bf9f1b6bcd002797ac1f8336810cdbd08878f1a52776fede089355a51f5"

log() {
    printf '%s\n' "$*"
    printf '%s\n' "$*" >> "$CONSOLE_LOG"
}

cleanup() {
    rm -f "$BUILD_LOG" "$RUN_LOG" \
        "$DIFF_12" "$DIFF_1A" "$DIFF_2A" "$ALL_DIFFS" \
        "$TOKEN_DIFF_IDS" "$RECORD_DIFF_IDS" "$FIELD_DIFFS"
}

fail() {
    log "$*"
    log "Stage4E-B4bR-A4=FAIL"
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

compare_reports() {
    awk '
    function get_key(line, equal_pos) {
        equal_pos = index(line, "=")
        if (equal_pos == 0) return "<malformed-line>"
        return substr(line, 1, equal_pos - 1)
    }
    function get_value(line, equal_pos) {
        equal_pos = index(line, "=")
        if (equal_pos == 0) return line
        return substr(line, equal_pos + 1)
    }
    FNR == NR {
        left_line[FNR] = $0
        left_key[FNR] = get_key($0)
        key = left_key[FNR]
        if (key == "record.record_id" || key == "record.id")
            left_record = get_value($0)
        left_record_for_line[FNR] = left_record
        left_lines = FNR
        next
    }
    {
        key = get_key($0)
        if (key == "record.record_id" || key == "record.id")
            right_record = get_value($0)
        right_lines = FNR
        if (FNR > left_lines || left_line[FNR] != $0) {
            leftkey = (FNR <= left_lines) ? left_key[FNR] : "<missing>"
            rightkey = key
            if (leftkey != "<missing>") different[leftkey] = 1
            if (rightkey != "<malformed-line>") different[rightkey] = 1
            if (leftkey ~ /^record\./ || rightkey ~ /^record\./) {
                lid = left_record_for_line[FNR]
                if (lid != "") differing_records[lid] = 1
                if (right_record != "") differing_records[right_record] = 1
                if (leftkey == "record.generated_token_ids" ||
                    rightkey == "record.generated_token_ids") {
                    if (lid != "") token_mismatch_records[lid] = 1
                    if (right_record != "") token_mismatch_records[right_record] = 1
                }
            }
        }
    }
    END {
        if (left_lines != right_lines)
            different["report_line_count"] = 1
        for (key in different) print "field\t" key
        for (id in differing_records) print "record_id\t" id
        for (id in token_mismatch_records) print "token_id_mismatch\t" id
    }
    ' "$1" "$2" > "$3"
}

max_loss_difference() {
    awk -F= '
    FNR == NR { left[$1] = $2; left_seen[$1] = 1; next }
    {
        if (($1 ~ /total_target_loss$/ || $1 ~ /average_target_loss$/) &&
            ($1 in left_seen)) {
            delta = left[$1] - $2
            if (delta < 0) delta = -delta
            if (delta > maximum) maximum = delta
        }
    }
    END { printf "%.12f\n", maximum }
    ' "$1" "$2"
}

max_token_count_difference() {
    awk -F= '
    FNR == NR { left[$1] = $2; left_seen[$1] = 1; next }
    {
        if ($1 ~ /target_top1_correct$/ && ($1 in left_seen)) {
            delta = left[$1] - $2
            if (delta < 0) delta = -delta
            if (delta > maximum) maximum = delta
        }
    }
    END { printf "%d\n", maximum }
    ' "$1" "$2"
}

: > "$CONSOLE_LOG" || exit 1
log "========================================"
log "Stage 4E-B4bR-A4 final generation repeatability"
log "========================================"

check_file "$BASE_CHECKPOINT"
check_file "$TOKENIZER"
check_file "$DATASET"
check_file "$ACCEPTED_REPORT"
check_file "Makefile"

BASE_ACTUAL=$(sha256_hex "$BASE_CHECKPOINT")
TOKENIZER_ACTUAL=$(sha256_hex "$TOKENIZER")
DATASET_ACTUAL=$(sha256_hex "$DATASET")
ACCEPTED_ACTUAL=$(sha256_hex "$ACCEPTED_REPORT")

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
if [ "$ACCEPTED_ACTUAL" != "$EXPECTED_ACCEPTED_REPORT" ]; then
    log "accepted_generation_report_identity=FAIL"
    log "expected=$EXPECTED_ACCEPTED_REPORT"
    log "actual=$ACCEPTED_ACTUAL"
    fail "FAIL: accepted B3R-A4 report identity mismatch"
fi
log "base_checkpoint_identity=PASS"
log "tokenizer_identity=PASS"
log "dataset_identity=PASS"
log "accepted_generation_report_identity=PASS"

make stage4eBaselineGenerationEval CC=clang > "$BUILD_LOG" 2>&1
BUILD_STATUS=$?
cat "$BUILD_LOG" | tee -a "$CONSOLE_LOG"
if [ "$BUILD_STATUS" -ne 0 ]; then
    fail "FAIL: generation evaluator build failed"
fi

./stage4e_baseline_unadapted_generation "$BASE_CHECKPOINT" \
    -d "$DATASET" -z "$TOKENIZER" -o "$RUN1" \
    --max-visible-output-tokens 64 > "$RUN_LOG" 2>&1 || fail "FAIL: generation run 1 failed"
cat "$RUN_LOG" | tee -a "$CONSOLE_LOG"
./stage4e_baseline_unadapted_generation "$BASE_CHECKPOINT" \
    -d "$DATASET" -z "$TOKENIZER" -o "$RUN2" \
    --max-visible-output-tokens 64 > "$RUN_LOG" 2>&1 || fail "FAIL: generation run 2 failed"
cat "$RUN_LOG" | tee -a "$CONSOLE_LOG"
check_file "$RUN1"
check_file "$RUN2"

for report in "$RUN1" "$RUN2"; do
    if [ "$(report_value "$report" adapter_loaded)" != no ] ||
        [ "$(report_value "$report" generation_temperature)" != 0 ] ||
        [ "$(report_value "$report" newline_terminator_token_id)" != 13 ] ||
        [ "$(report_value "$report" max_visible_output_tokens)" != 64 ] ||
        [ "$(report_value "$report" prompt_context)" != full_causal_prefill ]; then
        fail "FAIL: generation configuration mismatch in $report"
    fi
    awk -F= '
    /^record.id=/ {
        id = substr($0, index($0, "=") + 1)
        seen[id] = 1
        records++
        next
    }
    /^record.split=/ && id != "" { split_for[id] = substr($0, index($0, "=") + 1); next }
    /^record.task=/ && id != "" { task_for[id] = substr($0, index($0, "=") + 1); next }
    END {
        if (records != 30 || task_for["sysconf-ai-model-priority-val-021"] != "source_navigation" ||
            ("sysconf-ai-model-quantum-val-022" in seen) ||
            !("sample-zero-temperature-val-031" in seen) ||
            split_for["sample-zero-temperature-val-031"] != "validation" ||
            task_for["sample-zero-temperature-val-031"] != "explanation")
            exit 1
    }
    ' "$report" || fail "FAIL: A4 dataset semantic report checks failed in $report"
done
log "adapter_loaded=no"
log "temperature=0"
log "prompt_context=full_causal_prefill"
log "record021_task=source_navigation"
log "record022_present=no"
log "record031_present=yes"
log "record031_split=validation"
log "record031_task=explanation"

compare_reports "$RUN1" "$RUN2" "$DIFF_12"
compare_reports "$RUN1" "$ACCEPTED_REPORT" "$DIFF_1A"
compare_reports "$RUN2" "$ACCEPTED_REPORT" "$DIFF_2A"
RUN1_SHA=$(sha256_hex "$RUN1")
RUN2_SHA=$(sha256_hex "$RUN2")

if [ "$RUN1_SHA" = "$RUN2_SHA" ] && [ ! -s "$DIFF_12" ]; then
    DETERMINISTIC_MATCH=yes
else
    DETERMINISTIC_MATCH=no
fi
if [ "$RUN1_SHA" = "$EXPECTED_ACCEPTED_REPORT" ] && [ ! -s "$DIFF_1A" ]; then
    ACCEPTED_RUN1_MATCH=yes
else
    ACCEPTED_RUN1_MATCH=no
fi
if [ "$RUN2_SHA" = "$EXPECTED_ACCEPTED_REPORT" ] && [ ! -s "$DIFF_2A" ]; then
    ACCEPTED_RUN2_MATCH=yes
else
    ACCEPTED_RUN2_MATCH=no
fi

cat "$DIFF_12" "$DIFF_1A" "$DIFF_2A" | sort -u > "$ALL_DIFFS"
awk -F '\t' '$1 == "record_id" { print $2 }' "$ALL_DIFFS" | sort -u > "$RECORD_DIFF_IDS"
awk -F '\t' '$1 == "field" { print $2 }' "$ALL_DIFFS" | sort -u > "$FIELD_DIFFS"
awk -F '\t' '$1 == "token_id_mismatch" { print $2 }' "$ALL_DIFFS" | sort -u > "$TOKEN_DIFF_IDS"
DIFFERING_RECORD_COUNT=$(awk 'END { print NR + 0 }' "$RECORD_DIFF_IDS")
TOKEN_MISMATCH_COUNT=$(awk 'END { print NR + 0 }' "$TOKEN_DIFF_IDS")
DIFFERING_RECORD_IDS=$(awk 'BEGIN { ORS = "" } NF { printf "%s%s", separator, $0; separator = "," } END { print "" }' "$RECORD_DIFF_IDS")
DIFFERING_FIELDS=$(awk 'BEGIN { ORS = "" } NF { printf "%s%s", separator, $0; separator = "," } END { print "" }' "$FIELD_DIFFS")
if [ -z "$DIFFERING_RECORD_IDS" ]; then DIFFERING_RECORD_IDS=none; fi
if [ -z "$DIFFERING_FIELDS" ]; then DIFFERING_FIELDS=none; fi

{
    printf 'generation_run1_sha256=%s\n' "$RUN1_SHA"
    printf 'generation_run2_sha256=%s\n' "$RUN2_SHA"
    printf 'accepted_generation_sha256=%s\n' "$ACCEPTED_ACTUAL"
    printf 'deterministic_generation_match=%s\n' "$DETERMINISTIC_MATCH"
    printf 'accepted_generation_run1_match=%s\n' "$ACCEPTED_RUN1_MATCH"
    printf 'accepted_generation_run2_match=%s\n' "$ACCEPTED_RUN2_MATCH"
    printf 'differing_record_count=%s\n' "$DIFFERING_RECORD_COUNT"
    printf 'differing_record_ids=%s\n' "$DIFFERING_RECORD_IDS"
    printf 'differing_fields=%s\n' "$DIFFERING_FIELDS"
    printf 'generated_token_mismatch_count=%s\n' "$TOKEN_MISMATCH_COUNT"
} > "$RESULT_REPORT" || fail "FAIL: cannot write repeatability report"

log "adapter_loaded=no"
log "temperature=0"
log "prompt_context=full_causal_prefill"
log "total_records=30"
log "generation_run1_sha256=$RUN1_SHA"
log "generation_run2_sha256=$RUN2_SHA"
log "deterministic_generation_match=$DETERMINISTIC_MATCH"
log "accepted_generation_run1_match=$ACCEPTED_RUN1_MATCH"
log "accepted_generation_run2_match=$ACCEPTED_RUN2_MATCH"
log "differing_record_count=$DIFFERING_RECORD_COUNT"
log "generated_token_mismatch_count=$TOKEN_MISMATCH_COUNT"
log "differing_fields=$DIFFERING_FIELDS"

if [ "$DETERMINISTIC_MATCH" = yes ] &&
    [ "$ACCEPTED_RUN1_MATCH" = yes ] && [ "$ACCEPTED_RUN2_MATCH" = yes ] &&
    [ "$DIFFERING_RECORD_COUNT" = 0 ] && [ "$TOKEN_MISMATCH_COUNT" = 0 ] &&
    [ "$DIFFERING_FIELDS" = none ]; then
    log "Stage4E-B4bR-A4=PASS"
    cleanup
    exit 0
fi

log "Stage4E-B4bR-A4=FAIL"
cleanup
exit 1
