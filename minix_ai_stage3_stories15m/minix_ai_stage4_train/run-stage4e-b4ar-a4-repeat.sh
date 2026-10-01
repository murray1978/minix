#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" 2>/dev/null && pwd)
cd "$SCRIPT_DIR" || exit 1

BASE_CHECKPOINT="/usr/src/minix_ai_stage3_stories15m/stories15M.bin"
TOKENIZER="/usr/src/minix_ai_stage3_stories15m/tokenizer.bin"
DATASET="stage4e-approved.records"
B1_REPORT="stage4e-baseline-unadapted-full-context-a4-report.txt"
B2_REPORT="stage4e-baseline-unadapted-full-context-a4-top1-report.txt"
B1_RUN1="stage4e-b4ar-a4-b1-run1.txt"
B1_RUN2="stage4e-b4ar-a4-b1-run2.txt"
B2_RUN1="stage4e-b4ar-a4-b2-run1.txt"
B2_RUN2="stage4e-b4ar-a4-b2-run2.txt"
RESULT_REPORT="stage4e-b4ar-a4-repeatability-report.txt"
CONSOLE_LOG="stage4e-b4ar-a4-repeat-console.txt"
BUILD_B1_LOG=".stage4e-b4ar-a4-build-b1.tmp"
BUILD_B2_LOG=".stage4e-b4ar-a4-build-b2.tmp"
CHECK_LOG=".stage4e-b4ar-a4-record0-check.tmp"
RUN_LOG=".stage4e-b4ar-a4-evaluator.tmp"
B1_FIELDS1=".stage4e-b4ar-a4-b1-fields1.tmp"
B1_FIELDS2=".stage4e-b4ar-a4-b1-fields2.tmp"
B1_ACCEPTED=".stage4e-b4ar-a4-b1-accepted.tmp"
B2_FIELDS1=".stage4e-b4ar-a4-b2-fields1.tmp"
B2_FIELDS2=".stage4e-b4ar-a4-b2-fields2.tmp"
B2_ACCEPTED=".stage4e-b4ar-a4-b2-accepted.tmp"
DIFF_B1=".stage4e-b4ar-a4-diff-b1.tmp"
DIFF_B1_ACCEPTED=".stage4e-b4ar-a4-diff-b1-accepted.tmp"
DIFF_B2=".stage4e-b4ar-a4-diff-b2.tmp"
DIFF_B2_ACCEPTED=".stage4e-b4ar-a4-diff-b2-accepted.tmp"
ALL_DIFFS=".stage4e-b4ar-a4-all-diffs.tmp"

EXPECTED_BASE="cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a"
EXPECTED_TOKENIZER="50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361"
EXPECTED_DATASET="71744cea4de049e2e192cb2a4494c8a4a68e769b3a7f3b8ed4880433e4583b84"
EXPECTED_B1_REPORT="b92264d62bfa603d98b7f00580f1e7e0bf6b2c6cefb8e5ec5db7753e17249b2a"
EXPECTED_B2_REPORT="1c50c0868358349977d627529ae9f9b04ee155580cbf1e3740d276a5fd968c40"
EXPECTED_KV_CHECKSUM="abb568d3ab418288"

log() {
    printf '%s\n' "$*"
    printf '%s\n' "$*" >> "$CONSOLE_LOG"
}

cleanup() {
    rm -f "$BUILD_B1_LOG" "$BUILD_B2_LOG" "$CHECK_LOG" "$RUN_LOG" \
        "$B1_FIELDS1" "$B1_FIELDS2" "$B1_ACCEPTED" \
        "$B2_FIELDS1" "$B2_FIELDS2" "$B2_ACCEPTED" \
        "$DIFF_B1" "$DIFF_B1_ACCEPTED" "$DIFF_B2" "$DIFF_B2_ACCEPTED" "$ALL_DIFFS"
}

fail() {
    log "$*"
    log "Stage4E-B4aR-A4=FAIL"
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

normalize_b1() {
    awk -F= '
    /^(mode|dataset_sha256|dataset_bytes|base_checkpoint_sha256|base_checkpoint_sha256_expected|tokenizer_sha256|tokenizer_sha256_expected|adapter_loaded|evaluation_updates_parameters|test_split_used_for_selection|validation_baseline_loss|validation_required_relative_improvement|validation_acceptance_loss_max|total_records|total_target_prediction_tokens|total_target_loss|average_target_loss|target_perplexity)=/ { print }
    /^split\.(train|validation|test)\.(records|target_prediction_tokens|total_target_loss|average_target_loss|target_perplexity)=/ { print }
    ' "$1" | sort
}

normalize_b2() {
    awk -F= '
    /^(mode|dataset_sha256|dataset_bytes|base_checkpoint_sha256|base_checkpoint_sha256_expected|tokenizer_sha256|tokenizer_sha256_expected|adapter_loaded|evaluation_updates_parameters|test_split_used_for_selection|validation_baseline_loss|validation_required_relative_improvement|validation_acceptance_loss_max|total_records|target_prediction_count|total_target_loss|average_target_loss|target_perplexity|target_top1_correct|target_top1_accuracy)=/ { print }
    /^split\.(train|validation|test)\.(records|target_prediction_count|total_target_loss|average_target_loss|target_perplexity|target_top1_correct|target_top1_accuracy)=/ { print }
    /^split\.(train|validation|test)\.task\.[A-Za-z_]+\.(records|target_prediction_tokens|average_target_loss|target_top1_accuracy)=/ { print }
    ' "$1" | sort
}

compare_fields() {
    awk -F= '
    FNR == NR { left[$1] = $2; left_seen[$1] = 1; next }
    { right[$1] = $2; right_seen[$1] = 1 }
    END {
        for (key in left_seen)
            if (!(key in right_seen) || left[key] != right[key]) different[key] = 1
        for (key in right_seen)
            if (!(key in left_seen)) different[key] = 1
        for (key in different) print key
    }
    ' "$1" "$2" | sort -u > "$3"
}

max_numeric_difference() {
    awk -F= '
    FNR == NR { left[$1] = $2; left_seen[$1] = 1; next }
    {
        if ($1 in left_seen && $1 ~ /(total_target_loss|average_target_loss)$/) {
            delta = left[$1] - $2
            if (delta < 0) delta = -delta
            if (delta > maximum) maximum = delta
        }
    }
    END { printf "%.12f\n", maximum }
    ' "$1" "$2"
}

max_top1_count_difference() {
    awk -F= '
    FNR == NR { left[$1] = $2; left_seen[$1] = 1; next }
    {
        if ($1 in left_seen && $1 ~ /target_top1_correct$/) {
            delta = left[$1] - $2
            if (delta < 0) delta = -delta
            if (delta > maximum) maximum = delta
        }
    }
    END { printf "%d\n", maximum }
    ' "$1" "$2"
}

join_fields() {
    awk 'BEGIN { ORS = "" } NF { printf "%s%s", separator, $0; separator = "," } END { print "" }' "$1"
}

: > "$CONSOLE_LOG" || exit 1
log "========================================"
log "Stage 4E-B4aR-A4 final A4 baseline repeatability"
log "========================================"

check_file "$BASE_CHECKPOINT"
check_file "$TOKENIZER"
check_file "$DATASET"
check_file "$B1_REPORT"
check_file "$B2_REPORT"
check_file "Makefile"

BASE_ACTUAL=$(sha256_hex "$BASE_CHECKPOINT")
TOKENIZER_ACTUAL=$(sha256_hex "$TOKENIZER")
DATASET_ACTUAL=$(sha256_hex "$DATASET")
B1_ACCEPTED_SHA=$(sha256_hex "$B1_REPORT")
B2_ACCEPTED_SHA=$(sha256_hex "$B2_REPORT")

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
if [ "$B1_ACCEPTED_SHA" != "$EXPECTED_B1_REPORT" ]; then
    fail "FAIL: accepted B1R-A4 report identity mismatch: $B1_ACCEPTED_SHA"
fi
if [ "$B2_ACCEPTED_SHA" != "$EXPECTED_B2_REPORT" ]; then
    fail "FAIL: accepted B2R-A4 report identity mismatch: $B2_ACCEPTED_SHA"
fi
log "base_checkpoint_identity=PASS"
log "tokenizer_identity=PASS"
log "dataset_identity=PASS"

make stage4eBaselineFullContextEval CC=clang > "$BUILD_B1_LOG" 2>&1
BUILD_STATUS=$?
cat "$BUILD_B1_LOG" | tee -a "$CONSOLE_LOG"
if [ "$BUILD_STATUS" -ne 0 ]; then
    fail "FAIL: B1R-A4 evaluator build failed"
fi
make stage4eBaselineFullContextTop1Eval CC=clang > "$BUILD_B2_LOG" 2>&1
BUILD_STATUS=$?
cat "$BUILD_B2_LOG" | tee -a "$CONSOLE_LOG"
if [ "$BUILD_STATUS" -ne 0 ]; then
    fail "FAIL: B2R-A4 evaluator build failed"
fi

./stage4e_baseline_full_context_eval "$BASE_CHECKPOINT" -d "$DATASET" \
    -z "$TOKENIZER" --record0-context-check > "$CHECK_LOG" 2>&1
CHECK_STATUS=$?
cat "$CHECK_LOG" | tee -a "$CONSOLE_LOG"
if [ "$CHECK_STATUS" -ne 0 ]; then
    fail "FAIL: record-0 full-context regression check failed"
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
    fail "FAIL: record-0 full-context check mismatch"
fi
log "record0_full_context_check=PASS"
log "record0_kv_checksum_before_position_15=$RECORD0_CHECKSUM"

./stage4e_baseline_full_context_eval "$BASE_CHECKPOINT" -d "$DATASET" \
    -z "$TOKENIZER" -o "$B1_RUN1" > "$RUN_LOG" 2>&1 || fail "FAIL: B1R-A4 run 1 failed"
cat "$RUN_LOG" | tee -a "$CONSOLE_LOG"
./stage4e_baseline_full_context_eval "$BASE_CHECKPOINT" -d "$DATASET" \
    -z "$TOKENIZER" -o "$B1_RUN2" > "$RUN_LOG" 2>&1 || fail "FAIL: B1R-A4 run 2 failed"
cat "$RUN_LOG" | tee -a "$CONSOLE_LOG"
./stage4e_baseline_full_context_top1_eval "$BASE_CHECKPOINT" -d "$DATASET" \
    -z "$TOKENIZER" -o "$B2_RUN1" > "$RUN_LOG" 2>&1 || fail "FAIL: B2R-A4 run 1 failed"
cat "$RUN_LOG" | tee -a "$CONSOLE_LOG"
./stage4e_baseline_full_context_top1_eval "$BASE_CHECKPOINT" -d "$DATASET" \
    -z "$TOKENIZER" -o "$B2_RUN2" > "$RUN_LOG" 2>&1 || fail "FAIL: B2R-A4 run 2 failed"
cat "$RUN_LOG" | tee -a "$CONSOLE_LOG"

check_file "$B1_RUN1"
check_file "$B1_RUN2"
check_file "$B2_RUN1"
check_file "$B2_RUN2"

normalize_b1 "$B1_RUN1" > "$B1_FIELDS1"
normalize_b1 "$B1_RUN2" > "$B1_FIELDS2"
normalize_b1 "$B1_REPORT" > "$B1_ACCEPTED"
normalize_b2 "$B2_RUN1" > "$B2_FIELDS1"
normalize_b2 "$B2_RUN2" > "$B2_FIELDS2"
normalize_b2 "$B2_REPORT" > "$B2_ACCEPTED"
compare_fields "$B1_FIELDS1" "$B1_FIELDS2" "$DIFF_B1"
compare_fields "$B1_FIELDS1" "$B1_ACCEPTED" "$DIFF_B1_ACCEPTED"
compare_fields "$B2_FIELDS1" "$B2_FIELDS2" "$DIFF_B2"
compare_fields "$B2_FIELDS1" "$B2_ACCEPTED" "$DIFF_B2_ACCEPTED"

B1_RUN1_SHA=$(sha256_hex "$B1_RUN1")
B1_RUN2_SHA=$(sha256_hex "$B1_RUN2")
B2_RUN1_SHA=$(sha256_hex "$B2_RUN1")
B2_RUN2_SHA=$(sha256_hex "$B2_RUN2")
if [ -s "$DIFF_B1" ]; then B1_MATCH=no; else B1_MATCH=yes; fi
if [ -s "$DIFF_B2" ]; then B2_MATCH=no; else B2_MATCH=yes; fi
if [ -s "$DIFF_B1_ACCEPTED" ] ||
    [ "$B1_RUN1_SHA" != "$EXPECTED_B1_REPORT" ] ||
    [ "$B1_RUN2_SHA" != "$EXPECTED_B1_REPORT" ]; then
    B1_ACCEPTED_MATCH=no
else
    B1_ACCEPTED_MATCH=yes
fi
if [ -s "$DIFF_B2_ACCEPTED" ] ||
    [ "$B2_RUN1_SHA" != "$EXPECTED_B2_REPORT" ] ||
    [ "$B2_RUN2_SHA" != "$EXPECTED_B2_REPORT" ]; then
    B2_ACCEPTED_MATCH=no
else
    B2_ACCEPTED_MATCH=yes
fi

MAX_B1_REPEAT=$(max_numeric_difference "$B1_FIELDS1" "$B1_FIELDS2")
MAX_B1_ACCEPTED=$(max_numeric_difference "$B1_FIELDS1" "$B1_ACCEPTED")
MAX_B2_REPEAT=$(max_numeric_difference "$B2_FIELDS1" "$B2_FIELDS2")
MAX_B2_ACCEPTED=$(max_numeric_difference "$B2_FIELDS1" "$B2_ACCEPTED")
MAX_LOSS_DIFF=$(awk -v a="$MAX_B1_REPEAT" -v b="$MAX_B1_ACCEPTED" \
    -v c="$MAX_B2_REPEAT" -v d="$MAX_B2_ACCEPTED" \
    'BEGIN { m=a; if (b>m) m=b; if (c>m) m=c; if (d>m) m=d; printf "%.12f\n", m }')
TOP1_COUNT_DIFF=$(max_top1_count_difference "$B2_FIELDS1" "$B2_FIELDS2")
B2_ACCEPTED_COUNT_DIFF=$(max_top1_count_difference "$B2_FIELDS1" "$B2_ACCEPTED")
if [ "$B2_ACCEPTED_COUNT_DIFF" -gt "$TOP1_COUNT_DIFF" ]; then
    TOP1_COUNT_DIFF=$B2_ACCEPTED_COUNT_DIFF
fi
cat "$DIFF_B1" "$DIFF_B1_ACCEPTED" "$DIFF_B2" "$DIFF_B2_ACCEPTED" | sort -u > "$ALL_DIFFS"
DIFFERING_FIELDS=$(join_fields "$ALL_DIFFS")
if [ -z "$DIFFERING_FIELDS" ]; then DIFFERING_FIELDS=none; fi

{
    printf 'b1_run1_sha256=%s\n' "$B1_RUN1_SHA"
    printf 'b1_run2_sha256=%s\n' "$B1_RUN2_SHA"
    printf 'b2_run1_sha256=%s\n' "$B2_RUN1_SHA"
    printf 'b2_run2_sha256=%s\n' "$B2_RUN2_SHA"
    printf 'deterministic_b1_match=%s\n' "$B1_MATCH"
    printf 'deterministic_b2_match=%s\n' "$B2_MATCH"
    printf 'accepted_b1r_a4_match=%s\n' "$B1_ACCEPTED_MATCH"
    printf 'accepted_b2r_a4_match=%s\n' "$B2_ACCEPTED_MATCH"
    printf 'maximum_loss_difference=%s\n' "$MAX_LOSS_DIFF"
    printf 'top1_count_difference=%s\n' "$TOP1_COUNT_DIFF"
    printf 'differing_fields=%s\n' "$DIFFERING_FIELDS"
} > "$RESULT_REPORT" || fail "FAIL: cannot write repeatability report"

log "deterministic_b1_match=$B1_MATCH"
log "deterministic_b2_match=$B2_MATCH"
log "accepted_b1r_a4_match=$B1_ACCEPTED_MATCH"
log "accepted_b2r_a4_match=$B2_ACCEPTED_MATCH"
log "maximum_loss_difference=$MAX_LOSS_DIFF"
log "top1_count_difference=$TOP1_COUNT_DIFF"
log "differing_fields=$DIFFERING_FIELDS"

if [ "$B1_MATCH" = yes ] && [ "$B2_MATCH" = yes ] &&
    [ "$B1_ACCEPTED_MATCH" = yes ] && [ "$B2_ACCEPTED_MATCH" = yes ] &&
    [ "$MAX_LOSS_DIFF" = "0.000000000000" ] &&
    [ "$TOP1_COUNT_DIFF" = 0 ] && [ "$DIFFERING_FIELDS" = none ]; then
    log "Stage4E-B4aR-A4=PASS"
    cleanup
    exit 0
fi

log "Stage4E-B4aR-A4=FAIL"
cleanup
exit 1
