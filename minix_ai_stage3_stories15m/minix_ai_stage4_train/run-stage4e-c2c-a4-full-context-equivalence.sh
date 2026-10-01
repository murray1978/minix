#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" 2>/dev/null && pwd)
cd "$SCRIPT_DIR" || exit 1

BASE_CHECKPOINT="/usr/src/minix_ai_stage3_stories15m/stories15M.bin"
TOKENIZER="/usr/src/minix_ai_stage3_stories15m/tokenizer.bin"
DATASET="stage4e-approved.records"
CACHE="stage4e-target-cache-a4.bin"
B1R_REPORT="stage4e-baseline-unadapted-full-context-a4-report.txt"
B2R_REPORT="stage4e-baseline-unadapted-full-context-a4-top1-report.txt"
REPORT="stage4e-c2c-a4-full-context-equivalence-report.txt"
CONSOLE_LOG="stage4e-c2c-a4-full-context-equivalence-console.txt"
BUILD_LOG=".stage4e-c2c-a4-build.tmp"
CHECK_LOG=".stage4e-c2c-a4-check.tmp"
PREFLIGHT_LOG=".stage4e-c2c-a4-record0.tmp"

EXPECTED_BASE="cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a"
EXPECTED_TOKENIZER="50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361"
EXPECTED_DATASET="71744cea4de049e2e192cb2a4494c8a4a68e769b3a7f3b8ed4880433e4583b84"
EXPECTED_CACHE="d6288ad3d128c7182c20088aa57a1bca8bcaa1e13314fb0cdbd60eba14278def"
EXPECTED_B1R="b92264d62bfa603d98b7f00580f1e7e0bf6b2c6cefb8e5ec5db7753e17249b2a"
EXPECTED_B2R="1c50c0868358349977d627529ae9f9b04ee155580cbf1e3740d276a5fd968c40"
EXPECTED_KV_CHECKSUM="abb568d3ab418288"

log() {
    printf '%s\n' "$*"
    printf '%s\n' "$*" >> "$CONSOLE_LOG"
}

cleanup() {
    rm -f "$BUILD_LOG" "$CHECK_LOG" "$PREFLIGHT_LOG"
}

fail() {
    log "$*"
    log "Stage4E-C2c-A4=FAIL"
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
log "Stage 4E-C2c-A4 full-context cache equivalence"
log "=============================================="

check_file "$BASE_CHECKPOINT"
check_file "$TOKENIZER"
check_file "$DATASET"
check_file "$CACHE"
check_file "$B1R_REPORT"
check_file "$B2R_REPORT"
check_file "Makefile"

BASE_ACTUAL=$(sha256_hex "$BASE_CHECKPOINT")
TOKENIZER_ACTUAL=$(sha256_hex "$TOKENIZER")
DATASET_ACTUAL=$(sha256_hex "$DATASET")
CACHE_ACTUAL=$(sha256_hex "$CACHE")
B1R_ACTUAL=$(sha256_hex "$B1R_REPORT")
B2R_ACTUAL=$(sha256_hex "$B2R_REPORT")

if [ "$BASE_ACTUAL" != "$EXPECTED_BASE" ] ||
    [ "$TOKENIZER_ACTUAL" != "$EXPECTED_TOKENIZER" ] ||
    [ "$DATASET_ACTUAL" != "$EXPECTED_DATASET" ] ||
    [ "$CACHE_ACTUAL" != "$EXPECTED_CACHE" ] ||
    [ "$B1R_ACTUAL" != "$EXPECTED_B1R" ] ||
    [ "$B2R_ACTUAL" != "$EXPECTED_B2R" ]; then
    log "base_checkpoint_sha256=$BASE_ACTUAL"
    log "tokenizer_sha256=$TOKENIZER_ACTUAL"
    log "dataset_sha256=$DATASET_ACTUAL"
    log "cache_sha256=$CACHE_ACTUAL"
    log "b1r_a4_report_sha256=$B1R_ACTUAL"
    log "b2r_a4_report_sha256=$B2R_ACTUAL"
    fail "FAIL: one or more pinned input identities differ"
fi
log "base_checkpoint_identity=PASS"
log "tokenizer_identity=PASS"
log "dataset_identity=PASS"
log "cache_identity=PASS"
log "b1r_a4_report_identity=PASS"
log "b2r_a4_report_identity=PASS"

make stage4eCacheEquivalenceA4 CC=clang > "$BUILD_LOG" 2>&1
BUILD_STATUS=$?
cat "$BUILD_LOG" | tee -a "$CONSOLE_LOG"
if [ "$BUILD_STATUS" -ne 0 ]; then
    fail "FAIL: A4 equivalence checker build failed"
fi

./stage4e_cache_equivalence_a4 "$BASE_CHECKPOINT" \
    -d "$DATASET" -z "$TOKENIZER" -c "$CACHE" -r "$REPORT" \
    -1 "$B1R_REPORT" -2 "$B2R_REPORT" --record0-context-check \
    > "$PREFLIGHT_LOG" 2>&1
PREFLIGHT_STATUS=$?
cat "$PREFLIGHT_LOG" | tee -a "$CONSOLE_LOG"
if [ "$PREFLIGHT_STATUS" -ne 0 ]; then
    fail "FAIL: record-0 full-context preflight failed; comparison not run"
fi

RECORD0_COUNT=$(report_value "$PREFLIGHT_LOG" record0.sequence_count)
RECORD0_BOUNDARY=$(report_value "$PREFLIGHT_LOG" record0.first_target_index)
RECORD0_CALLS_BEFORE_POSITION=$(report_value "$PREFLIGHT_LOG" record0.forward_calls_before_position_15)
RECORD0_CALLS=$(report_value "$PREFLIGHT_LOG" record0.forward_calls_before_first_target)
RECORD0_POSITION=$(report_value "$PREFLIGHT_LOG" record0.first_supervised_input_position)
RECORD0_TARGET=$(report_value "$PREFLIGHT_LOG" record0.first_supervised_target_token_id)
RECORD0_CHECKSUM=$(report_value "$PREFLIGHT_LOG" record0.kv_checksum_before_position_15)
RECORD0_GATE=$(report_value "$PREFLIGHT_LOG" record0_full_context_check)
if [ "$RECORD0_COUNT" != 19 ] || [ "$RECORD0_BOUNDARY" != 16 ] ||
    [ "$RECORD0_CALLS_BEFORE_POSITION" != 15 ] ||
    [ "$RECORD0_CALLS" != 16 ] || [ "$RECORD0_POSITION" != 15 ] ||
    [ "$RECORD0_TARGET" != 2669 ] ||
    [ "$RECORD0_CHECKSUM" != "$EXPECTED_KV_CHECKSUM" ] ||
    [ "$RECORD0_GATE" != PASS ]; then
    fail "FAIL: record-0 causal-context values differ from the accepted gate"
fi
log "record0_full_context_check=PASS"
log "record0_kv_checksum_before_position_15=$RECORD0_CHECKSUM"

./stage4e_cache_equivalence_a4 "$BASE_CHECKPOINT" \
    -d "$DATASET" -z "$TOKENIZER" -c "$CACHE" -r "$REPORT" \
    -1 "$B1R_REPORT" -2 "$B2R_REPORT" > "$CHECK_LOG" 2>&1
CHECK_STATUS=$?
cat "$CHECK_LOG" | tee -a "$CONSOLE_LOG"
check_file "$REPORT"

REPORT_CACHE=$(report_value "$REPORT" cache_sha256)
REPORT_DATASET=$(report_value "$REPORT" dataset_sha256)
REPORT_BASE=$(report_value "$REPORT" base_checkpoint_sha256)
REPORT_TOKENIZER=$(report_value "$REPORT" tokenizer_sha256)
REPORT_B1R=$(report_value "$REPORT" b1r_a4_report_sha256)
REPORT_B2R=$(report_value "$REPORT" b2r_a4_report_sha256)
EXPECTED_ROWS=$(report_value "$REPORT" expected_cache_rows)
CONSUMED_ROWS=$(report_value "$REPORT" consumed_cache_rows)
REMAINING_ROWS=$(report_value "$REPORT" remaining_cache_rows)
HIDDEN_ROWS=$(report_value "$REPORT" hidden_rows_compared)
HIDDEN_FLOATS=$(report_value "$REPORT" hidden_floats_compared)
HIDDEN_MISMATCHES=$(report_value "$REPORT" hidden_bit_mismatch_count)
HIDDEN_MAX=$(report_value "$REPORT" maximum_hidden_absolute_difference)
LOGIT_ROWS=$(report_value "$REPORT" logit_rows_compared)
LOGIT_FLOATS=$(report_value "$REPORT" logits_compared)
LOGIT_MISMATCHES=$(report_value "$REPORT" logit_bit_mismatch_count)
LOGIT_MAX=$(report_value "$REPORT" maximum_logit_absolute_difference)
META_MISMATCHES=$(report_value "$REPORT" metadata_mismatch_count)
RECORD_MISMATCHES=$(report_value "$REPORT" metadata_record_index_mismatches)
POSITION_MISMATCHES=$(report_value "$REPORT" metadata_input_position_mismatches)
TARGET_MISMATCHES=$(report_value "$REPORT" metadata_target_token_mismatches)
SPLIT_MISMATCHES=$(report_value "$REPORT" metadata_split_id_mismatches)
LOSS_MISMATCHES=$(report_value "$REPORT" per_row_loss_mismatch_count)
LOSS_MAX=$(report_value "$REPORT" maximum_per_row_loss_difference)
TOP1_MISMATCHES=$(report_value "$REPORT" top1_token_mismatch_count)
RECORD0_GATE=$(report_value "$REPORT" record0_full_context_check)
LIVE_B1R=$(report_value "$REPORT" live_b1r_a4_loss_metrics_match)
CACHED_B1R=$(report_value "$REPORT" cached_b1r_a4_loss_metrics_match)
LIVE_B2R=$(report_value "$REPORT" live_b2r_a4_top1_metrics_match)
CACHED_B2R=$(report_value "$REPORT" cached_b2r_a4_top1_metrics_match)
EQUIVALENCE_PASS=$(report_value "$REPORT" equivalence_pass)

if [ "$REPORT_CACHE" != "$EXPECTED_CACHE" ] ||
    [ "$REPORT_DATASET" != "$EXPECTED_DATASET" ] ||
    [ "$REPORT_BASE" != "$EXPECTED_BASE" ] ||
    [ "$REPORT_TOKENIZER" != "$EXPECTED_TOKENIZER" ] ||
    [ "$REPORT_B1R" != "$EXPECTED_B1R" ] ||
    [ "$REPORT_B2R" != "$EXPECTED_B2R" ]; then
    fail "FAIL: equivalence report identities do not match pinned inputs"
fi

CACHE_AFTER=$(sha256_hex "$CACHE")
if [ "$CACHE_AFTER" != "$EXPECTED_CACHE" ]; then
    fail "FAIL: cache identity changed during equivalence checking"
fi
REPORT_SHA=$(sha256_hex "$REPORT")
if [ -z "$REPORT_SHA" ]; then
    fail "FAIL: unable to obtain equivalence report SHA-256"
fi

log "expected_cache_rows=$EXPECTED_ROWS"
log "consumed_cache_rows=$CONSUMED_ROWS"
log "remaining_cache_rows=$REMAINING_ROWS"
log "hidden_rows_compared=$HIDDEN_ROWS"
log "hidden_floats_compared=$HIDDEN_FLOATS"
log "hidden_bit_mismatch_count=$HIDDEN_MISMATCHES"
log "maximum_hidden_absolute_difference=$HIDDEN_MAX"
log "logit_rows_compared=$LOGIT_ROWS"
log "logits_compared=$LOGIT_FLOATS"
log "logit_bit_mismatch_count=$LOGIT_MISMATCHES"
log "maximum_logit_absolute_difference=$LOGIT_MAX"
log "metadata_mismatch_count=$META_MISMATCHES"
log "metadata_record_index_mismatches=$RECORD_MISMATCHES"
log "metadata_input_position_mismatches=$POSITION_MISMATCHES"
log "metadata_target_token_mismatches=$TARGET_MISMATCHES"
log "metadata_split_id_mismatches=$SPLIT_MISMATCHES"
log "per_row_loss_mismatch_count=$LOSS_MISMATCHES"
log "maximum_per_row_loss_difference=$LOSS_MAX"
log "top1_token_mismatch_count=$TOP1_MISMATCHES"
log "record0_full_context_check=$RECORD0_GATE"
log "live_b1r_a4_loss_metrics_match=$LIVE_B1R"
log "cached_b1r_a4_loss_metrics_match=$CACHED_B1R"
log "live_b2r_a4_top1_metrics_match=$LIVE_B2R"
log "cached_b2r_a4_top1_metrics_match=$CACHED_B2R"
log "equivalence_pass=$EQUIVALENCE_PASS"
log "equivalence_report_sha256=$REPORT_SHA"

if [ "$CHECK_STATUS" -eq 0 ] &&
    [ "$EXPECTED_ROWS" = 165 ] && [ "$CONSUMED_ROWS" = 165 ] &&
    [ "$REMAINING_ROWS" = 0 ] && [ "$HIDDEN_ROWS" = 165 ] &&
    [ "$HIDDEN_FLOATS" = 47520 ] && [ "$HIDDEN_MISMATCHES" = 0 ] &&
    [ "$HIDDEN_MAX" = 0 ] && [ "$LOGIT_ROWS" = 165 ] &&
    [ "$LOGIT_FLOATS" = 5280000 ] && [ "$LOGIT_MISMATCHES" = 0 ] &&
    [ "$LOGIT_MAX" = 0 ] && [ "$META_MISMATCHES" = 0 ] &&
    [ "$RECORD_MISMATCHES" = 0 ] && [ "$POSITION_MISMATCHES" = 0 ] &&
    [ "$TARGET_MISMATCHES" = 0 ] && [ "$SPLIT_MISMATCHES" = 0 ] &&
    [ "$LOSS_MISMATCHES" = 0 ] && [ "$LOSS_MAX" = "0.000000000000" ] &&
    [ "$TOP1_MISMATCHES" = 0 ] && [ "$RECORD0_GATE" = PASS ] &&
    [ "$LIVE_B1R" = yes ] && [ "$CACHED_B1R" = yes ] &&
    [ "$LIVE_B2R" = yes ] && [ "$CACHED_B2R" = yes ] &&
    [ "$EQUIVALENCE_PASS" = yes ]; then
    log "Stage4E-C2c-A4=PASS"
    cleanup
    exit 0
fi

fail "FAIL: one or more full-context equivalence acceptance gates failed; no tolerance was applied"
