#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" 2>/dev/null && pwd)
cd "$SCRIPT_DIR" || exit 1

BASE_CHECKPOINT="/usr/src/minix_ai_stage3_stories15m/stories15M.bin"
TOKENIZER="/usr/src/minix_ai_stage3_stories15m/tokenizer.bin"
DATASET="stage4e-approved.records"
CACHE="stage4e-target-cache-a4.bin"
D1_REPORT="stage4e-d1-a4-zero-update-trainer-report.txt"
REPORT="stage4e-d2-a4-one-step-report.txt"
CONSOLE_LOG="stage4e-d2-a4-one-step-console.txt"
BUILD_LOG=".stage4e-d2-a4-build.tmp"
RUN_LOG=".stage4e-d2-a4-run.tmp"

EXPECTED_BASE="cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a"
EXPECTED_TOKENIZER="50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361"
EXPECTED_DATASET="71744cea4de049e2e192cb2a4494c8a4a68e769b3a7f3b8ed4880433e4583b84"
EXPECTED_CACHE="d6288ad3d128c7182c20088aa57a1bca8bcaa1e13314fb0cdbd60eba14278def"
EXPECTED_D1_REPORT="45ebaa750f17d02d80feed332b16c9c29c3782a544aed7544da72a62161c76a6"

log() {
    printf '%s\n' "$*"
    printf '%s\n' "$*" >> "$CONSOLE_LOG"
}

cleanup() {
    rm -f "$BUILD_LOG" "$RUN_LOG"
}

fail() {
    log "$*"
    log "Stage4E-D2-A4=FAIL"
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

numeric_positive() {
    awk -v value="$1" 'BEGIN { exit !((value + 0) > 0) }'
}

train_improved() {
    awk -v post="$1" -v pre="$2" 'BEGIN { exit !((post + 0) < (pre + 0)) }'
}

clip_rule_matches() {
    awk -v before="$1" -v after="$2" 'BEGIN {
        before += 0
        after += 0
        if (before <= 1.0)
            exit !(after == before)
        delta = after - 1.0
        if (delta < 0)
            delta = -delta
        exit !(delta <= 0.000001)
    }'
}

: > "$CONSOLE_LOG" || exit 1
log "=============================================="
log "Stage 4E-D2-A4 single train-only Adam update"
log "=============================================="

check_file "$BASE_CHECKPOINT"
check_file "$TOKENIZER"
check_file "$DATASET"
check_file "$CACHE"
check_file "$D1_REPORT"
check_file "Makefile"

BASE_ACTUAL=$(sha256_hex "$BASE_CHECKPOINT")
TOKENIZER_ACTUAL=$(sha256_hex "$TOKENIZER")
DATASET_ACTUAL=$(sha256_hex "$DATASET")
CACHE_ACTUAL=$(sha256_hex "$CACHE")
D1_REPORT_ACTUAL=$(sha256_hex "$D1_REPORT")
if [ "$BASE_ACTUAL" != "$EXPECTED_BASE" ] ||
    [ "$TOKENIZER_ACTUAL" != "$EXPECTED_TOKENIZER" ] ||
    [ "$DATASET_ACTUAL" != "$EXPECTED_DATASET" ] ||
    [ "$CACHE_ACTUAL" != "$EXPECTED_CACHE" ] ||
    [ "$D1_REPORT_ACTUAL" != "$EXPECTED_D1_REPORT" ]; then
    log "base_checkpoint_sha256=$BASE_ACTUAL"
    log "tokenizer_sha256=$TOKENIZER_ACTUAL"
    log "dataset_sha256=$DATASET_ACTUAL"
    log "cache_sha256=$CACHE_ACTUAL"
    log "d1_report_sha256=$D1_REPORT_ACTUAL"
    fail "FAIL: one or more pinned D2 input identities differ"
fi
log "base_checkpoint_identity=PASS"
log "tokenizer_identity=PASS"
log "dataset_identity=PASS"
log "cache_identity=PASS"
log "d1_report_identity=PASS"

make stage4eD2A4OneStep CC=clang > "$BUILD_LOG" 2>&1
BUILD_STATUS=$?
cat "$BUILD_LOG" | tee -a "$CONSOLE_LOG"
if [ "$BUILD_STATUS" -ne 0 ]; then
    fail "FAIL: D2 A4 one-step executable build failed"
fi

./stage4e_d2_a4_one_step "$BASE_CHECKPOINT" \
    -z "$TOKENIZER" -d "$DATASET" -c "$CACHE" \
    -1 "$D1_REPORT" -r "$REPORT" > "$RUN_LOG" 2>&1
RUN_STATUS=$?
cat "$RUN_LOG" | tee -a "$CONSOLE_LOG"
if [ "$RUN_STATUS" -ne 0 ]; then
    fail "FAIL: D2 one-step execution failed"
fi
check_file "$REPORT"

REPORT_DATASET=$(report_value "$REPORT" dataset_sha256)
REPORT_CACHE=$(report_value "$REPORT" cache_sha256)
REPORT_BASE=$(report_value "$REPORT" base_checkpoint_sha256)
REPORT_TOKENIZER=$(report_value "$REPORT" tokenizer_sha256)
REPORT_D1=$(report_value "$REPORT" d1_report_sha256)
RANK=$(report_value "$REPORT" rank)
SCALE=$(report_value "$REPORT" scale)
PARAMETERS=$(report_value "$REPORT" adapter_parameter_count)
LEARNING_RATE=$(report_value "$REPORT" learning_rate)
BETA1=$(report_value "$REPORT" beta1)
BETA2=$(report_value "$REPORT" beta2)
EPSILON=$(report_value "$REPORT" epsilon)
WEIGHT_DECAY=$(report_value "$REPORT" weight_decay)
CLIP_THRESHOLD=$(report_value "$REPORT" gradient_clip_threshold)
GRADIENT_TRAIN_ROWS=$(report_value "$REPORT" gradient_train_rows)
GRADIENT_VALIDATION_ROWS=$(report_value "$REPORT" gradient_validation_rows)
GRADIENT_TEST_ROWS=$(report_value "$REPORT" gradient_test_rows)
GRAD_A_NONZERO=$(report_value "$REPORT" grad_A_nonzero_elements)
GRAD_A_MAX=$(report_value "$REPORT" grad_A_max_abs)
GRAD_B_NONZERO=$(report_value "$REPORT" grad_B_nonzero_elements)
GRAD_B_MAX=$(report_value "$REPORT" grad_B_max_abs)
NORM_BEFORE=$(report_value "$REPORT" gradient_norm_before_clip)
NORM_AFTER=$(report_value "$REPORT" gradient_norm_after_clip)
CLIP_APPLIED=$(report_value "$REPORT" gradient_clip_applied)
CLIP_CONSISTENCY=$(report_value "$REPORT" gradient_clip_consistency)
A_CHANGED=$(report_value "$REPORT" adapter_A_changed)
B_CHANGED=$(report_value "$REPORT" adapter_B_changed)
A_DELTA=$(report_value "$REPORT" adapter_A_max_abs_delta)
B_DELTA=$(report_value "$REPORT" adapter_B_max_abs_delta)
OPTIMIZER_BEFORE=$(report_value "$REPORT" optimizer_steps_before)
OPTIMIZER_AFTER=$(report_value "$REPORT" optimizer_steps_after)
M1_A=$(report_value "$REPORT" m1_A_nonzero_elements)
M2_A=$(report_value "$REPORT" m2_A_nonzero_elements)
M1_B=$(report_value "$REPORT" m1_B_nonzero_elements)
M2_B=$(report_value "$REPORT" m2_B_nonzero_elements)
PREUPDATE_MATCH=$(report_value "$REPORT" preupdate_d1_metrics_match)
PRE_TRAIN_COUNT=$(report_value "$REPORT" pre_train_target_count)
PRE_TRAIN_TOTAL=$(report_value "$REPORT" pre_train_total_loss)
PRE_TRAIN_AVERAGE=$(report_value "$REPORT" pre_train_average_loss)
PRE_TRAIN_TOP1=$(report_value "$REPORT" pre_train_top1_correct)
PRE_VALIDATION_COUNT=$(report_value "$REPORT" pre_validation_target_count)
PRE_VALIDATION_TOTAL=$(report_value "$REPORT" pre_validation_total_loss)
PRE_VALIDATION_AVERAGE=$(report_value "$REPORT" pre_validation_average_loss)
PRE_VALIDATION_TOP1=$(report_value "$REPORT" pre_validation_top1_correct)
PRE_TEST_COUNT=$(report_value "$REPORT" pre_test_target_count)
PRE_TEST_TOTAL=$(report_value "$REPORT" pre_test_total_loss)
PRE_TEST_AVERAGE=$(report_value "$REPORT" pre_test_average_loss)
PRE_TEST_TOP1=$(report_value "$REPORT" pre_test_top1_correct)
POST_TRAIN_COUNT=$(report_value "$REPORT" post_train_target_count)
POST_TRAIN_TOTAL=$(report_value "$REPORT" post_train_total_loss)
POST_TRAIN_AVERAGE=$(report_value "$REPORT" post_train_average_loss)
POST_TRAIN_TOP1=$(report_value "$REPORT" post_train_top1_correct)
POST_VALIDATION_COUNT=$(report_value "$REPORT" post_validation_target_count)
POST_VALIDATION_TOTAL=$(report_value "$REPORT" post_validation_total_loss)
POST_VALIDATION_AVERAGE=$(report_value "$REPORT" post_validation_average_loss)
POST_VALIDATION_TOP1=$(report_value "$REPORT" post_validation_top1_correct)
POST_TEST_COUNT=$(report_value "$REPORT" post_test_target_count)
POST_TEST_TOTAL=$(report_value "$REPORT" post_test_total_loss)
POST_TEST_AVERAGE=$(report_value "$REPORT" post_test_average_loss)
POST_TEST_TOP1=$(report_value "$REPORT" post_test_top1_correct)
TRAIN_DELTA=$(report_value "$REPORT" train_loss_delta)
VALIDATION_DELTA=$(report_value "$REPORT" validation_loss_delta)
TEST_DELTA=$(report_value "$REPORT" test_loss_delta)
POST_ZERO_ROWS=$(report_value "$REPORT" postupdate_zero_correction_rows)
POST_NONZERO_ROWS=$(report_value "$REPORT" postupdate_nonzero_correction_rows)
POST_EVAL_COMPLETED=$(report_value "$REPORT" post_update_evaluation_completed)

if [ "$REPORT_DATASET" != "$EXPECTED_DATASET" ] ||
    [ "$REPORT_CACHE" != "$EXPECTED_CACHE" ] ||
    [ "$REPORT_BASE" != "$EXPECTED_BASE" ] ||
    [ "$REPORT_TOKENIZER" != "$EXPECTED_TOKENIZER" ] ||
    [ "$REPORT_D1" != "$EXPECTED_D1_REPORT" ]; then
    fail "FAIL: D2 report identities differ from pinned inputs"
fi

BASE_AFTER=$(sha256_hex "$BASE_CHECKPOINT")
TOKENIZER_AFTER=$(sha256_hex "$TOKENIZER")
DATASET_AFTER=$(sha256_hex "$DATASET")
CACHE_AFTER=$(sha256_hex "$CACHE")
D1_REPORT_AFTER=$(sha256_hex "$D1_REPORT")
if [ "$BASE_AFTER" != "$EXPECTED_BASE" ] ||
    [ "$TOKENIZER_AFTER" != "$EXPECTED_TOKENIZER" ] ||
    [ "$DATASET_AFTER" != "$EXPECTED_DATASET" ] ||
    [ "$CACHE_AFTER" != "$EXPECTED_CACHE" ] ||
    [ "$D1_REPORT_AFTER" != "$EXPECTED_D1_REPORT" ]; then
    fail "FAIL: a pinned D2 input identity changed during evaluation"
fi
REPORT_SHA=$(sha256_hex "$REPORT")
if [ -z "$REPORT_SHA" ]; then
    fail "FAIL: unable to obtain D2 report SHA-256"
fi

EXPECTED_CLIP_APPLIED=$(awk -v norm="$NORM_BEFORE" 'BEGIN {
    if ((norm + 0) > 1.0) print "yes"
    else print "no"
}')
if [ "$CLIP_APPLIED" != "$EXPECTED_CLIP_APPLIED" ] ||
    ! clip_rule_matches "$NORM_BEFORE" "$NORM_AFTER"; then
    fail "FAIL: clipping flag or clipped norm does not match the Stage 4 clipping rule"
fi
if ! numeric_positive "$GRAD_B_NONZERO" ||
    ! numeric_positive "$GRAD_B_MAX" ||
    ! numeric_positive "$NORM_BEFORE" ||
    ! numeric_positive "$NORM_AFTER" ||
    ! numeric_positive "$B_DELTA" ||
    ! numeric_positive "$M1_B" ||
    ! numeric_positive "$M2_B" ||
    ! numeric_positive "$POST_NONZERO_ROWS"; then
    fail "FAIL: a required positive gradient, B update/moment, or correction count is zero"
fi
if ! train_improved "$POST_TRAIN_AVERAGE" "$PRE_TRAIN_AVERAGE"; then
    fail "FAIL: post-update train average loss did not improve"
fi

if [ "$RANK" != 8 ] || [ "$SCALE" != 1.0 ] ||
    [ "$PARAMETERS" != 258304 ] ||
    [ "$LEARNING_RATE" != 0.001 ] || [ "$BETA1" != 0.9 ] ||
    [ "$BETA2" != 0.999 ] || [ "$EPSILON" != 1e-08 ] ||
    [ "$WEIGHT_DECAY" != 0 ] || [ "$CLIP_THRESHOLD" != 1.0 ] ||
    [ "$GRADIENT_TRAIN_ROWS" != 111 ] ||
    [ "$GRADIENT_VALIDATION_ROWS" != 0 ] || [ "$GRADIENT_TEST_ROWS" != 0 ] ||
    [ "$GRAD_A_NONZERO" != 0 ] || [ "$GRAD_A_MAX" != 0 ] ||
    [ "$A_CHANGED" != no ] || [ "$B_CHANGED" != yes ] ||
    [ "$A_DELTA" != 0 ] || [ "$OPTIMIZER_BEFORE" != 0 ] ||
    [ "$OPTIMIZER_AFTER" != 1 ] || [ "$M1_A" != 0 ] || [ "$M2_A" != 0 ] ||
    [ "$PREUPDATE_MATCH" != yes ] || [ "$CLIP_CONSISTENCY" != yes ] ||
    [ "$PRE_TRAIN_COUNT" != 111 ] ||
    [ "$PRE_TRAIN_TOTAL" != 1158.750195944566 ] ||
    [ "$PRE_TRAIN_AVERAGE" != 10.439190954456 ] || [ "$PRE_TRAIN_TOP1" != 0 ] ||
    [ "$PRE_VALIDATION_COUNT" != 33 ] ||
    [ "$PRE_VALIDATION_TOTAL" != 300.506654007981 ] ||
    [ "$PRE_VALIDATION_AVERAGE" != 9.106262242666 ] ||
    [ "$PRE_VALIDATION_TOP1" != 2 ] ||
    [ "$PRE_TEST_COUNT" != 21 ] ||
    [ "$PRE_TEST_TOTAL" != 223.961209448908 ] ||
    [ "$PRE_TEST_AVERAGE" != 10.664819497567 ] || [ "$PRE_TEST_TOP1" != 0 ] ||
    [ "$POST_TRAIN_COUNT" != 111 ] || [ "$POST_VALIDATION_COUNT" != 33 ] ||
    [ "$POST_TEST_COUNT" != 21 ] || [ -z "$POST_TRAIN_TOTAL" ] ||
    [ -z "$POST_TRAIN_AVERAGE" ] ||
    [ -z "$POST_TRAIN_TOP1" ] || [ -z "$POST_VALIDATION_TOTAL" ] ||
    [ -z "$POST_VALIDATION_AVERAGE" ] ||
    [ -z "$POST_VALIDATION_TOP1" ] || [ -z "$POST_TEST_TOTAL" ] ||
    [ -z "$POST_TEST_AVERAGE" ] || [ -z "$POST_TEST_TOP1" ] ||
    [ -z "$POST_ZERO_ROWS" ] || [ -z "$TRAIN_DELTA" ] ||
    [ -z "$VALIDATION_DELTA" ] || [ -z "$TEST_DELTA" ] ||
    [ "$POST_EVAL_COMPLETED" != yes ]; then
    fail "FAIL: one or more D2 one-step acceptance fields do not match"
fi

log "base_checkpoint_identity=PASS"
log "tokenizer_identity=PASS"
log "dataset_identity=PASS"
log "cache_identity=PASS"
log "d1_report_identity=PASS"
log "rank=$RANK"
log "scale=$SCALE"
log "adapter_parameter_count=$PARAMETERS"
log "gradient_train_rows=$GRADIENT_TRAIN_ROWS"
log "gradient_validation_rows=$GRADIENT_VALIDATION_ROWS"
log "gradient_test_rows=$GRADIENT_TEST_ROWS"
log "grad_A_nonzero_elements=$GRAD_A_NONZERO"
log "grad_A_max_abs=$GRAD_A_MAX"
log "grad_B_nonzero_elements=$GRAD_B_NONZERO"
log "grad_B_max_abs=$GRAD_B_MAX"
log "gradient_norm_before_clip=$NORM_BEFORE"
log "gradient_norm_after_clip=$NORM_AFTER"
log "gradient_clip_applied=$CLIP_APPLIED"
log "adapter_A_changed=$A_CHANGED"
log "adapter_B_changed=$B_CHANGED"
log "adapter_A_max_abs_delta=$A_DELTA"
log "adapter_B_max_abs_delta=$B_DELTA"
log "optimizer_steps_before=$OPTIMIZER_BEFORE"
log "optimizer_steps_after=$OPTIMIZER_AFTER"
log "m1_A_nonzero_elements=$M1_A"
log "m2_A_nonzero_elements=$M2_A"
log "m1_B_nonzero_elements=$M1_B"
log "m2_B_nonzero_elements=$M2_B"
log "pre_train_average_loss=$PRE_TRAIN_AVERAGE"
log "post_train_average_loss=$POST_TRAIN_AVERAGE"
log "train_loss_delta=$TRAIN_DELTA"
log "pre_validation_average_loss=$PRE_VALIDATION_AVERAGE"
log "post_validation_average_loss=$POST_VALIDATION_AVERAGE"
log "validation_loss_delta=$VALIDATION_DELTA"
log "pre_test_average_loss=$PRE_TEST_AVERAGE"
log "post_test_average_loss=$POST_TEST_AVERAGE"
log "test_loss_delta=$TEST_DELTA"
log "postupdate_nonzero_correction_rows=$POST_NONZERO_ROWS"
log "preupdate_d1_metrics_match=$PREUPDATE_MATCH"
log "d2_report_sha256=$REPORT_SHA"
log "Stage4E-D2-A4=PASS"

cleanup
exit 0
