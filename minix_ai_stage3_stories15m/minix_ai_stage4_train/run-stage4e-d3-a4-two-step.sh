#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" 2>/dev/null && pwd)
cd "$SCRIPT_DIR" || exit 1

BASE_CHECKPOINT="/usr/src/minix_ai_stage3_stories15m/stories15M.bin"
TOKENIZER="/usr/src/minix_ai_stage3_stories15m/tokenizer.bin"
DATASET="stage4e-approved.records"
CACHE="stage4e-target-cache-a4.bin"
D1_REPORT="stage4e-d1-a4-zero-update-trainer-report.txt"
D2_REPORT="stage4e-d2-a4-one-step-report.txt"
D3_SOURCE="stage4e_d3_a4_two_step.c"
D3_MAKE_TARGET="stage4eD3A4TwoStep"
D3_EXECUTABLE="stage4e_d3_a4_two_step"
EXPECTED_D3_BUILD_PROVENANCE="stored-double-compare-fix-2026-10-01-c"
REPORT="stage4e-d3-a4-two-step-report.txt"
CONSOLE_LOG="stage4e-d3-a4-two-step-console.txt"
BUILD_LOG=".stage4e-d3-a4-build.tmp"
RUN_LOG=".stage4e-d3-a4-run.tmp"

EXPECTED_BASE="cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a"
EXPECTED_TOKENIZER="50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361"
EXPECTED_DATASET="71744cea4de049e2e192cb2a4494c8a4a68e769b3a7f3b8ed4880433e4583b84"
EXPECTED_CACHE="d6288ad3d128c7182c20088aa57a1bca8bcaa1e13314fb0cdbd60eba14278def"
EXPECTED_D1_REPORT="45ebaa750f17d02d80feed332b16c9c29c3782a544aed7544da72a62161c76a6"
EXPECTED_D2_REPORT="c27aad7474df093f5a3ff781bb3619b71b3e284edda46290f5814c350e99d97e"

log() {
    printf '%s\n' "$*"
    printf '%s\n' "$*" >> "$CONSOLE_LOG"
}

cleanup() {
    rm -f "$BUILD_LOG" "$RUN_LOG"
}

fail() {
    log "$*"
    log "Stage4E-D3-A4=FAIL"
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

train_below() {
    awk -v value="$1" -v bound="$2" 'BEGIN { exit !((value + 0) < (bound + 0)) }'
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
log "Stage 4E-D3-A4 deterministic two-step validation"
log "=============================================="

check_file "$BASE_CHECKPOINT"
check_file "$TOKENIZER"
check_file "$DATASET"
check_file "$CACHE"
check_file "$D1_REPORT"
check_file "$D2_REPORT"
check_file "$D3_SOURCE"
check_file "Makefile"

BASE_ACTUAL=$(sha256_hex "$BASE_CHECKPOINT")
TOKENIZER_ACTUAL=$(sha256_hex "$TOKENIZER")
DATASET_ACTUAL=$(sha256_hex "$DATASET")
CACHE_ACTUAL=$(sha256_hex "$CACHE")
D1_ACTUAL=$(sha256_hex "$D1_REPORT")
D2_ACTUAL=$(sha256_hex "$D2_REPORT")
if [ "$BASE_ACTUAL" != "$EXPECTED_BASE" ] ||
    [ "$TOKENIZER_ACTUAL" != "$EXPECTED_TOKENIZER" ] ||
    [ "$DATASET_ACTUAL" != "$EXPECTED_DATASET" ] ||
    [ "$CACHE_ACTUAL" != "$EXPECTED_CACHE" ] ||
    [ "$D1_ACTUAL" != "$EXPECTED_D1_REPORT" ] ||
    [ "$D2_ACTUAL" != "$EXPECTED_D2_REPORT" ]; then
    log "base_checkpoint_sha256=$BASE_ACTUAL"
    log "tokenizer_sha256=$TOKENIZER_ACTUAL"
    log "dataset_sha256=$DATASET_ACTUAL"
    log "cache_sha256=$CACHE_ACTUAL"
    log "d1_report_sha256=$D1_ACTUAL"
    log "d2_report_sha256=$D2_ACTUAL"
    fail "FAIL: one or more pinned D3 input identities differ"
fi
log "base_checkpoint_identity=PASS"
log "tokenizer_identity=PASS"
log "dataset_identity=PASS"
log "cache_identity=PASS"
log "d1_report_identity=PASS"
log "d2_report_identity=PASS"

D3_SOURCE_SHA=$(sha256_hex "$D3_SOURCE")
if [ -z "$D3_SOURCE_SHA" ]; then
    fail "FAIL: unable to hash the current D3 source"
fi

rm -f "$D3_EXECUTABLE"
make "$D3_MAKE_TARGET" CC=clang > "$BUILD_LOG" 2>&1
BUILD_STATUS=$?
cat "$BUILD_LOG" | tee -a "$CONSOLE_LOG"
if [ "$BUILD_STATUS" -ne 0 ]; then
    fail "FAIL: D3 A4 two-step executable build failed"
fi
check_file "$D3_EXECUTABLE"
D3_SOURCE_AFTER=$(sha256_hex "$D3_SOURCE")
if [ "$D3_SOURCE_AFTER" != "$D3_SOURCE_SHA" ]; then
    fail "FAIL: D3 source changed during build"
fi
D3_EXECUTABLE_SHA=$(sha256_hex "$D3_EXECUTABLE")
if [ -z "$D3_EXECUTABLE_SHA" ]; then
    fail "FAIL: unable to hash the rebuilt D3 executable"
fi
log "d3_source_sha256=$D3_SOURCE_SHA"
log "d3_executable_sha256=$D3_EXECUTABLE_SHA"

"./$D3_EXECUTABLE" "$BASE_CHECKPOINT" \
    -z "$TOKENIZER" -d "$DATASET" -c "$CACHE" \
    -1 "$D1_REPORT" -2 "$D2_REPORT" -r "$REPORT" \
    > "$RUN_LOG" 2>&1
RUN_STATUS=$?
cat "$RUN_LOG" | tee -a "$CONSOLE_LOG"
D3_RUNTIME_PROVENANCE=$(report_value "$RUN_LOG" d3_build_provenance)
if [ "$D3_RUNTIME_PROVENANCE" != "$EXPECTED_D3_BUILD_PROVENANCE" ]; then
    fail "FAIL: rebuilt D3 executable did not print the expected build provenance"
fi
if [ "$RUN_STATUS" -ne 0 ]; then
    fail "FAIL: D3 two-step execution failed"
fi
check_file "$REPORT"

REPORT_DATASET=$(report_value "$REPORT" dataset_sha256)
REPORT_CACHE=$(report_value "$REPORT" cache_sha256)
REPORT_BASE=$(report_value "$REPORT" base_checkpoint_sha256)
REPORT_TOKENIZER=$(report_value "$REPORT" tokenizer_sha256)
REPORT_D1=$(report_value "$REPORT" d1_report_sha256)
REPORT_D2=$(report_value "$REPORT" d2_report_sha256)
RANK=$(report_value "$REPORT" rank)
SCALE=$(report_value "$REPORT" scale)
PARAMETERS=$(report_value "$REPORT" adapter_parameter_count)
LEARNING_RATE=$(report_value "$REPORT" learning_rate)
BETA1=$(report_value "$REPORT" beta1)
BETA2=$(report_value "$REPORT" beta2)
EPSILON=$(report_value "$REPORT" epsilon)
WEIGHT_DECAY=$(report_value "$REPORT" weight_decay)
CLIP_THRESHOLD=$(report_value "$REPORT" gradient_clip_threshold)
INITIAL_MATCH=$(report_value "$REPORT" initial_d1_metrics_match)
STEP1_MATCH=$(report_value "$REPORT" step1_d2_metrics_match)
STEP1_TRAIN_ROWS=$(report_value "$REPORT" step1_gradient_train_rows)
STEP1_VALIDATION_ROWS=$(report_value "$REPORT" step1_gradient_validation_rows)
STEP1_TEST_ROWS=$(report_value "$REPORT" step1_gradient_test_rows)
STEP1_GRAD_A=$(report_value "$REPORT" step1_grad_A_nonzero_elements)
STEP1_GRAD_A_MAX=$(report_value "$REPORT" step1_grad_A_max_abs)
STEP1_GRAD_B=$(report_value "$REPORT" step1_grad_B_nonzero_elements)
STEP1_GRAD_B_MAX=$(report_value "$REPORT" step1_grad_B_max_abs)
STEP1_NORM_BEFORE=$(report_value "$REPORT" step1_gradient_norm_before_clip)
STEP1_NORM_AFTER=$(report_value "$REPORT" step1_gradient_norm_after_clip)
STEP1_CLIP=$(report_value "$REPORT" step1_gradient_clip_applied)
STEP1_A_CHANGED=$(report_value "$REPORT" step1_adapter_A_changed)
STEP1_B_CHANGED=$(report_value "$REPORT" step1_adapter_B_changed)
STEP1_A_DELTA=$(report_value "$REPORT" step1_adapter_A_max_abs_delta)
STEP1_B_DELTA=$(report_value "$REPORT" step1_adapter_B_max_abs_delta)
STEP1_STEPS=$(report_value "$REPORT" step1_optimizer_steps_after)
STEP1_M1_A=$(report_value "$REPORT" step1_m1_A_nonzero_elements)
STEP1_M2_A=$(report_value "$REPORT" step1_m2_A_nonzero_elements)
STEP1_M1_B=$(report_value "$REPORT" step1_m1_B_nonzero_elements)
STEP1_M2_B=$(report_value "$REPORT" step1_m2_B_nonzero_elements)
STEP2_TRAIN_ROWS=$(report_value "$REPORT" step2_gradient_train_rows)
STEP2_VALIDATION_ROWS=$(report_value "$REPORT" step2_gradient_validation_rows)
STEP2_TEST_ROWS=$(report_value "$REPORT" step2_gradient_test_rows)
STEP2_GRAD_A=$(report_value "$REPORT" step2_grad_A_nonzero_elements)
STEP2_GRAD_A_MAX=$(report_value "$REPORT" step2_grad_A_max_abs)
STEP2_GRAD_B=$(report_value "$REPORT" step2_grad_B_nonzero_elements)
STEP2_GRAD_B_MAX=$(report_value "$REPORT" step2_grad_B_max_abs)
STEP2_NORM_BEFORE=$(report_value "$REPORT" step2_gradient_norm_before_clip)
STEP2_NORM_AFTER=$(report_value "$REPORT" step2_gradient_norm_after_clip)
STEP2_CLIP=$(report_value "$REPORT" step2_gradient_clip_applied)
STEP2_A_CHANGED=$(report_value "$REPORT" step2_adapter_A_changed)
STEP2_B_CHANGED=$(report_value "$REPORT" step2_adapter_B_changed)
STEP2_A_DELTA=$(report_value "$REPORT" step2_adapter_A_max_abs_delta)
STEP2_B_DELTA=$(report_value "$REPORT" step2_adapter_B_max_abs_delta)
FINAL_A_CHANGED=$(report_value "$REPORT" final_adapter_A_changed_from_initial)
FINAL_B_CHANGED=$(report_value "$REPORT" final_adapter_B_changed_from_initial)
OPTIMIZER_BEFORE=$(report_value "$REPORT" optimizer_steps_before)
OPTIMIZER_AFTER=$(report_value "$REPORT" optimizer_steps_after)
FINAL_M1_A=$(report_value "$REPORT" final_m1_A_nonzero_elements)
FINAL_M2_A=$(report_value "$REPORT" final_m2_A_nonzero_elements)
FINAL_M1_B=$(report_value "$REPORT" final_m1_B_nonzero_elements)
FINAL_M2_B=$(report_value "$REPORT" final_m2_B_nonzero_elements)
INITIAL_TRAIN_COUNT=$(report_value "$REPORT" initial_train_target_count)
INITIAL_TRAIN_TOTAL=$(report_value "$REPORT" initial_train_total_loss)
INITIAL_TRAIN_AVERAGE=$(report_value "$REPORT" initial_train_average_loss)
INITIAL_TRAIN_TOP1=$(report_value "$REPORT" initial_train_top1_correct)
INITIAL_VALIDATION_COUNT=$(report_value "$REPORT" initial_validation_target_count)
INITIAL_VALIDATION_TOTAL=$(report_value "$REPORT" initial_validation_total_loss)
INITIAL_VALIDATION_AVERAGE=$(report_value "$REPORT" initial_validation_average_loss)
INITIAL_VALIDATION_TOP1=$(report_value "$REPORT" initial_validation_top1_correct)
INITIAL_TEST_COUNT=$(report_value "$REPORT" initial_test_target_count)
INITIAL_TEST_TOTAL=$(report_value "$REPORT" initial_test_total_loss)
INITIAL_TEST_AVERAGE=$(report_value "$REPORT" initial_test_average_loss)
INITIAL_TEST_TOP1=$(report_value "$REPORT" initial_test_top1_correct)
STEP1_TRAIN_COUNT=$(report_value "$REPORT" step1_train_target_count)
STEP1_TRAIN_TOTAL=$(report_value "$REPORT" step1_train_total_loss)
STEP1_TRAIN_AVERAGE=$(report_value "$REPORT" step1_train_average_loss)
STEP1_TRAIN_TOP1=$(report_value "$REPORT" step1_train_top1_correct)
STEP1_VALIDATION_COUNT=$(report_value "$REPORT" step1_validation_target_count)
STEP1_VALIDATION_TOTAL=$(report_value "$REPORT" step1_validation_total_loss)
STEP1_VALIDATION_AVERAGE=$(report_value "$REPORT" step1_validation_average_loss)
STEP1_VALIDATION_TOP1=$(report_value "$REPORT" step1_validation_top1_correct)
STEP1_TEST_COUNT=$(report_value "$REPORT" step1_test_target_count)
STEP1_TEST_TOTAL=$(report_value "$REPORT" step1_test_total_loss)
STEP1_TEST_AVERAGE=$(report_value "$REPORT" step1_test_average_loss)
STEP1_TEST_TOP1=$(report_value "$REPORT" step1_test_top1_correct)
STEP2_TRAIN_COUNT=$(report_value "$REPORT" step2_train_target_count)
STEP2_TRAIN_TOTAL=$(report_value "$REPORT" step2_train_total_loss)
STEP2_TRAIN_AVERAGE=$(report_value "$REPORT" step2_train_average_loss)
STEP2_TRAIN_TOP1=$(report_value "$REPORT" step2_train_top1_correct)
STEP2_VALIDATION_COUNT=$(report_value "$REPORT" step2_validation_target_count)
STEP2_VALIDATION_TOTAL=$(report_value "$REPORT" step2_validation_total_loss)
STEP2_VALIDATION_AVERAGE=$(report_value "$REPORT" step2_validation_average_loss)
STEP2_VALIDATION_TOP1=$(report_value "$REPORT" step2_validation_top1_correct)
STEP2_TEST_COUNT=$(report_value "$REPORT" step2_test_target_count)
STEP2_TEST_TOTAL=$(report_value "$REPORT" step2_test_total_loss)
STEP2_TEST_AVERAGE=$(report_value "$REPORT" step2_test_average_loss)
STEP2_TEST_TOP1=$(report_value "$REPORT" step2_test_top1_correct)
STEP2_VS_STEP1_DELTA=$(report_value "$REPORT" step2_vs_step1_train_loss_delta)
STEP2_VS_INITIAL_DELTA=$(report_value "$REPORT" step2_vs_initial_train_loss_delta)
VALIDATION_ACCEPTANCE=$(report_value "$REPORT" step2_validation_acceptance_reached)
FINAL_ZERO_ROWS=$(report_value "$REPORT" final_zero_correction_rows)
FINAL_NONZERO_ROWS=$(report_value "$REPORT" final_nonzero_correction_rows)
D3_PASS=$(report_value "$REPORT" d3_two_step_pass)

if [ "$REPORT_DATASET" != "$EXPECTED_DATASET" ] ||
    [ "$REPORT_CACHE" != "$EXPECTED_CACHE" ] ||
    [ "$REPORT_BASE" != "$EXPECTED_BASE" ] ||
    [ "$REPORT_TOKENIZER" != "$EXPECTED_TOKENIZER" ] ||
    [ "$REPORT_D1" != "$EXPECTED_D1_REPORT" ] ||
    [ "$REPORT_D2" != "$EXPECTED_D2_REPORT" ]; then
    fail "FAIL: D3 report identities differ from pinned inputs"
fi

BASE_AFTER=$(sha256_hex "$BASE_CHECKPOINT")
TOKENIZER_AFTER=$(sha256_hex "$TOKENIZER")
DATASET_AFTER=$(sha256_hex "$DATASET")
CACHE_AFTER=$(sha256_hex "$CACHE")
D1_AFTER=$(sha256_hex "$D1_REPORT")
D2_AFTER=$(sha256_hex "$D2_REPORT")
if [ "$BASE_AFTER" != "$EXPECTED_BASE" ] ||
    [ "$TOKENIZER_AFTER" != "$EXPECTED_TOKENIZER" ] ||
    [ "$DATASET_AFTER" != "$EXPECTED_DATASET" ] ||
    [ "$CACHE_AFTER" != "$EXPECTED_CACHE" ] ||
    [ "$D1_AFTER" != "$EXPECTED_D1_REPORT" ] ||
    [ "$D2_AFTER" != "$EXPECTED_D2_REPORT" ]; then
    fail "FAIL: a pinned D3 input identity changed during execution"
fi
REPORT_SHA=$(sha256_hex "$REPORT")
if [ -z "$REPORT_SHA" ]; then
    fail "FAIL: unable to obtain D3 report SHA-256"
fi

if ! clip_rule_matches "$STEP1_NORM_BEFORE" "$STEP1_NORM_AFTER" ||
    ! clip_rule_matches "$STEP2_NORM_BEFORE" "$STEP2_NORM_AFTER"; then
    fail "FAIL: one step's gradient norm does not follow the Stage 4 clipping rule"
fi
EXPECTED_STEP2_CLIP=$(awk -v norm="$STEP2_NORM_BEFORE" 'BEGIN {
    if ((norm + 0) > 1.0) print "yes"
    else print "no"
}')
if [ "$STEP2_CLIP" != "$EXPECTED_STEP2_CLIP" ]; then
    fail "FAIL: step-2 clipping flag disagrees with the pre-clip norm"
fi
if [ "$STEP1_CLIP" != no ] || [ "$STEP1_NORM_BEFORE" != 0.381956843381 ] ||
    [ "$STEP1_NORM_AFTER" != 0.381956843381 ]; then
    fail "FAIL: step 1 clipping/norm differs from accepted D2"
fi
if ! numeric_positive "$STEP2_GRAD_A" || ! numeric_positive "$STEP2_GRAD_A_MAX" ||
    ! numeric_positive "$STEP2_GRAD_B" || ! numeric_positive "$STEP2_GRAD_B_MAX" ||
    ! numeric_positive "$STEP2_NORM_BEFORE" ||
    ! numeric_positive "$STEP2_NORM_AFTER" ||
    ! numeric_positive "$STEP2_A_DELTA" || ! numeric_positive "$STEP2_B_DELTA" ||
    ! numeric_positive "$FINAL_M1_A" || ! numeric_positive "$FINAL_M2_A" ||
    ! numeric_positive "$FINAL_M1_B" || ! numeric_positive "$FINAL_M2_B" ||
    ! numeric_positive "$FINAL_NONZERO_ROWS"; then
    fail "FAIL: required step-2 gradient, parameter, moment, or correction value is zero"
fi
if ! train_below "$STEP2_TRAIN_AVERAGE" "$INITIAL_TRAIN_AVERAGE"; then
    fail "FAIL: step-2 train loss is not below initialization loss"
fi

if [ "$RANK" != 8 ] || [ "$SCALE" != 1.0 ] || [ "$PARAMETERS" != 258304 ] ||
    [ "$LEARNING_RATE" != 0.001 ] || [ "$BETA1" != 0.9 ] ||
    [ "$BETA2" != 0.999 ] || [ "$EPSILON" != 1e-08 ] ||
    [ "$WEIGHT_DECAY" != 0 ] || [ "$CLIP_THRESHOLD" != 1.0 ] ||
    [ "$INITIAL_MATCH" != yes ] || [ "$STEP1_MATCH" != yes ] ||
    [ "$STEP1_TRAIN_ROWS" != 111 ] || [ "$STEP1_VALIDATION_ROWS" != 0 ] ||
    [ "$STEP1_TEST_ROWS" != 0 ] || [ "$STEP1_GRAD_A" != 0 ] ||
    [ "$STEP1_GRAD_A_MAX" != 0 ] || [ "$STEP1_GRAD_B" != 256000 ] ||
    [ "$STEP1_GRAD_B_MAX" != 0.145234793425 ] ||
    [ "$STEP1_NORM_BEFORE" != 0.381956843381 ] ||
    [ "$STEP1_NORM_AFTER" != 0.381956843381 ] || [ "$STEP1_CLIP" != no ] ||
    [ "$STEP1_A_CHANGED" != no ] || [ "$STEP1_B_CHANGED" != yes ] ||
    [ "$STEP1_A_DELTA" != 0 ] || [ "$STEP1_B_DELTA" != 0.000999999931082 ] ||
    [ "$STEP1_STEPS" != 1 ] || [ "$STEP1_M1_A" != 0 ] || [ "$STEP1_M2_A" != 0 ] ||
    [ "$STEP1_M1_B" != 256000 ] || [ "$STEP1_M2_B" != 256000 ] ||
    [ "$STEP2_TRAIN_ROWS" != 111 ] || [ "$STEP2_VALIDATION_ROWS" != 0 ] ||
    [ "$STEP2_TEST_ROWS" != 0 ] || [ "$STEP2_A_CHANGED" != yes ] ||
    [ "$STEP2_B_CHANGED" != yes ] || [ "$FINAL_A_CHANGED" != yes ] ||
    [ "$FINAL_B_CHANGED" != yes ] || [ "$OPTIMIZER_BEFORE" != 0 ] ||
    [ "$OPTIMIZER_AFTER" != 2 ] || [ "$INITIAL_TRAIN_COUNT" != 111 ] ||
    [ "$INITIAL_TRAIN_TOTAL" != 1158.750195944566 ] ||
    [ "$INITIAL_TRAIN_AVERAGE" != 10.439190954456 ] || [ "$INITIAL_TRAIN_TOP1" != 0 ] ||
    [ "$INITIAL_VALIDATION_COUNT" != 33 ] ||
    [ "$INITIAL_VALIDATION_TOTAL" != 300.506654007981 ] ||
    [ "$INITIAL_VALIDATION_AVERAGE" != 9.106262242666 ] ||
    [ "$INITIAL_VALIDATION_TOP1" != 2 ] || [ "$INITIAL_TEST_COUNT" != 21 ] ||
    [ "$INITIAL_TEST_TOTAL" != 223.961209448908 ] ||
    [ "$INITIAL_TEST_AVERAGE" != 10.664819497567 ] || [ "$INITIAL_TEST_TOP1" != 0 ] ||
    [ "$STEP1_TRAIN_COUNT" != 111 ] || [ "$STEP1_TRAIN_TOTAL" != 1158.085676086679 ] ||
    [ "$STEP1_TRAIN_AVERAGE" != 10.433204289069 ] || [ "$STEP1_TRAIN_TOP1" != 0 ] ||
    [ "$STEP1_VALIDATION_COUNT" != 33 ] ||
    [ "$STEP1_VALIDATION_TOTAL" != 300.451596646300 ] ||
    [ "$STEP1_VALIDATION_AVERAGE" != 9.104593837767 ] ||
    [ "$STEP1_VALIDATION_TOP1" != 2 ] || [ "$STEP1_TEST_COUNT" != 21 ] ||
    [ "$STEP1_TEST_TOTAL" != 223.897562610626 ] ||
    [ "$STEP1_TEST_AVERAGE" != 10.661788695744 ] || [ "$STEP1_TEST_TOP1" != 0 ] ||
    [ "$STEP2_TRAIN_COUNT" != 111 ] || [ "$STEP2_VALIDATION_COUNT" != 33 ] ||
    [ "$STEP2_TEST_COUNT" != 21 ] || [ -z "$STEP2_TRAIN_TOTAL" ] ||
    [ -z "$STEP2_TRAIN_TOP1" ] || [ -z "$STEP2_VALIDATION_TOTAL" ] ||
    [ -z "$STEP2_VALIDATION_AVERAGE" ] || [ -z "$STEP2_VALIDATION_TOP1" ] ||
    [ -z "$STEP2_TEST_TOTAL" ] || [ -z "$STEP2_TEST_AVERAGE" ] ||
    [ -z "$STEP2_TEST_TOP1" ] || [ -z "$STEP2_VS_STEP1_DELTA" ] ||
    [ -z "$STEP2_VS_INITIAL_DELTA" ] ||
    { [ "$VALIDATION_ACCEPTANCE" != yes ] && [ "$VALIDATION_ACCEPTANCE" != no ]; } ||
    [ -z "$FINAL_ZERO_ROWS" ] || [ "$D3_PASS" != yes ]; then
    fail "FAIL: one or more D3 two-step acceptance fields do not match"
fi

log "base_checkpoint_identity=PASS"
log "tokenizer_identity=PASS"
log "dataset_identity=PASS"
log "cache_identity=PASS"
log "d1_report_identity=PASS"
log "d2_report_identity=PASS"
log "rank=$RANK"
log "scale=$SCALE"
log "adapter_parameter_count=$PARAMETERS"
log "step1_d2_metrics_match=$STEP1_MATCH"
log "step1_grad_A_nonzero_elements=$STEP1_GRAD_A"
log "step1_grad_B_nonzero_elements=$STEP1_GRAD_B"
log "step2_gradient_train_rows=$STEP2_TRAIN_ROWS"
log "step2_gradient_validation_rows=$STEP2_VALIDATION_ROWS"
log "step2_gradient_test_rows=$STEP2_TEST_ROWS"
log "step2_grad_A_nonzero_elements=$STEP2_GRAD_A"
log "step2_grad_A_max_abs=$STEP2_GRAD_A_MAX"
log "step2_grad_B_nonzero_elements=$STEP2_GRAD_B"
log "step2_grad_B_max_abs=$STEP2_GRAD_B_MAX"
log "step2_gradient_norm_before_clip=$STEP2_NORM_BEFORE"
log "step2_gradient_norm_after_clip=$STEP2_NORM_AFTER"
log "step2_gradient_clip_applied=$STEP2_CLIP"
log "step2_adapter_A_changed=$STEP2_A_CHANGED"
log "step2_adapter_B_changed=$STEP2_B_CHANGED"
log "optimizer_steps_before=$OPTIMIZER_BEFORE"
log "optimizer_steps_after=$OPTIMIZER_AFTER"
log "final_m1_A_nonzero_elements=$FINAL_M1_A"
log "final_m2_A_nonzero_elements=$FINAL_M2_A"
log "final_m1_B_nonzero_elements=$FINAL_M1_B"
log "final_m2_B_nonzero_elements=$FINAL_M2_B"
log "initial_train_average_loss=$INITIAL_TRAIN_AVERAGE"
log "step1_train_average_loss=$STEP1_TRAIN_AVERAGE"
log "step2_train_average_loss=$STEP2_TRAIN_AVERAGE"
log "initial_validation_average_loss=$INITIAL_VALIDATION_AVERAGE"
log "step1_validation_average_loss=$STEP1_VALIDATION_AVERAGE"
log "step2_validation_average_loss=$STEP2_VALIDATION_AVERAGE"
log "step2_validation_acceptance_reached=$VALIDATION_ACCEPTANCE"
log "final_nonzero_correction_rows=$FINAL_NONZERO_ROWS"
log "d3_report_sha256=$REPORT_SHA"
log "Stage4E-D3-A4=PASS"

cleanup
exit 0
