#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" 2>/dev/null && pwd)
cd "$SCRIPT_DIR" || exit 1

BASE_CHECKPOINT="/usr/src/minix_ai_stage3_stories15m/stories15M.bin"
TOKENIZER="/usr/src/minix_ai_stage3_stories15m/tokenizer.bin"
DATASET="stage4e-approved.records"
CACHE="stage4e-target-cache-a4.bin"
D1_REPORT="stage4e-d1-a4-zero-update-trainer-report.txt"
D2_REPORT="stage4e-d2-a4-one-step-report.txt"
D3_REPORT="stage4e-d3-a4-two-step-report.txt"
D4_SOURCE="stage4e_d4_a4_trajectory.c"
D4_EXECUTABLE="stage4e_d4_a4_trajectory"
D4_MAKE_TARGET="stage4eD4A4Trajectory"
EXPECTED_PROVENANCE="ten-step-trajectory-2026-10-01-a"
REPORT="stage4e-d4-a4-trajectory-report.txt"
TRAJECTORY="stage4e-d4-a4-trajectory.tsv"
CONSOLE_LOG="stage4e-d4-a4-trajectory-console.txt"
BUILD_LOG=".stage4e-d4-a4-build.tmp"
RUN_LOG=".stage4e-d4-a4-run.tmp"

EXPECTED_BASE="cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a"
EXPECTED_TOKENIZER="50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361"
EXPECTED_DATASET="71744cea4de049e2e192cb2a4494c8a4a68e769b3a7f3b8ed4880433e4583b84"
EXPECTED_CACHE="d6288ad3d128c7182c20088aa57a1bca8bcaa1e13314fb0cdbd60eba14278def"
EXPECTED_D1="45ebaa750f17d02d80feed332b16c9c29c3782a544aed7544da72a62161c76a6"
EXPECTED_D2="c27aad7474df093f5a3ff781bb3619b71b3e284edda46290f5814c350e99d97e"
EXPECTED_D3="f418305f3fb9263480b8a80f34b02a966ebbb94f305f2aa5897d5c2f4356b94e"

log() {
    printf '%s\n' "$*"
    printf '%s\n' "$*" >> "$CONSOLE_LOG"
}

cleanup() {
    rm -f "$BUILD_LOG" "$RUN_LOG"
}

fail() {
    log "$*"
    log "Stage4E-D4-A4=FAIL"
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
    awk -v a="$1" -v b="$2" 'BEGIN { exit !((a + 0) < (b + 0)) }'
}

: > "$CONSOLE_LOG" || exit 1
log "=============================================="
log "Stage 4E-D4-A4 deterministic ten-step trajectory"
log "=============================================="

check_file "$BASE_CHECKPOINT"
check_file "$TOKENIZER"
check_file "$DATASET"
check_file "$CACHE"
check_file "$D1_REPORT"
check_file "$D2_REPORT"
check_file "$D3_REPORT"
check_file "$D4_SOURCE"
check_file "Makefile"

BASE_ACTUAL=$(sha256_hex "$BASE_CHECKPOINT")
TOKENIZER_ACTUAL=$(sha256_hex "$TOKENIZER")
DATASET_ACTUAL=$(sha256_hex "$DATASET")
CACHE_ACTUAL=$(sha256_hex "$CACHE")
D1_ACTUAL=$(sha256_hex "$D1_REPORT")
D2_ACTUAL=$(sha256_hex "$D2_REPORT")
D3_ACTUAL=$(sha256_hex "$D3_REPORT")
if [ "$BASE_ACTUAL" != "$EXPECTED_BASE" ] ||
    [ "$TOKENIZER_ACTUAL" != "$EXPECTED_TOKENIZER" ] ||
    [ "$DATASET_ACTUAL" != "$EXPECTED_DATASET" ] ||
    [ "$CACHE_ACTUAL" != "$EXPECTED_CACHE" ] ||
    [ "$D1_ACTUAL" != "$EXPECTED_D1" ] ||
    [ "$D2_ACTUAL" != "$EXPECTED_D2" ] ||
    [ "$D3_ACTUAL" != "$EXPECTED_D3" ]; then
    log "base_checkpoint_sha256=$BASE_ACTUAL"
    log "tokenizer_sha256=$TOKENIZER_ACTUAL"
    log "dataset_sha256=$DATASET_ACTUAL"
    log "cache_sha256=$CACHE_ACTUAL"
    log "d1_report_sha256=$D1_ACTUAL"
    log "d2_report_sha256=$D2_ACTUAL"
    log "d3_report_sha256=$D3_ACTUAL"
    fail "FAIL: one or more pinned D4 input identities differ"
fi
log "base_checkpoint_identity=PASS"
log "tokenizer_identity=PASS"
log "dataset_identity=PASS"
log "cache_identity=PASS"
log "d1_report_identity=PASS"
log "d2_report_identity=PASS"
log "d3_report_identity=PASS"

D4_SOURCE_SHA=$(sha256_hex "$D4_SOURCE")
if [ -z "$D4_SOURCE_SHA" ]; then
    fail "FAIL: unable to hash current D4 source"
fi
rm -f "$D4_EXECUTABLE"
make "$D4_MAKE_TARGET" CC=clang > "$BUILD_LOG" 2>&1
BUILD_STATUS=$?
cat "$BUILD_LOG" | tee -a "$CONSOLE_LOG"
if [ "$BUILD_STATUS" -ne 0 ]; then
    fail "FAIL: D4 trajectory executable build failed"
fi
check_file "$D4_EXECUTABLE"
D4_SOURCE_AFTER=$(sha256_hex "$D4_SOURCE")
if [ "$D4_SOURCE_AFTER" != "$D4_SOURCE_SHA" ]; then
    fail "FAIL: D4 source changed during build"
fi
D4_EXECUTABLE_SHA=$(sha256_hex "$D4_EXECUTABLE")
if [ -z "$D4_EXECUTABLE_SHA" ]; then
    fail "FAIL: unable to hash rebuilt D4 executable"
fi
log "d4_source_sha256=$D4_SOURCE_SHA"
log "d4_executable_sha256=$D4_EXECUTABLE_SHA"

"./$D4_EXECUTABLE" "$BASE_CHECKPOINT" -z "$TOKENIZER" \
    -d "$DATASET" -c "$CACHE" -1 "$D1_REPORT" -2 "$D2_REPORT" \
    -3 "$D3_REPORT" > "$RUN_LOG" 2>&1
RUN_STATUS=$?
cat "$RUN_LOG" | tee -a "$CONSOLE_LOG"
D4_RUNTIME_PROVENANCE=$(report_value "$RUN_LOG" d4_build_provenance)
if [ "$D4_RUNTIME_PROVENANCE" != "$EXPECTED_PROVENANCE" ]; then
    fail "FAIL: rebuilt D4 executable provenance mismatch"
fi
if [ "$RUN_STATUS" -ne 0 ]; then
    fail "FAIL: D4 ten-step trajectory execution failed"
fi
check_file "$REPORT"
check_file "$TRAJECTORY"

for pair in \
    "dataset_sha256:$EXPECTED_DATASET" \
    "cache_sha256:$EXPECTED_CACHE" \
    "base_checkpoint_sha256:$EXPECTED_BASE" \
    "tokenizer_sha256:$EXPECTED_TOKENIZER" \
    "d1_report_sha256:$EXPECTED_D1" \
    "d2_report_sha256:$EXPECTED_D2" \
    "d3_report_sha256:$EXPECTED_D3"; do
    KEY=${pair%%:*}
    EXPECTED=${pair#*:}
    ACTUAL=$(report_value "$REPORT" "$KEY")
    if [ "$ACTUAL" != "$EXPECTED" ]; then
        fail "FAIL: D4 report identity mismatch for $KEY"
    fi
done

STEP0=$(report_value "$REPORT" step0_d1_metrics_match)
STEP1=$(report_value "$REPORT" step1_d2_metrics_match)
STEP2=$(report_value "$REPORT" step2_d3_metrics_match)
OPT_BEFORE=$(report_value "$REPORT" optimizer_steps_before)
OPT_AFTER=$(report_value "$REPORT" optimizer_steps_after)
TEST_GRAD=$(report_value "$REPORT" test_gradient_rows_seen)
TEST_EVAL=$(report_value "$REPORT" test_evaluation_rows_seen)
ALL_GRAD=$(report_value "$REPORT" all_gradients_finite)
ALL_PARAMS=$(report_value "$REPORT" all_parameters_finite)
ALL_MOMENTS=$(report_value "$REPORT" all_optimizer_moments_finite)
ALL_LOSSES=$(report_value "$REPORT" all_losses_finite)
FINAL_A=$(report_value "$REPORT" final_adapter_A_changed_from_initial)
FINAL_B=$(report_value "$REPORT" final_adapter_B_changed_from_initial)
TRAIN_INITIAL=$(report_value "$REPORT" initial_train_average_loss)
TRAIN_FINAL=$(report_value "$REPORT" final_train_average_loss)
D4_PASS=$(report_value "$REPORT" d4_trajectory_pass)
FINAL_VALIDATION=$(report_value "$REPORT" final_validation_average_loss)
FINAL_VALIDATION_ACCEPTANCE=$(report_value "$REPORT" final_validation_acceptance_reached)
VALIDATION_ACCEPTANCE_MAX=$(report_value "$REPORT" validation_acceptance_loss_max)
TRAIN_IMPROVED=$(report_value "$REPORT" train_steps_improved_from_previous)
TRAIN_SAME=$(report_value "$REPORT" train_steps_unchanged_from_previous)
TRAIN_WORSE=$(report_value "$REPORT" train_steps_worse_than_previous)
VAL_IMPROVED=$(report_value "$REPORT" validation_steps_improved_from_previous)
VAL_SAME=$(report_value "$REPORT" validation_steps_unchanged_from_previous)
VAL_WORSE=$(report_value "$REPORT" validation_steps_worse_than_previous)
M1_A=$(report_value "$REPORT" final_m1_A_nonzero_elements)
M2_A=$(report_value "$REPORT" final_m2_A_nonzero_elements)
M1_B=$(report_value "$REPORT" final_m1_B_nonzero_elements)
M2_B=$(report_value "$REPORT" final_m2_B_nonzero_elements)
PARAMETERS=$(report_value "$REPORT" adapter_parameter_count)

if [ "$STEP0" != yes ] || [ "$STEP1" != yes ] || [ "$STEP2" != yes ] ||
    [ "$OPT_BEFORE" != 0 ] || [ "$OPT_AFTER" != 10 ] ||
    [ "$TEST_GRAD" != 0 ] || [ "$TEST_EVAL" != 0 ] ||
    [ "$ALL_GRAD" != yes ] || [ "$ALL_PARAMS" != yes ] ||
    [ "$ALL_MOMENTS" != yes ] || [ "$ALL_LOSSES" != yes ] ||
    [ "$FINAL_A" != yes ] || [ "$FINAL_B" != yes ] ||
    [ "$M1_A" -le 0 ] || [ "$M2_A" -le 0 ] ||
    [ "$M1_B" -le 0 ] || [ "$M2_B" -le 0 ] ||
    [ "$PARAMETERS" != 258304 ] ||
    [ "$D4_PASS" != yes ] || ! train_below "$TRAIN_FINAL" "$TRAIN_INITIAL"; then
    fail "FAIL: D4 trajectory report did not satisfy the fixed acceptance gates"
fi

log "step0_d1_metrics_match=$STEP0"
log "step1_d2_metrics_match=$STEP1"
log "step2_d3_metrics_match=$STEP2"
log "optimizer_steps_before=$OPT_BEFORE"
log "optimizer_steps_after=$OPT_AFTER"
log "test_gradient_rows_seen=$TEST_GRAD"
log "test_evaluation_rows_seen=$TEST_EVAL"
log "initial_train_average_loss=$TRAIN_INITIAL"
log "final_train_average_loss=$TRAIN_FINAL"
log "final_vs_initial_train_loss_delta=$(report_value "$REPORT" final_vs_initial_train_loss_delta)"
log "initial_validation_average_loss=$(report_value "$REPORT" initial_validation_average_loss)"
log "final_validation_average_loss=$FINAL_VALIDATION"
log "final_vs_initial_validation_loss_delta=$(report_value "$REPORT" final_vs_initial_validation_loss_delta)"
log "train_steps_improved_from_previous=$TRAIN_IMPROVED"
log "train_steps_unchanged_from_previous=$TRAIN_SAME"
log "train_steps_worse_than_previous=$TRAIN_WORSE"
log "validation_steps_improved_from_previous=$VAL_IMPROVED"
log "validation_steps_unchanged_from_previous=$VAL_SAME"
log "validation_steps_worse_than_previous=$VAL_WORSE"
log "validation_acceptance_loss_max=$VALIDATION_ACCEPTANCE_MAX"
log "final_validation_acceptance_reached=$FINAL_VALIDATION_ACCEPTANCE"
log "final_adapter_A_changed_from_initial=$FINAL_A_CHANGED"
log "final_adapter_B_changed_from_initial=$FINAL_B_CHANGED"
log "final_m1_A_nonzero_elements=$M1_A"
log "final_m2_A_nonzero_elements=$M2_A"
log "final_m1_B_nonzero_elements=$M1_B"
log "final_m2_B_nonzero_elements=$M2_B"
log "all_gradients_finite=$ALL_GRAD"
log "all_parameters_finite=$ALL_PARAMS"
log "all_optimizer_moments_finite=$ALL_MOMENTS"
log "all_losses_finite=$ALL_LOSSES"
log "Stage4E-D4-A4=PASS"
cleanup
exit 0
