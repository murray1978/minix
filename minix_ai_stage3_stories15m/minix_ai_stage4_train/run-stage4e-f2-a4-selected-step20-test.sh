#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" 2>/dev/null && pwd)
cd "$SCRIPT_DIR" || exit 1

BASE_CHECKPOINT="/usr/src/minix_ai_stage3_stories15m/stories15M.bin"
TOKENIZER="/usr/src/minix_ai_stage3_stories15m/tokenizer.bin"
DATASET="stage4e-approved.records"
CACHE="stage4e-target-cache-a4.bin"
D6_CHECKPOINT="stage4e-d6-a4-step5-checkpoint.bin"
D6_REPORT="stage4e-d6-a4-checkpoint-report.txt"
D7_REPORT="stage4e-d7-a4-resume-report.txt"
D7_TRAJECTORY="stage4e-d7-a4-resume-trajectory.tsv"
F1_REPORT="stage4e-f1-a4-fixed-horizon-report.txt"
F1_TSV="stage4e-f1-a4-fixed-horizon-trajectory.tsv"
SOURCE="stage4e_f2_a4_selected_step20_test.c"
EXECUTABLE="stage4e_f2_a4_selected_step20_test"
REPORT="stage4e-f2-a4-selected-step20-audit-test-report.txt"
TRAJECTORY="stage4e-f2-a4-selected-step20-audit-reconstruction.tsv"
CHECKPOINT="stage4e-f2-a4-selected-step20-audit-checkpoint.bin"
CONSOLE_LOG="stage4e-f2-a4-selected-step20-audit-console.txt"
BUILD_LOG=".stage4e-f2-a4-build.tmp"
RUN_LOG=".stage4e-f2-a4-run.tmp"

EXPECTED_BASE="cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a"
EXPECTED_TOKENIZER="50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361"
EXPECTED_DATASET="71744cea4de049e2e192cb2a4494c8a4a68e769b3a7f3b8ed4880433e4583b84"
EXPECTED_CACHE="d6288ad3d128c7182c20088aa57a1bca8bcaa1e13314fb0cdbd60eba14278def"
EXPECTED_D6="ecb75d52881df04a0c47fcd4b852909a865c62cb54929ab33b230c7cf1e5ec89"

log() { printf '%s\n' "$*"; printf '%s\n' "$*" >> "$CONSOLE_LOG"; }
cleanup() { rm -f "$BUILD_LOG" "$RUN_LOG"; }
fail() { log "$*"; log "Stage4E-F2-A4=FAIL"; cleanup; exit 1; }
check_file() { [ -f "$1" ] || fail "FAIL: missing file: $1"; }
sha256_hex() { openssl dgst -sha256 "$1" 2>/dev/null | awk '{print $NF}'; }
report_value() { awk -F= -v wanted="$2" '$1 == wanted { print substr($0, index($0, "=") + 1); exit }' "$1"; }

if [ -e "$REPORT" ] || [ -e "$TRAJECTORY" ] || [ -e "$CHECKPOINT" ]; then
    printf '%s\n' 'FAIL: refusing to overwrite an existing F2 output artifact' >&2
    exit 1
fi
: > "$CONSOLE_LOG" || exit 1
log 'Stage 4E-F2-A4 selected step-20 test-sealed reconstruction'
for file in "$BASE_CHECKPOINT" "$TOKENIZER" "$DATASET" "$CACHE" "$D6_CHECKPOINT" "$D6_REPORT" "$D7_REPORT" "$D7_TRAJECTORY" "$F1_REPORT" "$F1_TSV" "$SOURCE" stage4_adapter_train.c stage4_adapter_train.h ../minix_llama.c; do
    check_file "$file"
done

BASE_HASH=$(sha256_hex "$BASE_CHECKPOINT")
TOKENIZER_HASH=$(sha256_hex "$TOKENIZER")
DATASET_HASH=$(sha256_hex "$DATASET")
CACHE_HASH=$(sha256_hex "$CACHE")
D6_HASH=$(sha256_hex "$D6_CHECKPOINT")
F1_REPORT_HASH=$(sha256_hex "$F1_REPORT")
F1_TSV_HASH=$(sha256_hex "$F1_TSV")
SOURCE_HASH=$(sha256_hex "$SOURCE")
if [ "$BASE_HASH" != "$EXPECTED_BASE" ] || [ "$TOKENIZER_HASH" != "$EXPECTED_TOKENIZER" ] || [ "$DATASET_HASH" != "$EXPECTED_DATASET" ] || [ "$CACHE_HASH" != "$EXPECTED_CACHE" ] || [ "$D6_HASH" != "$EXPECTED_D6" ] || [ -z "$F1_REPORT_HASH" ] || [ -z "$F1_TSV_HASH" ] || [ -z "$SOURCE_HASH" ]; then
    fail 'FAIL: a pinned identity or prerequisite hash differs'
fi
if [ "$(report_value "$F1_REPORT" f1_fixed_horizon_pass)" != yes ] || [ "$(report_value "$F1_REPORT" lowest_observed_validation_loss_step)" != 20 ] || [ "$(report_value "$F1_REPORT" lowest_observed_validation_loss)" != 8.328697455428 ]; then
    fail 'FAIL: F1 selection evidence mismatch'
fi
log "f1_report_sha256=$F1_REPORT_HASH"
log "f1_trajectory_sha256=$F1_TSV_HASH"

rm -f "$EXECUTABLE"
clang -O2 -std=c99 -D_POSIX_C_SOURCE=200112L -D_LARGEFILE_SOURCE -Wno-unused-function -o "$EXECUTABLE" "$SOURCE" stage4_adapter_train.c -lm > "$BUILD_LOG" 2>&1
BUILD_STATUS=$?
cat "$BUILD_LOG" | tee -a "$CONSOLE_LOG"
[ "$BUILD_STATUS" -eq 0 ] || fail 'FAIL: F2 clang build failed'
check_file "$EXECUTABLE"
[ "$(sha256_hex "$SOURCE")" = "$SOURCE_HASH" ] || fail 'FAIL: F2 source changed during build'

"./$EXECUTABLE" "$BASE_CHECKPOINT" "$TOKENIZER" "$DATASET" "$CACHE" "$D6_CHECKPOINT" "$F1_REPORT" "$F1_TSV" > "$RUN_LOG" 2>&1
RUN_STATUS=$?
cat "$RUN_LOG" | tee -a "$CONSOLE_LOG"
[ "$RUN_STATUS" -eq 0 ] || fail 'FAIL: F2 execution failed'
for file in "$REPORT" "$TRAJECTORY" "$CHECKPOINT"; do check_file "$file"; done
if [ "$(report_value "$REPORT" selected_step)" != 20 ] || [ "$(report_value "$REPORT" selection_unique_minimum)" != yes ] || [ "$(report_value "$REPORT" selected_train_average_loss)" != 7.026346899236 ] || [ "$(report_value "$REPORT" selected_validation_average_loss)" != 8.328697455428 ] || [ "$(report_value "$REPORT" f1_reconstruction_fields_match)" != yes ] || [ "$(report_value "$REPORT" checkpoint_reloaded_optimizer_step)" != 20 ] || [ "$(report_value "$REPORT" selected_state_byte_mismatches)" != 0 ] || [ "$(report_value "$REPORT" test_gradient_rows_seen)" != 0 ] || [ "$(report_value "$REPORT" test_evaluation_rows_seen)" != 21 ] || [ "$(report_value "$REPORT" f2_selected_step20_test_pass)" != yes ]; then
    fail 'FAIL: F2 acceptance gate failed'
fi
for key in stage mode dataset_pathname dataset_sha256 cache_pathname cache_sha256 base_checkpoint_pathname base_checkpoint_sha256 tokenizer_pathname tokenizer_sha256 d6_checkpoint_pathname d6_checkpoint_sha256 d6_report_sha256 d7_report_sha256 d7_trajectory_sha256 f1_report_pathname f1_report_sha256 f1_trajectory_pathname f1_trajectory_sha256 selection_source selection_metric selection_test_access validation_acceptance_loss_max selected_step selected_validation_average_loss selected_step_unique_minimum selected_validation_acceptance_pass selection_frozen reconstruction_start_step reconstruction_end_step reconstruction_update_count trajectory_rows_compared trajectory_field_mismatches deterministic_trajectory_match selected_train_average_loss all_training_values_finite checkpoint_format_version checkpoint_output checkpoint_optimizer_step checkpoint_A_byte_mismatches checkpoint_B_byte_mismatches checkpoint_m1A_byte_mismatches checkpoint_m2A_byte_mismatches checkpoint_m1B_byte_mismatches checkpoint_m2B_byte_mismatches checkpoint_total_state_byte_mismatches checkpoint_roundtrip_bitwise_match test_access_before_selection_frozen preselection_test_loss_rows preselection_test_top1_rows preselection_test_gradient_rows preselection_test_update_rows heldout_test_calls test_records test_target_rows test_total_target_loss test_average_target_loss test_top1_correct test_top1_total test_top1_fraction test_gradient_rows_consumed test_update_rows_consumed optimizer_step_before_test optimizer_step_after_test test_values_finite baseline_test_average_loss baseline_test_top1_correct baseline_test_top1_total test_loss_delta_from_baseline test_relative_loss_change_from_baseline test_top1_correct_delta_from_baseline gate_selection_validation_only gate_selected_step20 gate_reconstruction_matches_f1 gate_step20_checkpoint_roundtrip gate_test_sealed_until_selection gate_test_no_gradient gate_test_no_update gate_optimizer_remains_step20 gate_all_finite gate_stage4e_f2; do
    [ -n "$(report_value "$REPORT" "$key")" ] || fail "FAIL: missing F2 report field: $key"
done
if [ "$(report_value "$REPORT" stage)" != 4E-F2 ] || [ "$(report_value "$REPORT" mode)" != validation_selected_step20_heldout_test ] || [ "$(report_value "$REPORT" selection_frozen)" != yes ] || [ "$(report_value "$REPORT" selected_step_unique_minimum)" != yes ] || [ "$(report_value "$REPORT" selected_validation_acceptance_pass)" != yes ] || [ "$(report_value "$REPORT" deterministic_trajectory_match)" != yes ] || [ "$(report_value "$REPORT" checkpoint_roundtrip_bitwise_match)" != yes ] || [ "$(report_value "$REPORT" test_access_before_selection_frozen)" != no ] || [ "$(report_value "$REPORT" preselection_test_loss_rows)" != 0 ] || [ "$(report_value "$REPORT" preselection_test_top1_rows)" != 0 ] || [ "$(report_value "$REPORT" preselection_test_gradient_rows)" != 0 ] || [ "$(report_value "$REPORT" preselection_test_update_rows)" != 0 ] || [ "$(report_value "$REPORT" heldout_test_calls)" != 1 ] || [ "$(report_value "$REPORT" test_gradient_rows_consumed)" != 0 ] || [ "$(report_value "$REPORT" test_update_rows_consumed)" != 0 ] || [ "$(report_value "$REPORT" optimizer_step_before_test)" != 20 ] || [ "$(report_value "$REPORT" optimizer_step_after_test)" != 20 ] || [ "$(report_value "$REPORT" test_records)" != 5 ] || [ "$(report_value "$REPORT" test_target_rows)" != 21 ] || [ "$(report_value "$REPORT" test_top1_total)" != 21 ] || [ "$(report_value "$REPORT" gate_selection_validation_only)" != yes ] || [ "$(report_value "$REPORT" gate_selected_step20)" != yes ] || [ "$(report_value "$REPORT" gate_reconstruction_matches_f1)" != yes ] || [ "$(report_value "$REPORT" gate_step20_checkpoint_roundtrip)" != yes ] || [ "$(report_value "$REPORT" gate_test_sealed_until_selection)" != yes ] || [ "$(report_value "$REPORT" gate_test_no_gradient)" != yes ] || [ "$(report_value "$REPORT" gate_test_no_update)" != yes ] || [ "$(report_value "$REPORT" gate_optimizer_remains_step20)" != yes ] || [ "$(report_value "$REPORT" gate_all_finite)" != yes ] || [ "$(report_value "$REPORT" gate_stage4e_f2)" != yes ]; then
    fail 'FAIL: F2 audit-report gate failed'
fi
awk -F '\t' 'NR == 1 { next } { if ($1 == 5) { if ($2 != "NA" || $3 != "NA" || $4 != "NA" || $13 != "yes") bad++ } else { expected=($1 <= 10 || $1 % 5 == 0); if ($1 < 6 || $1 > 20 || $2 != 111 || $3 != 0 || $4 != 0 || (($13 == "yes") != expected)) bad++ } rows++ } END { exit !(rows == 16 && bad == 0) }' "$TRAJECTORY" || fail 'FAIL: F2 reconstruction TSV gate failed'
log 'Stage4E-F2-A4=PASS'
cleanup
exit 0