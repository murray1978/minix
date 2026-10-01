#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" 2>/dev/null && pwd)
cd "$SCRIPT_DIR" || exit 1

BASE_MODEL="/usr/src/minix_ai_stage3_stories15m/stories15M.bin"
TOKENIZER="/usr/src/minix_ai_stage3_stories15m/tokenizer.bin"
DATASET="stage4e-approved.records"
ADAPTER="stage4g1-selected-step20.adapter.bin"
G1_REPORT="stage4g1-selected-step20-export-report.txt"
SOURCE="stage4g2_deployment_inference_equivalence.c"
EXECUTABLE="stage4g2_deployment_inference_equivalence"
REPORT="stage4g2-deployment-inference-equivalence-report.txt"
EXPECTED_ADAPTER="a248d7a6f988684404e86e5b0cf6b298eefc54ffc5fe41f016e4c8add3683f5e"
EXPECTED_G1_REPORT="37874c6356dbd2189793d0795afc68d75340f06ec2f26f30cb3c4a5ad9a45eb4"
EXPECTED_DATASET="71744cea4de049e2e192cb2a4494c8a4a68e769b3a7f3b8ed4880433e4583b84"
EXPECTED_BASE="cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a"
EXPECTED_TOKENIZER="50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361"

sha256_hex() { openssl dgst -sha256 "$1" 2>/dev/null | awk '{print $NF}'; }
report_value() { awk -F= -v wanted="$2" '$1 == wanted { print substr($0, index($0, "=") + 1); exit }' "$1"; }
fail() {
    printf '%s\n' "$*" 'Stage4G2=FAIL' >&2
    exit 1
}
check_file() { [ -f "$1" ] || fail "FAIL: missing file: $1"; }

if [ -e "$REPORT" ]; then
    printf '%s\n' 'FAIL: refusing to overwrite an existing G2 report' >&2
    exit 1
fi
check_file "$BASE_MODEL"
check_file "$TOKENIZER"
check_file "$DATASET"
check_file "$ADAPTER"
check_file "$G1_REPORT"
check_file "$SOURCE"
check_file stage4_adapter_train.c
check_file stage4_adapter_train.h
check_file ../minix_llama.c

[ "$(sha256_hex "$ADAPTER")" = "$EXPECTED_ADAPTER" ] || fail 'FAIL: accepted G1 adapter SHA-256 mismatch'
[ "$(sha256_hex "$G1_REPORT")" = "$EXPECTED_G1_REPORT" ] || fail 'FAIL: accepted G1 report SHA-256 mismatch'
[ "$(sha256_hex "$DATASET")" = "$EXPECTED_DATASET" ] || fail 'FAIL: accepted dataset SHA-256 mismatch'
[ "$(sha256_hex "$BASE_MODEL")" = "$EXPECTED_BASE" ] || fail 'FAIL: accepted base model SHA-256 mismatch'
[ "$(sha256_hex "$TOKENIZER")" = "$EXPECTED_TOKENIZER" ] || fail 'FAIL: accepted tokenizer SHA-256 mismatch'

rm -f "$EXECUTABLE"
clang -O2 -std=c99 -D_POSIX_C_SOURCE=200112L -D_LARGEFILE_SOURCE -Wno-unused-function -o "$EXECUTABLE" "$SOURCE" stage4_adapter_train.c -lm || fail 'FAIL: G2 clang build failed'
check_file "$EXECUTABLE"
"./$EXECUTABLE" || fail 'FAIL: G2 deployment inference execution failed'
check_file "$REPORT"

[ "$(report_value "$REPORT" stage)" = 4G2 ] || fail 'FAIL: G2 report stage mismatch'
[ "$(report_value "$REPORT" mode)" = frozen_adapter_live_full_context_equivalence ] || fail 'FAIL: G2 report mode mismatch'
[ "$(report_value "$REPORT" adapter_values_finite)" = yes ] || fail 'FAIL: G2 adapter values are non-finite'
[ "$(report_value "$REPORT" adapter_sha256)" = "$EXPECTED_ADAPTER" ] || fail 'FAIL: G2 report adapter identity mismatch'
[ "$(report_value "$REPORT" g1_report_sha256)" = "$EXPECTED_G1_REPORT" ] || fail 'FAIL: G2 report G1 identity mismatch'
[ "$(report_value "$REPORT" training_checkpoint_loaded)" = no ] || fail 'FAIL: G2 loaded a training checkpoint'
[ "$(report_value "$REPORT" optimizer_state_loaded)" = no ] || fail 'FAIL: G2 loaded optimizer state'
[ "$(report_value "$REPORT" hidden_state_cache_used_for_inference)" = no ] || fail 'FAIL: G2 used a hidden-state cache'
[ "$(report_value "$REPORT" train_records_evaluated)" = 20 ] || fail 'FAIL: G2 train record count mismatch'
[ "$(report_value "$REPORT" train_target_rows_evaluated)" = 111 ] || fail 'FAIL: G2 train target-row count mismatch'
[ "$(report_value "$REPORT" validation_records_evaluated)" = 5 ] || fail 'FAIL: G2 validation record count mismatch'
[ "$(report_value "$REPORT" validation_target_rows_evaluated)" = 33 ] || fail 'FAIL: G2 validation target-row count mismatch'
[ "$(report_value "$REPORT" test_records_evaluated)" = 0 ] || fail 'FAIL: G2 evaluated test records'
[ "$(report_value "$REPORT" test_target_rows_evaluated)" = 0 ] || fail 'FAIL: G2 evaluated test targets'
[ "$(report_value "$REPORT" gradient_rows_consumed)" = 0 ] || fail 'FAIL: G2 consumed gradients'
[ "$(report_value "$REPORT" optimizer_updates)" = 0 ] || fail 'FAIL: G2 performed optimizer updates'
[ "$(report_value "$REPORT" train_loss_reference_match)" = yes ] || fail 'FAIL: G2 train loss reference mismatch'
[ "$(report_value "$REPORT" validation_loss_reference_match)" = yes ] || fail 'FAIL: G2 validation loss reference mismatch'
[ "$(report_value "$REPORT" train_top1_reference_match)" = yes ] || fail 'FAIL: G2 train top-1 reference mismatch'
[ "$(report_value "$REPORT" validation_top1_reference_match)" = yes ] || fail 'FAIL: G2 validation top-1 reference mismatch'
[ "$(report_value "$REPORT" gate_adapter_identity)" = yes ] || fail 'FAIL: G2 adapter identity gate failed'
[ "$(report_value "$REPORT" gate_live_full_context)" = yes ] || fail 'FAIL: G2 live full-context gate failed'
[ "$(report_value "$REPORT" gate_no_training_checkpoint)" = yes ] || fail 'FAIL: G2 training-checkpoint gate failed'
[ "$(report_value "$REPORT" gate_no_optimizer)" = yes ] || fail 'FAIL: G2 optimizer gate failed'
[ "$(report_value "$REPORT" gate_no_test_access)" = yes ] || fail 'FAIL: G2 test-isolation gate failed'
[ "$(report_value "$REPORT" gate_all_finite)" = yes ] || fail 'FAIL: G2 finite-values gate failed'
[ "$(report_value "$REPORT" gate_stage4g2)" = PASS ] || fail 'FAIL: G2 report gate failed'

printf '%s\n' 'train=7.026346899236 (4/111)' 'validation=8.328697455428 (4/33)' 'test_access=none' 'Stage4G2=PASS'
