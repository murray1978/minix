#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" 2>/dev/null && pwd)
cd "$SCRIPT_DIR" || exit 1

CHECKPOINT="stage4e-f2-a4-selected-step20-audit-checkpoint.bin"
F2_REPORT="stage4e-f2-a4-selected-step20-audit-test-report.txt"
SOURCE="stage4g1_export_selected_adapter.c"
EXECUTABLE="stage4g1_export_selected_adapter"
ADAPTER="stage4g1-selected-step20.adapter.bin"
REPORT="stage4g1-selected-step20-export-report.txt"
BUILD_LOG=".stage4g1-build.tmp"
EXPECTED_CHECKPOINT="87148919dfc4d5da22e7a8494216dd47f9b30064bae9a347bee6e37ddd50b1f0"
EXPECTED_F2_REPORT="b9d8c727538d94554cfe9851d14f92cbb896a8f9faf08c929d5ad678320ee707"

sha256_hex() { openssl dgst -sha256 "$1" 2>/dev/null | awk '{print $NF}'; }
report_value() { awk -F= -v wanted="$2" '$1 == wanted { print substr($0, index($0, "=") + 1); exit }' "$1"; }
fail() {
    printf '%s\n' "$*" 'Stage4G1=FAIL' >&2
    rm -f "$ADAPTER" "$REPORT" "$BUILD_LOG"
    exit 1
}
check_file() { [ -f "$1" ] || fail "FAIL: missing file: $1"; }

if [ -e "$ADAPTER" ] || [ -e "$REPORT" ]; then
    printf '%s\n' 'FAIL: refusing to overwrite an existing G1 output artifact' >&2
    exit 1
fi
check_file "$CHECKPOINT"
check_file "$F2_REPORT"
check_file "$SOURCE"
check_file stage4_adapter_train.c
check_file stage4_adapter_train.h
check_file ../minix_llama.c

CHECKPOINT_HASH=$(sha256_hex "$CHECKPOINT")
F2_REPORT_HASH=$(sha256_hex "$F2_REPORT")
[ "$CHECKPOINT_HASH" = "$EXPECTED_CHECKPOINT" ] || fail 'FAIL: accepted F2 checkpoint SHA-256 mismatch'
[ "$F2_REPORT_HASH" = "$EXPECTED_F2_REPORT" ] || fail 'FAIL: accepted F2 report SHA-256 mismatch'

rm -f "$EXECUTABLE"
clang -O2 -std=c99 -D_POSIX_C_SOURCE=200112L -D_LARGEFILE_SOURCE -Wno-unused-function -o "$EXECUTABLE" "$SOURCE" stage4_adapter_train.c -lm > "$BUILD_LOG" 2>&1
BUILD_STATUS=$?
cat "$BUILD_LOG"
[ "$BUILD_STATUS" -eq 0 ] || fail 'FAIL: G1 clang build failed'
check_file "$EXECUTABLE"

"./$EXECUTABLE"
RUN_STATUS=$?
[ "$RUN_STATUS" -eq 0 ] || fail 'FAIL: G1 export execution failed'
check_file "$ADAPTER"
check_file "$REPORT"
[ "$(report_value "$REPORT" input_checkpoint_sha256)" = "$EXPECTED_CHECKPOINT" ] || fail 'FAIL: G1 report checkpoint identity mismatch'
[ "$(report_value "$REPORT" f2_report_sha256)" = "$EXPECTED_F2_REPORT" ] || fail 'FAIL: G1 report F2 identity mismatch'
[ "$(report_value "$REPORT" checkpoint_step20_verified)" = yes ] || fail 'FAIL: G1 step-20 verification failed'
[ "$(report_value "$REPORT" deployment_adapter_roundtrip_bitwise_match)" = yes ] || fail 'FAIL: G1 adapter round-trip gate failed'
[ "$(report_value "$REPORT" A_byte_mismatches)" = 0 ] || fail 'FAIL: G1 A byte mismatches'
[ "$(report_value "$REPORT" B_byte_mismatches)" = 0 ] || fail 'FAIL: G1 B byte mismatches'
[ "$(report_value "$REPORT" total_adapter_byte_mismatches)" = 0 ] || fail 'FAIL: G1 total byte mismatches'
[ "$(report_value "$REPORT" all_exported_values_finite)" = yes ] || fail 'FAIL: G1 exported values are non-finite'
[ "$(report_value "$REPORT" all_reloaded_values_finite)" = yes ] || fail 'FAIL: G1 reloaded values are non-finite'
[ "$(report_value "$REPORT" train_rows_evaluated)" = 0 ] || fail 'FAIL: G1 unexpectedly evaluated training rows'
[ "$(report_value "$REPORT" validation_rows_evaluated)" = 0 ] || fail 'FAIL: G1 unexpectedly evaluated validation rows'
[ "$(report_value "$REPORT" test_rows_evaluated)" = 0 ] || fail 'FAIL: G1 unexpectedly evaluated test rows'
[ "$(report_value "$REPORT" gradient_rows_consumed)" = 0 ] || fail 'FAIL: G1 unexpectedly consumed gradients'
[ "$(report_value "$REPORT" optimizer_updates)" = 0 ] || fail 'FAIL: G1 unexpectedly updated the optimizer'
[ "$(report_value "$REPORT" gate_stage4g1)" = PASS ] || fail 'FAIL: G1 report gate failed'

rm -f "$BUILD_LOG"
printf '%s\n' 'Stage4G1=PASS'
exit 0
