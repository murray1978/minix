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
D4_REPORT="stage4e-d4-a4-trajectory-report.txt"
D6_REPORT="stage4e-d6-a4-checkpoint-report.txt"
D6_CHECKPOINT="stage4e-d6-a4-step5-checkpoint.bin"
D7_REPORT="stage4e-d7-a4-resume-report.txt"
D7_TSV="stage4e-d7-a4-resume-trajectory.tsv"
F1_SOURCE="stage4e_f1_a4_fixed_horizon.c"
F1_EXECUTABLE="stage4e_f1_a4_fixed_horizon"
F1_TARGET="stage4eF1A4FixedHorizon"
EXPECTED_PROVENANCE="fixed-horizon-collapse-regression-2026-10-01-a"
REPORT="stage4e-f1-a4-fixed-horizon-report.txt"
TRAJECTORY="stage4e-f1-a4-fixed-horizon-trajectory.tsv"
CONSOLE_LOG="stage4e-f1-a4-fixed-horizon-console.txt"
BUILD_LOG=".stage4e-f1-a4-build.tmp"
RUN_LOG=".stage4e-f1-a4-run.tmp"

EXPECTED_BASE="cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a"
EXPECTED_TOKENIZER="50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361"
EXPECTED_DATASET="71744cea4de049e2e192cb2a4494c8a4a68e769b3a7f3b8ed4880433e4583b84"
EXPECTED_CACHE="d6288ad3d128c7182c20088aa57a1bca8bcaa1e13314fb0cdbd60eba14278def"
EXPECTED_D1="45ebaa750f17d02d80feed332b16c9c29c3782a544aed7544da72a62161c76a6"
EXPECTED_D2="c27aad7474df093f5a3ff781bb3619b71b3e284edda46290f5814c350e99d97e"
EXPECTED_D3="f418305f3fb9263480b8a80f34b02a966ebbb94f305f2aa5897d5c2f4356b94e"
EXPECTED_D4="d57d81ed4608df0c4a662404911c16a72fa475f0aea1459e68533e522cebace1"
EXPECTED_D6_REPORT="59e3370ab4420128caa46f1cf6a83714ffdb1485b57bd71d5658df90d41faf9a"
EXPECTED_D6_CHECKPOINT="ecb75d52881df04a0c47fcd4b852909a865c62cb54929ab33b230c7cf1e5ec89"
EXPECTED_D7_REPORT="05b78313572cfd14242185ffa5e7fdf535ad103f2c4bcc2ee4d0faf439f253f2"
EXPECTED_D7_TSV="ed5505733443f3b57b49db71dc87398bec860e04aeac5752a5e60e24e57fd46a"

log() {
    printf '%s\n' "$*"
    printf '%s\n' "$*" >> "$CONSOLE_LOG"
}

cleanup() {
    rm -f "$BUILD_LOG" "$RUN_LOG"
}

fail() {
    log "$*"
    log "Stage4E-F1-A4=FAIL"
    cleanup
    exit 1
}

check_file() {
    if [ ! -f "$1" ]; then fail "FAIL: missing file: $1"; fi
}

sha256_hex() {
    openssl dgst -sha256 "$1" 2>/dev/null | awk '{print $NF}'
}

report_value() {
    awk -F= -v wanted="$2" '$1 == wanted { print substr($0, index($0, "=") + 1); exit }' "$1"
}

if [ -e "$REPORT" ] || [ -e "$TRAJECTORY" ]; then
    printf '%s\n' "FAIL: refusing to overwrite an existing F1 output artifact" >&2
    exit 1
fi
: > "$CONSOLE_LOG" || exit 1
log "=============================================="
log "Stage 4E-F1-A4 fixed-horizon continuation"
log "=============================================="

check_file "$BASE_CHECKPOINT"
check_file "$TOKENIZER"
check_file "$DATASET"
check_file "$CACHE"
check_file "$D1_REPORT"
check_file "$D2_REPORT"
check_file "$D3_REPORT"
check_file "$D4_REPORT"
check_file "$D6_REPORT"
check_file "$D6_CHECKPOINT"
check_file "$D7_REPORT"
check_file "$D7_TSV"
check_file "$F1_SOURCE"
check_file "Makefile"

BASE_ACTUAL=$(sha256_hex "$BASE_CHECKPOINT")
TOKENIZER_ACTUAL=$(sha256_hex "$TOKENIZER")
DATASET_ACTUAL=$(sha256_hex "$DATASET")
CACHE_ACTUAL=$(sha256_hex "$CACHE")
D1_ACTUAL=$(sha256_hex "$D1_REPORT")
D2_ACTUAL=$(sha256_hex "$D2_REPORT")
D3_ACTUAL=$(sha256_hex "$D3_REPORT")
D4_ACTUAL=$(sha256_hex "$D4_REPORT")
D6_REPORT_ACTUAL=$(sha256_hex "$D6_REPORT")
D6_CHECKPOINT_ACTUAL=$(sha256_hex "$D6_CHECKPOINT")
D7_REPORT_ACTUAL=$(sha256_hex "$D7_REPORT")
D7_TSV_ACTUAL=$(sha256_hex "$D7_TSV")
if [ "$BASE_ACTUAL" != "$EXPECTED_BASE" ] ||
    [ "$TOKENIZER_ACTUAL" != "$EXPECTED_TOKENIZER" ] ||
    [ "$DATASET_ACTUAL" != "$EXPECTED_DATASET" ] ||
    [ "$CACHE_ACTUAL" != "$EXPECTED_CACHE" ] ||
    [ "$D1_ACTUAL" != "$EXPECTED_D1" ] ||
    [ "$D2_ACTUAL" != "$EXPECTED_D2" ] ||
    [ "$D3_ACTUAL" != "$EXPECTED_D3" ] ||
    [ "$D4_ACTUAL" != "$EXPECTED_D4" ] ||
    [ "$D6_REPORT_ACTUAL" != "$EXPECTED_D6_REPORT" ] ||
    [ "$D6_CHECKPOINT_ACTUAL" != "$EXPECTED_D6_CHECKPOINT" ] ||
    [ "$D7_REPORT_ACTUAL" != "$EXPECTED_D7_REPORT" ] ||
    [ "$D7_TSV_ACTUAL" != "$EXPECTED_D7_TSV" ]; then
    log "base_checkpoint_sha256=$BASE_ACTUAL"
    log "tokenizer_sha256=$TOKENIZER_ACTUAL"
    log "dataset_sha256=$DATASET_ACTUAL"
    log "cache_sha256=$CACHE_ACTUAL"
    log "d1_report_sha256=$D1_ACTUAL"
    log "d2_report_sha256=$D2_ACTUAL"
    log "d3_report_sha256=$D3_ACTUAL"
    log "d4_report_sha256=$D4_ACTUAL"
    log "d6_report_sha256=$D6_REPORT_ACTUAL"
    log "d6_checkpoint_sha256=$D6_CHECKPOINT_ACTUAL"
    log "d7_report_sha256=$D7_REPORT_ACTUAL"
    log "d7_trajectory_sha256=$D7_TSV_ACTUAL"
    fail "FAIL: a pinned F1 input identity differs"
fi
if [ "$(report_value "$D4_REPORT" d4_trajectory_pass)" != yes ] ||
    [ "$(report_value "$D6_REPORT" checkpoint_roundtrip_pass)" != yes ] ||
    [ "$(report_value "$D6_REPORT" checkpoint_sha256)" != "$EXPECTED_D6_CHECKPOINT" ] ||
    [ "$(report_value "$D7_REPORT" d7_resume_pass)" != yes ] ||
    [ "$(report_value "$D7_REPORT" deterministic_resume_trajectory_match)" != yes ]; then
    fail "FAIL: D4/D6/D7 reference acceptance fields mismatch"
fi
log "all_pinned_input_identities=PASS"

F1_SOURCE_SHA=$(sha256_hex "$F1_SOURCE")
if [ -z "$F1_SOURCE_SHA" ]; then fail "FAIL: unable to hash F1 source"; fi
rm -f "$F1_EXECUTABLE"
make "$F1_TARGET" CC=clang > "$BUILD_LOG" 2>&1
BUILD_STATUS=$?
cat "$BUILD_LOG" | tee -a "$CONSOLE_LOG"
if [ "$BUILD_STATUS" -ne 0 ]; then fail "FAIL: F1 executable build failed"; fi
check_file "$F1_EXECUTABLE"
if [ "$(sha256_hex "$F1_SOURCE")" != "$F1_SOURCE_SHA" ]; then fail "FAIL: F1 source changed during build"; fi
F1_EXECUTABLE_SHA=$(sha256_hex "$F1_EXECUTABLE")
if [ -z "$F1_EXECUTABLE_SHA" ]; then fail "FAIL: unable to hash F1 executable"; fi
log "f1_source_sha256=$F1_SOURCE_SHA"
log "f1_executable_sha256=$F1_EXECUTABLE_SHA"

"./$F1_EXECUTABLE" "$BASE_CHECKPOINT" -z "$TOKENIZER" \
    -d "$DATASET" -c "$CACHE" -1 "$D1_REPORT" -2 "$D2_REPORT" \
    -3 "$D3_REPORT" -4 "$D4_REPORT" -6 "$D6_REPORT" \
    -7 "$D6_CHECKPOINT" -8 "$D7_REPORT" -9 "$D7_TSV" > "$RUN_LOG" 2>&1
RUN_STATUS=$?
cat "$RUN_LOG" | tee -a "$CONSOLE_LOG"
if [ "$(report_value "$RUN_LOG" f1_build_provenance)" != "$EXPECTED_PROVENANCE" ]; then
    fail "FAIL: F1 executable provenance mismatch"
fi
if [ "$RUN_STATUS" -ne 0 ]; then fail "FAIL: F1 fixed-horizon run failed"; fi
check_file "$REPORT"
check_file "$TRAJECTORY"

if [ "$(sha256_hex "$D4_REPORT")" != "$D4_ACTUAL" ] ||
    [ "$(sha256_hex "$D6_REPORT")" != "$D6_REPORT_ACTUAL" ] ||
    [ "$(sha256_hex "$D6_CHECKPOINT")" != "$D6_CHECKPOINT_ACTUAL" ] ||
    [ "$(sha256_hex "$D7_REPORT")" != "$D7_REPORT_ACTUAL" ] ||
    [ "$(sha256_hex "$D7_TSV")" != "$D7_TSV_ACTUAL" ]; then
    fail "FAIL: an accepted reference artifact changed during F1"
fi
if [ "$(report_value "$REPORT" f1_fixed_horizon_pass)" != yes ] ||
    [ "$(report_value "$REPORT" step5_checkpoint_metrics_match)" != yes ] ||
    [ "$(report_value "$REPORT" steps6_to_10_d7_trajectory_match)" != yes ] ||
    [ "$(report_value "$REPORT" final_optimizer_step)" != 50 ] ||
    [ "$(report_value "$REPORT" total_updates_executed)" != 45 ] ||
    [ "$(report_value "$REPORT" test_gradient_rows_seen)" != 0 ] ||
    [ "$(report_value "$REPORT" test_evaluation_rows_seen)" != 0 ]; then
    fail "FAIL: F1 report failed fixed-horizon acceptance gates"
fi
awk -F '\t' 'NR == 1 { next } { step=$1+0; expected=(step >= 6 && step <= 10) || (step >= 15 && step <= 50 && step % 5 == 0); actual=($13 == "yes"); if ($2 != 111 || $3 != 0 || $4 != 0 || expected != actual) bad++ ; rows++ } END { if (rows != 45 || bad != 0) exit 1 }' "$TRAJECTORY" ||
    fail "FAIL: F1 trajectory schedule or split-isolation check failed"
log "optimizer_step=50"
log "updates_executed=45"
log "test_gradient_rows_seen=0"
log "test_evaluation_rows_seen=0"
log "validation_acceptance_is_observational=yes"
log "Stage4E-F1-A4=PASS"
cleanup
exit 0
