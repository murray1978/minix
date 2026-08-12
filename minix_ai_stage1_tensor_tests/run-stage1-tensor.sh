#!/bin/sh

set -u

ITERATIONS="${ITERATIONS:-500}"
BENCH_REPS="${BENCH_REPS:-32}"
CC_VALUE="${CC:-clang}"
RESULT_DIR="${RESULT_DIR:-results}"
TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"
LOG_FILE="${RESULT_DIR}/stage1-tensor-${TIMESTAMP}.log"
LATEST_FILE="${RESULT_DIR}/stage1-tensor-latest.log"

mkdir -p "$RESULT_DIR" || exit 1

make CC="$CC_VALUE" || exit 1

{
    echo "MINIX AI Stage 1 tensor suite"
    echo "date: $(date)"
    echo "compiler: $CC_VALUE"
    echo "iterations: $ITERATIONS"
    echo "bench repetitions: $BENCH_REPS"
    echo
    ./ai_stage1_tensor --verbose \
        --iterations "$ITERATIONS" \
        --bench-reps "$BENCH_REPS"
} >"$LOG_FILE" 2>&1
STATUS=$?

cat "$LOG_FILE"
rm -f "$LATEST_FILE"
ln -s "$(basename "$LOG_FILE")" "$LATEST_FILE" 2>/dev/null ||
    cp "$LOG_FILE" "$LATEST_FILE"

printf '\nSaved: %s\n' "$LOG_FILE"
exit "$STATUS"
