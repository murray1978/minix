#!/bin/sh

set -u

ITERATIONS="${ITERATIONS:-100}"
BENCH_TOKENS="${BENCH_TOKENS:-4096}"
RESULT_DIR="${RESULT_DIR:-results}"
TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"
LOG="${RESULT_DIR}/stage2-tiny-llm-${TIMESTAMP}.log"
LATEST="${RESULT_DIR}/stage2-tiny-llm-latest.log"

mkdir -p "$RESULT_DIR" || exit 1

make CC="${CC:-clang}" || exit 1

./ai_stage2_tiny_llm \
    --iterations "$ITERATIONS" \
    --bench-tokens "$BENCH_TOKENS" \
    --verbose 2>&1 | tee "$LOG"
STATUS=$?

rm -f "$LATEST"
ln -s "$(basename "$LOG")" "$LATEST" 2>/dev/null || cp "$LOG" "$LATEST"

exit "$STATUS"
