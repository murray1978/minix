#!/bin/sh

# Build and run tests 1-11, preserving a timestamped result log.
# Usage:
#   ./run-stage1-gate.sh
#   MAX_MIB=64 ./run-stage1-gate.sh

set -u

MAX_MIB="${MAX_MIB:-128}"
WORKDIR="${WORKDIR:-.}"
RESULT_DIR="${RESULT_DIR:-results}"
TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"
RESULT_LOG="${RESULT_DIR}/stage1-gate-${TIMESTAMP}.log"
LATEST_LOG="${RESULT_DIR}/stage1-gate-latest.log"

mkdir -p "$RESULT_DIR" || exit 1

if [ ! -x ./ai_stage1_probe ]; then
    echo "Building ai_stage1_probe..."
    make || exit $?
fi

{
    echo "MINIX AI Stage 1 platform gate"
    echo "Date: $(date)"
    echo "System: $(uname -a)"
    echo "MAX_MIB=$MAX_MIB"
    echo
    ./ai_stage1_probe --max-mib "$MAX_MIB" --workdir "$WORKDIR"
} >"$RESULT_LOG" 2>&1
STATUS=$?

cat "$RESULT_LOG"

rm -f "$LATEST_LOG"
ln -s "$(basename "$RESULT_LOG")" "$LATEST_LOG" 2>/dev/null ||     cp "$RESULT_LOG" "$LATEST_LOG"

printf '
Result log: %s
' "$RESULT_LOG"
exit "$STATUS"
