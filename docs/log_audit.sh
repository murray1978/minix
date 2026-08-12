#!/bin/sh

# collect-minix-logs.sh
#
# Collect a MINIX system baseline for the AI Stage 1 audit.
#
# Run as root:
#   su root -c '/usr/src/docs/audit/collect-minix-logs.sh'
#
# Optional output directory:
#   su root -c '/usr/src/docs/audit/collect-minix-logs.sh /tmp/minix-logs'

set -u

DEFAULT_OUTPUT_DIR="/usr/src/docs/audit/logs"
OUTPUT_DIR="${1:-$DEFAULT_OUTPUT_DIR}"

TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"
FULL_LOG="${OUTPUT_DIR}/minix-baseline-${TIMESTAMP}.log"
FILTERED_LOG="${OUTPUT_DIR}/minix-baseline-${TIMESTAMP}-filtered.log"
LATEST_LOG="${OUTPUT_DIR}/minix-baseline-latest.log"
LATEST_FILTERED="${OUTPUT_DIR}/minix-baseline-latest-filtered.log"

ERROR_PATTERN='panic|fatal|assert|error|failed|failure|warning|restart|reincarn|killed|segmentation|out of memory|ENOMEM|VM:|VFS:|PM:|RS:'

create_output_directory()
{
    if [ ! -d "$OUTPUT_DIR" ]; then
        mkdir -p "$OUTPUT_DIR" || {
            echo "Unable to create output directory: $OUTPUT_DIR" >&2
            exit 1
        }
    fi
}

check_root()
{
    USER_ID="$(id -u 2>/dev/null || echo unknown)"

    if [ "$USER_ID" != "0" ]; then
        echo "Warning: this script is not running as root." >&2
        echo "Some kernel or system log information may be unavailable." >&2
        echo >&2
    fi
}

section()
{
    echo
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

run_command()
{
    DESCRIPTION="$1"
    shift

    section "$DESCRIPTION"

    echo "\$ $*"
    "$@" 2>&1

    STATUS=$?

    if [ "$STATUS" -ne 0 ]; then
        echo
        echo "[Command exited with status $STATUS]"
    fi

    return 0
}

collect_logs()
{
    {
        section "MINIX LOG BASELINE"

        echo "Collection time:"
        date

        echo
        echo "Hostname:"
        hostname 2>&1 || true

        echo
        echo "Output file:"
        echo "$FULL_LOG"

        run_command "SYSTEM VERSION" uname -a
        run_command "SYSTEM UPTIME" uptime
        run_command "CURRENT USER" id

        section "/var/log CONTENTS"

        if [ -d /var/log ]; then
            ls -la /var/log 2>&1
        else
            echo "/var/log does not exist."
        fi

        section "KERNEL MESSAGE BUFFER"

        if command -v dmesg >/dev/null 2>&1; then
            dmesg 2>&1
        else
            echo "dmesg command was not found."
        fi

        section "SYSTEM LOG: /var/log/messages"

        if [ -r /var/log/messages ]; then
            echo "Showing the last 1000 lines:"
            tail -n 1000 /var/log/messages 2>&1
        else
            echo "/var/log/messages is missing or unreadable."
        fi

        run_command "PROCESS LIST" ps -ax
        run_command "FILESYSTEM USAGE" df
        run_command "MOUNTED FILESYSTEMS" mount

        section "MEMORY INFORMATION"

        if command -v top >/dev/null 2>&1; then
            echo "The top command is installed."
            echo "Interactive top output is not collected automatically."
        else
            echo "top command was not found."
        fi

        if command -v sysenv >/dev/null 2>&1; then
            echo
            echo "Boot environment:"
            sysenv 2>&1 || true
        fi

        section "AUDIT COLLECTION COMPLETE"

        echo "Full log:"
        echo "$FULL_LOG"

        echo
        echo "Filtered log:"
        echo "$FILTERED_LOG"

    } >"$FULL_LOG" 2>&1
}

filter_logs()
{
    if grep -Ein "$ERROR_PATTERN" "$FULL_LOG" >"$FILTERED_LOG" 2>/dev/null; then
        :
    else
        # grep returns 1 when there are no matches.
        : >"$FILTERED_LOG"
    fi
}

create_latest_links()
{
    rm -f "$LATEST_LOG" "$LATEST_FILTERED"

    ln -s "$(basename "$FULL_LOG")" "$LATEST_LOG" 2>/dev/null ||
        cp "$FULL_LOG" "$LATEST_LOG"

    ln -s "$(basename "$FILTERED_LOG")" "$LATEST_FILTERED" 2>/dev/null ||
        cp "$FILTERED_LOG" "$LATEST_FILTERED"
}

set_permissions()
{
    chmod 644 "$FULL_LOG" "$FILTERED_LOG" 2>/dev/null || true
    chmod 644 "$LATEST_LOG" "$LATEST_FILTERED" 2>/dev/null || true
}

print_summary()
{
    FULL_LINES="$(wc -l <"$FULL_LOG" | tr -d ' ')"
    FILTERED_LINES="$(wc -l <"$FILTERED_LOG" | tr -d ' ')"

    echo "MINIX log collection complete."
    echo
    echo "Full log:"
    echo "  $FULL_LOG"
    echo "  $FULL_LINES lines"
    echo
    echo "Filtered log:"
    echo "  $FILTERED_LOG"
    echo "  $FILTERED_LINES matching lines"
    echo
    echo "Latest copies or links:"
    echo "  $LATEST_LOG"
    echo "  $LATEST_FILTERED"
}

check_root
create_output_directory
collect_logs
filter_logs
create_latest_links
set_permissions
print_summary

exit 0