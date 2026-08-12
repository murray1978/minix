#!/bin/sh
set -eu

CHECKER=./stage4e_dataset_check
DATASET=./stage4e-one-record-test.records
TOKENIZER=/usr/src/minix_ai_stage3_stories15m/tokenizer.bin
OUT=./stage4e-one-record-report.txt

$CHECKER -d "$DATASET" -z "$TOKENIZER" -o "$OUT" >/dev/null

require_line() {
	key="$1"
	value="$2"
	actual=$(grep "^${key}=" "$OUT" | sed -e "s/^${key}=//")
	if [ -z "$actual" ]; then
		echo "missing key: $key" >&2
		exit 1
	fi
	if [ "$actual" != "$value" ]; then
		echo "mismatch for $key: expected '$value' got '$actual'" >&2
		exit 1
	fi
}

require_line total_records 1
require_line total_prompt_tokens 17
require_line total_visible_target_tokens 2
require_line total_training_target_tokens 3
require_line minimum_sequence_length 20
require_line maximum_sequence_length 20
require_line malformed_records 0

echo "PASS: stage4e one-record regression"
