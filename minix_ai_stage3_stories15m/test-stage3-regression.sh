#!/bin/sh

# Stage 3 regression guardrail for stories15M inference.
# This script is read-only with respect to model artifacts.

set -eu

CHECKPOINT="stories15M.bin"
TOKENIZER="tokenizer.bin"
PROMPT="Once upon a time"
EXPECTED_BYTES=60816028
EXPECTED_SHA256="cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a"
MIN_OUTPUT_CHARS=16

PASS=0
FAIL=0

pass() {
	PASS=$((PASS + 1))
	echo "PASS: $1"
}

fail() {
	FAIL=$((FAIL + 1))
	echo "FAIL: $1"
}

have_cmd() {
	command -v "$1" >/dev/null 2>&1
}

sha256_file() {
	if have_cmd sha256; then
		sha256 "$1" | awk '{print $4}'
		return 0
	fi
	if have_cmd sha256sum; then
		sha256sum "$1" | awk '{print $1}'
		return 0
	fi
	if have_cmd openssl; then
		openssl dgst -sha256 "$1" | awk '{print $NF}'
		return 0
	fi
	return 1
}

filesize_bytes() {
	if have_cmd stat; then
		# MINIX stat supports -f '%z' in this environment.
		stat -f "%z" "$1"
		return 0
	fi
	wc -c < "$1" | tr -d ' '
}

if [ ! -r "$CHECKPOINT" ]; then
	echo "FAIL: missing checkpoint $CHECKPOINT"
	exit 1
fi
if [ ! -r "$TOKENIZER" ]; then
	echo "FAIL: missing tokenizer $TOKENIZER"
	exit 1
fi

bytes=`filesize_bytes "$CHECKPOINT"`
if [ "$bytes" = "$EXPECTED_BYTES" ]; then
	pass "checkpoint byte length"
else
	fail "checkpoint byte length (got $bytes expected $EXPECTED_BYTES)"
fi

sha=""
if sha=`sha256_file "$CHECKPOINT"`; then
	if [ "$sha" = "$EXPECTED_SHA256" ]; then
		pass "checkpoint SHA-256"
	else
		fail "checkpoint SHA-256 mismatch (got $sha)"
	fi
else
	fail "no SHA-256 command available (need sha256, sha256sum, or openssl)"
fi

if ./minix_llama "$CHECKPOINT" -z "$TOKENIZER" -c -i "$PROMPT" >/tmp/stage3_check.out 2>/tmp/stage3_check.err; then
	pass "checkpoint/tokenizer validation"
else
	fail "checkpoint/tokenizer validation"
fi

if ./minix_llama "$CHECKPOINT" -z "$TOKENIZER" -t 0 -n 64 -i "$PROMPT" >/tmp/stage3_gen.out 2>/tmp/stage3_gen.err; then
	pass "deterministic greedy generation"
else
	fail "deterministic greedy generation"
fi

gen_text=`cat /tmp/stage3_gen.out`
if [ "${#gen_text}" -ge "$MIN_OUTPUT_CHARS" ]; then
	pass "generated non-empty output"
else
	fail "generated output too short"
fi

if echo "$gen_text" | grep "<unk><unk><unk><unk><unk><unk><unk><unk>" >/dev/null 2>&1; then
	fail "repeated <unk> sequence detected"
else
	pass "no repeated <unk> run"
fi

if [ -c /dev/ai_driver ]; then
	if echo "-t 0" > /dev/ai_driver 2>/dev/null && \
	   echo "-n 64" > /dev/ai_driver 2>/dev/null && \
	   echo "i $PROMPT" > /dev/ai_driver 2>/dev/null; then
		resp=`cat /dev/ai_driver`
		if echo "$resp" | grep "backend=ai_model" >/dev/null 2>&1; then
			pass "/dev/ai_driver integration"
		else
			fail "/dev/ai_driver missing backend marker"
		fi
	else
		fail "/dev/ai_driver optional check failed"
	fi
else
	echo "SKIP: /dev/ai_driver optional check (device missing)"
fi

if [ "$FAIL" -eq 0 ]; then
	echo "SUMMARY: PASS ($PASS checks)"
	exit 0
fi

echo "SUMMARY: FAIL ($FAIL failed, $PASS passed)"
exit 1
