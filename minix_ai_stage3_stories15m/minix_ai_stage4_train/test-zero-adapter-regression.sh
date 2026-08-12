#!/bin/sh

set -eu

CHECKPOINT="/usr/src/minix_ai_stage3_stories15m/stories15M.bin"
TOKENIZER="/usr/src/minix_ai_stage3_stories15m/tokenizer.bin"
ADAPTER="/usr/src/minix_ai_stage3_stories15m/minix_ai_stage4_train/stories15M.adapter.bin"
PROMPT="Once upon a time"
EXPECTED_SHA="cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a"

sha256_file() {
	if command -v sha256 >/dev/null 2>&1; then
		sha256 "$1" | awk '{print $4}'
		return 0
	fi
	if command -v sha256sum >/dev/null 2>&1; then
		sha256sum "$1" | awk '{print $1}'
		return 0
	fi
	if command -v openssl >/dev/null 2>&1; then
		OPENSSL_CONF=/dev/null openssl dgst -sha256 "$1" | awk '{print $NF}'
		return 0
	fi
	return 1
}

echo "[1/5] base checkpoint hash before"
before_hash=`sha256_file "$CHECKPOINT"`
echo "checkpoint_sha256_before=$before_hash"
if [ "$before_hash" != "$EXPECTED_SHA" ]; then
	echo "FAIL: unexpected checkpoint hash before test"
	exit 1
fi

echo "[2/5] initialize adapter"
./stage4_train_adapter "$CHECKPOINT" "$ADAPTER" -r 8 > /tmp/stage4_adapter_init.out
cat /tmp/stage4_adapter_init.out

echo "[3/5] base checkpoint hash after adapter init"
after_hash=`sha256_file "$CHECKPOINT"`
echo "checkpoint_sha256_after=$after_hash"
if [ "$after_hash" != "$EXPECTED_SHA" ]; then
	echo "FAIL: unexpected checkpoint hash after test"
	exit 1
fi
if [ "$before_hash" != "$after_hash" ]; then
	echo "FAIL: checkpoint hash changed"
	exit 1
fi

echo "[4/5] deterministic base inference token IDs"
/usr/src/minix_ai_stage3_stories15m/minix_llama \
	"$CHECKPOINT" -z "$TOKENIZER" -t 0 -s 1 -n 64 \
	-i "$PROMPT" -T /tmp/stage4_base_tokens.txt >/tmp/stage4_base_text.out

echo "[5/5] deterministic zero-adapter inference token IDs"
/usr/src/minix_ai_stage3_stories15m/minix_llama \
	"$CHECKPOINT" -z "$TOKENIZER" -t 0 -s 1 -n 64 \
	-i "$PROMPT" -a "$ADAPTER" -y 1.0 \
	-T /tmp/stage4_adapter_tokens.txt >/tmp/stage4_adapter_text.out

if cmp -s /tmp/stage4_base_tokens.txt /tmp/stage4_adapter_tokens.txt; then
	echo "PASS: base and zero-adapter token IDs match"
else
	echo "FAIL: base and zero-adapter token IDs differ"
	diff /tmp/stage4_base_tokens.txt /tmp/stage4_adapter_tokens.txt || true
	exit 1
fi

adapter_sha=`sha256_file "$ADAPTER"`
adapter_bytes=`wc -c < "$ADAPTER" | tr -d ' '`
echo "adapter_sha256=$adapter_sha"
echo "adapter_bytes=$adapter_bytes"

echo "PASS: zero-adapter regression"
