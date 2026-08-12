#!/bin/sh
set -eu

MODEL=${MODEL:-stories15M.bin}
TOKENIZER=${TOKENIZER:-tokenizer.bin}
STEPS=${STEPS:-128}
TEMPERATURE=${TEMPERATURE:-0}
TOPP=${TOPP:-0.9}
SEED=${SEED:-1}
PROMPT=${PROMPT:-Once upon a time}

exec ./minix_llama "$MODEL" \
    -z "$TOKENIZER" \
    -n "$STEPS" \
    -t "$TEMPERATURE" \
    -p "$TOPP" \
    -s "$SEED" \
    -i "$PROMPT" \
    -v
