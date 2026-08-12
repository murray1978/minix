#!/bin/sh
set -eu

MODEL_URL='https://huggingface.co/karpathy/tinyllamas/resolve/58e1696980dcbdf80b2dfe876819104a174f5e78/stories15M.bin?download=true'
TOKENIZER_URL='https://raw.githubusercontent.com/karpathy/llama2.c/refs/heads/master/tokenizer.bin'
MODEL_SHA256='cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a'

#if command -v curl >/dev/null 2>&1; then
#    curl -L --fail -o -k stories15M.bin "$MODEL_URL"
#    curl -L --fail -o -k tokenizer.bin "$TOKENIZER_URL"
if command -v wget >/dev/null 2>&1; then
    wget -O stories15M.bin "$MODEL_URL"
    wget -O tokenizer.bin "$TOKENIZER_URL"
else
    echo "curl or wget is required" >&2
    exit 1
fi

if command -v sha256sum >/dev/null 2>&1; then
    printf '%s  %s\n' "$MODEL_SHA256" stories15M.bin | sha256sum -c -
elif command -v shasum >/dev/null 2>&1; then
    actual=$(shasum -a 256 stories15M.bin | awk '{print $1}')
    test "$actual" = "$MODEL_SHA256"
else
    echo "warning: no SHA256 utility found; model hash not checked" >&2
fi

ls -l stories15M.bin tokenizer.bin
