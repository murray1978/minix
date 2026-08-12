# Host validation

The package source was compiled and exercised in this environment using the
included synthetic version-0 checkpoint and tokenizer fixture.

Compiler/build output:

```text
rm -f minix_llama *.o *.trace
cc -O2 -std=c99 -D_POSIX_C_SOURCE=200112L -Wall -Wextra -Wpedantic -o minix_llama minix_llama.c -lm
```

Smoke-run stderr:

```text
generated 13 token(s) in 0.000130 seconds (100290.843 token/s)
```

Checkpoint/tokenizer check ending:

```text
checkpoint and tokenizer validation: PASS
```

Trace self-comparison:

```text
records compared: 16
logits per record: 260
maximum absolute error: 0
PASS
```

This validates source portability and internal file-format handling. It does
not replace the required native MINIX run with the real `stories15M.bin`.
