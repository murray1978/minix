#!/usr/bin/env python3
"""Create a tiny version-0 llama2.c checkpoint and tokenizer smoke fixture."""

from __future__ import annotations

import argparse
import math
import random
import struct
from pathlib import Path


def write_tokenizer(path: Path, vocab_size: int) -> None:
    pieces = ["<unk>", "<s>", "</s>"]
    pieces.extend(f"<0x{i:02X}>" for i in range(256))
    pieces.append(" ")
    if len(pieces) != vocab_size:
        raise AssertionError((len(pieces), vocab_size))

    encoded = [piece.encode("utf-8") for piece in pieces]
    max_length = max(map(len, encoded))

    with path.open("wb") as f:
        f.write(struct.pack("<I", max_length))
        for i, piece in enumerate(encoded):
            score = -float(i)
            f.write(struct.pack("<fI", score, len(piece)))
            f.write(piece)


def product(*values: int) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dim = 8
    hidden = 16
    layers = 2
    heads = 2
    kv_heads = 2
    vocab = 260
    seq = 32
    head_size = dim // heads
    kv_dim = dim * kv_heads // heads
    shared = True

    counts = [
        vocab * dim,
        layers * dim,
        layers * dim * dim,
        layers * dim * kv_dim,
        layers * dim * kv_dim,
        layers * dim * dim,
        layers * dim,
        layers * hidden * dim,
        layers * dim * hidden,
        layers * hidden * dim,
        dim,
        seq * head_size // 2,
        seq * head_size // 2,
    ]
    if not shared:
        counts.append(vocab * dim)

    total = sum(counts)
    rng = random.Random(1234)
    weights = [(rng.random() - 0.5) * 0.08 for _ in range(total)]

    checkpoint = args.output_dir / "smoke_model.bin"
    with checkpoint.open("wb") as f:
        f.write(
            struct.pack(
                "<7i",
                dim,
                hidden,
                layers,
                heads,
                kv_heads,
                vocab if shared else -vocab,
                seq,
            )
        )
        f.write(struct.pack(f"<{total}f", *weights))

    tokenizer = args.output_dir / "smoke_tokenizer.bin"
    write_tokenizer(tokenizer, vocab)

    print(checkpoint)
    print(tokenizer)
    print(f"weights={total}, checkpoint_bytes={checkpoint.stat().st_size}")


if __name__ == "__main__":
    main()
