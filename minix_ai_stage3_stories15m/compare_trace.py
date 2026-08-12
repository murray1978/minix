#!/usr/bin/env python3
"""Compare two MLTR logit traces produced by minix_llama."""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

HEADER = struct.Struct("<4sIIII")
RECORD = struct.Struct("<II")
MAGIC = b"MLTR"
VERSION = 1


def read_trace(path: Path):
    data = path.read_bytes()
    if len(data) < HEADER.size:
        raise ValueError(f"{path}: truncated header")

    magic, version, vocab, entries, _reserved = HEADER.unpack_from(data, 0)
    if magic != MAGIC or version != VERSION:
        raise ValueError(f"{path}: unsupported trace format")

    record_bytes = RECORD.size + vocab * 4
    expected = HEADER.size + entries * record_bytes
    if len(data) != expected:
        raise ValueError(
            f"{path}: expected {expected} bytes for {entries} records, got {len(data)}"
        )

    offset = HEADER.size
    records = []
    floats = struct.Struct(f"<{vocab}f")
    for _ in range(entries):
        pos, token = RECORD.unpack_from(data, offset)
        offset += RECORD.size
        logits = floats.unpack_from(data, offset)
        offset += vocab * 4
        records.append((pos, token, logits))

    return vocab, records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    ref_vocab, ref_records = read_trace(args.reference)
    cand_vocab, cand_records = read_trace(args.candidate)

    if ref_vocab != cand_vocab:
        print(f"FAIL: vocabulary differs: {ref_vocab} vs {cand_vocab}")
        return 1
    if len(ref_records) != len(cand_records):
        print(f"FAIL: record count differs: {len(ref_records)} vs {len(cand_records)}")
        return 1

    maximum = 0.0
    worst = None

    for record_index, (left, right) in enumerate(zip(ref_records, cand_records)):
        lpos, ltoken, llogits = left
        rpos, rtoken, rlogits = right
        if (lpos, ltoken) != (rpos, rtoken):
            print(
                "FAIL: trace path differs at record "
                f"{record_index}: {(lpos, ltoken)} vs {(rpos, rtoken)}"
            )
            return 1

        for token_index, (a, b) in enumerate(zip(llogits, rlogits)):
            if not (math.isfinite(a) and math.isfinite(b)):
                if a == b:
                    continue
                print(
                    f"FAIL: non-finite mismatch at record {record_index}, "
                    f"token {token_index}: {a!r} vs {b!r}"
                )
                return 1
            error = abs(a - b)
            if error > maximum:
                maximum = error
                worst = (record_index, lpos, token_index, a, b)

    print(f"records compared: {len(ref_records)}")
    print(f"logits per record: {ref_vocab}")
    print(f"maximum absolute error: {maximum:.9g}")
    if worst is not None:
        print(
            "worst location: "
            f"record={worst[0]} position={worst[1]} token={worst[2]} "
            f"reference={worst[3]:.9g} candidate={worst[4]:.9g}"
        )

    if maximum > args.tolerance:
        print(f"FAIL: error exceeds tolerance {args.tolerance:g}")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
