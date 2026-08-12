#!/usr/bin/env python3
"""Generate a deterministic tiny Llama-style checkpoint and reference trace.

The generated files are consumed by ai_stage2_tiny_llm.c on MINIX.
Requires Python 3 and NumPy only when regenerating the bundled artifacts.
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import numpy as np

MODEL_MAGIC = 0x4D584C31  # "MXL1"
REF_MAGIC = 0x4D585231    # "MXR1"
VERSION = 1
ENDIAN_TAG = 0x01020304

VOCAB = 256
CONTEXT = 16
D_MODEL = 16
N_HEADS = 4
N_LAYERS = 2
D_FF = 32
PROMPT = b"MINIX"
GEN_LEN = 8
SEED = 3300


def fnv1a32(data: bytes) -> int:
    value = 2166136261
    for byte in data:
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def f32(x):
    return np.asarray(x, dtype=np.float32)


def rmsnorm(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    mean_square = np.mean(x * x, dtype=np.float32)
    scale = np.float32(1.0) / np.sqrt(np.float32(mean_square + np.float32(1.0e-5)))
    return f32(weight * x * scale)


def silu(x: np.ndarray) -> np.ndarray:
    return f32(x / (np.float32(1.0) + np.exp(-x, dtype=np.float32)))


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = f32(values - np.max(values))
    exps = np.exp(shifted, dtype=np.float32)
    return f32(exps / np.sum(exps, dtype=np.float32))


def apply_rope(q: np.ndarray, k: np.ndarray, pos: int) -> None:
    head_size = D_MODEL // N_HEADS
    for head in range(N_HEADS):
        base = head * head_size
        for i in range(0, head_size, 2):
            freq = np.float32(math.pow(10000.0, -float(i) / float(head_size)))
            angle = np.float32(pos) * freq
            c = np.float32(math.cos(float(angle)))
            s = np.float32(math.sin(float(angle)))
            q0 = np.float32(q[base + i])
            q1 = np.float32(q[base + i + 1])
            k0 = np.float32(k[base + i])
            k1 = np.float32(k[base + i + 1])
            q[base + i] = np.float32(q0 * c - q1 * s)
            q[base + i + 1] = np.float32(q0 * s + q1 * c)
            k[base + i] = np.float32(k0 * c - k1 * s)
            k[base + i + 1] = np.float32(k0 * s + k1 * c)


def matvec(weight: np.ndarray, vector: np.ndarray) -> np.ndarray:
    # Explicit float32 accumulation order matching the C implementation.
    out = np.empty(weight.shape[0], dtype=np.float32)
    for row in range(weight.shape[0]):
        total = np.float32(0.0)
        for col in range(weight.shape[1]):
            total = np.float32(total + np.float32(weight[row, col] * vector[col]))
        out[row] = total
    return out


def make_weights():
    rng = np.random.default_rng(SEED)

    def rand(shape, scale):
        return f32(rng.uniform(-scale, scale, size=shape))

    token_embedding = rand((VOCAB, D_MODEL), 0.16)

    # Give printable bytes a mild deterministic structure. This increases the
    # separation between top logits without hard-coding the generated trace.
    for token in range(32, 127):
        token_embedding[token, token % D_MODEL] = np.float32(
            token_embedding[token, token % D_MODEL] + np.float32(0.10)
        )

    layers = []
    for _ in range(N_LAYERS):
        layers.append({
            "rms_att": f32(1.0 + rng.uniform(-0.04, 0.04, D_MODEL)),
            "wq": rand((D_MODEL, D_MODEL), 0.10),
            "wk": rand((D_MODEL, D_MODEL), 0.10),
            "wv": rand((D_MODEL, D_MODEL), 0.10),
            "wo": rand((D_MODEL, D_MODEL), 0.10),
            "rms_ffn": f32(1.0 + rng.uniform(-0.04, 0.04, D_MODEL)),
            "w1": rand((D_FF, D_MODEL), 0.08),
            "w3": rand((D_FF, D_MODEL), 0.08),
            "w2": rand((D_MODEL, D_FF), 0.08),
        })

    final_rms = f32(1.0 + rng.uniform(-0.04, 0.04, D_MODEL))
    return token_embedding, layers, final_rms


def flatten_weights(token_embedding, layers, final_rms) -> np.ndarray:
    pieces = [token_embedding.reshape(-1)]
    for layer in layers:
        for name in ("rms_att", "wq", "wk", "wv", "wo", "rms_ffn", "w1", "w3", "w2"):
            pieces.append(layer[name].reshape(-1))
    pieces.append(final_rms.reshape(-1))
    return np.concatenate(pieces).astype("<f4", copy=False)


class Runtime:
    def __init__(self, token_embedding, layers, final_rms):
        self.token_embedding = token_embedding
        self.layers = layers
        self.final_rms = final_rms
        self.key_cache = np.zeros((N_LAYERS, CONTEXT, D_MODEL), dtype=np.float32)
        self.value_cache = np.zeros((N_LAYERS, CONTEXT, D_MODEL), dtype=np.float32)

    def reset(self):
        self.key_cache.fill(0.0)
        self.value_cache.fill(0.0)

    def forward(self, token: int, pos: int) -> np.ndarray:
        x = self.token_embedding[token].copy()
        head_size = D_MODEL // N_HEADS

        for layer_index, layer in enumerate(self.layers):
            xb = rmsnorm(x, layer["rms_att"])
            q = matvec(layer["wq"], xb)
            k = matvec(layer["wk"], xb)
            v = matvec(layer["wv"], xb)
            apply_rope(q, k, pos)
            self.key_cache[layer_index, pos] = k
            self.value_cache[layer_index, pos] = v

            attention_output = np.zeros(D_MODEL, dtype=np.float32)
            for head in range(N_HEADS):
                base = head * head_size
                scores = np.empty(pos + 1, dtype=np.float32)
                for timestep in range(pos + 1):
                    total = np.float32(0.0)
                    for i in range(head_size):
                        total = np.float32(
                            total
                            + np.float32(
                                q[base + i]
                                * self.key_cache[layer_index, timestep, base + i]
                            )
                        )
                    scores[timestep] = np.float32(total / np.float32(math.sqrt(head_size)))
                probs = softmax(scores)
                for i in range(head_size):
                    total = np.float32(0.0)
                    for timestep in range(pos + 1):
                        total = np.float32(
                            total
                            + np.float32(
                                probs[timestep]
                                * self.value_cache[layer_index, timestep, base + i]
                            )
                        )
                    attention_output[base + i] = total

            x = f32(x + matvec(layer["wo"], attention_output))

            xb = rmsnorm(x, layer["rms_ffn"])
            h1 = matvec(layer["w1"], xb)
            h3 = matvec(layer["w3"], xb)
            hidden = f32(silu(h1) * h3)
            x = f32(x + matvec(layer["w2"], hidden))

        x = rmsnorm(x, self.final_rms)
        logits = np.empty(VOCAB, dtype=np.float32)
        for token_index in range(VOCAB):
            total = np.float32(0.0)
            for i in range(D_MODEL):
                total = np.float32(
                    total + np.float32(self.token_embedding[token_index, i] * x[i])
                )
            logits[token_index] = total
        return logits


def generate_reference(token_embedding, layers, final_rms):
    runtime = Runtime(token_embedding, layers, final_rms)
    runtime.reset()
    prompt_tokens = list(PROMPT)
    generated = []
    traces = []

    logits = None
    pos = 0
    for token in prompt_tokens:
        logits = runtime.forward(token, pos)
        traces.append(logits.copy())
        pos += 1

    assert logits is not None
    for _ in range(GEN_LEN):
        next_token = int(np.argmax(logits))
        generated.append(next_token)
        logits = runtime.forward(next_token, pos)
        traces.append(logits.copy())
        pos += 1

    return prompt_tokens, generated, np.stack(traces).astype("<f4", copy=False)


def write_model(path: Path, weights: np.ndarray) -> None:
    payload = weights.tobytes(order="C")
    header = [
        MODEL_MAGIC,
        VERSION,
        ENDIAN_TAG,
        VOCAB,
        CONTEXT,
        D_MODEL,
        N_HEADS,
        N_LAYERS,
        D_FF,
        1,  # tied token embedding / LM head
        weights.size,
        fnv1a32(payload),
        0, 0, 0, 0,
    ]
    path.write_bytes(struct.pack("<16I", *header) + payload)


def write_reference(path: Path, prompt, generated, traces: np.ndarray) -> None:
    prompt_bytes = struct.pack(f"<{len(prompt)}I", *prompt)
    generated_bytes = struct.pack(f"<{len(generated)}I", *generated)
    logits_bytes = traces.tobytes(order="C")
    payload = prompt_bytes + generated_bytes + logits_bytes
    header = [
        REF_MAGIC,
        VERSION,
        VOCAB,
        len(prompt),
        len(generated),
        traces.shape[0],
        fnv1a32(payload),
        0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]
    path.write_bytes(struct.pack("<16I", *header) + payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".", help="output directory")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    token_embedding, layers, final_rms = make_weights()
    weights = flatten_weights(token_embedding, layers, final_rms)
    prompt, generated, traces = generate_reference(token_embedding, layers, final_rms)

    model_path = output / "tiny_minix_llm.bin"
    reference_path = output / "tiny_minix_llm.ref"
    write_model(model_path, weights)
    write_reference(reference_path, prompt, generated, traces)

    printable = "".join(chr(t) if 32 <= t < 127 else f"\\x{t:02x}" for t in generated)
    print(f"wrote {model_path} ({model_path.stat().st_size} bytes)")
    print(f"wrote {reference_path} ({reference_path.stat().st_size} bytes)")
    print(f"parameters: {weights.size}")
    print(f"prompt: {PROMPT!r}")
    print(f"generated token ids: {generated}")
    print(f"generated bytes: {printable}")


if __name__ == "__main__":
    main()
