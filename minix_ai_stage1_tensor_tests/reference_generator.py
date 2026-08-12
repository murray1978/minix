#!/usr/bin/env python3
"""Regenerate the trusted reference vectors used by ai_stage1_tensor.c.

This script uses Python double precision and the same explicit operation order
as the C test where practical. It is not required on MINIX.
"""

import math


def pattern(count, multiplier, addend, scale):
    return [(((i * multiplier + addend) % 17) - 8) * scale
            for i in range(count)]


def matmul(left, right, rows, shared, columns):
    out = [0.0] * (rows * columns)
    for row in range(rows):
        for column in range(columns):
            total = 0.0
            for inner in range(shared):
                total += (left[row * shared + inner] *
                          right[inner * columns + column])
            out[row * columns + column] = total
    return out


def rmsnorm_rows(values, weight, rows, width, epsilon):
    out = []
    for row in range(rows):
        vector = values[row * width:(row + 1) * width]
        mean_square = sum(value * value for value in vector) / width
        inverse_rms = 1.0 / math.sqrt(mean_square + epsilon)
        out.extend(value * inverse_rms * weight[index]
                   for index, value in enumerate(vector))
    return out


def softmax(values):
    maximum = max(values)
    exponents = [math.exp(value - maximum) for value in values]
    total = sum(exponents)
    return [value / total for value in exponents]


def causal_attention(query, key, value, sequence, width):
    out = [0.0] * (sequence * width)
    weights = [0.0] * (sequence * sequence)
    scale = 1.0 / math.sqrt(width)
    for token in range(sequence):
        scores = []
        for source in range(token + 1):
            dot = sum(query[token * width + i] * key[source * width + i]
                      for i in range(width))
            scores.append(dot * scale)
        probabilities = softmax(scores)
        for source, probability in enumerate(probabilities):
            weights[token * sequence + source] = probability
            for i in range(width):
                out[token * width + i] += (
                    probability * value[source * width + i])
    return out, weights


def tiny_block():
    sequence = 3
    width = 4
    hidden_width = 6
    epsilon = 1.0e-5
    values = [
        0.5, -1.0, 0.25, 2.0,
        1.5, 0.0, -0.5, 1.0,
        -1.0, 0.75, 1.25, -0.25,
    ]
    rms_att = [1.0, 0.95, 1.05, 0.9]
    rms_ffn = [0.9, 1.1, 1.0, 1.05]
    wq = pattern(width * width, 3, 1, 0.04)
    wk = pattern(width * width, 5, 2, 0.035)
    wv = pattern(width * width, 7, 3, 0.03)
    wo = pattern(width * width, 11, 4, 0.025)
    w1 = pattern(width * hidden_width, 13, 5, 0.03)
    w3 = pattern(width * hidden_width, 9, 6, 0.028)
    w2 = pattern(hidden_width * width, 4, 7, 0.026)

    norm1 = rmsnorm_rows(values, rms_att, sequence, width, epsilon)
    query = matmul(norm1, wq, sequence, width, width)
    key = matmul(norm1, wk, sequence, width, width)
    value = matmul(norm1, wv, sequence, width, width)
    attention, weights = causal_attention(query, key, value, sequence, width)
    projected = matmul(attention, wo, sequence, width, width)
    residual = [left + right for left, right in zip(values, projected)]
    norm2 = rmsnorm_rows(residual, rms_ffn, sequence, width, epsilon)
    gate = matmul(norm2, w1, sequence, width, hidden_width)
    up = matmul(norm2, w3, sequence, width, hidden_width)
    hidden = [(a / (1.0 + math.exp(-a))) * b
              for a, b in zip(gate, up)]
    feed_forward = matmul(hidden, w2, sequence, hidden_width, width)
    output = [left + right for left, right in zip(residual, feed_forward)]
    return output, weights


def print_c_array(name, values):
    print(f"{name} = {{")
    for index, value in enumerate(values):
        suffix = "," if index + 1 != len(values) else ""
        print(f"    {value:.10f}f{suffix}")
    print("};")


if __name__ == "__main__":
    output, _ = tiny_block()
    print_c_array("transformer_expected", output)
