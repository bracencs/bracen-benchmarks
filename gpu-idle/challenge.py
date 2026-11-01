#!/usr/bin/env python3
"""challenge.py — feel the memory wall in 30 seconds, on your own machine.

A matrix times a vector (one chat, batch 1) reads the whole matrix to do one multiply per number.
The same matrix times a wide batch of vectors reads it once and reuses it. Same memory, far more math.

    python3 challenge.py

Works on CPU with NumPy.
"""
import time
import numpy as np

N = 8192
W = np.random.randn(N, N).astype(np.float32)


def timed(x, reps=20):
    for _ in range(3):
        W @ x
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); W @ x; t.append(time.perf_counter() - t0)
    t.sort()
    return t[len(t) // 2]


vec = np.random.randn(N, 1).astype(np.float32)      # batch 1
batch = np.random.randn(N, 256).astype(np.float32)  # batch 256

tv = timed(vec)
tb = timed(batch)
print(f"matrix x vector  (batch 1):   {tv*1e3:7.2f} ms  -> {1/tv:8.0f} results/s")
print(f"matrix x 256     (batch 256): {tb*1e3:7.2f} ms  -> {256/tb:8.0f} results/s")
print(f"256x the work took only {tb/tv:.1f}x the time: the matrix read is the cost, not the math.")
