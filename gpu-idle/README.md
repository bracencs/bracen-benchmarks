# gpu-idle — why chatting barely uses a GPU's math

Generating one token from a language model reads every weight once and does about **two** arithmetic
operations per weight. Two operations for every two bytes fetched: roughly **one operation per byte**.
A modern accelerator can do hundreds of operations for every byte it can pull from memory, so during a
single chat the math units mostly sit idle, waiting on memory. Serving many chats at once reads each
weight once and reuses it for all of them — that is the only thing that fills the machine.

This folder measures the effect on **this machine's** GPU (Apple Silicon via PyTorch MPS) with random
weights shaped like a real 1B model — no model download — and derives the H100 figure from NVIDIA's
published datasheet.

## Files

| file | what it shows |
|---|---|
| `decode_bench.py` | times one token's matmuls at batch 1, 2, 4, ... ; tokens/s, achieved TFLOP/s, GB/s, and arithmetic intensity |
| `h100.py` | the "under 1%" number derived from the H100 datasheet (dense BF16 TFLOPS, HBM3 bandwidth) and the roofline model |
| `measure.py` | measures the machine's compute roof, memory roof, ridge point, and the full decode table; writes `../recordings/gpu-idle/measure.json` |
| `challenge.py` | 30-second NumPy demo: matrix×vector vs matrix×256 — 256× the work in a fraction more time |

## Run it

```
python3 gpu-idle/measure.py     # the full table + the H100 derivation
python3 gpu-idle/h100.py         # just the headline number
python3 gpu-idle/challenge.py    # feel the memory wall, CPU-only
```

Measured on an Apple M1 Pro (14-core GPU), PyTorch MPS, fp16:

- compute roof (a big GEMM): **~4 TFLOP/s**; memory roof (a big copy): **~160 GB/s** (Apple's spec: up to 200 GB/s)
- ridge point: **~25 operations per byte** — below that, you are memory-bound and the math idles
- batch-1 decode: **~26 tokens/s**, using only **~1.6% of the compute roof**
- crossover to compute-bound near **batch 32**: batching is what fills the machine

Derived for an H100 SXM from the datasheet (BF16 989.5 TFLOPS dense, 3.35 TB/s):

- ridge point: **~295 operations per byte**
- a single batch-1 chat can reach only **~0.34%** of its BF16 math — **under 1%**

## Honesty

The M1 numbers are a live demonstration of the *shape* (memory-bound at batch 1, compute-bound when
batched). The "under 1%" headline is the **H100 datasheet derivation**, not a measurement of an H100.
The datasheet prints the BF16 number *with sparsity* (1,979 TFLOPS); the dense figure used here is half.
