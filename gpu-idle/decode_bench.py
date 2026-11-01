#!/usr/bin/env python3
"""decode_bench.py — why an LLM barely touches a GPU's math while chatting.

Generating one token means reading every weight once and doing about two floating-point operations
per weight. Two ops for every two bytes read: the arithmetic intensity is about 1 operation per byte.
A modern accelerator can do hundreds of operations for every byte it can fetch, so the math sits idle
waiting on memory. Batching many chats together reads each weight once but reuses it for every chat,
which is the only thing that lifts the intensity.

This times a real transformer's per-token matmuls at batch 1, 2, 4, ... on this machine's GPU, using
random weights of the exact shapes of a shipping model (values don't affect speed). No model download.

    python3 decode_bench.py            # prints a table; writes nothing
Called by measure.py, which records the JSON.
"""
import time, json, argparse
import torch

# Llama-3.2-1B config.json shapes (unsloth/Llama-3.2-1B): the numbers are the architecture, not weights.
CFG_1B = dict(name="1B", hidden=2048, inter=8192, layers=16, heads=32, kv_heads=8, head_dim=64, vocab=128256)

DTYPE = torch.float16   # MPS bf16 matmul support is patchy; fp16 is 2 bytes/param, same story
DEV = "mps"


def layers_and_params(cfg):
    """The per-layer projection matrices (in, out) and the final lm_head, plus the parameter count
    that the matmuls actually read per token."""
    h, i, kv = cfg["hidden"], cfg["inter"], cfg["kv_heads"] * cfg["head_dim"]
    per_layer = [
        (h, h),    # q_proj
        (h, kv),   # k_proj
        (h, kv),   # v_proj
        (h, h),    # o_proj
        (h, i),    # gate_proj
        (h, i),    # up_proj
        (i, h),    # down_proj
    ]
    mats = per_layer * cfg["layers"] + [(h, cfg["vocab"])]  # + lm_head
    params = sum(a * b for a, b in mats)
    return mats, params


def build(cfg):
    mats, params = layers_and_params(cfg)
    weights = [torch.randn(a, b, device=DEV, dtype=DTYPE) for a, b in mats]
    return weights, params


def time_step(weights, cfg, batch, reps, warmup=3):
    h = cfg["hidden"]
    x = torch.randn(batch, h, device=DEV, dtype=DTYPE)
    def one():
        cur = x
        for w in weights:
            # each projection reads x[batch, in] @ w[in, out]; feed a right-shaped vector in
            v = cur if cur.shape[1] == w.shape[0] else torch.randn(batch, w.shape[0], device=DEV, dtype=DTYPE)
            cur = v @ w
        return cur
    for _ in range(warmup):
        one()
    torch.mps.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        one()
        torch.mps.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2], ts[0]   # median, min


def peak_flops(reps=20):
    """Achievable compute roof: a big square fp16 GEMM."""
    n = 4096
    a = torch.randn(n, n, device=DEV, dtype=DTYPE)
    b = torch.randn(n, n, device=DEV, dtype=DTYPE)
    for _ in range(3):
        a @ b
    torch.mps.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); a @ b; torch.mps.synchronize(); ts.append(time.perf_counter() - t0)
    ts.sort()
    flops = 2 * n ** 3
    return flops / ts[len(ts) // 2]


def peak_bandwidth(reps=30):
    """Achievable memory roof: a large read+write copy."""
    n = 256 * 1024 * 1024 // 2   # 256 MB of fp16
    a = torch.randn(n, device=DEV, dtype=DTYPE)
    b = torch.empty_like(a)
    for _ in range(3):
        b.copy_(a)
    torch.mps.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); b.copy_(a); torch.mps.synchronize(); ts.append(time.perf_counter() - t0)
    ts.sort()
    bytes_moved = a.numel() * 2 * 2   # read + write
    return bytes_moved / ts[len(ts) // 2]


def run(cfg, batches, reps, on_row=None):
    """Time one decode step at each batch size. `on_row` is called as each row is measured so a live
    run prints the table filling in, instead of sitting blank for minutes and dumping it at the end."""
    weights, params = build(cfg)
    weight_bytes = params * 2  # fp16
    rows = []
    for bsz in batches:
        med, mn = time_step(weights, cfg, bsz, reps)
        flops = 2 * params * bsz            # 2 ops per weight per token, bsz tokens
        ach_flops = flops / med
        ach_bytes = weight_bytes / med      # weights read once per step, whatever the batch
        intensity = flops / weight_bytes    # ~= bsz
        row = dict(batch=bsz, step_ms=med * 1e3, step_min_ms=mn * 1e3,
                   tokens_per_s=bsz / med, ach_tflops=ach_flops / 1e12,
                   ach_gbps=ach_bytes / 1e9, intensity=intensity)
        rows.append(row)
        if on_row:
            on_row(row)
    del weights
    torch.mps.empty_cache()
    return dict(params=params, weight_bytes=weight_bytes, rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--batches", type=str, default="1,2,4,8,16,32,64,128,256")
    args = ap.parse_args()
    batches = [int(b) for b in args.batches.split(",")]

    mats, params = layers_and_params(CFG_1B)
    print(f"model {CFG_1B['name']}: {params/1e9:.3f} B matmul params, {params*2/1e9:.2f} GB fp16", flush=True)
    print(f"{'batch':>6} {'tok/s':>10} {'ms':>8} {'TFLOP/s':>9} {'GB/s':>8} {'intensity':>10}", flush=True)

    def show(row):
        print(f"{row['batch']:>6} {row['tokens_per_s']:>10.0f} {row['step_ms']:>8.2f} "
              f"{row['ach_tflops']:>9.2f} {row['ach_gbps']:>8.0f} {row['intensity']:>10.1f}", flush=True)

    run(CFG_1B, batches, args.reps, on_row=show)


if __name__ == "__main__":
    main()
