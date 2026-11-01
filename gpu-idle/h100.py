#!/usr/bin/env python3
"""h100.py — the "under 1%" number, derived from NVIDIA's own datasheet.

This does no measurement. It takes two published H100 numbers and one fact about how decoding works,
and computes what fraction of an H100's math a single chat can use.

  NVIDIA H100 SXM datasheet (doc 2287922, Sep 2022):
    BF16 Tensor Core: 1,979 TFLOPS *with sparsity*  ->  989.5 TFLOPS dense (the asterisk halves it)
    GPU memory bandwidth: 3.35 TB/s
  How decoding works (roofline survey, Yuan et al. 2024; Pope et al. 2022):
    one token ~= 2 FLOPs per parameter, and 2 bytes read per parameter at BF16
    -> arithmetic intensity ~= 1 FLOP per byte at batch 1

The ridge point is where compute and memory balance: peak FLOPs / peak bytes. Below it, you are
memory-bound and the math sits idle.
"""

# datasheet figures (dense = half the printed sparsity number)
BF16_DENSE_TFLOPS = 989.5        # 1,979 with sparsity / 2
HBM3_TBPS = 3.35                 # TB/s

DECODE_INTENSITY = 1.0           # FLOP per byte at batch 1 (survey Table 1: ~0.99)


def derive():
    peak_flops = BF16_DENSE_TFLOPS * 1e12
    peak_bytes = HBM3_TBPS * 1e12
    ridge = peak_flops / peak_bytes                 # FLOP/byte to saturate the math
    utilization = DECODE_INTENSITY / ridge          # fraction of compute a batch-1 chat can reach
    return {
        "bf16_dense_tflops": BF16_DENSE_TFLOPS,
        "hbm3_tbps": HBM3_TBPS,
        "ridge_flops_per_byte": ridge,
        "decode_intensity": DECODE_INTENSITY,
        "batch1_utilization": utilization,
        "batch1_percent": utilization * 100,
        "under_one_percent": utilization < 0.01,
    }


if __name__ == "__main__":
    d = derive()
    print(f"H100 SXM ridge point: {d['ridge_flops_per_byte']:.0f} FLOPs per byte")
    print(f"batch-1 decode intensity: ~{d['decode_intensity']:.0f} FLOP/byte")
    print(f"-> a single chat can use {d['batch1_percent']:.2f}% of the H100's BF16 math")
    print(f"under 1%: {d['under_one_percent']}")
