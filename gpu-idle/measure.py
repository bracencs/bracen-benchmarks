#!/usr/bin/env python3
"""measure.py — every number the gpu-idle video uses.

Writes recordings/gpu-idle/measure.json. One command:
    python3 gpu-idle/measure.py

It measures this machine's GPU (compute roof, memory roof, and per-token decode cost across batch
sizes), then derives the H100 figure from the datasheet (h100.py). The M1 numbers demonstrate the
SHAPE; the "under 1%" headline is the H100 datasheet derivation, never presented as an H100
measurement.
"""
import json, os, sys, platform, subprocess, re
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from decode_bench import run, peak_flops, peak_bandwidth, CFG_1B
import h100

OUT = os.path.join(HERE, "..", "recordings", "gpu-idle", "measure.json")
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def gpu_cores():
    try:
        out = subprocess.check_output(["system_profiler", "SPDisplaysDataType"], text=True)
        m = re.search(r"Total Number of Cores:\s*(\d+)", out)
        chip = re.search(r"Chipset Model:\s*(.+)", out)
        return {"cores": int(m.group(1)) if m else None, "chip": chip.group(1).strip() if chip else None}
    except Exception:
        return {"cores": None, "chip": None}


def main():
    reps = 40
    gpu = gpu_cores()
    pf = peak_flops(reps=20)
    pb = peak_bandwidth(reps=30)
    ridge = pf / pb

    bench = run(CFG_1B, BATCHES, reps)
    rows = bench["rows"]
    for r in rows:
        r["compute_util"] = (r["ach_tflops"] * 1e12) / pf
        r["bandwidth_util"] = (r["ach_gbps"] * 1e9) / pb
    b1 = rows[0]
    # crossover: first batch whose achieved intensity >= measured ridge (memory-bound -> compute-bound)
    crossover = next((r["batch"] for r in rows if r["intensity"] >= ridge), rows[-1]["batch"])

    data = {
        "machine": {"platform": platform.platform(), "torch": torch.__version__, "device": "mps",
                    "dtype": "float16", **gpu},
        "peaks": {"compute_tflops": pf / 1e12, "bandwidth_gbps": pb / 1e9,
                  "ridge_flops_per_byte": ridge, "apple_spec_bandwidth_gbps": 200},
        "model": {"name": CFG_1B["name"], "matmul_params": bench["params"],
                  "weight_bytes_fp16": bench["weight_bytes"], "config": CFG_1B},
        "decode": rows,
        "batch1": {"tokens_per_s": b1["tokens_per_s"], "compute_util": b1["compute_util"],
                   "bandwidth_util": b1["bandwidth_util"], "intensity": b1["intensity"]},
        "crossover_batch": crossover,
        "h100": h100.derive(),
        "commands": {"bench": "python3 decode_bench.py", "h100": "python3 h100.py",
                     "measure": "python3 measure.py", "challenge": "python3 challenge.py"},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(data, f, indent=2)

    print(f"{gpu['chip']} {gpu['cores']}-core GPU, torch {torch.__version__}")
    print(f"peak compute {pf/1e12:.2f} TFLOP/s, peak bandwidth {pb/1e9:.0f} GB/s, ridge {ridge:.0f} FLOP/byte")
    print(f"batch-1 decode: {b1['tokens_per_s']:.0f} tok/s, {b1['compute_util']*100:.1f}% of compute")
    print(f"crossover to compute-bound near batch {crossover}")
    h = data["h100"]
    print(f"H100 datasheet ridge {h['ridge_flops_per_byte']:.0f}; batch-1 chat uses {h['batch1_percent']:.2f}% of its math")
    print(f"wrote {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
