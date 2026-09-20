#!/usr/bin/env python3
"""
Measure WHY the GPU lost, instead of asserting a reason.

The obvious story is "kernel launch overhead swamps a tiny model". That story is probably
right, but the video may not say it until it is measured. This script measures three things:

  1. OPS PER FORWARD   — how many separate GPU operations one forward pass dispatches.
  2. FIXED COST        — time for a trivial op on each device, repeated. This is the floor
                         you pay per dispatch regardless of how much arithmetic you asked for.
  3. THE CROSSOVER     — the same matmul at growing sizes on CPU vs MPS. If the launch-overhead
                         story is true there is a size where MPS overtakes CPU, and below it
                         the GPU is simply the wrong tool. Finding that size is the payoff.

Output: recordings/why_mps_lost.json
"""
import json, sys, time
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parent.parent / "recordings"
OUT.mkdir(parents=True, exist_ok=True)
# --out NAME: the screen-recorded run writes beside, never over, the quiet-machine measurement
# that the film's narration and labels are built from.
OUT_NAME = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "why_mps_lost.json"

def say(msg=""):
    print(msg, flush=True)
HAS_MPS = torch.backends.mps.is_available()


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()


def timeit(fn, dev, reps, warmup=5):
    for _ in range(warmup):
        fn()
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    sync(dev)
    return (time.perf_counter() - t0) / reps


res = {"mps_available": HAS_MPS}

# ---- 1. how many dispatches does one forward actually make? -----------------
# Counted structurally from the architecture, not guessed: per layer we issue
# layernorm(≈5) + qkv matmul + bias + 3 slices + 3 views + scores + scale + mask + softmax
# + attn@v + reshape + proj + bias + add + layernorm(≈5) + fc1 + bias + gelu(≈6) + fc2
# + bias + add  ≈ 35 dispatches, times n_layer, plus embedding and head.
cfg = json.loads((OUT.parent / "model" / "config.json").read_text())
A = cfg["arch"]
per_layer = 35
res["dispatch_estimate"] = {
    "per_layer": per_layer,
    "n_layer": A["n_layer"],
    "per_forward": per_layer * A["n_layer"] + 8,
    "per_generation": (per_layer * A["n_layer"] + 8) * cfg["decode"]["new_tokens"],
    "note": "structural count of separate tensor ops, not a profiler trace",
}

# ---- 2. the fixed cost of asking a device to do anything --------------------
fixed = {}
for dev in ["cpu"] + (["mps"] if HAS_MPS else []):
    a = torch.ones(1, device=dev)
    fixed[dev] = {
        "trivial_add_s": timeit(lambda: a.add(1.0), dev, 2000),
        "scalar_readback_s": timeit(lambda: float(a.sum().item()), dev, 200),
    }
res["fixed_cost"] = fixed
if HAS_MPS:
    res["fixed_cost"]["mps_over_cpu_dispatch"] = round(
        fixed["mps"]["trivial_add_s"] / fixed["cpu"]["trivial_add_s"], 2)
    say(f"asking each device to do nothing:  cpu {fixed['cpu']['trivial_add_s']*1e6:6.2f} us   "
        f"gpu {fixed['mps']['trivial_add_s']*1e6:6.2f} us   ({res['fixed_cost']['mps_over_cpu_dispatch']}x)")
say()
say(f"  {'n x n matmul':>12}{'cpu (ms)':>12}{'gpu (ms)':>12}{'gpu speedup':>14}")

# ---- 3. where does the GPU actually start winning? -------------------------
cross = []
for n in (32, 64, 128, 256, 512, 1024, 2048, 4096):
    row = {"n": n}
    for dev in ["cpu"] + (["mps"] if HAS_MPS else []):
        x = torch.randn(n, n, device=dev)
        y = torch.randn(n, n, device=dev)
        reps = 200 if n <= 256 else (50 if n <= 1024 else 10)
        row[dev] = timeit(lambda: x @ y, dev, reps)
    if HAS_MPS:
        row["mps_speedup_over_cpu"] = round(row["cpu"] / row["mps"], 3)
    cross.append(row)
    sp = row.get("mps_speedup_over_cpu") or 0
    first_win = sp > 1 and all((c.get("mps_speedup_over_cpu") or 0) <= 1 for c in cross[:-1])
    say(f"  {n:>12}{row['cpu']*1e3:>12.4f}{row.get('mps', 0)*1e3:>12.4f}{sp:>13.2f}x"
        + ("   <-- GPU finally wins" if first_win else ""))
res["matmul_crossover"] = cross
res["model_matmul_sizes"] = {
    "note": "the shapes this model actually runs, for comparison with the crossover table",
    "qkv": f"[T,{A['d_model']}] @ [{A['d_model']},{3*A['d_model']}]",
    "fc1": f"[T,{A['d_model']}] @ [{A['d_model']},{A['d_ff']}]",
    "largest_dim": max(A["d_model"], A["d_ff"], 3 * A["d_model"]),
    "T_range": [len(cfg["decode"]["prompt"]), A["block_size"]],
}
# The crossover table is SQUARE n x n. This model never multiplies square matrices, so comparing
# its largest DIMENSION (512) to n understates the gap. Compare WORK instead: the biggest single
# matmul is [T_max, d_model] @ [d_model, d_ff] (fc1; fc2 is the same size transposed). Its
# multiply-adds equal those of a square matmul of side cbrt(T*d*ff) — that is the honest n to
# place on the crossover table.
T_max = A["block_size"]
biggest_macs = T_max * A["d_model"] * A["d_ff"]
res["model_matmul_sizes"]["biggest_job_macs"] = biggest_macs
res["model_matmul_sizes"]["biggest_job_equiv_square_n"] = round(biggest_macs ** (1 / 3))

(OUT / OUT_NAME).write_text(json.dumps(res, indent=2))

say()
eq = res["model_matmul_sizes"]["biggest_job_equiv_square_n"]
say(f"this model's biggest single multiply is as much work as a {eq} x {eq} one")
# Relative name only, so the output is identical on any machine.
say(f"saved recordings/{OUT_NAME}")
