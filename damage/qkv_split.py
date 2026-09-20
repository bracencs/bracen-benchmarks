#!/usr/bin/env python3
"""
Why does one top-exponent flip in an attention weight break the model only ~3 times in 10?

    python qkv_split.py        -> recordings/bitflip/qkv_split.json

qkv.w is stored [in=128, out=384]: output columns 0-127 build the QUERY, 128-255 the KEY, 256-383 the VALUE.
per-tensor (bitflip.py) chose elements uniformly over all 384 columns, so it mixed the three.
This splits them. For each block and each part (q/k/v, weight and bias), REPS single bit-30 flips,
with the element drawn uniformly inside that part only. Same break rule as bitflip.py, which this imports.

Hypothesis (to be confirmed or killed here, not assumed): a huge query or key only sharpens the softmax,
so attention collapses onto one position but stays finite. A huge value is copied straight into the
residual stream. The prediction is that v breaks nearly always and q/k nearly never.
"""
import json, os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # see bitflip.py
import numpy as np

from bitflip import BASE, BREAK_AT, broken, flip
from model import TENSORS, d, loss, nl, pristine

OUT = Path(__file__).resolve().parent.parent / "recordings" / "bitflip"
REPS = 30
PARTS = {"q": 0, "k": 1, "v": 2}
IX = {t["name"]: t for t in TENSORS}


def element(rng, block: int, which: str, part: int) -> int:
    t = IX[f"b{block}.qkv.{which}"]
    col = part * d + int(rng.integers(0, d))
    if which == "b":
        return t["offset"] + col
    row = int(rng.integers(0, t["shape"][0]))
    return t["offset"] + row * t["shape"][1] + col


rows = []
for block in range(nl):
    for which in ("w", "b"):
        for pname, part in PARTS.items():
            losses = []
            for r in range(REPS):
                rng = np.random.default_rng(31_000 + 1000 * block + 100 * part + r + (50 if which == "b" else 0))
                w = pristine()
                flip(w, np.array([element(rng, block, which, part)]), np.array([30]))
                losses.append(loss(w))
            n_broken = sum(broken(x) for x in losses)
            fin = [x for x in losses if np.isfinite(x)]
            rows.append({"block": block, "tensor": f"b{block}.qkv.{which}", "part": pname, "reps": REPS,
                         "broken": n_broken, "non_finite": sum(not np.isfinite(x) for x in losses),
                         "median_finite_loss": float(np.median(fin)) if fin else None})
            print(f"  b{block}.qkv.{which}  {pname}   one bit-30 flip breaks it {n_broken:>2}/{REPS}", flush=True)

totals = {p: {"broken": sum(r["broken"] for r in rows if r["part"] == p),
              "trials": sum(r["reps"] for r in rows if r["part"] == p)} for p in PARTS}
(OUT / "qkv_split.json").write_text(json.dumps({"baseline": BASE, "break_at": BREAK_AT, "reps": REPS,
                                                "totals": totals, "rows": rows}, indent=2))
print("\n" + "  ".join(f"{p}: {v['broken']}/{v['trials']} broken" for p, v in totals.items()))
print("saved recordings/bitflip/qkv_split.json")
