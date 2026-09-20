#!/usr/bin/env python3
"""
Delete (zero) the model's weights and measure how much it can lose before it stops being itself.

    python prune.py            -> recordings/prune/sweep.json  (+ text samples per level)

What is pruned: every 2-D weight MATRIX — token/position embeddings, qkv, proj, fc1, fc2, head. Biases and
LayerNorm gains are excluded (they are ~2% of the parameters and pruning them is not what "pruning" means).
Both fractions are reported: of the prunable weights, and of ALL parameters, so no headline can blur them.

Three modes, because the interesting question is WHICH weights matter:
  magnitude-global    zero the smallest |w| across all matrices together
  magnitude-per-layer zero the same fraction of the smallest |w| inside each matrix
  random              zero the same fraction chosen at random — the control

THRESHOLDS, DECLARED BEFORE ANY RUN:
  "same model"  loss within +5% of the undamaged model's loss
  "broken"      loss non-finite or more than halfway to a uniform-random guesser (as in bitflip.py)
"""
import json, math, os, sys
from pathlib import Path

# One BLAS thread: see bitflip.py. Must be set before numpy loads.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

from model import TENSORS, V, check, loss, pristine, sample

OUT = Path(__file__).resolve().parent.parent / "recordings" / "prune"
OUT.mkdir(parents=True, exist_ok=True)

BASE = check()
SAME_AT = BASE * 1.05
BREAK_AT = (BASE + math.log(V)) / 2
PRUNABLE = [t for t in TENSORS if len(t["shape"]) == 2]
N_ALL = sum(t["count"] for t in TENSORS)
N_PRUNABLE = sum(t["count"] for t in PRUNABLE)
LEVELS = [0, 10, 20, 30, 40, 50, 60, 70, 75, 80, 85, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]


def prune(mode: str, frac: float) -> np.ndarray:
    w = pristine()
    if frac <= 0:
        return w
    if mode == "magnitude-global":
        idx = np.concatenate([np.arange(t["offset"], t["offset"] + t["count"]) for t in PRUNABLE])
        k = int(round(frac * len(idx)))
        order = np.argsort(np.abs(w[idx]), kind="stable")
        w[idx[order[:k]]] = 0.0
    elif mode == "magnitude-per-layer":
        for t in PRUNABLE:
            seg = w[t["offset"]: t["offset"] + t["count"]]
            k = int(round(frac * t["count"]))
            seg[np.argsort(np.abs(seg), kind="stable")[:k]] = 0.0
    elif mode == "random":
        rng = np.random.default_rng(424242)
        idx = np.concatenate([np.arange(t["offset"], t["offset"] + t["count"]) for t in PRUNABLE])
        w[rng.choice(idx, int(round(frac * len(idx))), replace=False)] = 0.0
    return w


DIM, BOLD, RED, GRN, YEL, RST = "\033[2m", "\033[1m", "\033[31m", "\033[32m", "\033[33m", "\033[0m"


def cmd_show(mode: str):
    """FILMED. Delete weights level by level and show the loss plus the first line the model now writes.
    Re-measured live (same deterministic pruning as the sweep); writes nothing."""
    label = {"random": "a RANDOM share", "magnitude": "the SMALLEST weights first"}[mode]
    m = "random" if mode == "random" else "magnitude-global"
    print(f"{DIM}deleting {label} · {N_PRUNABLE:,} matrix weights · deleted = set to zero{RST}")
    print(f"  {'deleted':>8}{'loss':>9}   {'verdict':<10} what it writes after \"What say\"")
    for pct in (0, 10, 20, 30, 40, 50, 60, 70, 80, 90):
        w = prune(m, pct / 100); l = loss(w)
        text = sample(w)[1][len("ROMEO:\nWhat say"):].replace("\n", " / ")[:34]
        word, col = (("same", GRN) if math.isfinite(l) and l <= SAME_AT else
                     ("BROKEN", RED + BOLD) if (not math.isfinite(l)) or l > BREAK_AT else ("degraded", YEL))
        print(f"  {pct:>7}%{l:>9.4f}   {col}{word:<10}{RST} {text}", flush=True)


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "show":
    cmd_show(sys.argv[2])
elif __name__ == "__main__":
    rows = []
    for mode in ("magnitude-global", "magnitude-per-layer", "random"):
        for pct in LEVELS:
            w = prune(mode, pct / 100)
            l = loss(w)
            zeroed = int((w == 0).sum()) - int((pristine() == 0).sum())
            _, text = sample(w)
            rows.append({"mode": mode, "pct_of_prunable": pct, "pct_of_all_params": round(100 * zeroed / N_ALL, 2),
                         "loss": l, "same_model": math.isfinite(l) and l <= SAME_AT,
                         "broken": (not math.isfinite(l)) or l > BREAK_AT, "sample": text})
            r = rows[-1]
            tag = "same" if r["same_model"] else ("BROKEN" if r["broken"] else "degraded")
            print(f"  {mode:<20}{pct:>3}% pruned ({r['pct_of_all_params']:>5}% of all)  loss {l:7.4f}  {tag}", flush=True)


    def last(mode, key):
        ok = [r["pct_of_prunable"] for r in rows if r["mode"] == mode and r[key]]
        return max(ok) if ok else None


    summary = {m: {"max_pct_still_same_model": last(m, "same_model"),
                   "max_pct_not_broken": max([r["pct_of_prunable"] for r in rows if r["mode"] == m and not r["broken"]], default=None)}
               for m in ("magnitude-global", "magnitude-per-layer", "random")}
    (OUT / "sweep.json").write_text(json.dumps({"baseline": BASE, "same_at": SAME_AT, "break_at": BREAK_AT,
        "n_all_params": N_ALL, "n_prunable": N_PRUNABLE, "summary": summary, "rows": rows}, indent=2))
    print("\n" + json.dumps(summary, indent=2))
    print("saved recordings/prune/sweep.json")
