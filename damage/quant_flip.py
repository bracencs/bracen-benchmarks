#!/usr/bin/env python3
"""
The same until-break experiment as bitflip.py, but with the model stored as 8-bit INTEGERS instead of floats.

    python quant_flip.py      -> recordings/bitflip/int8_until_break.json

Why: in float32, bit 30 multiplies a weight by 2^128. An int8 weight has no exponent, so no single flip can do that.
The worst a flip can do is the top (sign) bit of the two's-complement byte, which moves the value by 128 steps
of the tensor's scale, about the size of the largest weight in that tensor. Does that make the model
measurably harder to break? Measured, same break rule, same trial count, same search as bitflip.py.

Quantization: symmetric, one scale per tensor, scale = max|w| / 127, q = round(w / scale) as int8, and every tensor
is quantized (matrices, biases, LayerNorm), so every stored bit is a target.
The quantized model's own loss is measured and reported. It is the baseline this run is judged against.
"""
import json, math, os, sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # see bitflip.py
import numpy as np

from model import N_FLOATS, TENSORS, V, loss, pristine, sample, race_tokens

OUT = Path(__file__).resolve().parent.parent / "recordings" / "bitflip"
TRIALS, CAP = 100, 2_000_000


def quantize(w: np.ndarray):
    q = np.empty(w.shape, dtype=np.int8); scale = np.empty(w.shape, dtype=np.float32)
    for t in TENSORS:
        s = slice(t["offset"], t["offset"] + t["count"])
        sc = float(np.abs(w[s]).max()) / 127.0
        q[s] = np.clip(np.round(w[s] / sc), -127, 127).astype(np.int8)
        scale[s] = sc
    return q, scale


W0 = pristine()
Q0, SCALE = quantize(W0)
BITS = N_FLOATS * 8
BASE_Q = loss(Q0.astype(np.float32) * SCALE)
BREAK_AT = (BASE_Q + math.log(V)) / 2


def broken(l): return (not math.isfinite(l)) or l > BREAK_AT


def at(e, b, n):
    q = Q0.copy(); u = q.view(np.uint8)
    np.bitwise_xor.at(u, e[:n], (1 << b[:n]).astype(np.uint8))   # .at: a repeated element flips twice, as it should
    return loss(q.astype(np.float32) * SCALE)


DIM, BOLD, RED, GRN, YEL, RST = "\033[2m", "\033[1m", "\033[31m", "\033[32m", "\033[33m", "\033[0m"


def cmd_watch():
    """FILMED. One trial, live: the same doubling-then-halving search as the recorded run, printing every probe.
    Trial chosen by rule: the first trial within 5% of the recorded median (trial 0 if the run is not recorded yet)."""
    f = OUT / "int8_until_break.json"
    t = 0
    if f.exists():
        ub = json.loads(f.read_text())
        t = next(r["trial"] for r in ub["results"] if r["flips_to_break"] and abs(r["flips_to_break"] / ub["median"] - 1) <= 0.05)
    rng = np.random.default_rng(90_000 + t)
    e = rng.integers(0, N_FLOATS, CAP); b = rng.integers(0, 8, CAP)
    def show(n, l):
        word, col = ("BROKEN", RED + BOLD) if broken(l) else (("same", GRN) if l <= BASE_Q * 1.05 else ("degraded", YEL))
        print(f"  {n:>9,} flips   loss {l:8.4f}   {col}{word}{RST}", flush=True)
        return broken(l)
    print(f"{DIM}the same model as 8-bit integers · {BITS:,} bits · random flips · trial {t}{RST}\n")
    lo, hi = 0, 1
    while not show(hi, at(e, b, hi)):
        lo, hi = hi, hi * 2
    print(f"\n{DIM}  broken somewhere between {lo:,} and {hi:,} flips: narrowing it down{RST}")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if broken(at(e, b, mid)): hi = mid
        else: lo = mid
    print(f"\n  first broken at flip {YEL}{BOLD}{hi:,}{RST}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "watch":
    cmd_watch()
elif __name__ == "__main__":
    same_tokens = sample(Q0.astype(np.float32) * SCALE)[0] == race_tokens()
    print(f"int8 model: loss {BASE_Q:.4f} (float32 was {loss(W0):.4f}) · greedy text identical to the race: {same_tokens}"
          f" · {BITS:,} bits · BROKEN above {BREAK_AT:.4f}\n", flush=True)
    results = []
    for t in range(TRIALS):
        rng = np.random.default_rng(90_000 + t)          # same seeds as bitflip.py's until-break
        e = rng.integers(0, N_FLOATS, CAP); b = rng.integers(0, 8, CAP)
        lo, hi = 0, 1
        while hi <= CAP and not broken(at(e, b, hi)):
            lo, hi = hi, hi * 2
        hi = min(hi, CAP)
        if not broken(at(e, b, hi)):
            results.append({"trial": t, "flips_to_break": None}); print(f"  trial {t:>3}: not broken by {CAP:,}", flush=True); continue
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if broken(at(e, b, mid)): hi = mid
            else: lo = mid
        # spot-check the monotone assumption: 8 evenly spaced earlier prefixes must all be unbroken
        spot = None if t >= 5 else not any(broken(at(e, b, k)) for k in np.linspace(1, hi - 1, 8, dtype=int) if k > 0)
        results.append({"trial": t, "flips_to_break": hi, "share_of_bits": hi / BITS, "spot_check_agrees": spot})
        print(f"  trial {t:>3}: broke at flip {hi:>9,}" + ("" if spot is None else f"   spot-check agrees: {spot}"), flush=True)
    ns = [r["flips_to_break"] for r in results if r["flips_to_break"]]
    summary = {"trials": TRIALS, "bits_total": BITS, "baseline_int8": BASE_Q, "baseline_float32": loss(W0),
               "int8_text_matches_race": same_tokens, "break_at": BREAK_AT,
               "median": float(np.median(ns)), "p10": float(np.percentile(ns, 10)), "p90": float(np.percentile(ns, 90)),
               "min": min(ns), "max": max(ns), "never_broke_within_cap": sum(r["flips_to_break"] is None for r in results),
               "results": results}
    (OUT / "int8_until_break.json").write_text(json.dumps(summary, indent=2))
    print(f"\nint8 median flips to break: {summary['median']:,.0f} (p10 {summary['p10']:,.0f}, p90 {summary['p90']:,.0f})")
    print("saved recordings/bitflip/int8_until_break.json")
