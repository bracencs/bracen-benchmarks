#!/usr/bin/env python3
"""
Flip random bits in the model's weights and measure what breaks.

    python bitflip.py sweep        by bit class x flip count x seeds  -> recordings/bitflip/sweep.json
    python bitflip.py until-break  uniformly random flips until broken -> recordings/bitflip/until_break.json
    python bitflip.py per-tensor   one top-exponent flip per tensor    -> recordings/bitflip/per_tensor.json
    python bitflip.py one          the single-flip demo (filmed)       -> prints + writes one_flip.json

"BROKEN" IS DECLARED HERE, BEFORE ANY RUN, and never tuned afterwards:
    loss is non-finite, OR loss is more than halfway from the trained model's loss to the loss of a model
    that guesses uniformly at random (ln 65). The threshold is computed from the baseline, not chosen.

Every flip is logged (tensor, element index, bit, old value, new value), so any number in the film can be
traced to the exact flips that produced it.
"""
import json, math, os, sys
from pathlib import Path

# One BLAS thread. These matrices are small, and OpenBLAS starts 8 spinning threads per process. With two
# experiments running at once, a loss() took 276 s instead of 1.9 s (measured 2026-09-18). Must be set before numpy loads.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

from model import N_FLOATS, TENSORS, V, check, loss, pristine, sample

OUT = Path(__file__).resolve().parent.parent / "recordings" / "bitflip"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = {
    "sign":       [31],
    "exp_top":    [30],
    "exp_rest":   list(range(23, 30)),
    "mant_high":  list(range(16, 23)),
    "mant_low":   list(range(0, 16)),
}
BITS_TOTAL = N_FLOATS * 32

BASE = check()
RANDOM_LOSS = math.log(V)
BREAK_AT = (BASE + RANDOM_LOSS) / 2


def broken(l: float) -> bool:
    return (not math.isfinite(l)) or l > BREAK_AT


def tensor_of(idx: int) -> str:
    for t in TENSORS:
        if t["offset"] <= idx < t["offset"] + t["count"]:
            return t["name"]
    return "?"


def flip(w: np.ndarray, elems: np.ndarray, bits: np.ndarray, log: list | None = None) -> None:
    u = w.view(np.uint32)
    for e, b in zip(elems.tolist(), bits.tolist()):
        old = float(w[e])
        u[e] ^= np.uint32(1 << b)
        if log is not None:
            log.append({"tensor": tensor_of(e), "elem": e, "bit": b, "old": old, "new": float(w[e])})


def draw(rng, n: int, bit_choices: list[int]):
    return rng.integers(0, N_FLOATS, n), rng.choice(bit_choices, n)


def cmd_sweep(seeds: int = 20, counts=(1, 10, 100, 1000, 10000)):
    rows = []
    for cls, bits in CLASSES.items():
        for n in counts:
            losses = []
            for s in range(seeds):
                rng = np.random.default_rng(1000 * s + n)
                w = pristine()
                e, b = draw(rng, n, bits)
                flip(w, e, b)
                losses.append(loss(w))
            fin = [x for x in losses if math.isfinite(x)]
            rows.append({"class": cls, "flips": n, "seeds": seeds,
                         "broken": sum(broken(x) for x in losses),
                         "non_finite": sum(not math.isfinite(x) for x in losses),
                         "median_loss": float(np.median(fin)) if fin else None,
                         "worst_finite": max(fin) if fin else None})
            r = rows[-1]
            print(f"  {cls:<10}{n:>7} flips   broken {r['broken']:>2}/{seeds}   "
                  f"median loss {r['median_loss'] if r['median_loss'] is None else round(r['median_loss'], 4)}", flush=True)
    (OUT / "sweep.json").write_text(json.dumps({"baseline": BASE, "break_at": BREAK_AT, "rows": rows}, indent=2))


def cmd_until_break(trials: int = 100, cap: int = 60000):
    """Uniformly random flips over ALL bits of ALL weights. For each trial: the first flip count at which
    the model is broken — exponential then binary search over one fixed random sequence per trial.
    The search assumes damage is monotone in the number of flips; the first 5 trials are ALSO scanned
    linearly (flip by flip, up to their answer) to check that assumption on real data."""
    results = []
    for t in range(trials):
        rng = np.random.default_rng(90_000 + t)
        e = rng.integers(0, N_FLOATS, cap)
        b = rng.integers(0, 32, cap)

        def at(n):
            w = pristine(); flip(w, e[:n], b[:n]); return loss(w)

        lo, hi = 0, 1
        while hi <= cap and not broken(at(hi)):
            lo, hi = hi, hi * 2
        hi = min(hi, cap)
        if not broken(at(hi)):
            results.append({"trial": t, "flips_to_break": None}); continue
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if broken(at(mid)): hi = mid
            else: lo = mid
        n = hi
        linear_ok = None
        if t < 5:   # validate the monotone assumption: no earlier prefix may already be broken
            linear_ok = not any(broken(at(k)) for k in range(1, n))
        results.append({"trial": t, "flips_to_break": n, "breaking_bit": int(b[n - 1]),
                        "breaking_tensor": tensor_of(int(e[n - 1])), "linear_scan_agrees": linear_ok})
        print(f"  trial {t:>3}: broke at flip {n:>6}  (bit {int(b[n-1]):>2}, {tensor_of(int(e[n-1]))})"
              + ("" if linear_ok is None else f"  linear-scan agrees: {linear_ok}"), flush=True)
    ns = [r["flips_to_break"] for r in results if r["flips_to_break"]]
    summary = {"trials": trials, "bits_total": BITS_TOTAL, "baseline": BASE, "break_at": BREAK_AT,
               "median": float(np.median(ns)), "p10": float(np.percentile(ns, 10)),
               "p90": float(np.percentile(ns, 90)), "min": min(ns), "max": max(ns),
               "share_breaking_bit_30": sum(r.get("breaking_bit") == 30 for r in results) / len(ns),
               "never_broke_within_cap": sum(r["flips_to_break"] is None for r in results),
               "results": results}
    (OUT / "until_break.json").write_text(json.dumps(summary, indent=2))
    print(f"\nmedian flips to break: {summary['median']:.0f}  (p10 {summary['p10']:.0f}, p90 {summary['p90']:.0f})"
          f" · broken by bit 30 in {summary['share_breaking_bit_30']:.0%} of trials")


def cmd_per_tensor(reps: int = 10):
    rows = []
    for t in TENSORS:
        losses = []
        for r in range(reps):
            rng = np.random.default_rng(7_000 + r)
            w = pristine()
            e = np.array([t["offset"] + int(rng.integers(0, t["count"]))])
            flip(w, e, np.array([30]))
            losses.append(loss(w))
        rows.append({"tensor": t["name"], "count": t["count"], "reps": reps,
                     "broken": sum(broken(x) for x in losses),
                     "median_loss": float(np.median([x if math.isfinite(x) else 99 for x in losses]))})
        print(f"  {t['name']:<14}{t['count']:>8}   one bit-30 flip breaks it {rows[-1]['broken']:>2}/{reps}", flush=True)
    (OUT / "per_tensor.json").write_text(json.dumps({"baseline": BASE, "break_at": BREAK_AT, "rows": rows}, indent=2))


def cmd_one(elem: int | None = None):
    """The filmed demo: ONE bit, in one weight. Picks the first qkv weight of block 0 unless told otherwise."""
    w = pristine()
    if elem is None:
        elem = next(t for t in TENSORS if t["name"] == "b0.qkv.w")["offset"]
    before_ids, before = sample(w)
    log: list = []
    flip(w, np.array([elem]), np.array([30]), log)
    after_ids, after = sample(w)
    rec = {"flip": log[0], "loss_before": BASE, "loss_after": loss(w), "before": before, "after": after}
    (OUT / "one_flip.json").write_text(json.dumps(rec, indent=2))
    f = log[0]
    print(f"flipped bit {f['bit']} of {f['tensor']}[{f['elem']}]:  {f['old']!r}  ->  {f['new']!r}")
    print(f"loss {BASE:.4f} -> {rec['loss_after']}\n\nbefore:\n{before}\n\nafter:\n{after}")


# ------------------------------------------------------------------ filmed commands
# These run on camera. Each one re-measures live (nothing is replayed from a file except `report`, which says so),
# prints only relative paths, and never overwrites the recorded experiment files above.

DIM, BOLD, RED, GRN, YEL, RST = "\033[2m", "\033[1m", "\033[31m", "\033[32m", "\033[33m", "\033[0m"


def verdict(l: float) -> str:
    if broken(l):
        return f"{RED}{BOLD}BROKEN{RST}"
    return f"{GRN}same{RST}" if l <= BASE * 1.05 else f"{YEL}degraded{RST}"


def fmt_loss(l: float) -> str:
    return "     inf" if not math.isfinite(l) else f"{l:8.4f}"


def cmd_classes(seed: int = 1):
    """One seed. For each class of bit, flip 1, 100 and 10,000 random bits of that class and measure the loss.
    One row per class, each cell measured live and coloured by the declared thresholds."""
    names = {"mant_low": "low mantissa  (bits 0-15)", "mant_high": "high mantissa (bits 16-22)", "sign": "sign          (bit 31)",
             "exp_rest": "exponent      (bits 23-29)", "exp_top": "top exponent  (bit 30)"}
    col = lambda l: RED + BOLD if broken(l) else (GRN if l <= BASE * 1.05 else YEL)
    print(f"  {'which bits get flipped':<28}{'1 flip':>11}{'100 flips':>12}{'10,000 flips':>15}")
    for cls in ("mant_low", "mant_high", "sign", "exp_rest", "exp_top"):
        print(f"  {names[cls]:<28}", end="", flush=True)
        for n, wd in ((1, 11), (100, 12), (10_000, 15)):
            rng = np.random.default_rng(1000 * seed + n)
            w = pristine(); e, b = draw(rng, n, CLASSES[cls]); flip(w, e, b)
            l = loss(w)
            print(f"{col(l)}{fmt_loss(l).strip():>{wd}}{RST}", end="", flush=True)
        print()
    print(f"\n  {GRN}same{RST}   {YEL}degraded{RST}   {RED}{BOLD}BROKEN{RST}   {DIM}(loss: lower is better){RST}")


def cmd_watch():
    """Replays ONE until-break trial flip by flip, measuring the loss after every flip.
    Which trial is declared by rule, not picked by eye: the first trial whose count is within 3 of the median."""
    ub = json.loads((OUT / "until_break.json").read_text())
    med = ub["median"]
    t = next(r["trial"] for r in ub["results"] if r["flips_to_break"] and abs(r["flips_to_break"] - med) <= 3)
    rng = np.random.default_rng(90_000 + t)
    e = rng.integers(0, N_FLOATS, 60000); b = rng.integers(0, 32, 60000)
    print(f"{DIM}trial {t} of 100 (the first within 3 of the median) · one random bit at a time, anywhere in the model{RST}\n")
    print(f"  {'flip':>5}  {'bit':>4}  {'where':<12}{'loss':>10}")
    w = pristine()
    for n in range(1, 60001):
        log: list = []
        flip(w, e[n - 1:n], b[n - 1:n], log)
        l = loss(w); f = log[0]
        mark = f"   {DIM}{f['old']:.4g} -> {f['new']:.4g}{RST}" if f["bit"] == 30 else ""
        print(f"  {n:>5}  {f['bit']:>4}  {f['tensor']:<12}{fmt_loss(l):>10}   {verdict(l)}{mark}", flush=True)
        if broken(l):
            break


def cmd_report():
    """The 100-trial result, read from recordings/bitflip/until_break.json (measured earlier; this only prints it)."""
    ub = json.loads((OUT / "until_break.json").read_text())
    ns = sorted(r["flips_to_break"] for r in ub["results"] if r["flips_to_break"])
    by30 = sum(r.get("breaking_bit") == 30 for r in ub["results"])
    print(f"{DIM}read from recordings/bitflip/until_break.json · {ub['trials']} trials, each a fresh random sequence{RST}")
    edges = [1, 2, 4, 8, 16, 32, 64, 128, 256]          # doubling bins: equal width on a log scale
    for lo, hi in zip(edges, edges[1:]):
        k = sum(lo <= n < hi for n in ns)
        rng_ = f"{lo}" if hi - lo == 1 else f"{lo}-{hi - 1}"
        print(f"  {rng_:>9} flips  {'#' * k} {k}")
    print(f"  median {BOLD}{ub['median']:.0f}{RST} flips   {DIM}(10% of trials broke by {ub['p10']:.0f}, 90% by {ub['p90']:.0f};"
          f" fewest {ub['min']}, most {ub['max']}){RST}")
    print(f"  the flip that broke it was bit 30:  {YEL}{BOLD}{by30} of {ub['trials']}{RST}")


def cmd_one_part(part: str):
    """ONE bit-30 flip in block 0's attention weights, row 0, column 0 of the QUERY or the VALUE block.
    Same row, same column offset, same bit: the only difference is which third of the matrix it lands in."""
    t = next(x for x in TENSORS if x["name"] == "b0.qkv.w")
    col = {"query": 0, "key": 1, "value": 2}[part] * t["shape"][1] // 3
    elem = t["offset"] + col
    w = pristine(); _, before = sample(w)
    log: list = []; flip(w, np.array([elem]), np.array([30]), log)
    l = loss(w); _, after = sample(w)
    f = log[0]
    (OUT / f"one_flip_{part}.json").write_text(json.dumps({"flip": f, "part": part, "loss_before": BASE, "loss_after": l,
                                                          "before": before, "after": after}, indent=2))
    print(f"  one weight in the {BOLD}{part}{RST} part of the first attention layer:  {f['old']:.4f}  ->  {YEL}{f['new']:.3g}{RST}")
    print(f"  {DIM}bit 30 flipped: the weight is now 2 to the power 128 times bigger{RST}")
    print(f"  loss  {BASE:.4f}  ->  {fmt_loss(l).strip()}   {verdict(l)}")
    print(f"{DIM}before:{RST}")
    for line in before.split("\n"): print(f"    {line}")
    print(f"{DIM}after:{RST}")
    for line in after.split("\n"): print(f"    {line}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if cmd not in ("one", "report"):
        print(f"{BITS_TOTAL:,} bits in the model · trained loss {BASE:.4f} · BROKEN above {BREAK_AT:.4f}\n", flush=True)
    if cmd == "one" and len(sys.argv) > 3 and sys.argv[2] == "--part":
        cmd_one_part(sys.argv[3])
    else:
        {"sweep": cmd_sweep, "until-break": cmd_until_break, "per-tensor": cmd_per_tensor, "one": cmd_one,
         "classes": cmd_classes, "watch": cmd_watch, "report": cmd_report}[cmd]()
