#!/usr/bin/env python3
"""
Does Python's GIL stop my model from using more than one core?  Measured, on the race's model.

    python gil.py            -> recordings/gil/scaling.json   (RUN ON A QUIET MACHINE — this times things)

The common belief: "Python threads can't run in parallel because of the GIL, so threading won't speed up
CPU work." The more precise truth: the GIL serialises PYTHON BYTECODE. Code that spends its time inside
C that releases the GIL — NumPy's BLAS calls and ufunc loops — can run on several cores at once.

Four workloads, same model, same forward pass, 1..8 worker THREADS each (processes as a control):
  pure-python          the race's pure-Python forward (lists and loops): pure bytecode, holds the GIL
  numpy-blas1          NumPy forward, BLAS pinned to 1 thread per call (threadpoolctl) so only OUR threads
                       provide parallelism — the honest test of "does threading help"
  numpy-blas-default   NumPy forward with OpenBLAS at its default thread count (recorded in the output) — threads on top of threads
  pure-python-procs    pure-Python forward in PROCESSES (each has its own GIL) — the control

Measured per config: total forward passes completed in a fixed window, speedup vs 1 worker, and
CORES ACTUALLY USED = process CPU time / wall time (counts every thread; no psutil needed).
CORRECTNESS: every worker's argmax must equal the single-thread reference, or the run is invalid.

Hardware context for reading the numbers: this M1 Pro has 6 performance + 2 efficiency cores.
"""
import array, json, math, os, sys, threading, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from model import A, BLK, TENSORS, VOCAB, STOI, _forward_all, pristine, views  # noqa: E402

OUT = ROOT / "recordings" / "gil"
OUT.mkdir(parents=True, exist_ok=True)
WINDOW_S = float(os.environ.get("GIL_WINDOW", "8"))
WORKERS = [1, 2, 4, 6, 8]
CTX = [STOI[c] for c in "ROMEO:\nWhat say the son, and"]          # fixed context, T=27
NP_BATCH = 64                                                     # enough work per call that compute dominates

# ---------- pure-Python forward: the race's 01 implementation, exposed as a function ----------
d, nh, nl, dff = A["d_model"], A["n_head"], A["n_layer"], A["d_ff"]
hd = d // nh
_flat = array.array("f"); _flat.frombytes((ROOT / "model" / "weights.bin").read_bytes())
_ix = {t["name"]: t for t in TENSORS}
def _L(n): t = _ix[n]; return _flat[t["offset"]: t["offset"] + t["count"]].tolist()
_W = {t["name"]: _L(t["name"]) for t in TENSORS}
_SQ = math.sqrt(2.0 / math.pi)


def _ln(x, g, b, n):
    out = [0.0] * len(x)
    for r in range(0, len(x), n):
        row = x[r: r + n]; m = sum(row) / n
        inv = 1.0 / math.sqrt(sum((z - m) ** 2 for z in row) / n + 1e-5)
        for j in range(n): out[r + j] = (row[j] - m) * inv * g[j] + b[j]
    return out


def _mm(x, T, K, w, N, bias):
    out = [0.0] * (T * N)
    for t in range(T):
        xo, oo = t * K, t * N
        for j in range(N):
            acc = bias[j]
            for k in range(K): acc += x[xo + k] * w[k * N + j]
            out[oo + j] = acc
    return out


def py_forward(ids):
    T = len(ids); W = _W
    x = [W["tok.weight"][i * d + j] + W["pos.weight"][t * d + j] for t, i in enumerate(ids) for j in range(d)]
    for L in range(nl):
        g = lambda k: W[f"b{L}.{k}"]
        qkv = _mm(_ln(x, g("ln1.g"), g("ln1.b"), d), T, d, g("qkv.w"), 3 * d, g("qkv.b"))
        o = [0.0] * (T * d)
        for h in range(nh):
            base = h * hd
            for p in range(T):
                sc = [sum(qkv[p * 3 * d + base + j] * qkv[q * 3 * d + d + base + j] for j in range(hd)) / math.sqrt(hd)
                      for q in range(p + 1)]
                mx = max(sc); ex = [math.exp(s - mx) for s in sc]; tot = sum(ex)
                for q in range(p + 1):
                    wq = ex[q] / tot; vo = q * 3 * d + 2 * d + base
                    for j in range(hd): o[p * d + base + j] += wq * qkv[vo + j]
        pr = _mm(o, T, d, g("proj.w"), d, g("proj.b")); x = [a + b for a, b in zip(x, pr)]
        f1 = _mm(_ln(x, g("ln2.g"), g("ln2.b"), d), T, d, g("fc1.w"), dff, g("fc1.b"))
        f1 = [0.5 * z * (1.0 + math.tanh(_SQ * (z + 0.044715 * z ** 3))) for z in f1]
        f2 = _mm(f1, T, dff, g("fc2.w"), d, g("fc2.b")); x = [a + b for a, b in zip(x, f2)]
    last = _ln(x, W["lnf.g"], W["lnf.b"], d)[(T - 1) * d: T * d]
    V = len(VOCAB)
    logits = [W["head.b"][j] + sum(last[k] * W["head.w"][k * V + j] for k in range(d)) for j in range(V)]
    return max(range(V), key=logits.__getitem__)


# ---------- NumPy forward: the damage evaluator's batched forward, one call = NP_BATCH sequences ----------
_NPW = views(pristine())
_NPX = np.array([CTX] * NP_BATCH)


def np_forward():
    return int(np.argmax(_forward_all(_NPW, _NPX)[0, -1]))


def _run_threads(fn, n):
    count = [0] * n; stop = time.perf_counter() + WINDOW_S; answers = set()
    def work(i):
        c = 0
        while time.perf_counter() < stop:
            answers.add(fn()); c += 1
        count[i] = c
    cpu0, t0 = time.process_time(), time.perf_counter()
    ts = [threading.Thread(target=work, args=(i,)) for i in range(n)]
    for t in ts: t.start()
    for t in ts: t.join()
    wall, cpu = time.perf_counter() - t0, time.process_time() - cpu0
    return sum(count), wall, cpu, answers


def _proc_worker(_):
    """Each process times itself: its last forward runs past the window, so rate = count / its OWN elapsed time."""
    t0 = time.perf_counter(); stop = t0 + WINDOW_S; c = 0; ans = set()
    cpu0 = time.process_time()
    while time.perf_counter() < stop:
        ans.add(py_forward(CTX)); c += 1
    return c, time.process_time() - cpu0, ans, time.perf_counter() - t0


def measure(name, n, runner):
    done, wall, cpu, answers = runner(n)
    return {"workload": name, "workers": n, "forwards": done, "wall_s": round(wall, 3),
            "throughput": round(done / wall, 3), "cores_used": round(cpu / wall, 2), "answers": sorted(answers)}


def _arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


if __name__ == "__main__":
    ONLY = set(_arg("--only", "pure-python,numpy-blas1,numpy-blas-default,pure-python-procs").split(","))
    WORKERS = [int(x) for x in _arg("--workers", ",".join(map(str, WORKERS))).split(",")]
    OUT_NAME = _arg("--out", "scaling.json")
    ref_py, ref_np = py_forward(CTX), np_forward()
    assert ref_py == ref_np, f"pure-Python ({ref_py}) and NumPy ({ref_np}) forwards disagree — invalid"
    print(f"every worker must predict {VOCAB[ref_py]!r} · {WINDOW_S:.0f} s per row · 6 performance + 2 efficiency cores\n", flush=True)
    rows = []
    print(f"  {'workload':<20}{'workers':>8}{'forwards/s':>12}{'speedup':>9}{'cores used':>12}", flush=True)
    def py_threads(n):
        return _run_threads(lambda: py_forward(CTX), n)

    def np_threads_blas1(n):
        with threadpool_limits(1):          # BLAS may use 1 thread per call: only OUR threads add cores
            return _run_threads(np_forward, n)

    def np_threads_blas_default(n):
        return _run_threads(np_forward, n)  # OpenBLAS at its default thread count, underneath ours

    plans = [("pure-python", py_threads), ("numpy-blas1", np_threads_blas1),
             ("numpy-blas-default", np_threads_blas_default)]
    for name, runner in plans:
        if name not in ONLY:
            continue
        base = None
        for n in WORKERS:
            r = measure(name, n, runner)
            base = base or r["throughput"]
            r["speedup"] = round(r["throughput"] / base, 2)
            r["correct"] = r["answers"] == [ref_py]
            rows.append(r)
            print(f"  {name:<20}{n:>8}{r['throughput']:>12.2f}{r['speedup']:>8.2f}x{r['cores_used']:>12.2f}"
                  + ("" if r["correct"] else "   !! WRONG ANSWER"), flush=True)
    base = None
    for n in (WORKERS if "pure-python-procs" in ONLY else []):
        t0 = time.perf_counter()
        with Pool(n) as pool:
            res = pool.map(_proc_worker, range(n))
        wall = time.perf_counter() - t0
        done = sum(x[0] for x in res); cpu = sum(x[1] for x in res); span = max(x[3] for x in res)
        ans = sorted(set().union(*[x[2] for x in res]))
        r = {"workload": "pure-python-procs", "workers": n, "forwards": done, "wall_s": round(wall, 3),   # wall includes process start-up
             "throughput": round(sum(x[0] / x[3] for x in res), 3), "cores_used": round(cpu / span, 2), "answers": ans}
        base = base or r["throughput"]; r["speedup"] = round(r["throughput"] / base, 2); r["correct"] = ans == [ref_py]
        rows.append(r)
        print(f"  {'pure-python-procs':<20}{n:>8}{r['throughput']:>12.2f}{r['speedup']:>8.2f}x{r['cores_used']:>12.2f}", flush=True)
    valid = all(r["correct"] for r in rows)
    (OUT / OUT_NAME).write_text(json.dumps({"valid": valid, "window_s": WINDOW_S, "np_batch": NP_BATCH,
        "context_len": len(CTX), "cores": {"performance": 6, "efficiency": 2},
        "blas_default": [{k: i[k] for k in ("internal_api", "num_threads", "version")} for i in threadpool_info()], "rows": rows}, indent=2))
    print(f"\n{'all workers produced the reference answer' if valid else '!! INVALID: some worker disagreed'}")
    print(f"saved recordings/gil/{OUT_NAME}")
