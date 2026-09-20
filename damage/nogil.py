#!/usr/bin/env python3
"""
Take the GIL away and run the SAME code again.  The gil film's last measurement.

    python3.14t nogil.py bench --out nogil.freethreaded.json     (GIL off  — the free-threading build)
    python3.14t -X gil=1 nogil.py bench --out nogil.gilback.json (GIL on   — the SAME binary, one switch)
    python3.14  nogil.py bench --out nogil.standard.json         (GIL on   — the ordinary build)
    python3.14t nogil.py compare                                 (FILMED: the two headline legs, side by side)

WHY THIS FILE EXISTS SEPARATELY FROM gil.py
gil.py imports numpy, and the point here is the interpreter, not the maths library. So this file loads the
weights with `array` and runs the race's pure-Python forward with nothing but the standard library — which
means the byte-for-byte same source runs on an ordinary CPython and on a free-threading CPython.

THE ONE-VARIABLE EXPERIMENT
A free-threading build can put the GIL back at startup (-X gil=1). Same binary, same build, same bytecode:
the only thing that changes is whether the lock is there. That is the comparison the film shows. The
ordinary 3.14 build is measured too, because the free-threading build pays for its parallelism even when
the lock is back — and a film that showed only the payoff and not the price would be lying by omission.

INTEGRITY
  * `py_forward` is copied from gil.py unchanged, so the workload is the one already on film.
  * The reference answer is NOT trusted from this file. It is read from recordings/gil/scaling.run1.json,
    which gil.py wrote only after asserting that the pure-Python forward and the NumPy evaluator agree.
    If this file's forward ever stops reproducing that token, the run is INVALID and nothing is written.
  * Each thread keeps its OWN answer list. A shared set would be a lock, and putting a lock inside the
    experiment that measures what happens when you remove a lock is how you measure your own harness.
"""
import array, json, math, os, subprocess, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = ROOT / "model"
OUT = ROOT / "recordings" / "gil"
OUT.mkdir(parents=True, exist_ok=True)

CFG = json.loads((MODEL / "config.json").read_text())
VOCAB = json.loads((MODEL / "tokenizer.json").read_text())["vocab"]
STOI = {c: i for i, c in enumerate(VOCAB)}
TENSORS = json.loads((MODEL / "weights.json").read_text())["tensors"]
A = CFG["arch"]
d, nh, nl, dff = A["d_model"], A["n_head"], A["n_layer"], A["d_ff"]
hd = d // nh
WINDOW_S = float(os.environ.get("GIL_WINDOW", "8"))
CTX = [STOI[c] for c in "ROMEO:\nWhat say the son, and"]          # the same fixed context gil.py used, T=27

_flat = array.array("f"); _flat.frombytes((MODEL / "weights.bin").read_bytes())
_ix = {t["name"]: t for t in TENSORS}
_W = {t["name"]: _flat[_ix[t["name"]]["offset"]: _ix[t["name"]]["offset"] + _ix[t["name"]]["count"]].tolist()
      for t in TENSORS}
_SQ = math.sqrt(2.0 / math.pi)


# ---------- the race's 01 pure-Python forward, copied from gil.py unchanged ----------
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


# ---------- the reference answer comes from the recording gil.py validated, not from here ----------
def reference_token() -> int:
    rec = json.loads((OUT / "scaling.run1.json").read_text())
    if not rec.get("valid"):
        raise SystemExit("!! recordings/gil/scaling.run1.json is not a valid run — cannot take a reference from it")
    ans = {a for r in rec["rows"] if r["workload"] == "pure-python" for a in r["answers"]}
    if len(ans) != 1:
        raise SystemExit(f"!! scaling.run1.json's pure-Python rows disagree on the answer: {sorted(ans)}")
    return ans.pop()


def load_avg() -> list:
    """1/5/15-minute load average, recorded with every run.

    Throughput on a contended machine is depressed, and the whole point of these rows is throughput.
    Recording the load means a later reader can see whether a run was measured on a quiet machine
    instead of taking the word of whoever ran it.
    """
    return [round(x, 2) for x in os.getloadavg()]


def interp() -> dict:
    import sysconfig
    return {"version": sys.version.split()[0],
            "freethreaded_build": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
            "gil_enabled": bool(sys._is_gil_enabled()) if hasattr(sys, "_is_gil_enabled") else True,
            "executable": Path(sys.executable).name}


def label(i: dict) -> str:
    if not i["freethreaded_build"]:
        return f"{i['version']} standard"
    return f"{i['version']}t  GIL {'on' if i['gil_enabled'] else 'OFF'}"


def run_threads(n: int):
    """n threads, each counting its own forwards into its own slot. No shared mutable state in the hot loop."""
    counts = [0] * n; answers = [None] * n
    stop = time.perf_counter() + WINDOW_S

    def work(i):
        c = 0; a = set()
        while time.perf_counter() < stop:
            a.add(py_forward(CTX)); c += 1
        counts[i] = c; answers[i] = sorted(a)

    cpu0, t0 = time.process_time(), time.perf_counter()
    ts = [threading.Thread(target=work, args=(i,)) for i in range(n)]
    for t in ts: t.start()
    for t in ts: t.join()
    wall, cpu = time.perf_counter() - t0, time.process_time() - cpu0
    return sum(counts), wall, cpu, sorted({x for a in answers for x in a})


def bench(workers, out_name, quiet=False):
    ref = reference_token()
    me = interp()
    load0 = load_avg()
    if py_forward(CTX) != ref:
        raise SystemExit("!! EVALUATOR BROKEN: this file's forward does not reproduce the recorded token")
    if not quiet:
        print(f"{label(me)}  ·  every worker must predict {VOCAB[ref]!r}  ·  {WINDOW_S:.0f} s per row"
              f"  ·  6 performance + 2 efficiency cores\n", flush=True)
        print(f"  {'threads':>8}{'forwards/s':>13}{'speedup':>10}{'cores used':>13}", flush=True)
    rows, base = [], None
    for n in workers:
        done, wall, cpu, ans = run_threads(n)
        r = {"workers": n, "forwards": done, "wall_s": round(wall, 3), "throughput": round(done / wall, 3),
             "cores_used": round(cpu / wall, 2), "answers": ans, "correct": ans == [ref]}
        base = base or r["throughput"]
        r["speedup"] = round(r["throughput"] / base, 2)
        rows.append(r)
        if not quiet:
            print(f"  {n:>8}{r['throughput']:>13.2f}{r['speedup']:>9.2f}x{r['cores_used']:>13.2f}"
                  + ("" if r["correct"] else "   !! WRONG ANSWER"), flush=True)
    valid = all(r["correct"] for r in rows)
    if out_name:
        (OUT / out_name).write_text(json.dumps(
            {"valid": valid, "window_s": WINDOW_S, "context_len": len(CTX), "reference_token": ref,
             "cores": {"performance": 6, "efficiency": 2}, "interpreter": me, "label": label(me),
             "load_avg_start": load0, "load_avg_end": load_avg(),
             "workload": "pure-python", "rows": rows}, indent=2))
    if not quiet:
        print(f"\n{'all threads produced the reference token' if valid else '!! INVALID: a thread disagreed'}")
        if out_name: print(f"saved recordings/gil/{out_name}")  # noqa: E701
    return {"valid": valid, "interpreter": me, "label": label(me), "rows": rows}


def compare(workers):
    """FILMED. The same file, the same threads, one switch: -X gil=1 puts the lock back in the same binary."""
    ft = sys.executable
    legs = [("GIL back on   (-X gil=1)", [ft, "-X", "gil=1"]), ("GIL off  (free-threaded)", [ft])]
    print(f"  {Path(ft).name}   ·   the same file, the same threads, one switch\n", flush=True)
    print(f"  {'':<26}{'threads':>8}{'forwards/s':>13}{'speedup':>10}{'cores used':>13}", flush=True)
    for name, cmd in legs:
        env = dict(os.environ, NOGIL_CHILD="1")
        p = subprocess.run(cmd + [__file__, "_leg", ",".join(map(str, workers))],
                           capture_output=True, text=True, env=env)
        if p.returncode != 0:
            print(f"  {name:<26}  !! {p.stderr.strip().splitlines()[-1] if p.stderr.strip() else 'failed'}", flush=True)
            continue
        data = json.loads(p.stdout)
        for j, r in enumerate(data["rows"]):
            head = name if j == 0 else ""
            print(f"  {head:<26}{r['workers']:>8}{r['throughput']:>13.2f}{r['speedup']:>9.2f}x"
                  f"{r['cores_used']:>13.2f}", flush=True)
        print(flush=True)


def _arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "bench"
    if cmd == "_leg":                                   # internal: one interpreter's rows, as JSON on stdout
        print(json.dumps(bench([int(x) for x in sys.argv[2].split(",")], None, quiet=True)))
    elif cmd == "compare":
        compare([int(x) for x in _arg("--workers", "1,8").split(",")])
    else:
        bench([int(x) for x in _arg("--workers", "1,2,4,6,8").split(",")], _arg("--out", "nogil.json"))
