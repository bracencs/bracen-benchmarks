#!/usr/bin/env python3
"""
bench — the only entry point that produces a number this video is allowed to quote.

    bench.py run     [--reps N] [--only IMPL]   run every implementation, write a recording
    bench.py verify  [--recording ID]           re-run and check the recording still holds
    bench.py report  [--recording ID]           print the leaderboard

Modelled on a sibling project's `sim` CLI: a run is identified by a spec hash over everything that
could change the answer, so a recording can never be silently attributed to a different model.

THE GATE IS THE TOKENS, NOT THE TIMES. If two implementations disagree on a single output
token the run is INVALID and no leaderboard is written — a race between things computing
different answers is not a race.
"""
import argparse, hashlib, json, os, platform, statistics, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = ROOT / "model"
# NEVER .resolve() this path. bin/python is a SYMLINK into the venv; resolving it follows the
# link out to Homebrew's system python3.11, which has no numpy and no torch — so three of the
# six implementations die with ModuleNotFoundError while the path still "exists". A venv only
# works when you invoke it THROUGH its own bin/ directory. Resolve the directory, keep the name.
# WHICH PYTHON. This used to be hard-wired to a venv two levels above the repo, which exists only
# on the machine the videos were made on — so `bench.py run` exited immediately for anyone who
# cloned this repo (found 2026-09-20, before the first public push). Default to the interpreter
# running this file, and let BENCH_PYTHON override. The requirement is unchanged and still checked:
# it must import numpy AND torch, because three of the six implementations are built on them.
PY = os.environ.get("BENCH_PYTHON") or sys.executable
_probe = subprocess.run([PY, "-c", "import numpy, torch"], capture_output=True)
if _probe.returncode != 0:
    sys.exit(f"{PY} cannot import numpy + torch, and three of the six implementations need them.\n"
             f"Install them for this interpreter, or point BENCH_PYTHON at one that has them:\n"
             f"    BENCH_PYTHON=/path/to/python python3 harness/bench.py run\n"
             f"{_probe.stderr.decode()[:300]}")

IMPLS = [
    {"id": "01-pure-python",   "label": "pure Python",     "cmd": [PY, "01_pure_python.py"],      "cwd": ROOT / "impls"},
    {"id": "02-numpy",         "label": "NumPy",           "cmd": [PY, "02_numpy.py"],            "cwd": ROOT / "impls"},
    # The C binaries are BUILT, not shipped: harness/build_c.sh compiles both from impls/03_04_c.c.
    # A repo whose whole claim is "measured, not asserted" should not ask anyone to run an opaque
    # prebuilt executable it cannot read.
    {"id": "03-c-naive",       "label": "C, my matmul",    "cmd": ["../bin/03_c_naive", "../model"],      "cwd": ROOT / "impls"},
    {"id": "04-c-accelerate",  "label": "C + Accelerate",  "cmd": ["../bin/04_c_accelerate", "../model"], "cwd": ROOT / "impls"},
    {"id": "05-torch-cpu",     "label": "PyTorch CPU",     "cmd": [PY, "05_06_torch.py", "cpu"],  "cwd": ROOT / "impls"},
    {"id": "06-torch-mps",     "label": "PyTorch MPS/GPU", "cmd": [PY, "05_06_torch.py", "mps"],  "cwd": ROOT / "impls"},
]


def _require_c_binaries():
    """`run` needs the two C executables. Say so plainly instead of failing with ENOENT."""
    missing = [b for b in ("03_c_naive", "04_c_accelerate") if not (ROOT / "bin" / b).exists()]
    if missing:
        sys.exit(f"missing C binaries: {', '.join(missing)}\n"
                 f"Build them first:  bash harness/build_c.sh")


def spec_hash():
    """Hash everything that could change the ANSWER. Not the timings — those are the measurement."""
    h = hashlib.sha256()
    for p in (MODEL / "config.json", MODEL / "weights.json", MODEL / "tokenizer.json",
              MODEL / "layout.txt", MODEL / "weights.bin"):
        h.update(p.read_bytes())
    for src in ("01_pure_python.py", "02_numpy.py", "03_04_c.c", "05_06_torch.py", "common.py"):
        h.update((ROOT / "impls" / src).read_bytes())
    return h.hexdigest()[:8]


def machine():
    def sysctl(k):
        try:
            return subprocess.run(["/usr/sbin/sysctl", "-n", k], capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return None
    return {
        "model": sysctl("hw.model"),
        "cpu": sysctl("machdep.cpu.brand_string"),
        "cores": sysctl("hw.ncpu"),
        "mem_bytes": sysctl("hw.memsize"),
        "os": platform.platform(),
        "python": platform.python_version(),
    }


def run_one(impl, reps):
    """Run one implementation `reps` times. Returns every rep, never just the best."""
    reps_out, result = [], None
    for r in range(reps):
        t0 = time.time()
        proc = subprocess.run(impl["cmd"], cwd=impl["cwd"], capture_output=True, text=True)
        wall = time.time() - t0
        if proc.returncode != 0:
            return None, f"{impl['id']} exited {proc.returncode}: {proc.stderr.strip()[:300]}"
        try:
            result = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception as e:
            return None, f"{impl['id']} unparseable stdout: {e}"
        reps_out.append({"rep": r + 1, "elapsed_s": result["elapsed_s"],
                         "tokens_per_s": result["tokens_per_s"], "wall_with_startup_s": round(wall, 4)})
        print(f"    rep {r+1}/{reps}  {result['elapsed_s']:10.4f}s  {result['tokens_per_s']:10.3f} tok/s",
              flush=True)
    times = [x["elapsed_s"] for x in reps_out]
    return {
        "impl": impl["id"], "label": impl["label"],
        "cmd": " ".join(str(c) for c in impl["cmd"]),
        "reps": reps_out,
        "best_s": min(times), "median_s": statistics.median(times),
        "worst_s": max(times),
        "tokens_per_s_best": round(result["forward_passes"] / min(times), 4),
        "tokens": result["tokens"], "text": result["text"],
        "logit_sum": result["logit_sum"], "forward_passes": result["forward_passes"],
    }, None


def cmd_run(args):
    sh = spec_hash()
    only = set(args.only.split(",")) if args.only else None
    chosen = [i for i in IMPLS if not only or i["id"] in only]
    if any(i["id"].startswith(("03-", "04-")) for i in chosen):
        _require_c_binaries()

    print(f"spec {sh} · {args.reps} rep(s) · {len(chosen)} implementation(s)\n")
    results, errors = [], []
    for impl in chosen:
        print(f"  {impl['id']}  ({impl['label']})", flush=True)
        res, err = run_one(impl, args.reps)
        if err:
            print(f"    ERROR {err}", flush=True)
            errors.append(err)
        else:
            results.append(res)
        print(flush=True)

    # ---------------- THE GATE ----------------
    agree, disagreements = True, []
    if len(results) > 1:
        ref = results[0]
        for r in results[1:]:
            if r["tokens"] != ref["tokens"]:
                agree = False
                first = next((i for i, (a, b) in enumerate(zip(ref["tokens"], r["tokens"])) if a != b), None)
                disagreements.append({
                    "a": ref["impl"], "b": r["impl"], "first_divergent_index": first,
                    "a_token": ref["tokens"][first] if first is not None else None,
                    "b_token": r["tokens"][first] if first is not None else None,
                })
            if abs(r["logit_sum"] - ref["logit_sum"]) > 0.5:
                disagreements.append({"a": ref["impl"], "b": r["impl"],
                                      "logit_sum_delta": round(r["logit_sum"] - ref["logit_sum"], 4)})

    rec = {
        "spec": sh,
        "reps": args.reps,
        "machine": machine(),
        "config": json.loads((MODEL / "config.json").read_text()),
        "n_params": json.loads((MODEL / "weights.json").read_text())["n_params"],
        "results": results,
        "errors": errors,
        "tokens_agree": agree,
        "disagreements": disagreements,
        "valid": agree and not errors and len(results) == len(IMPLS),
    }
    out = ROOT / "recordings" / f"race@{sh}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "recording.json").write_text(json.dumps(rec, indent=2))
    print(f"wrote {out/'recording.json'}")

    if not agree:
        print("\n!! INVALID: implementations disagree on output tokens. No leaderboard.", file=sys.stderr)
        for d in disagreements:
            print(f"   {d}", file=sys.stderr)
        return 2
    if errors:
        print(f"\n!! INCOMPLETE: {len(errors)} implementation(s) failed.", file=sys.stderr)
        return 3
    print_leaderboard(rec)
    return 0


def print_leaderboard(rec):
    rs = sorted(rec["results"], key=lambda r: r["best_s"])
    if not rs:
        print("no results")
        return
    fastest = rs[0]["best_s"]
    print(f"\n  {'#':<3}{'implementation':<20}{'best':>11}{'median':>11}{'tok/s':>11}{'vs fastest':>13}")
    print("  " + "-" * 69)
    for i, r in enumerate(rs, 1):
        print(f"  {i:<3}{r['label']:<20}{r['best_s']:>10.4f}s{r['median_s']:>10.4f}s"
              f"{r['tokens_per_s_best']:>11.1f}{r['best_s']/fastest:>12.1f}x")
    print("  " + "-" * 69)
    print(f"  spread {rs[-1]['best_s']/fastest:,.0f}x   ·   all {len(rs)} produced identical tokens "
          f"(logit_sum {rs[0]['logit_sum']})")


def load_recording(rid):
    d = ROOT / "recordings"
    if rid:
        p = d / rid / "recording.json"
    else:
        cands = sorted(d.glob("race@*/recording.json"), key=lambda p: p.stat().st_mtime)
        if not cands:
            sys.exit("no recordings")
        p = cands[-1]
    return json.loads(p.read_text()), p


def cmd_report(args):
    rec, p = load_recording(args.recording)
    print(f"recording {p.parent.name}  ·  spec {rec['spec']}  ·  {rec['n_params']:,} params  "
          f"·  valid={rec['valid']}")
    print(f"machine: {rec['machine']['model']} · {rec['machine']['cores']} cores · "
          f"{int(rec['machine']['mem_bytes'])/1e9:.0f} GB · {rec['machine']['os']}")
    print_leaderboard(rec)
    return 0


def cmd_gate(args):
    """Show ONLY what the gate checks — token identity — with no times and no ranking.

    Deliberately narrower than `report`: this prints ONLY the identity check, not the leaderboard,
    so you can confirm the six implementations agree without seeing the finishing order. Rows are in implementation order,
    never finishing order, and no timing column exists.
    """
    import hashlib
    rec, p = load_recording(args.recording)
    rs = sorted(rec["results"], key=lambda r: r["impl"])
    n = len(rs[0]["tokens"]) - len(rec["config"]["decode"]["prompt"])
    print(f"recording {p.parent.name} · {n} new tokens from each implementation\n")
    print(f"  {'implementation':<18}{'token sequence (sha-256)':<28}{'logit_sum':>12}")
    ref = rs[0]
    for r in rs:
        h = hashlib.sha256(",".join(map(str, r["tokens"])).encode()).hexdigest()[:16]
        same = r["tokens"] == ref["tokens"] and abs(r["logit_sum"] - ref["logit_sum"]) <= 0.5
        print(f"  {r['impl']:<18}{h + '…':<28}{r['logit_sum']:>12}   {'MATCH' if same else 'DIFFER'}")
    ok = all(r["tokens"] == ref["tokens"] for r in rs)
    print(f"\n  {'all ' + str(len(rs)) + ' identical — gate PASS' if ok else 'tokens DIFFER — no leaderboard'}")
    return 0 if ok else 2


def cmd_verify(args):
    """Re-run and confirm the recording still holds. Tokens must match exactly; timings may not."""
    rec, p = load_recording(args.recording)
    sh = spec_hash()
    print(f"recording {p.parent.name} (spec {rec['spec']}) vs current spec {sh}")
    if sh != rec["spec"]:
        print("!! spec hash changed — the model or an implementation was edited since this run.",
              file=sys.stderr)
        return 2
    bad = 0
    for r in rec["results"]:
        impl = next(i for i in IMPLS if i["id"] == r["impl"])
        proc = subprocess.run(impl["cmd"], cwd=impl["cwd"], capture_output=True, text=True)
        fresh = json.loads(proc.stdout.strip().splitlines()[-1])
        ok = fresh["tokens"] == r["tokens"]
        drift = fresh["elapsed_s"] / r["best_s"]
        print(f"  {r['impl']:<18} tokens {'MATCH' if ok else 'DIFFER'}   "
              f"time {fresh['elapsed_s']:9.4f}s ({drift:5.2f}x recorded best)")
        if not ok:
            bad += 1
    print("\nverify:", "PASS" if bad == 0 else f"FAIL ({bad} mismatched)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--reps", type=int, default=3); r.add_argument("--only")
    v = sub.add_parser("verify"); v.add_argument("--recording")
    p = sub.add_parser("report"); p.add_argument("--recording")
    g = sub.add_parser("gate"); g.add_argument("--recording")
    a = ap.parse_args()
    sys.exit({"run": cmd_run, "verify": cmd_verify, "report": cmd_report, "gate": cmd_gate}[a.cmd](a))
