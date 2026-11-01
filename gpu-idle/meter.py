#!/usr/bin/env python3
"""meter.py — what the operating system's GPU meter says while one chat is being answered.

The point of the video is that "the GPU is busy" and "the GPU's math is working" are different claims.
decode_bench.py measures the second one (achieved FLOP/s against the chip's roof). This measures the
FIRST one: macOS's own "Device Utilization %", sampled from ioreg while a batch-1 decode loop runs.

Run it on a QUIET machine (no renders, no browser) — anything else using the GPU inflates the reading.

    python3 meter.py            # prints idle baseline, then the reading during batch-1 decode
"""
import json, os, re, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SAMPLE_S = 0.25


def device_utilization():
    """macOS's own GPU busy percentage, straight from the accelerator's PerformanceStatistics."""
    try:
        out = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    m = re.search(r'"Device Utilization %"=(\d+)', out)
    return int(m.group(1)) if m else None


def sample(seconds, stop=None):
    vals = []
    t0 = time.time()
    while time.time() - t0 < seconds and not (stop and stop.is_set()):
        v = device_utilization()
        if v is not None:
            vals.append(v)
        time.sleep(SAMPLE_S)
    return vals


def stats(vals):
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    return {"n": len(s), "median": s[len(s) // 2], "max": s[-1], "min": s[0]}


def main():
    import torch
    from decode_bench import build, CFG_1B, DEV, DTYPE

    print("idle baseline (3 s, nothing running)…")
    idle = stats(sample(3))
    print(f"  idle: median {idle.get('median')}%  max {idle.get('max')}%")

    weights, params = build(CFG_1B)
    x = torch.randn(1, CFG_1B["hidden"], device=DEV, dtype=DTYPE)

    stop = threading.Event()

    def spin():                      # one chat, answered over and over
        while not stop.is_set():
            cur = x
            for w in weights:
                v = cur if cur.shape[1] == w.shape[0] else torch.randn(1, w.shape[0], device=DEV, dtype=DTYPE)
                cur = v @ w
            torch.mps.synchronize()

    th = threading.Thread(target=spin, daemon=True)
    th.start()
    time.sleep(1.5)                  # let it reach steady state
    print("sampling while one chat is being answered (12 s)…")
    busy = stats(sample(12))
    stop.set(); th.join(timeout=5)
    print(f"  during batch-1 decode: median {busy.get('median')}%  max {busy.get('max')}%")

    out = {"idle": idle, "batch1": busy, "sample_interval_s": SAMPLE_S,
           "source": 'ioreg -c IOAccelerator → PerformanceStatistics["Device Utilization %"]'}
    dest = os.path.join(HERE, "..", "recordings", "gpu-idle", "meter.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {os.path.relpath(dest)}")
    print("\nThe meter reports the chip as busy; decode_bench.py shows how little of its MATH is working.")


if __name__ == "__main__":
    main()
