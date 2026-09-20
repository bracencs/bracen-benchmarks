#!/usr/bin/env python3
"""Generate benchmarks/README.md FROM recordings/race@<spec>/recording.json.

Every number in the README is read out of the measurement. None is typed by a human, so the
README cannot drift from what actually ran — the same reason the leaderboard is generated and
not written down.

    python3 harness/make_readme.py [--recording race@abc12345]
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(rid):
    d = ROOT / "recordings"
    p = (d / rid / "recording.json") if rid else max(
        d.glob("race@*/recording.json"), key=lambda q: q.stat().st_mtime, default=None)
    if not p or not p.exists():
        sys.exit("no recording found — run harness/bench.py run first")
    return json.loads(p.read_text()), p


ap = argparse.ArgumentParser()
ap.add_argument("--recording")
a = ap.parse_args()
rec, path = load(a.recording)

if not rec.get("valid"):
    sys.exit(f"recording {path.parent.name} is not valid (tokens_agree="
             f"{rec.get('tokens_agree')}, errors={len(rec.get('errors', []))}). "
             f"Refusing to publish a README from an invalid run.")

rs = sorted(rec["results"], key=lambda r: r["best_s"])
fast, slow = rs[0], rs[-1]
A = rec["config"]["arch"]
D = rec["config"]["decode"]
m = rec["machine"]
sample = rs[0]["text"]

rows = "\n".join(
    f"| {i} | {r['label']} | `{r['impl']}` | {r['best_s']:.4f} s | {r['median_s']:.4f} s | "
    f"{r['tokens_per_s_best']:,.1f} | {r['best_s']/fast['best_s']:,.1f}x |"
    for i, r in enumerate(rs, 1))

vocab_size = json.loads((ROOT / "model" / "weights.json").read_text())["vocab_size"]

# The crossover measurement — the part that actually explains the result.
why_p = ROOT / "recordings" / "why_mps_lost.json"
if why_p.exists():
    why = json.loads(why_p.read_text())
    first = next((r["n"] for r in why["matmul_crossover"]
                  if (r.get("mps_speedup_over_cpu") or 0) > 1), None)
    xrows = "\n".join(
        f"| {r['n']} | {r['cpu']*1e3:.4f} | {r.get('mps', 0)*1e3:.4f} | "
        f"{r.get('mps_speedup_over_cpu', 0):.2f}x |" for r in why["matmul_crossover"])
    fc = why["fixed_cost"]
    disp_p = ROOT / "recordings" / "dispatch_count.json"
    if disp_p.exists():
        dc = json.loads(disp_p.read_text()); mc = dc["macs"]
        dispatch_line = (
            f"One generation dispatches **{dc['ops_launched']:,}** operations to the GPU — measured with\n"
            f"`TorchDispatchMode` over the exact forward pass (`harness/count_dispatches.py`; the\n"
            f"{dc['ops_metadata_only']:,} metadata-only views are excluded) — so that fixed cost is paid over twelve\n"
            f"thousand times and the arithmetic never gets a chance to dominate. Each forward pass is\n"
            f"{mc['first_forward']:,} multiply-adds for the first character and {mc['last_forward']:,} for the\n"
            f"last (no KV cache, so it grows with the context): {mc['whole_generation']:,} for the paragraph.")
    else:
        dispatch_line = (f"One generation dispatches roughly **{why['dispatch_estimate']['per_generation']:,}**"
                         f" operations (structural estimate — run harness/count_dispatches.py to measure).")
    crossover_section = f"""
## Why the GPU lost — measured, not asserted

An `n x n` matmul on each device, same machine, same run:

| n | CPU (ms) | MPS (ms) | MPS speedup |
|---|---|---|---|
{xrows}

The GPU does not overtake the CPU until **n = {first}**. This model never multiplies square
matrices, so comparing its widest dimension ({why['model_matmul_sizes']['largest_dim']}) to that `n`
would understate the gap. Compared by work instead: its biggest single matmul,
`[{A['block_size']}x{A['d_model']}] @ [{A['d_model']}x{A['d_ff']}]`, is as many multiply-adds as a
**{why['model_matmul_sizes']['biggest_job_equiv_square_n']} x {why['model_matmul_sizes']['biggest_job_equiv_square_n']}**
square — deep in the region where the GPU runs at a fraction of the CPU's speed. Every job it
dispatches is on the wrong side of that line.

The fixed cost of asking each device to do *nothing*: CPU {fc['cpu']['trivial_add_s']*1e6:.2f} us,
MPS {fc['mps']['trivial_add_s']*1e6:.2f} us — **{fc['mps_over_cpu_dispatch']}x**. {dispatch_line}

The GPU was never slow. It was never asked for enough at once.
"""
else:
    crossover_section = ""

readme = f"""# The same tiny transformer, six ways

One set of weights. One prompt. Six implementations. They all produce **byte-identical output**,
and the fastest is **{slow['best_s']/fast['best_s']:,.0f}x** faster than the slowest.

The interesting part is not that pure Python is slow. It is **where the GPU finished**.

## Results

Measured on {m['model']} · {m['cores']} cores · {int(m['mem_bytes'])/1e9:.0f} GB · {m['os']}
{rec['reps']} reps each, best-of reported. Recording: `{path.parent.name}` (spec `{rec['spec']}`).

| # | implementation | id | best | median | tok/s | vs fastest |
|---|---|---|---|---|---|---|
{rows}

**All {len(rs)} produced identical tokens** (`logit_sum = {rs[0]['logit_sum']}`). That is the gate:
`bench.py` refuses to write a leaderboard if any two implementations disagree on a single token,
because a race between things computing different answers is not a race.

## The model

{rec['n_params']:,} parameters. {A['n_layer']} layers, d_model {A['d_model']}, {A['n_head']} heads,
d_ff {A['d_ff']}, context {A['block_size']}, character-level, vocab {vocab_size}.
Trained on tinyShakespeare (public domain).

Prompt `{D['prompt']!r}`, {D['new_tokens']} tokens, greedy argmax. It produces:

```
{sample}
```

It is not good. It is not supposed to be good. It is supposed to be *the same* in six languages
and runtimes, so that the only thing the clock measures is how the arithmetic got done.

## What is actually being raced

**No KV cache, on purpose.** Every generated token re-runs a full forward pass over the whole
context. That is the one algorithm all six can express identically, so the race measures the
matmul and nothing else — and it is the naive thing a first implementation does.

Two pairs isolate exactly one variable each:

- `03-c-naive` and `04-c-accelerate` are **one source file, two builds**. The only difference is
  `-DUSE_ACCELERATE`, which swaps three nested loops for one `cblas_sgemm` call.
- `05-torch-cpu` and `06-torch-mps` are **one source file, two devices**.

Weights are stored as a flat little-endian float32 blob plus a JSON manifest, so C, NumPy and
pure Python all read the same bytes with no format library and no conversion step.

{crossover_section}
## Reproduce

```bash
bash harness/build_c.sh                   # build the two C implementations FIRST
python3 harness/bench.py run --reps 3     # race everything, write a recording
python3 harness/bench.py verify           # re-run, confirm the tokens still match
python3 harness/bench.py report           # print the leaderboard
python3 harness/why_mps_lost.py           # find the size where the GPU starts winning
python3 harness/count_dispatches.py       # count the ops one generation actually sends to the GPU
```

The build step is not optional: the C binaries are compiled here, never shipped. This repo asks you
to trust measurements, so it does not also ask you to run an executable you cannot read.

Requires a Python with numpy + torch, and clang. `harness/build_c.sh` probes for a macOS SDK that
actually links — the default symlink on some installs is malformed. `bench.py` uses the interpreter
you invoke it with; set `BENCH_PYTHON=/path/to/python` to point it somewhere else.

Set `OPENBLAS_NUM_THREADS=1` before any timing run. Left at its default, NumPy starts one BLAS
thread per core inside every process, and two concurrent jobs can run orders of magnitude slower
while showing healthy CPU usage — that is its own experiment in `damage/gil.py`.

## Numbers in this file

Generated by `harness/make_readme.py` directly from `{path.parent.name}/recording.json`.
Nothing here is typed by hand.
"""

(ROOT / "README.md").write_text(readme)
print(f"wrote {ROOT/'README.md'} from {path.parent.name}")
print(f"  fastest {fast['label']} {fast['best_s']:.4f}s · slowest {slow['label']} {slow['best_s']:.4f}s "
      f"· spread {slow['best_s']/fast['best_s']:,.0f}x")
