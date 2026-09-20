# The same tiny transformer, six ways

One set of weights. One prompt. Six implementations. They all produce **byte-identical output**,
and the fastest is **2,610x** faster than the slowest.

The interesting part is not that pure Python is slow. It is **where the GPU finished**.

## Results

Measured on MacBookPro18,3 · 8 cores · 17 GB · macOS-27.0-arm64-arm-64bit
3 reps each, best-of reported. Recording: `race@37024541` (spec `37024541`).

| # | implementation | id | best | median | tok/s | vs fastest |
|---|---|---|---|---|---|---|
| 1 | PyTorch CPU | `05-torch-cpu` | 0.0596 s | 0.0615 s | 1,074.6 | 1.0x |
| 2 | C + Accelerate | `04-c-accelerate` | 0.0786 s | 0.0786 s | 814.6 | 1.3x |
| 3 | NumPy | `02-numpy` | 0.1921 s | 0.2198 s | 333.2 | 3.2x |
| 4 | C, my matmul | `03-c-naive` | 2.1174 s | 2.1213 s | 30.2 | 35.6x |
| 5 | PyTorch MPS/GPU | `06-torch-mps` | 5.5890 s | 5.6026 s | 11.5 | 93.8x |
| 6 | pure Python | `01-pure-python` | 155.4282 s | 157.1421 s | 0.4 | 2,609.7x |

**All 6 produced identical tokens** (`logit_sum = -190.0184`). That is the gate:
`bench.py` refuses to write a leaderboard if any two implementations disagree on a single token,
because a race between things computing different answers is not a race.

## The model

822,337 parameters. 4 layers, d_model 128, 4 heads,
d_ff 512, context 96, character-level, vocab 65.
Trained on tinyShakespeare (public domain).

Prompt `'ROMEO:\nWhat say'`, 64 tokens, greedy argmax. It produces:

```
ROMEO:
What say the son, and the son, and the son,
And the shall the shall the 
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


## Why the GPU lost — measured, not asserted

An `n x n` matmul on each device, same machine, same run:

| n | CPU (ms) | MPS (ms) | MPS speedup |
|---|---|---|---|
| 32 | 0.0011 | 0.0494 | 0.02x |
| 64 | 0.0016 | 0.0535 | 0.03x |
| 128 | 0.0052 | 0.0631 | 0.08x |
| 256 | 0.0284 | 0.0826 | 0.34x |
| 512 | 0.1369 | 0.1870 | 0.73x |
| 1024 | 1.1658 | 0.7058 | 1.65x |
| 2048 | 9.5139 | 4.6547 | 2.04x |
| 4096 | 82.4014 | 35.8707 | 2.30x |

The GPU does not overtake the CPU until **n = 1024**. This model never multiplies square
matrices, so comparing its widest dimension (512) to that `n`
would understate the gap. Compared by work instead: its biggest single matmul,
`[96x128] @ [128x512]`, is as many multiply-adds as a
**185 x 185**
square — deep in the region where the GPU runs at a fraction of the CPU's speed. Every job it
dispatches is on the wrong side of that line.

The fixed cost of asking each device to do *nothing*: CPU 1.54 us,
MPS 38.50 us — **25.04x**. One generation dispatches **12,736** operations to the GPU — measured with
`TorchDispatchMode` over the exact forward pass (`harness/count_dispatches.py`; the
6,656 metadata-only views are excluded) — so that fixed cost is paid over twelve
thousand times and the arithmetic never gets a chance to dominate. Each forward pass is
12,035,200 multiply-adds for the first character and 67,580,032 for the
last (no KV cache, so it grows with the context): 2,505,023,488 for the paragraph.

The GPU was never slow. It was never asked for enough at once.

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

Generated by `harness/make_readme.py` directly from `race@37024541/recording.json`.
Nothing here is typed by hand.
