#!/usr/bin/env python3
"""
Two follow-ups to prune.py, both measured:

    python prune_retrain.py   -> recordings/prune/why_and_retrain.json

1. WHERE THE WEIGHT LIVES. For each deletion fraction: what share of the prunable weights' total squared size
   (sum of w^2) the deleted weights held, smallest-first vs random. This is why the order matters.
2. RETRAIN AFTER PRUNING. Prune, then keep training with the deleted weights held at exactly zero
   (the mask is re-applied after every optimizer step), and score it with the SAME evaluator as prune.py.
   The recipe is declared here and not tuned afterwards: the original training's batch size, the original
   learning rate, AdamW (betas 0.9/0.95, weight decay 0.1), grad-clip 1.0, the same corpus split.
   CONTROL: the undamaged model trained for the same steps, so "it recovered" can't be just "more training helped".
"""
import json, math, os, sys, time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # see bitflip.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import CFG, STOI, TENSORS, VOCAB, check, loss, pristine, sample
from prune import BREAK_AT, PRUNABLE, SAME_AT, prune

OUT = Path(__file__).resolve().parent.parent / "recordings" / "prune"
A, TR, C = CFG["arch"], CFG["train"], CFG["corpus"]
CHECKPOINTS = [0, 25, 50, 100, 200, 400, 800]
CASES = [("magnitude-global", 70), ("magnitude-global", 80), ("magnitude-global", 90),
         ("random", 20), ("random", 50), ("none", 0)]
dev = "mps" if torch.backends.mps.is_available() else "cpu"

# ---- the model, as defined in model/train.py (copied, because importing train.py would retrain it) ----
def gelu_tanh(x): return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))

class Block(nn.Module):
    def __init__(self):
        super().__init__(); d = A["d_model"]
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, A["d_ff"]), nn.Linear(A["d_ff"], d)
        self.nh, self.hd = A["n_head"], d // A["n_head"]
    def forward(self, x):
        B, t, d = x.shape
        q, k, v = (z.view(B, t, self.nh, self.hd).transpose(1, 2) for z in self.qkv(self.ln1(x)).split(d, dim=2))
        att = ((q @ k.transpose(-2, -1)) / math.sqrt(self.hd)).masked_fill(
            torch.tril(torch.ones(t, t, device=x.device)) == 0, float("-inf")).softmax(-1)
        x = x + self.proj((att @ v).transpose(1, 2).contiguous().view(B, t, d))
        return x + self.fc2(gelu_tanh(self.fc1(self.ln2(x))))

class Tiny(nn.Module):
    def __init__(self, V):
        super().__init__(); d = A["d_model"]
        self.tok, self.pos = nn.Embedding(V, d), nn.Embedding(A["block_size"], d)
        self.blocks = nn.ModuleList([Block() for _ in range(A["n_layer"])])
        self.lnf, self.head = nn.LayerNorm(d), nn.Linear(d, V)
    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks: x = b(x)
        return self.head(self.lnf(x))

def params_by_name(m: Tiny):
    """Manifest name -> (parameter, stored-transposed?) — the inverse of train.py's export."""
    out = {"tok.weight": (m.tok.weight, False), "pos.weight": (m.pos.weight, False),
           "lnf.g": (m.lnf.weight, False), "lnf.b": (m.lnf.bias, False),
           "head.w": (m.head.weight, True), "head.b": (m.head.bias, False)}
    for i, b in enumerate(m.blocks):
        for mod, short in ((b.ln1, "ln1"), (b.ln2, "ln2")):
            out[f"b{i}.{short}.g"] = (mod.weight, False); out[f"b{i}.{short}.b"] = (mod.bias, False)
        for mod, short in ((b.qkv, "qkv"), (b.proj, "proj"), (b.fc1, "fc1"), (b.fc2, "fc2")):
            out[f"b{i}.{short}.w"] = (mod.weight, True); out[f"b{i}.{short}.b"] = (mod.bias, False)
    return out

def load(m: Tiny, flat: np.ndarray):
    P = params_by_name(m)
    with torch.no_grad():
        for t in TENSORS:
            p, tr = P[t["name"]]
            a = torch.from_numpy(flat[t["offset"]: t["offset"] + t["count"]].reshape(t["shape"]).copy())
            p.copy_(a.T if tr else a)

def dump(m: Tiny) -> np.ndarray:
    P = params_by_name(m); flat = np.empty(sum(t["count"] for t in TENSORS), dtype=np.float32)
    for t in TENSORS:
        p, tr = P[t["name"]]; a = p.detach().cpu()
        flat[t["offset"]: t["offset"] + t["count"]] = (a.T if tr else a).contiguous().numpy().ravel()
    return flat

# ---- data: the same corpus split train.py uses ----
text = (Path(__file__).resolve().parent.parent / "model" / C["path"]).read_text(encoding="utf-8")
data = torch.tensor([STOI[c] for c in text], dtype=torch.long)
train_data = data[: int(len(data) * C["split"])]
BS, BLK = TR["batch_size"], A["block_size"]

def batch(g):
    ix = torch.randint(len(train_data) - BLK - 1, (BS,), generator=g)
    x = torch.stack([train_data[i: i + BLK] for i in ix]); y = torch.stack([train_data[i + 1: i + 1 + BLK] for i in ix])
    return x.to(dev), y.to(dev)


def energy_table():
    w = pristine()
    idx = np.concatenate([np.arange(t["offset"], t["offset"] + t["count"]) for t in PRUNABLE])
    sq = w[idx].astype(np.float64) ** 2; total = sq.sum()
    order = np.sort(sq)
    rng = np.random.default_rng(424242)
    rows = []
    for pct in (10, 20, 30, 40, 50, 60, 70, 80, 90):
        k = int(round(pct / 100 * len(idx)))
        rnd = rng.choice(len(idx), k, replace=False)
        rows.append({"pct_deleted": pct, "smallest_first_share_of_size": float(order[:k].sum() / total),
                     "random_share_of_size": float(sq[rnd].sum() / total)})
    a = np.abs(w[idx])
    return {"rows": rows, "n_prunable": int(len(idx)), "median_abs": float(np.median(a)), "max_abs": float(a.max()),
            "share_below_0_05": float(np.mean(a < 0.05)), "share_below_0_1": float(np.mean(a < 0.1))}


def retrain(mode: str, pct: int, camera: bool = False):
    torch.manual_seed(TR["seed"]); g = torch.Generator().manual_seed(TR["seed"] + 1)
    flat = prune(mode, pct / 100) if mode != "none" else pristine()
    m = Tiny(len(VOCAB)); load(m, flat); m.to(dev).train()
    P = params_by_name(m)
    masks = [(P[t["name"]][0], (P[t["name"]][0] != 0).float()) for t in PRUNABLE] if mode != "none" else []
    opt = torch.optim.AdamW(m.parameters(), lr=TR["lr"], betas=(0.9, 0.95), weight_decay=0.1)
    curve, t0 = [], time.time()
    for step in range(0, CHECKPOINTS[-1] + 1):
        if step in CHECKPOINTS:
            m.eval(); f = dump(m); l = loss(f); txt = sample(f)[1]; m.train()
            zero_share = float(np.mean([float((p == 0).float().mean()) for p, _ in masks])) if masks else 0.0
            curve.append({"step": step, "loss": l, "same_model": l <= SAME_AT, "broken": (not math.isfinite(l)) or l > BREAK_AT,
                          "still_zero_share": zero_share, "sample": txt})
            tag = 'same' if l <= SAME_AT else 'BROKEN' if l > BREAK_AT else 'degraded'
            if camera:
                col = GRN if tag == "same" else (RED + BOLD) if tag == "BROKEN" else YEL
                said = txt[len("ROMEO:\nWhat say"):].replace("\n", " / ")[:30]
                print(f"  step {step:>4}   loss {l:7.4f}   {col}{tag:<9}{RST} {said}", flush=True)
            else:
                print(f"  {mode:<17}{pct:>3}%  step {step:>4}  loss {l:7.4f}  {tag:<8}  {time.time() - t0:5.1f}s", flush=True)
        if step == CHECKPOINTS[-1]:
            break
        x, y = batch(g)
        l = F.cross_entropy(m(x).view(-1, len(VOCAB)), y.view(-1))
        opt.zero_grad(set_to_none=True); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        with torch.no_grad():
            for p, mk in masks: p.mul_(mk)
    return {"mode": mode, "pct": pct, "curve": curve}


DIM, BOLD, RED, GRN, YEL, RST = "\033[2m", "\033[1m", "\033[31m", "\033[32m", "\033[33m", "\033[0m"


def cmd_energy():
    """FILMED. Re-computes the table live from the weights (deterministic); writes nothing."""
    en = energy_table()
    print(f"{DIM}{en['n_prunable']:,} matrix weights · the typical one is {en['median_abs']:.3f} in size · the largest {en['max_abs']:.2f}{RST}")
    print(f"  {'delete':>7}   {'smallest first removes':>24}   {'random removes':>16}")
    print(f"  {'':>7}   {DIM}{'(share of the total squared size)':>40}{RST}")
    for r in en["rows"]:
        print(f"  {r['pct_deleted']:>6}%   {YEL}{r['smallest_first_share_of_size']:>23.2%}{RST}   {r['random_share_of_size']:>16.2%}", flush=True)


def cmd_watch(pct: int):
    """FILMED. One retrain run live (the same deterministic recipe and seed as the recorded run); writes nothing."""
    print(f"{DIM}the smallest {pct}% of the matrix weights deleted, then trained on, with every deleted weight held at zero{RST}\n")
    retrain("magnitude-global", pct, camera=True)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "energy":
    cmd_energy()
elif __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "watch":
    cmd_watch(int(sys.argv[2]))
elif __name__ == "__main__":
    base = check()
    # round-trip check: loading the weights into torch and dumping them back must be exact
    rt = Tiny(len(VOCAB)); load(rt, pristine()); assert np.array_equal(dump(rt), pristine()), "torch round-trip is not exact"
    en = energy_table()
    print(f"prunable weights {en['n_prunable']:,} · median |w| {en['median_abs']:.4f} · max |w| {en['max_abs']:.3f}")
    for r in en["rows"]:
        print(f"  delete {r['pct_deleted']:>2}%:  smallest-first removes {r['smallest_first_share_of_size']:6.2%} of the total size"
              f"   random removes {r['random_share_of_size']:6.2%}")
    print(f"\nretraining on {dev} · checkpoints {CHECKPOINTS}\n", flush=True)
    runs = [retrain(m, p) for m, p in CASES]
    (OUT / "why_and_retrain.json").write_text(json.dumps({"baseline": base, "same_at": SAME_AT, "break_at": BREAK_AT,
        "device": dev, "recipe": {"lr": TR["lr"], "batch_size": BS, "optimizer": "AdamW(0.9,0.95) wd 0.1", "clip": 1.0},
        "energy": en, "retrain": runs}, indent=2))
    print("saved recordings/prune/why_and_retrain.json")
