#!/usr/bin/env python3
"""
Train the one tiny transformer that all six implementations will run.

Output is deliberately language-neutral:
  weights.bin   — every tensor concatenated, raw little-endian float32
  weights.json  — the manifest (name, shape, offset, count) so C, NumPy and pure
                  Python all read the SAME bytes with no format library
  tokenizer.json— the char vocabulary
  trainlog.json — the real loss curve, for the film

Run:  python train.py            (needs torch; MPS is used when available)
"""
import json, math, os, struct, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text())
A, T, C = CFG["arch"], CFG["train"], CFG["corpus"]

torch.manual_seed(T["seed"])
np.random.seed(T["seed"])

dev = T["device"]
if dev == "mps" and not torch.backends.mps.is_available():
    print("!! MPS unavailable, falling back to cpu", file=sys.stderr)
    dev = "cpu"

# ---------------------------------------------------------------- corpus
text = (HERE / C["path"]).read_text(encoding="utf-8")
vocab = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for ch, i in stoi.items()}
V = len(vocab)
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(len(data) * C["split"])
train_data, val_data = data[:n], data[n:]
print(f"corpus {len(text):,} chars · vocab {V} · train {len(train_data):,} · val {len(val_data):,}")

BS, BLK = T["batch_size"], A["block_size"]


def batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLK - 1, (BS,))
    x = torch.stack([d[i : i + BLK] for i in ix])
    y = torch.stack([d[i + 1 : i + 1 + BLK] for i in ix])
    return x.to(dev), y.to(dev)


# ---------------------------------------------------------------- model
def gelu_tanh(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        d = A["d_model"]
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=A["bias"])
        self.proj = nn.Linear(d, d, bias=A["bias"])
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, A["d_ff"], bias=A["bias"])
        self.fc2 = nn.Linear(A["d_ff"], d, bias=A["bias"])
        self.nh = A["n_head"]
        self.hd = d // self.nh

    def attn(self, x):
        B, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        q = q.view(B, t, self.nh, self.hd).transpose(1, 2)
        k = k.view(B, t, self.nh, self.hd).transpose(1, 2)
        v = v.view(B, t, self.nh, self.hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        mask = torch.tril(torch.ones(t, t, device=x.device)).view(1, 1, t, t)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, t, d)
        return self.proj(y)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.fc2(gelu_tanh(self.fc1(self.ln2(x))))
        return x


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        d = A["d_model"]
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(A["block_size"], d)
        self.blocks = nn.ModuleList([Block() for _ in range(A["n_layer"])])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=A["bias"])

    def forward(self, idx):
        B, t = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))


model = Tiny().to(dev)
nparams = sum(p.numel() for p in model.parameters())
print(f"parameters {nparams:,}")

opt = torch.optim.AdamW(model.parameters(), lr=T["lr"], betas=(0.9, 0.95), weight_decay=0.1)


@torch.no_grad()
def evaluate():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(T["eval_batches"]):
            x, y = batch(split)
            logits = model(x)
            losses.append(F.cross_entropy(logits.view(-1, V), y.view(-1)).item())
        out[split] = float(np.mean(losses))
    model.train()
    return out


log, t0 = [], time.time()
for step in range(1, T["steps"] + 1):
    lr = T["lr"] * min(1.0, step / T["warmup"])
    for g in opt.param_groups:
        g["lr"] = lr
    x, y = batch("train")
    loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % T["eval_every"] == 0 or step == 1:
        e = evaluate()
        el = time.time() - t0
        log.append({"step": step, "train": e["train"], "val": e["val"], "elapsed_s": round(el, 2)})
        print(f"  step {step:>5}/{T['steps']}  train {e['train']:.4f}  val {e['val']:.4f}  {el:6.1f}s")

print(f"trained in {time.time()-t0:.1f}s on {dev}")

# ---------------------------------------------------------------- export
# One flat float32 blob + a manifest. No .npz, no pickle: C must read this too.
model.eval().to("cpu")
tensors, manifest, offset = [], [], 0
def put(name, t):
    global offset
    a = t.detach().cpu().numpy().astype("<f4", copy=True)
    tensors.append(a.ravel())
    manifest.append({"name": name, "shape": list(a.shape), "offset": offset, "count": int(a.size)})
    offset += int(a.size)

put("tok.weight", model.tok.weight)
put("pos.weight", model.pos.weight)
for i, b in enumerate(model.blocks):
    put(f"b{i}.ln1.g", b.ln1.weight); put(f"b{i}.ln1.b", b.ln1.bias)
    # store Linear weights TRANSPOSED to [in, out] so every implementation does x @ W
    put(f"b{i}.qkv.w", b.qkv.weight.T); put(f"b{i}.qkv.b", b.qkv.bias)
    put(f"b{i}.proj.w", b.proj.weight.T); put(f"b{i}.proj.b", b.proj.bias)
    put(f"b{i}.ln2.g", b.ln2.weight); put(f"b{i}.ln2.b", b.ln2.bias)
    put(f"b{i}.fc1.w", b.fc1.weight.T); put(f"b{i}.fc1.b", b.fc1.bias)
    put(f"b{i}.fc2.w", b.fc2.weight.T); put(f"b{i}.fc2.b", b.fc2.bias)
put("lnf.g", model.lnf.weight); put("lnf.b", model.lnf.bias)
put("head.w", model.head.weight.T); put("head.b", model.head.bias)

blob = np.concatenate(tensors).astype("<f4")
(HERE / "weights.bin").write_bytes(blob.tobytes())
(HERE / "weights.json").write_text(json.dumps({
    "dtype": "float32", "endian": "little", "total_floats": int(blob.size),
    "vocab_size": V, "n_params": int(nparams), "tensors": manifest,
}, indent=2))
(HERE / "tokenizer.json").write_text(json.dumps({"type": "char", "vocab": vocab}, indent=2))
(HERE / "trainlog.json").write_text(json.dumps({
    "device": dev, "seed": T["seed"], "steps": T["steps"],
    "n_params": int(nparams), "vocab_size": V, "curve": log,
    "final": log[-1] if log else None,
}, indent=2))

print(f"wrote weights.bin ({blob.nbytes/1e6:.2f} MB, {blob.size:,} floats) + manifest + tokenizer + trainlog")

# sanity: greedy sample straight from torch, the reference the six must match
with torch.no_grad():
    ids = [stoi[c] for c in CFG["decode"]["prompt"]]
    for _ in range(CFG["decode"]["new_tokens"]):
        ctx = ids[-A["block_size"]:]
        logits = model(torch.tensor([ctx]))[0, -1]
        ids.append(int(torch.argmax(logits).item()))
    print("\n--- reference greedy sample (torch, fp32 cpu) ---")
    print("".join(itos[i] for i in ids))
