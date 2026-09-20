#!/usr/bin/env python3
"""05 / 06 — the same PyTorch, on two devices.

  python 05_06_torch.py cpu   -> 05-torch-cpu
  python 05_06_torch.py mps   -> 06-torch-mps

One source file, like the C pair. The device string is the only difference, so anything
between the two times is the device and not the code.

Weights are loaded from the SAME weights.bin every other implementation reads — this does not
reload the training checkpoint, because that would be a different code path from the one the
other five are running.
"""
import math, sys, time
import numpy as np
import torch
from common import Progress, load, report

DEV = sys.argv[1] if len(sys.argv) > 1 else "cpu"
IMPL = {"cpu": "05-torch-cpu", "mps": "06-torch-mps"}[DEV]

if DEV == "mps" and not torch.backends.mps.is_available():
    sys.exit("mps unavailable on this machine")

cfg, man, tokj, raw, index = load()
A, D = cfg["arch"], cfg["decode"]
d, nh, nl, dff = A["d_model"], A["n_head"], A["n_layer"], A["d_ff"]
hd = d // nh
scale = 1.0 / math.sqrt(hd)
vocab = tokj["vocab"]
V = len(vocab)
stoi = {c: i for i, c in enumerate(vocab)}

flat = np.frombuffer(raw, dtype="<f4")


def W(name):
    t = index[name]
    a = flat[t["offset"] : t["offset"] + t["count"]].reshape(t["shape"])
    return torch.from_numpy(np.array(a, dtype=np.float32)).to(DEV)


tok_w, pos_w = W("tok.weight"), W("pos.weight")
lnf_g, lnf_b = W("lnf.g"), W("lnf.b")
head_w, head_b = W("head.w"), W("head.b")
B = [{k: W(f"b{i}.{k}") for k in
      ("ln1.g", "ln1.b", "qkv.w", "qkv.b", "proj.w", "proj.b",
       "ln2.g", "ln2.b", "fc1.w", "fc1.b", "fc2.w", "fc2.b")} for i in range(nl)]

S = math.sqrt(2.0 / math.pi)


def layernorm(x, g, b):
    m = x.mean(-1, keepdim=True)
    v = ((x - m) ** 2).mean(-1, keepdim=True)
    return (x - m) / torch.sqrt(v + 1e-5) * g + b


def gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(S * (x + 0.044715 * x ** 3)))


@torch.no_grad()
def forward(ids_t):
    T = ids_t.shape[0]
    x = tok_w[ids_t] + pos_w[:T]
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=DEV))
    for blk in B:
        h = layernorm(x, blk["ln1.g"], blk["ln1.b"])
        qkv = h @ blk["qkv.w"] + blk["qkv.b"]
        q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]
        q = q.view(T, nh, hd).transpose(0, 1)
        k = k.view(T, nh, hd).transpose(0, 1)
        v = v.view(T, nh, hd).transpose(0, 1)
        att = (q @ k.transpose(1, 2)) * scale
        att = att.masked_fill(~causal, float("-inf"))
        att = torch.softmax(att, dim=-1)
        o = (att @ v).transpose(0, 1).reshape(T, d)
        x = x + (o @ blk["proj.w"] + blk["proj.b"])
        h = layernorm(x, blk["ln2.g"], blk["ln2.b"])
        x = x + (gelu(h @ blk["fc1.w"] + blk["fc1.b"]) @ blk["fc2.w"] + blk["fc2.b"])
    x = layernorm(x, lnf_g, lnf_b)
    return x[-1] @ head_w + head_b


ids = [stoi[c] for c in D["prompt"]]
N, blk = D["new_tokens"], A["block_size"]

# Warm the device once BEFORE the clock starts. On MPS the first kernel launch pays for
# shader compilation; timing that would measure the driver warming up, not the matmul.
_ = forward(torch.tensor(ids[-blk:], dtype=torch.long, device=DEV))
if DEV == "mps":
    torch.mps.synchronize()

pr = Progress(IMPL, N)
t0 = time.time()
logits = None
for i in range(N):
    logits = forward(torch.tensor(ids[-blk:], dtype=torch.long, device=DEV))
    ids.append(int(torch.argmax(logits).item()))
    pr.tick(i + 1, vocab[ids[-1]])
if DEV == "mps":
    torch.mps.synchronize()
el = time.time() - t0
pr.done()
report(IMPL, ids, "".join(vocab[i] for i in ids), el, N, float(logits.sum().item()))
