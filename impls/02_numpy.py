#!/usr/bin/env python3
"""02 — NumPy. The same arithmetic, handed to someone else's BLAS."""
import json, math, sys, time
import numpy as np
from common import MODEL, Progress, load, report

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
    return flat[t["offset"] : t["offset"] + t["count"]].reshape(t["shape"])


tok_w, pos_w = W("tok.weight"), W("pos.weight")
lnf_g, lnf_b = W("lnf.g"), W("lnf.b")
head_w, head_b = W("head.w"), W("head.b")
B = [{k: W(f"b{i}.{k}") for k in
      ("ln1.g", "ln1.b", "qkv.w", "qkv.b", "proj.w", "proj.b",
       "ln2.g", "ln2.b", "fc1.w", "fc1.b", "fc2.w", "fc2.b")} for i in range(nl)]

S = math.sqrt(2.0 / math.pi)


def layernorm(x, g, b):
    m = x.mean(axis=-1, keepdims=True)
    v = ((x - m) ** 2).mean(axis=-1, keepdims=True)
    return ((x - m) / np.sqrt(v + 1e-5)) * g + b


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(S * (x + 0.044715 * x ** 3)))


def forward(ids):
    T = len(ids)
    x = (tok_w[np.asarray(ids)] + pos_w[:T]).astype(np.float32)
    causal = np.tril(np.ones((T, T), dtype=bool))
    for blk in B:
        h = layernorm(x, blk["ln1.g"], blk["ln1.b"])
        qkv = h @ blk["qkv.w"] + blk["qkv.b"]
        q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]
        q = q.reshape(T, nh, hd).transpose(1, 0, 2)
        k = k.reshape(T, nh, hd).transpose(1, 0, 2)
        v = v.reshape(T, nh, hd).transpose(1, 0, 2)
        att = (q @ k.transpose(0, 2, 1)) * scale
        att = np.where(causal, att, -np.inf)
        att = att - att.max(axis=-1, keepdims=True)
        e = np.exp(att)
        att = e / e.sum(axis=-1, keepdims=True)
        o = (att @ v).transpose(1, 0, 2).reshape(T, d)
        x = x + (o @ blk["proj.w"] + blk["proj.b"])
        h = layernorm(x, blk["ln2.g"], blk["ln2.b"])
        x = x + (gelu(h @ blk["fc1.w"] + blk["fc1.b"]) @ blk["fc2.w"] + blk["fc2.b"])
    x = layernorm(x, lnf_g, lnf_b)
    return x[-1] @ head_w + head_b


ids = [stoi[c] for c in D["prompt"]]
N, blk = D["new_tokens"], A["block_size"]
pr = Progress("02-numpy", N)
t0 = time.time()
logits = None
for i in range(N):
    logits = forward(ids[-blk:])
    ids.append(int(np.argmax(logits)))
    pr.tick(i + 1, vocab[ids[-1]])
el = time.time() - t0
pr.done()
report("02-numpy", ids, "".join(vocab[i] for i in ids), el, N, float(np.sum(logits)))
