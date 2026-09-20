#!/usr/bin/env python3
"""Stream the model's greedy output one character at a time, for the screen recording.

This is NOT one of the six raced implementations and is never timed. It exists so scene 6 can
show the model actually writing, character by character, at whatever speed it really runs —
there is no sleep in here, and adding one would be faking the thing the video exists to show.

It runs the same weights with the same NumPy forward pass as 02_numpy.py and prints the same
first 64 tokens the race prints; it simply keeps going, so the viewer can watch it fall into the
loop the narration describes.
"""
import json, math, sys
from pathlib import Path
import numpy as np

MODEL = Path(__file__).resolve().parent.parent / "model"
cfg = json.loads((MODEL / "config.json").read_text())
man = json.loads((MODEL / "weights.json").read_text())
vocab = json.loads((MODEL / "tokenizer.json").read_text())["vocab"]
A, D = cfg["arch"], cfg["decode"]
d, nh, nl = A["d_model"], A["n_head"], A["n_layer"]
hd = d // nh
flat = np.frombuffer((MODEL / "weights.bin").read_bytes(), dtype="<f4")
idx = {t["name"]: t for t in man["tensors"]}
W = lambda n: flat[idx[n]["offset"]: idx[n]["offset"] + idx[n]["count"]].reshape(idx[n]["shape"])
S = math.sqrt(2 / math.pi)


def ln(x, g, b):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * g + b


def fwd(ids):
    T = len(ids)
    x = W("tok.weight")[np.asarray(ids)] + W("pos.weight")[:T]
    mask = np.tril(np.ones((T, T), bool))
    for i in range(nl):
        B = lambda k: W(f"b{i}.{k}")
        qkv = ln(x, B("ln1.g"), B("ln1.b")) @ B("qkv.w") + B("qkv.b")
        q, k, v = (qkv[:, j * d:(j + 1) * d].reshape(T, nh, hd).transpose(1, 0, 2) for j in range(3))
        a = np.where(mask, (q @ k.transpose(0, 2, 1)) / math.sqrt(hd), -np.inf)
        a = np.exp(a - a.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True)
        x = x + (a @ v).transpose(1, 0, 2).reshape(T, d) @ B("proj.w") + B("proj.b")
        h = ln(x, B("ln2.g"), B("ln2.b")) @ B("fc1.w") + B("fc1.b")
        x = x + (0.5 * h * (1 + np.tanh(S * (h + 0.044715 * h ** 3)))) @ B("fc2.w") + B("fc2.b")
    return ln(x, W("lnf.g"), W("lnf.b"))[-1] @ W("head.w") + W("head.b")


n = int(sys.argv[1]) if len(sys.argv) > 1 else 160
stoi = {c: i for i, c in enumerate(vocab)}
ids = [stoi[c] for c in D["prompt"]]
sys.stdout.write("\n\033[2m" + D["prompt"] + "\033[0m")
sys.stdout.flush()
for _ in range(n):
    nxt = int(np.argmax(fwd(ids[-A["block_size"]:])))
    ids.append(nxt)
    sys.stdout.write(vocab[nxt])
    sys.stdout.flush()
sys.stdout.write("\n")
