#!/usr/bin/env python3
"""01 — pure Python. Lists and loops. No NumPy, no BLAS, no C.

This is the control. Everything else in the race is a way of not doing this.
"""
import array, math, sys, time
from common import MODEL, Progress, load, report

cfg, man, tokj, raw, index = load()
A, D = cfg["arch"], cfg["decode"]
d, nh, nl, dff = A["d_model"], A["n_head"], A["n_layer"], A["d_ff"]
hd = d // nh
scale = 1.0 / math.sqrt(hd)
vocab = tokj["vocab"]
V = len(vocab)
stoi = {c: i for i, c in enumerate(vocab)}

# one flat array of floats; every tensor is a window into it
flat = array.array("f")
flat.frombytes(raw)
if sys.byteorder != "little":
    flat.byteswap()


def W(name):
    """Return a tensor as a flat list plus its shape. Lists beat array.array for random access."""
    t = index[name]
    o, n = t["offset"], t["count"]
    return flat[o : o + n].tolist(), t["shape"]


tok_w, _ = W("tok.weight")
pos_w, _ = W("pos.weight")
lnf_g, _ = W("lnf.g")
lnf_b, _ = W("lnf.b")
head_w, _ = W("head.w")
head_b, _ = W("head.b")

B = []
for i in range(nl):
    B.append({k: W(f"b{i}.{k}")[0] for k in
              ("ln1.g", "ln1.b", "qkv.w", "qkv.b", "proj.w", "proj.b",
               "ln2.g", "ln2.b", "fc1.w", "fc1.b", "fc2.w", "fc2.b")})

SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def layernorm(x, g, b, n):
    """x is a flat list of T*n; normalises each row of length n in place-ish."""
    out = [0.0] * len(x)
    for r in range(0, len(x), n):
        row = x[r : r + n]
        m = sum(row) / n
        v = sum((z - m) * (z - m) for z in row) / n
        inv = 1.0 / math.sqrt(v + 1e-5)
        for j in range(n):
            out[r + j] = (row[j] - m) * inv * g[j] + b[j]
    return out


def matmul(x, T, K, w, N, bias):
    """[T,K] @ [K,N] + [N]  — the thing this whole video is about."""
    out = [0.0] * (T * N)
    for t in range(T):
        xo, oo = t * K, t * N
        for j in range(N):
            acc = bias[j]
            wo = j
            for k in range(K):
                acc += x[xo + k] * w[k * N + wo]
            out[oo + j] = acc
    return out


def forward(ids):
    T = len(ids)
    x = [0.0] * (T * d)
    for t, tid in enumerate(ids):
        to, po, xo = tid * d, t * d, t * d
        for j in range(d):
            x[xo + j] = tok_w[to + j] + pos_w[po + j]

    for blk in B:
        h = layernorm(x, blk["ln1.g"], blk["ln1.b"], d)
        qkv = matmul(h, T, d, blk["qkv.w"], 3 * d, blk["qkv.b"])
        o = [0.0] * (T * d)
        for head in range(nh):
            base = head * hd
            for p in range(T):
                qo = p * 3 * d + base
                sc = [0.0] * (p + 1)
                mx = -1e30
                for q in range(p + 1):
                    ko = q * 3 * d + d + base
                    s = 0.0
                    for j in range(hd):
                        s += qkv[qo + j] * qkv[ko + j]
                    s *= scale
                    sc[q] = s
                    if s > mx:
                        mx = s
                tot = 0.0
                for q in range(p + 1):
                    e = math.exp(sc[q] - mx)
                    sc[q] = e
                    tot += e
                oo = p * d + base
                for q in range(p + 1):
                    wgt = sc[q] / tot
                    vo = q * 3 * d + 2 * d + base
                    for j in range(hd):
                        o[oo + j] += wgt * qkv[vo + j]
        proj = matmul(o, T, d, blk["proj.w"], d, blk["proj.b"])
        for i in range(T * d):
            x[i] += proj[i]

        h = layernorm(x, blk["ln2.g"], blk["ln2.b"], d)
        f1 = matmul(h, T, d, blk["fc1.w"], dff, blk["fc1.b"])
        for i in range(len(f1)):
            z = f1[i]
            f1[i] = 0.5 * z * (1.0 + math.tanh(SQRT_2_OVER_PI * (z + 0.044715 * z * z * z)))
        f2 = matmul(f1, T, dff, blk["fc2.w"], d, blk["fc2.b"])
        for i in range(T * d):
            x[i] += f2[i]

    x = layernorm(x, lnf_g, lnf_b, d)
    last = x[(T - 1) * d : T * d]
    logits = [0.0] * V
    for j in range(V):
        acc = head_b[j]
        for k in range(d):
            acc += last[k] * head_w[k * V + j]
        logits[j] = acc
    return logits


ids = [stoi[c] for c in D["prompt"]]
N = D["new_tokens"]
blk = A["block_size"]
pr = Progress("01-pure-python", N)
t0 = time.time()
logits = None
for i in range(N):
    logits = forward(ids[-blk:])
    best, bi = logits[0], 0
    for j in range(1, V):
        if logits[j] > best:
            best, bi = logits[j], j
    ids.append(bi)
    pr.tick(i + 1, vocab[bi])
el = time.time() - t0
pr.done()
report("01-pure-python", ids, "".join(vocab[i] for i in ids), el, N, sum(logits))
