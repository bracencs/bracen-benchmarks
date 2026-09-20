#!/usr/bin/env python3
"""
Count the operations the GPU run ACTUALLY dispatches — replacing a structural estimate.

why_mps_lost.py reported "9,472 dispatches per generation" from a hand count of ~35 ops per
layer. The film put that number on screen as if it were measured. A hand recount came to ~42 per
layer, so the estimate was not even self-consistent. This measures it.

Method: torch.utils._python_dispatch.TorchDispatchMode sees every aten operator call that reaches
a backend. Running the exact generation loop of impls/05_06_torch.py on MPS inside the mode counts
each op executed on an MPS tensor. Metadata-only ops (views, reshapes, transposes, slices) move no
data and launch no kernel, so they are counted separately and excluded from the headline number.

Also records the multiply-add count per forward pass, computed from the architecture, so the film's
narration can quote a derived number instead of a guess (the previous "four million per
character" was 3-17x too low).

Output: recordings/dispatch_count.json
"""
import json, math, sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "impls"))
from common import load  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else sys.exit("needs MPS")
cfg, man, tokj, raw, index = load()
A, D = cfg["arch"], cfg["decode"]
d, nh, nl, dff = A["d_model"], A["n_head"], A["n_layer"], A["d_ff"]
hd = d // nh
V = len(tokj["vocab"])
stoi = {c: i for i, c in enumerate(tokj["vocab"])}
flat = np.frombuffer(raw, dtype="<f4")
S = math.sqrt(2.0 / math.pi)


def W(name):
    t = index[name]
    return torch.from_numpy(np.array(flat[t["offset"]: t["offset"] + t["count"]].reshape(t["shape"]),
                                     dtype=np.float32)).to(DEV)


tok_w, pos_w, lnf_g, lnf_b, head_w, head_b = (W(n) for n in
    ("tok.weight", "pos.weight", "lnf.g", "lnf.b", "head.w", "head.b"))
B = [{k: W(f"b{i}.{k}") for k in ("ln1.g", "ln1.b", "qkv.w", "qkv.b", "proj.w", "proj.b",
                                  "ln2.g", "ln2.b", "fc1.w", "fc1.b", "fc2.w", "fc2.b")} for i in range(nl)]


# IDENTICAL to impls/05_06_torch.py's forward — copied, not imported, because that file runs its
# timed generation at import time. tests below assert the tokens still match the recording.
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
        att = (q @ k.transpose(1, 2)) * (1.0 / math.sqrt(hd))
        att = att.masked_fill(~causal, float("-inf"))
        att = torch.softmax(att, dim=-1)
        o = (att @ v).transpose(0, 1).reshape(T, d)
        x = x + (o @ blk["proj.w"] + blk["proj.b"])
        h = layernorm(x, blk["ln2.g"], blk["ln2.b"])
        x = x + (gelu(h @ blk["fc1.w"] + blk["fc1.b"]) @ blk["fc2.w"] + blk["fc2.b"])
    x = layernorm(x, lnf_g, lnf_b)
    return x[-1] @ head_w + head_b


METADATA = {"view", "_unsafe_view", "transpose", "t", "slice", "select", "reshape", "expand",
            "as_strided", "alias", "detach", "unsqueeze", "squeeze", "permute", "lift_fresh"}


class Count(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        tensors = [a for a in list(args) + list((kwargs or {}).values()) if isinstance(a, torch.Tensor)]
        if isinstance(out, torch.Tensor):
            tensors.append(out)
        if any(t.device.type == "mps" for t in tensors):
            self.ops[func.overloadpacket.__name__] += 1
        return out


ids = [stoi[c] for c in D["prompt"]]
blk = A["block_size"]
per_forward = []
with Count() as c:
    for _ in range(D["new_tokens"]):
        before = sum(c.ops.values())
        logits = forward(torch.tensor(ids[-blk:], dtype=torch.long, device=DEV))
        ids.append(int(torch.argmax(logits).item()))
        per_forward.append(sum(c.ops.values()) - before)

meta = sum(n for op, n in c.ops.items() if op in METADATA)
compute = sum(n for op, n in c.ops.items() if op not in METADATA)

rec = json.loads(max((ROOT / "recordings").glob("race@*/recording.json"),
                     key=lambda p: p.stat().st_mtime).read_text())
race_tokens = rec["results"][0]["tokens"]
assert ids == race_tokens, "counting run diverged from the recorded race — the forward pass drifted"

# multiply-adds, from the architecture: no KV cache, so forward pass t sees T = len(prompt)+t tokens
P = len(D["prompt"])
def macs(T):
    linear = nl * T * (d * 3 * d + d * d + d * dff + dff * d)
    attention = nl * 2 * T * T * d
    return linear + attention + d * V
Ts = [P + t for t in range(D["new_tokens"])]

out = {
    "method": "TorchDispatchMode over the exact 05_06_torch.py forward on MPS; tokens asserted equal to the recorded race",
    "ops_total": sum(c.ops.values()),
    "ops_metadata_only": meta,
    "ops_launched": compute,
    "ops_launched_per_forward_mean": round(compute / D["new_tokens"], 1),
    "per_forward_all_ops": [per_forward[0], per_forward[-1]],
    "by_op": dict(c.ops.most_common()),
    "structural_estimate_it_replaces": 9472,
    "macs": {
        "first_forward": macs(Ts[0]), "last_forward": macs(Ts[-1]),
        "whole_generation": sum(macs(T) for T in Ts),
        "context_range": [Ts[0], Ts[-1]],
        "formula": "n_layer*T*(4*d*d + 2*d*d_ff) + n_layer*2*T*T*d + d*vocab",
    },
}
(ROOT / "recordings" / "dispatch_count.json").write_text(json.dumps(out, indent=2))
print(f"ops that launch work on the GPU: {compute:,}  ({out['ops_launched_per_forward_mean']} per forward)")
print(f"metadata-only ops (views etc., no kernel): {meta:,}")
print(f"replaces the structural estimate: 9,472")
m = out["macs"]
print(f"multiply-adds: first char {m['first_forward']:,} · last {m['last_forward']:,} · "
      f"whole paragraph {m['whole_generation']:,}")
print("saved recordings/dispatch_count.json")
