#!/usr/bin/env python3
"""
WHY a top-exponent flip in the attention VALUE part is fatal and one in the QUERY/KEY part is not.
Measured, not asserted.

    python why_flip.py        -> recordings/bitflip/why_flip.json

For each case, ONE bit-30 flip (the same element choice rule as qkv_split.py), then three measurements taken
inside the forward pass on the fixed evaluation slice (64 windows x 96 positions):

  positions_destroyed   share of positions whose prediction is non-finite, or whose top guess differs
                        from the undamaged model's
  one_hot_attention     share of attention rows (in the flipped block) whose largest weight is > 0.99,
                        i.e. attention collapsed onto a single earlier character
  max_residual          the largest |value| in the residual stream after the flipped block

The claim this tests: a huge query or key only makes attention scores huge, and softmax turns huge scores
into a finite one-hot choice. A huge value is multiplied into what attention writes back, at every
position that looks at it, so it floods the residual stream.
"""
import json, math, os, sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # see bitflip.py
import numpy as np

from bitflip import BASE, broken, flip
from model import EVAL_X, S, TENSORS, _ln, d, hd, loss, nh, nl, pristine, views

OUT = Path(__file__).resolve().parent.parent / "recordings" / "bitflip"
IX = {t["name"]: t for t in TENSORS}
REPS = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 20
OUT_NAME = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "why_flip.json"
BLOCK = 1


def probe(w: np.ndarray, block: int):
    """Forward pass that records attention rows of `block` and the residual stream after it."""
    W = views(w); X = EVAL_X
    Bn, T = X.shape
    x = W["tok.weight"][X] + W["pos.weight"][:T]
    mask = np.tril(np.ones((T, T), dtype=bool))
    one_hot = max_res = None
    with np.errstate(all="ignore"):
        for i in range(nl):
            B = lambda k: W[f"b{i}.{k}"]
            qkv = _ln(x, B("ln1.g"), B("ln1.b")) @ B("qkv.w") + B("qkv.b")
            q, k, v = (qkv[..., j * d:(j + 1) * d].reshape(Bn, T, nh, hd).transpose(0, 2, 1, 3) for j in range(3))
            att = np.where(mask, (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(hd), -np.inf)
            att = np.exp(att - att.max(-1, keepdims=True))
            att = att / att.sum(-1, keepdims=True)
            x = x + (att @ v).transpose(0, 2, 1, 3).reshape(Bn, T, d) @ B("proj.w") + B("proj.b")
            h = _ln(x, B("ln2.g"), B("ln2.b")) @ B("fc1.w") + B("fc1.b")
            x = x + (0.5 * h * (1 + np.tanh(S * (h + 0.044715 * h ** 3)))) @ B("fc2.w") + B("fc2.b")
            if i == block:
                top = np.nanmax(np.where(np.isfinite(att), att, np.nan), -1)
                # rows 0 of every head are trivially one-hot (a single visible position): exclude them
                one_hot = float(np.mean(top[..., 1:] > 0.99))
                max_res = float(np.nanmax(np.abs(np.where(np.isfinite(x), x, np.inf))))
        logits = _ln(x, W["lnf.g"], W["lnf.b"]) @ W["head.w"] + W["head.b"]
    return logits, one_hot, max_res


def element(rng, block: int, part: int) -> int:
    t = IX[f"b{block}.qkv.w"]
    return t["offset"] + int(rng.integers(0, t["shape"][0])) * t["shape"][1] + part * d + int(rng.integers(0, d))


if __name__ == "__main__":
    ref_logits, ref_one_hot, ref_res = probe(pristine(), BLOCK)
    ref_top = ref_logits.argmax(-1)
    print(f"one bit-30 flip in attention layer {BLOCK + 1}, {REPS} times per part\n")
    print(f"  {'part':<7}{'broken':>8}{'predictions changed':>22}{'one-character attention':>26}{'largest number':>17}")
    print(f"  {'undamaged':<15}{'':>22}{ref_one_hot:>26.0%}{ref_res:>17.3g}")
    rows = []
    for pname, part in (("query", 0), ("key", 1), ("value", 2)):
        for r in range(REPS):
            rng = np.random.default_rng(55_000 + 100 * part + r)
            w = pristine(); e = element(rng, BLOCK, part)
            log = []; flip(w, np.array([e]), np.array([30]), log)
            lg, oh, mr = probe(w, BLOCK)
            bad = ~np.isfinite(lg).all(-1) | (lg.argmax(-1) != ref_top)
            l = loss(w)
            rows.append({"part": pname, "elem": e, "old": log[0]["old"], "new": log[0]["new"], "loss": l,
                         "broken": broken(l), "positions_destroyed": float(bad.mean()),
                         "one_hot_attention": oh, "max_residual": mr})
        sub = [x for x in rows if x["part"] == pname]
        med = lambda k: float(np.median([x[k] for x in sub]))
        print(f"  {pname:<7}{sum(x['broken'] for x in sub):>4}/{REPS:<3}{med('positions_destroyed'):>22.0%}"
              f"{med('one_hot_attention'):>26.0%}{med('max_residual'):>17.3g}", flush=True)
    summary = {p: {k: float(np.median([x[k] for x in rows if x["part"] == p]))
                   for k in ("positions_destroyed", "one_hot_attention", "max_residual")}
               | {"broken": sum(x["broken"] for x in rows if x["part"] == p), "reps": REPS} for p in ("query", "key", "value")}
    (OUT / OUT_NAME).write_text(json.dumps({"baseline": BASE, "block": BLOCK,
        "undamaged": {"one_hot_attention": ref_one_hot, "max_residual": ref_res},
        "summary": summary, "rows": rows}, indent=2))
    print(f"saved recordings/bitflip/{OUT_NAME}")
