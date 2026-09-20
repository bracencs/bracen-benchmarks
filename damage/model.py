"""The ONE evaluator the damage experiments (bit flips, pruning) trust — and how it proves itself.

Loads the transformer-race weights as a single mutable float32 array (a copy of weights.bin), so an
experiment can flip or zero raw values and measure what breaks. Two measurements:

  loss(w)    mean next-character cross-entropy over a FIXED held-out slice of the validation split
             (every position of every window predicts its successor: teacher forcing, one batched pass)
  sample(w)  greedy continuation of the race prompt — exactly the algorithm the six race
             implementations run, so an undamaged model must reproduce the race's tokens

SELF-CHECK (run this file directly): at zero damage, sample() must equal the recorded race tokens
token-for-token, and loss() must land near training's reported val loss. Any experiment built on a
broken evaluator measures nothing — so `check()` is called at the top of every experiment script.
"""
import json, math, warnings
from pathlib import Path

import numpy as np

# Damage legitimately overflows — that IS the experiment, so the warning carries no information here. It also
# prints the absolute path of the file that raised it, which makes output machine-dependent. Silence the category.
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "model"
CFG = json.loads((MODEL / "config.json").read_text())
MAN = json.loads((MODEL / "weights.json").read_text())
VOCAB = json.loads((MODEL / "tokenizer.json").read_text())["vocab"]
A, D = CFG["arch"], CFG["decode"]
d, nh, nl, dff, BLK = A["d_model"], A["n_head"], A["n_layer"], A["d_ff"], A["block_size"]
hd = d // nh
V = len(VOCAB)
STOI = {c: i for i, c in enumerate(VOCAB)}
TENSORS = MAN["tensors"]
N_FLOATS = MAN["total_floats"]
S = math.sqrt(2.0 / math.pi)

# Fixed evaluation slice: the first EVAL_WINDOWS non-overlapping windows of the validation split.
# Fixed, not random, so every damage level is scored on exactly the same text.
EVAL_WINDOWS = 64
_text = (MODEL / CFG["corpus"]["path"]).read_text(encoding="utf-8")
_val = np.array([STOI[c] for c in _text[int(len(_text) * CFG["corpus"]["split"]):]], dtype=np.int64)
_starts = np.arange(EVAL_WINDOWS) * (BLK + 1)
EVAL_X = np.stack([_val[s: s + BLK] for s in _starts])
EVAL_Y = np.stack([_val[s + 1: s + BLK + 1] for s in _starts])


def pristine() -> np.ndarray:
    """A fresh, writable copy of the weights. Experiments damage copies, never the file."""
    return np.frombuffer((MODEL / "weights.bin").read_bytes(), dtype="<f4").copy()


def views(w: np.ndarray) -> dict:
    return {t["name"]: w[t["offset"]: t["offset"] + t["count"]].reshape(t["shape"]) for t in TENSORS}


def _ln(x, g, b):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * g + b


def _forward_all(W: dict, X: np.ndarray) -> np.ndarray:
    """Logits for EVERY position. X: [B, T] int -> [B, T, V]. Same arithmetic as impls/02_numpy.py."""
    Bn, T = X.shape
    x = W["tok.weight"][X] + W["pos.weight"][:T]
    mask = np.tril(np.ones((T, T), dtype=bool))
    with np.errstate(all="ignore"):   # damage legitimately produces inf/nan; measure it, don't crash on it
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
        return _ln(x, W["lnf.g"], W["lnf.b"]) @ W["head.w"] + W["head.b"]


def loss(w: np.ndarray) -> float:
    """Mean cross-entropy on the fixed slice. Returns inf if the model produces non-finite logits."""
    logits = _forward_all(views(w), EVAL_X)
    if not np.isfinite(logits).all():
        return float("inf")
    m = logits.max(-1, keepdims=True)
    lse = (m + np.log(np.exp(logits - m).sum(-1, keepdims=True)))[..., 0]
    picked = np.take_along_axis(logits, EVAL_Y[..., None], -1)[..., 0]
    return float((lse - picked).mean())


def sample(w: np.ndarray, n: int | None = None, prompt: str | None = None) -> tuple[list[int], str]:
    """Greedy continuation, no KV cache — the race's algorithm exactly."""
    W = views(w)
    ids = [STOI[c] for c in (prompt or D["prompt"])]
    for _ in range(n if n is not None else D["new_tokens"]):
        lg = _forward_all(W, np.array([ids[-BLK:]]))[0, -1]
        ids.append(int(np.argmax(lg)) if np.isfinite(lg).all() else 0)
    return ids, "".join(VOCAB[i] for i in ids)


def race_tokens() -> list[int]:
    rec = max((ROOT / "recordings").glob("race@*/recording.json"), key=lambda p: p.stat().st_mtime)
    return json.loads(rec.read_text())["results"][0]["tokens"]


def check(verbose: bool = False) -> float:
    """Refuse to run an experiment on an evaluator that doesn't reproduce the race. Returns baseline loss."""
    w = pristine()
    ids, _ = sample(w)
    if ids != race_tokens():
        raise SystemExit("EVALUATOR BROKEN: undamaged sample() does not reproduce the race tokens")
    base = loss(w)
    if not (1.4 < base < 2.2):
        raise SystemExit(f"EVALUATOR SUSPECT: baseline loss {base:.4f} is far from training's val ~1.79")
    if verbose:
        print(f"evaluator OK: sample() == race tokens ({len(ids)}), baseline loss {base:.4f} "
              f"on {EVAL_WINDOWS} fixed windows x {BLK} predictions")
    return base


if __name__ == "__main__":
    check(verbose=True)
