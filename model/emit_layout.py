#!/usr/bin/env python3
"""Derive model/layout.txt from config.json + weights.json + tokenizer.json.

The C implementations read this instead of parsing JSON. It is GENERATED, never edited:
every number in it comes from the same manifest the Python implementations read, so C and
Python cannot drift apart by someone updating one and forgetting the other.

Format is line-based, whitespace-separated, trivially fscanf-able:
    scalar <name> <int>
    prompt <n> <id> <id> ...
    tensor <name> <offset> <count>
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
cfg = json.loads((HERE / "config.json").read_text())
man = json.loads((HERE / "weights.json").read_text())
tok = json.loads((HERE / "tokenizer.json").read_text())

A, D = cfg["arch"], cfg["decode"]
stoi = {c: i for i, c in enumerate(tok["vocab"])}
ids = [stoi[c] for c in D["prompt"]]

lines = []
for k, v in (
    ("d_model", A["d_model"]), ("n_head", A["n_head"]), ("n_layer", A["n_layer"]),
    ("d_ff", A["d_ff"]), ("block_size", A["block_size"]),
    ("vocab_size", man["vocab_size"]), ("new_tokens", D["new_tokens"]),
    ("total_floats", man["total_floats"]),
):
    lines.append(f"scalar {k} {v}")
lines.append("prompt " + str(len(ids)) + " " + " ".join(str(i) for i in ids))
for t in man["tensors"]:
    lines.append(f"tensor {t['name']} {t['offset']} {t['count']}")

(HERE / "layout.txt").write_text("\n".join(lines) + "\n")
# the vocabulary, one char per line as a decimal codepoint, so C can print the sample
(HERE / "vocab.txt").write_text(
    "\n".join(str(ord(c)) for c in tok["vocab"]) + "\n"
)
print(f"wrote layout.txt ({len(lines)} lines) and vocab.txt ({len(tok['vocab'])} entries)")
