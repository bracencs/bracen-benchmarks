# The contract every implementation obeys

Six implementations, one set of weights, one prompt, one answer. If any two disagree on the
output tokens, the race is meaningless — so the tokens are the gate, not the times.

## Inputs (all read from disk, nothing hardcoded)

| file | what |
|---|---|
| `../model/config.json` | architecture + decode settings |
| `../model/weights.json` | tensor manifest: name, shape, offset (in floats), count |
| `../model/weights.bin` | every tensor concatenated, raw little-endian float32 |
| `../model/tokenizer.json` | the char vocabulary |

Linear weights are stored **transposed to `[in, out]`**, so every implementation computes
`y = x @ W + b` with no transpose of its own.

## The forward pass, in exactly this order

```
x[t] = tok[id[t]] + pos[t]                       for t in 0..T-1
for each block b:
    h  = layernorm(x, b.ln1.g, b.ln1.b)
    qkv = h @ b.qkv.w + b.qkv.b                   # [T, 3d]
    q,k,v = qkv[:, 0:d], qkv[:, d:2d], qkv[:, 2d:3d]
    per head i of n_head, over columns [i*hd, (i+1)*hd):
        att[p][q] = dot(q[p], k[q]) / sqrt(hd)    for q <= p   (causal)
        att[p]    = softmax(att[p][0..p])
        o[p]     += att[p][q] * v[q]
    x += o @ b.proj.w + b.proj.b
    h  = layernorm(x, b.ln2.g, b.ln2.b)
    x += gelu(h @ b.fc1.w + b.fc1.b) @ b.fc2.w + b.fc2.b
x = layernorm(x, lnf.g, lnf.b)
logits = x[T-1] @ head.w + head.b
next = argmax(logits)
```

**No KV cache, on purpose.** Every generated token re-runs the whole context. It is the one
algorithm all six can express identically, so the race measures the matmul and nothing else.
It is also the naive thing a first implementation does, which is the point of the video.

## Numerics — the details that decide whether tokens match

- `layernorm`: `eps = 1e-5`, **biased** variance (divide by N, not N-1) — this is what
  `torch.nn.LayerNorm` does. Getting this wrong shifts every downstream value.
- `gelu`: the tanh approximation, `0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3)))`.
- `softmax`: subtract the row max before `exp`.
- attention scale: `1/sqrt(head_dim)` where `head_dim = d_model / n_head`.
- everything is float32; accumulation order may differ between implementations, and it does.

## Output — one JSON object on stdout

```json
{"impl":"02-numpy","tokens":[...],"text":"...","elapsed_s":1.234,
 "tokens_per_s":51.9,"logit_sum":-1234.5,"forward_passes":64}
```

`logit_sum` is the sum of the final logit vector on the LAST step, printed to catch a silent
numerical divergence that argmax happened to absorb. Times are wall clock around the generation
loop only — model load is excluded, because load time is not what is being raced.
