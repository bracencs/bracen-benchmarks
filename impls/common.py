"""Shared loading + reporting for the Python implementations. No math lives here."""
import json, sys, time
from pathlib import Path

MODEL = Path(__file__).resolve().parent.parent / "model"


def load():
    cfg = json.loads((MODEL / "config.json").read_text())
    man = json.loads((MODEL / "weights.json").read_text())
    tok = json.loads((MODEL / "tokenizer.json").read_text())
    raw = (MODEL / "weights.bin").read_bytes()
    index = {t["name"]: t for t in man["tensors"]}
    return cfg, man, tok, raw, index


def report(impl, tokens, text, elapsed, forwards, logit_sum):
    print(json.dumps({
        "impl": impl,
        "tokens": tokens,
        "text": text,
        "elapsed_s": round(elapsed, 6),
        "tokens_per_s": round(forwards / elapsed, 4) if elapsed > 0 else None,
        "logit_sum": round(float(logit_sum), 4),
        "forward_passes": forwards,
    }))


class Progress:
    """Prints a live line to stderr so the screen capture has something to show.

    stderr, never stdout: stdout carries the single JSON result the harness parses.
    """

    def __init__(self, impl, total, every=1):
        self.impl, self.total, self.every, self.t0 = impl, total, every, time.time()
        # Animate only for a human. Piped into the harness this would be thousands of
        # redraw lines in the log; on screen it is the entire point of the shot.
        self.live = sys.stderr.isatty()

    def tick(self, i, ch):
        if not self.live or (i % self.every and i != self.total):
            return
        el = time.time() - self.t0
        rate = i / el if el > 0 else 0
        bar = "#" * int(30 * i / self.total)
        sys.stderr.write(
            f"\r  {self.impl:<16} [{bar:<30}] {i:>3}/{self.total}  {el:7.2f}s  {rate:8.3f} tok/s"
        )
        sys.stderr.flush()

    def done(self):
        if self.live:
            sys.stderr.write("\n")
            sys.stderr.flush()
