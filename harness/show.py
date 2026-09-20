#!/usr/bin/env python3
"""Pretty-print one implementation's JSON result for the screen recording.

    python 02_numpy.py | show        (show is a shell function pointing at this file)

The implementations print one JSON line on stdout — the harness's contract — which is unreadable
on camera (it carries the full token list). This reformats THAT SAME line: nothing here computes,
rounds for effect, or substitutes a value. The progress bar above it is the implementation's own
stderr, drawn live by the implementation.

It must never print a filesystem path. Output should be machine-independent, so two people running
this on different computers see byte-identical lines.
"""
import json, sys

raw = sys.stdin.read().strip().splitlines()
if not raw:
    sys.exit("no result on stdin")
r = json.loads(raw[-1])

DIM, BOLD, YEL, RST = "\033[2m", "\033[1m", "\033[33m", "\033[0m"


def dur(s):
    return f"{s:.2f} s" if s >= 1 else f"{s*1000:.0f} ms"


def rate(t):
    return f"{t:,.0f}" if t >= 100 else (f"{t:.1f}" if t >= 10 else f"{t:.2f}")


print()
for line in r["text"].split("\n"):
    print(f"    {line}")
print()
print(f"  {DIM}{r['forward_passes']} tokens in{RST} {BOLD}{dur(r['elapsed_s'])}{RST}"
      f"   {DIM}·{RST}   {YEL}{BOLD}{rate(r['tokens_per_s'])} tokens/sec{RST}")
