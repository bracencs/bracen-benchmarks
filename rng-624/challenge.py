#!/usr/bin/env python3
"""challenge.py — run it yourself in under a minute.

It plays both sides: a generator seeded from your operating system, and an observer who sees only
624 of its outputs. The observer then calls the next ten before the generator makes them.

    python3 challenge.py

Then change `random.Random()` to `secrets.SystemRandom()` and watch the prediction collapse to noise.
"""
import random
import secrets
from untemper import untemper

victim = random.Random()                                  # unknown OS seed
observed = [victim.getrandbits(32) for _ in range(624)]   # all an observer ever sees

clone = random.Random()
clone.setstate((3, tuple(untemper(o) for o in observed) + (624,), None))

print("next | predicted vs actual")
for i in range(10):
    p, a = clone.getrandbits(32), victim.getrandbits(32)
    print(f"{i + 1:>4} | {p:>11} {'==' if p == a else '!='} {a}")
