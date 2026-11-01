#!/usr/bin/env python3
"""crack.py — rebuild Python's random generator from 624 of its outputs, then predict the rest.

Not a break of the algorithm. MT19937 was never meant to hide its state (its own paper says so).
Given 624 consecutive 32-bit outputs, untemper each (untemper.py), load them as the state, and you
own a perfect clone of the generator — every number it produces next, you produce first.

Everything below observes ONLY the generator's public outputs. It never reads getstate().

    python3 crack.py            # the headline demo, capture-friendly
"""
import random
from untemper import untemper

N_STATE = 624


def clone_from_outputs(outputs):
    """624 consecutive getrandbits(32) outputs -> a random.Random that is an exact clone."""
    assert len(outputs) == N_STATE, f"need {N_STATE} outputs, got {len(outputs)}"
    state = tuple(untemper(o) for o in outputs)
    clone = random.Random()
    # CPython's state is (version, 624 state words + index, gauss_next).
    # index = 624 means "all consumed" so the next draw regenerates the block, exactly matching
    # the victim, which has also drawn 624 words.
    clone.setstate((3, state + (N_STATE,), None))
    return clone


def demo():
    """Kept deliberately short: this output is filmed, so it has to fit a terminal strip."""
    victim = random.Random()          # seeded from os.urandom by default — no seed known to us
    print("A generator was seeded by the operating system. Nobody here knows its seed.")
    observed = [victim.getrandbits(32) for _ in range(N_STATE)]
    print(f"Watched {N_STATE} of its outputs:  " + "  ".join(f"{o:10d}" for o in observed[:4]) + "  ...")
    print()
    print("State rebuilt. Now calling numbers it has not produced yet:")
    print()
    print("      predicted        actual   match")
    clone = clone_from_outputs(observed)
    hits = 0
    for _ in range(5):
        p, a = clone.getrandbits(32), victim.getrandbits(32)
        hits += p == a
        print(f"   {p:12d}  {a:12d}   {'yes' if p == a else 'NO'}")
    print()
    big = 1_000_000
    more = sum(clone.getrandbits(32) == victim.getrandbits(32) for _ in range(big))
    print(f"Next {big:,} outputs: {hits + more:,} of {big + 5:,} correct.")
    return hits + more, big + 5


if __name__ == "__main__":
    demo()
