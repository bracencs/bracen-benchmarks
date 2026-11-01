#!/usr/bin/env python3
"""untemper.py — invert MT19937's output tempering.

Python's `random` module is the Mersenne Twister (MT19937). Each 32-bit output is one internal
state word run through four reversible "tempering" steps. Because every step is reversible, an
observer who sees the 32-bit output can run the steps backwards and recover the exact state word.
Collect 624 in a row and you have rebuilt the whole generator — see crack.py.

The four tempering steps, from CPython's _randommodule.c genrand_uint32 (constants are MT19937's):
    y ^= y >> 11
    y ^= (y << 7)  & 0x9D2C5680
    y ^= (y << 15) & 0xEFC60000
    y ^= y >> 18
This file inverts each one. It is deliberately short: it is shown on screen.
"""

U, S, T, L = 11, 7, 15, 18
B, C = 0x9D2C5680, 0xEFC60000
M32 = 0xFFFFFFFF


def _undo_rshift_xor(y, shift):
    """Invert  y = x ^ (x >> shift).  The top `shift` bits of x survive untouched, so we peel the
    word `shift` bits at a time from the most significant end down."""
    x = 0
    for i in range(0, 32, shift):
        # bits [32-shift-i .. 32-i) of x are now known; use them to reveal the next lower chunk
        x = y ^ (x >> shift)
    return x & M32


def _undo_lshift_xor(y, shift, mask):
    """Invert  y = x ^ ((x << shift) & mask).  Symmetric: the low `shift` bits of x survive, so we
    peel from the least significant end up."""
    x = 0
    for i in range(0, 32, shift):
        x = y ^ ((x << shift) & mask)
    return x & M32


def untemper(y):
    """A single 32-bit MT19937 output -> the state word it came from."""
    y = _undo_rshift_xor(y, L)
    y = _undo_lshift_xor(y, T, C)
    y = _undo_lshift_xor(y, S, B)
    y = _undo_rshift_xor(y, U)
    return y & M32


def temper(x):
    """The forward tempering, for the round-trip self-test only."""
    x ^= x >> U
    x ^= (x << S) & B
    x ^= (x << T) & C
    x ^= x >> L
    return x & M32


if __name__ == "__main__":
    import random
    r = random.Random(0)
    n = 200_000
    ok = sum(1 for _ in range(n) if untemper(temper(w := r.getrandbits(32))) == w)
    print(f"untemper(temper(w)) == w for {ok:,} of {n:,} random words")
    assert ok == n
