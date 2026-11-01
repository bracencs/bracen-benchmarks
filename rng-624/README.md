# rng-624 — predicting Python's `random` from its own output

Python's `random` module is the Mersenne Twister (MT19937). Its designers said in print that it is
not for security: *"from sufficiently long subsequence of the outputs, one can predict the rest."*
This folder does exactly that — with no download, no dependencies, stock Python 3.

Everything here observes only a generator's **public 32-bit outputs**. Nothing reads its internal
state. That is the point: the state is not hidden, it is reconstructable.

## The idea

Each 32-bit output is one internal state word run through four **reversible** "tempering" steps.
Reverse the steps and one output gives you back one state word. Collect **624** in a row — the size
of the state — and you own an exact clone of the generator. Whatever it produces next, you produce
first.

## Files

| file | what it shows |
|---|---|
| `untemper.py` | the four tempering steps and their inverses; a self-test that `untemper(temper(w)) == w` |
| `crack.py` | observe 624 outputs, rebuild the state, predict the next million and check every one |
| `measure.py` | writes `../recordings/rng-624/measure.json`: recover time, accuracy, the 623-fails case, downstream (floats, dice, a card shuffle), and `secrets` as the fix |
| `challenge.py` | the 20-second version: predict the next ten before the generator makes them |

## Run it

```
python3 rng-624/crack.py        # the headline: 1,000,005 / 1,000,005 predicted
python3 rng-624/measure.py       # all the numbers, into recordings/
python3 rng-624/challenge.py     # the fast demo
```

Expected (numbers are seed-independent):

- state: **624 words, a 19,937-bit state**
- recover a full clone: **under a millisecond**
- predict the next **1,000,000** outputs: **all correct**
- with only **623** outputs: **0** predicted — you need the whole state
- the clone also predicts the next floats, the next dice rolls, and the next 52-card shuffle exactly
- point the same attack at `secrets` (the OS CSPRNG): **0** predicted — there is no fixed state to rebuild

## The fix

Use `secrets` (PEP 506, Python 3.6+) for tokens, passwords, and anything an adversary should not
predict. `random` is for simulations and games, not secrets.

## Integrity

The WIRED (2017) slot-machine case is a real-world echo, reported exactly as WIRED reported it.
WIRED never named the machines' algorithm; this demo cracks *Python's* generator and never claims the
casinos ran Python or MT19937.
