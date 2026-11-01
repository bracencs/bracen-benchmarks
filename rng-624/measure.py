#!/usr/bin/env python3
"""measure.py — every number the video uses, produced from the real generator.

Writes recordings/rng-624/measure.json. Run from the repo root or from here:
    python3 rng-624/measure.py

Nothing here reads a generator's internal state. Every prediction is built only from observed
32-bit outputs, then checked against what the generator actually did next.
"""
import json, os, sys, time, platform, secrets, statistics
import random as random_mod

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from untemper import untemper
from crack import clone_from_outputs, N_STATE

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "recordings", "rng-624", "measure.json")


def recover_time(runs=200):
    """Median wall-clock to turn 624 observed outputs into a working clone."""
    ts = []
    for _ in range(runs):
        obs = [random_mod.Random().getrandbits(32) for _ in range(N_STATE)]
        t0 = time.perf_counter()
        clone_from_outputs(obs)
        ts.append((time.perf_counter() - t0) * 1000.0)  # ms
    return {"median_ms": statistics.median(ts), "min_ms": min(ts), "runs": runs}


def accuracy(big=1_000_000):
    v = random_mod.Random()
    obs = [v.getrandbits(32) for _ in range(N_STATE)]
    c = clone_from_outputs(obs)
    hits = sum(c.getrandbits(32) == v.getrandbits(32) for _ in range(big))
    return {"observed": N_STATE, "predicted_count": hits, "checked": big, "accuracy": hits / big}


def not_enough(short=N_STATE - 1, checknext=1000, trials=20):
    """One output short of a full state: how much can you still predict? (Spoiler: nothing.)"""
    worst = []
    for _ in range(trials):
        v = random_mod.Random()
        obs = [v.getrandbits(32) for _ in range(short)]
        # fill the one missing word with a guess (0); index still says "block consumed"
        state = tuple(untemper(o) for o in obs) + (0,) * (N_STATE - short)
        c = random_mod.Random()
        c.setstate((3, state + (N_STATE,), None))
        hits = sum(c.getrandbits(32) == v.getrandbits(32) for _ in range(checknext))
        worst.append(hits)
    return {"observed": short, "checked_each": checknext, "trials": trials,
            "max_predicted": max(worst), "mean_predicted": sum(worst) / len(worst)}


def downstream():
    """After cloning from 624 raw outputs, the clone predicts whatever the generator is asked for
    next — because it is the same generator, one step behind us."""
    v = random_mod.Random()
    obs = [v.getrandbits(32) for _ in range(N_STATE)]
    c = clone_from_outputs(obs)

    floats = [(round(c.random(), 12), round(v.random(), 12)) for _ in range(10)]
    floats_hits = sum(p == a for p, a in floats)

    dice = [(c.randint(1, 6), v.randint(1, 6)) for _ in range(20)]
    dice_hits = sum(p == a for p, a in dice)

    deck = list(range(52))
    cd, vd = deck[:], deck[:]
    c.shuffle(cd); v.shuffle(vd)
    shuffle_exact = cd == vd

    return {
        "floats": {"checked": len(floats), "hits": floats_hits, "sample": floats[:3]},
        "dice": {"checked": len(dice), "hits": dice_hits, "predicted": [p for p, _ in dice], "actual": [a for _, a in dice]},
        "shuffle": {"cards": 52, "exact": shuffle_exact, "predicted_top5": cd[:5], "actual_top5": vd[:5]},
    }


def the_fix(checknext=1000):
    """secrets uses the operating system's CSPRNG, not MT19937. The same attack sees nothing to
    clone: every draw is fresh OS entropy, so the predictor is right only by 1-in-4-billion chance."""
    # Observe 624 outputs from secrets' generator, try the identical clone attack, predict next.
    sysrand = secrets.SystemRandom()
    obs = [sysrand.getrandbits(32) for _ in range(N_STATE)]
    c = clone_from_outputs(obs)  # builds an MT clone from the observations...
    hits = sum(c.getrandbits(32) == sysrand.getrandbits(32) for _ in range(checknext))
    return {"observed": N_STATE, "checked": checknext, "predicted": hits,
            "note": "SystemRandom draws from os.urandom; there is no fixed state to rebuild"}


def state_math():
    """The numbers behind '624'."""
    return {
        "state_words": N_STATE,
        "word_bits": 32,
        "naive_bits": N_STATE * 32,
        "state_bits": N_STATE * 32 - 31,   # r = 31 bits discarded => 19937
        "period_exponent": 19937,
        "temper_steps": 4,
    }


def main():
    random_mod.seed(secrets.randbits(64))  # so a rerun differs; results are seed-independent
    data = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "generator": "random.Random (MT19937), default os.urandom seeding",
        "state": state_math(),
        "recover_time": recover_time(),
        "accuracy": accuracy(),
        "not_enough": not_enough(),
        "downstream": downstream(),
        "the_fix": the_fix(),
        "roundtrip_selftest": None,
        "commands": {
            "demo": "python3 crack.py",
            "count_type": "python3 measure.py",
            "challenge": "python3 challenge.py",
        },
    }
    # untemper self-test count for the results file
    rr = random_mod.Random(0)
    from untemper import temper
    data["roundtrip_selftest"] = sum(1 for _ in range(200_000) if untemper(temper(w := rr.getrandbits(32))) == w)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(data, f, indent=2)
    # capture-friendly summary
    print(f"python {data['python_version']}")
    print(f"state: {data['state']['state_words']} words, {data['state']['state_bits']:,}-bit state")
    print(f"recover: {data['recover_time']['median_ms']:.3f} ms (median of {data['recover_time']['runs']})")
    a = data["accuracy"]
    print(f"accuracy: {a['predicted_count']:,} of {a['checked']:,}")
    ne = data["not_enough"]
    print(f"623 outputs: at most {ne['max_predicted']} of {ne['checked_each']:,} predicted")
    d = data["downstream"]
    print(f"floats {d['floats']['hits']}/{d['floats']['checked']}  dice {d['dice']['hits']}/{d['dice']['checked']}  shuffle exact: {d['shuffle']['exact']}")
    fx = data["the_fix"]
    print(f"secrets: {fx['predicted']} of {fx['checked']:,} predicted")
    print(f"wrote {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
