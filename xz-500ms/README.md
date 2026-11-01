# xz-500ms — reconstructing the timing symptom that caught the xz backdoor

In March 2024, Andres Freund noticed SSH logins on his test machine had gotten about half a second
slower. Chasing that one number led him to CVE-2024-3094 — a backdoor hidden in the xz/liblzma library,
days before it would have reached stable Linux distributions everywhere.

This folder **reconstructs the symptom, not the attack**. It builds a benign shared library that does
nothing but burn a bounded amount of CPU when it is loaded into a process, reproducing the login
slowdown Freund measured — with **no xz code, no backdoor, no authentication bypass, and no network**.
The real 5.6.0 / 5.6.1 tarballs are never downloaded or run.

Everything runs inside a throwaway Linux container (Debian), because the tools Freund used — `time`,
`perf`, `strace` — are real there.

## Files

| file | what |
|---|---|
| `slow_recon.c` | the benign shim: a constructor burns ~500 ms of CPU (hashing a fake in-memory table) and returns; gated on `RECON_ACTIVE`, dialled down by `RECON_QUIET` |
| `Dockerfile` | Debian + openssh-server + perf/strace/time/hyperfine; builds the shim |
| `measure.sh` | times a FAILED login with and without the shim preloaded, runs `perf`/CPU counters, and prints the numbers as JSON |
| `challenge.sh` | the safe viewer version: time a failed login a few times, and `ldd` sshd (the login server) for lzma/systemd |

## Run it (on a Linux host or container with Docker)

```
docker build -t bracen-xz-recon .
docker run --rm --privileged bracen-xz-recon bash /lab/measure.sh
```

Expected shape: a failed login near **0.3 s** clean jumps to roughly **0.8 s** with the shim loaded —
the `real` (wall-clock) time moves. `perf` puts the self-time inside the injected library. Note: this
harness `LD_PRELOAD`s the shim into the **client** process, so the client's own `user`/`sys` DO move here.
The "client-flat, server-burns" split (client `user 0.202s`/`sys 0.006s` identical on both runs) is
**Freund's own 2024 measurement**, reproduced here only as the wall-clock jump; the video attributes that
observation to him, not to this reconstruction.

Freund's own numbers, from his 2024-03-29 oss-security post, were **0.299 s → 0.807 s** (a 2.7× jump).
This harness prints its own measured pair beside them.

## Safety

`slow_recon.c` contains no part of the xz backdoor. It hooks nothing, decrypts nothing, and bypasses no
authentication. It is loaded with `LD_PRELOAD`, in the open, and only spends CPU. It exists to show how a
purely *timing* signal — a stopwatch — was enough to expose one of the closest calls in open-source
history, and is labelled a reconstruction wherever it appears on screen.
