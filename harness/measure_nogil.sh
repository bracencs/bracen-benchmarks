#!/usr/bin/env bash
# The free-threading measurement, on a QUIET machine, twice.
#
# Three interpreters, the same file, the same threads:
#   1. the ordinary 3.14 build          — "normal Python", the GIL is there
#   2. the free-threading build, -X gil=1 — the SAME binary with the lock put back (the control)
#   3. the free-threading build           — the lock gone (the payoff)
# Leg 2 exists because without it you cannot tell a free-threading WIN from a free-threading BUILD
# difference. It is the one-variable comparison; legs 1 vs 2 price the build itself.
#
# Waits for quiet first: the gil numbers are throughput, and a busy machine silently depresses them.
set -euo pipefail
cd "$(dirname "$0")/.."                       # benchmarks/
FT="$HOME/.local/share/uv/python/cpython-3.14.2+freethreaded-macos-aarch64-none/bin/python3.14t"
# The ORDINARY build, deliberately the SAME PATCH VERSION as the free-threading one. Homebrew's
# 3.14.4 was used first, which made the one-thread price claim span two patch releases as well as
# two builds — a second variable in the one comparison that has only one thing to prove. uv's
# 3.14.2 removes it. Falls back to Homebrew if the matched build is not installed.
STD="$HOME/.local/share/uv/python/cpython-3.14.2-macos-aarch64-none/bin/python3.14"
[ -x "$STD" ] || STD="/opt/homebrew/bin/python3.14"
[ -x "$FT" ] || { echo "✋ no free-threading interpreter at $FT"; exit 1; }
WORKERS="${WORKERS:-1,2,4,6,8}"
RUNS="${RUNS:-2}"

wait_quiet() {
  # THE GATE IS IDLE CPU, NOT LOAD AVERAGE. On macOS the load average counts threads blocked in the
  # kernel as well as running ones, so this Mac sits at load 6-8 while 66% of the CPU is idle. Gating
  # on load never opens; gating on it loosely lets a genuinely busy machine through. What these rows
  # measure is throughput across 8 threads, so the quantity that matters is how many cores are free.
  #   idle >= 88%  ->  under one core busy  ->  8 threads have room
  # Every run records its own load average too, so a compromised run is visible in the data.
  local tries=0 idle
  while [ $tries -lt 360 ]; do
    idle=$(top -l 2 -n 0 -s 1 2>/dev/null | grep -E '^CPU usage' | tail -1 \
           | sed -E 's/.*, ([0-9.]+)% idle.*/\1/')
    if [ -n "$idle" ] && [ "$(echo "$idle >= ${QUIET_IDLE:-88}" | bc -l)" = "1" ]; then
      echo "  machine quiet (${idle}% idle)"; return 0
    fi
    [ $((tries % 10)) -eq 0 ] && echo "  waiting for quiet (${idle:-?}% idle, need ${QUIET_IDLE:-88}%)…"
    tries=$((tries+1))
  done
  echo "⚠ never went quiet in ~30 min — measuring anyway; the recorded load_avg shows it"; return 0
}

for r in $(seq 1 "$RUNS"); do
  echo "── run $r of $RUNS ─────────────────────────────────────────────"
  wait_quiet
  echo "[standard build]";            "$STD" damage/nogil.py bench --workers "$WORKERS" --out "nogil.standard.run$r.json"
  echo; echo "[free-threading, GIL back on]"; "$FT" -X gil=1 damage/nogil.py bench --workers "$WORKERS" --out "nogil.gilback.run$r.json"
  echo; echo "[free-threading, GIL off]";     "$FT"           damage/nogil.py bench --workers "$WORKERS" --out "nogil.freethreaded.run$r.json"
  echo
done
echo "✓ recordings/gil/nogil.*.json written"
