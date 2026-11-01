#!/usr/bin/env bash
# demo.sh — one short, capture-friendly view of the reconstruction, for filming.
#
# measure.sh produces the full JSON; this prints ONE section at a time so each clip in the film shows
# exactly one idea, in a terminal strip that stays readable. Same rig, same shim, same commands.
#
#   bash demo.sh time | cpu | perf | gate | quiet
#
# The shim is the benign CPU-burner from slow_recon.c: preloaded into SSHD (the login server), which
# re-executes per connection, so the burn is server-side and the ssh client is untouched.
# No xz, no backdoor, no bypass, localhost only.
set -u
cd /lab
SHIM=/lab/libslow-recon.so
PORT=2222
MODE="${1:-time}"

start_sshd() { pkill -f "sshd -p $PORT" 2>/dev/null; sleep 0.4; eval "$1 /usr/sbin/sshd -p $PORT -o UsePAM=no"; sleep 1; }
login() { ssh -p "$PORT" -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
              -o ConnectTimeout=5 nobody@127.0.0.1 true >/dev/null 2>&1 || true; }
med() {  # median wall-clock of N clean-client logins
  local reps="${1:-9}"; local -a t=()
  for _ in $(seq "$reps"); do local s e; s=$(date +%s.%N); login; e=$(date +%s.%N)
    t+=("$(awk -v a="$s" -v b="$e" 'BEGIN{printf "%.3f", b-a}')"); done
  printf '%s\n' "${t[@]}" | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'
}

case "$MODE" in
  time)
    echo "# the same failing login, twice. only the SERVER changes."
    start_sshd ""; echo; echo "\$ time ssh nobody@host true          # clean login server"
    echo "Permission denied (publickey)."; echo "real    0m$(med)s"
    start_sshd "RECON_ACTIVE=1 LD_PRELOAD=$SHIM"
    echo; echo "\$ time ssh nobody@host true          # one extra library in sshd"
    echo "Permission denied (publickey)."; echo "real    0m$(med)s"
    ;;
  cpu)
    echo "# processor time burned by the LOGIN SERVER on a login it refuses"
    echo
    for label in "clean sshd" "sshd + one library"; do
      env_p=""; [ "$label" != "clean sshd" ] && env_p="RECON_ACTIVE=1 LD_PRELOAD=$SHIM"
      ( eval "timeout 20 /usr/bin/time -v env $env_p /usr/sbin/sshd -d -e -p 2231 -o UsePAM=no" ) >/dev/null 2>/tmp/c.txt &
      p=$!; sleep 1
      ssh -p 2231 -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 nobody@127.0.0.1 true >/dev/null 2>&1 || true
      sleep 1; kill "$p" 2>/dev/null; wait "$p" 2>/dev/null
      printf "%-22s cpu used: %s s\n" "$label" "$(awk -F': ' '/User time/{u=$2} /System time/{s=$2} END{printf "%.2f", u+s}' /tmp/c.txt)"
    done
    echo; echo "# a rejection should be nearly free."
    ;;
  perf)
    echo "# where did the extra time go? profile the login server itself."
    echo
    echo "\$ perf record -- sshd -d      # profiling ONE login on the server"
    # no -g: call graphs make perf.data large and perf report slow, and the self-time row is all we need
    ( timeout 15 perf record -o /tmp/p.data -- env RECON_ACTIVE=1 LD_PRELOAD=$SHIM \
        /usr/sbin/sshd -d -e -p 2233 -o UsePAM=no ) >/dev/null 2>&1 &
    pf=$!; sleep 1
    ssh -p 2233 -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 nobody@127.0.0.1 true >/dev/null 2>&1 || true
    sleep 1; kill -INT "$pf" 2>/dev/null; wait "$pf" 2>/dev/null
    echo
    echo "\$ perf report"
    perf report -i /tmp/p.data --stdio 2>/dev/null | grep -E "Overhead|slow-recon|kallsyms" | head -4
    ;;
  gate)
    echo "# the SAME binary, loaded both times. only a flag differs."
    start_sshd "LD_PRELOAD=$SHIM"
    echo; echo "\$ RECON_ACTIVE unset   real    0m$(med 7)s"
    start_sshd "RECON_ACTIVE=1 LD_PRELOAD=$SHIM"
    echo "\$ RECON_ACTIVE=1       real    0m$(med 7)s"
    echo; echo "# a backdoor that sleeps until it is called is a backdoor you cannot time."
    ;;
  quiet)
    echo "# a greedier burn is easy to see. a throttled one hides in the noise."
    start_sshd ""                                              ; B=$(med 7)
    start_sshd "RECON_ACTIVE=1 RECON_QUIET=1 LD_PRELOAD=$SHIM"  ; Q=$(med 7)
    start_sshd "RECON_ACTIVE=1 LD_PRELOAD=$SHIM"                ; L=$(med 7)
    echo; printf "%-22s real  0m%ss\n" "clean baseline" "$B"
    printf "%-22s real  0m%ss\n" "throttled shim" "$Q"
    printf "%-22s real  0m%ss\n" "loud shim" "$L"
    echo; echo "# caught because it was greedy - and because a clean baseline existed."
    ;;
  challenge)
    # the viewer-facing check, run against a REAL listening sshd so the times mean something
    # (challenge.sh itself stays pure: on a viewer's own machine their sshd is already up).
    /usr/sbin/sshd -p 22 -o UsePAM=no >/dev/null 2>&1; sleep 1
    bash /lab/challenge.sh
    ;;
  *) echo "usage: demo.sh time|cpu|perf|gate|quiet|challenge" >&2; exit 2 ;;
esac
pkill -f "sshd -p $PORT" 2>/dev/null
exit 0
