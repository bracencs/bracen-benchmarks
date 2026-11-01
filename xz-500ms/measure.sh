#!/usr/bin/env bash
# measure.sh — reconstruct the xz timing symptom INSIDE the lab container and print the numbers.
#
# FAITHFUL to the attack's shape: the benign shim (slow_recon.c) is preloaded into SSHD, the login
# SERVER — not the client. OpenSSH re-executes sshd for every connection, so each login inherits the
# preload and the shim's constructor burns CPU server-side, per login. The ssh client is byte-for-byte
# unmodified across baseline and shim runs, so the extra half second lands entirely on the server —
# just as the real xz backdoor rode in through sshd -> libsystemd -> liblzma. The shim hooks nothing
# and bypasses nothing; it only spends CPU. No xz, no backdoor, no auth bypass, localhost only.
# Emits JSON on the last line for the harness to capture.
set -u   # NOT -e/pipefail: every ssh login here is expected to fail (we only time the reject path)
cd /lab
SHIM=/lab/libslow-recon.so
PORT=2222

start_sshd() {  # $1 = env prefix applied to SSHD (the server), e.g. "RECON_ACTIVE=1 LD_PRELOAD=$SHIM"
  pkill -f "sshd -p $PORT" 2>/dev/null || true; sleep 0.4
  eval "$1 /usr/sbin/sshd -p $PORT -o UsePAM=no"
  sleep 1
}

login() {  # the client is ALWAYS clean; only the server changes between runs
  ssh -p "$PORT" -o BatchMode=yes -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 nobody@127.0.0.1 true >/dev/null 2>&1 || true
}

median_login() {  # median wall-clock of N clean-client logins against the currently-running server
  local reps="${1:-15}" ; local -a t=()
  for _ in $(seq "$reps"); do
    local s e; s=$(date +%s.%N); login; e=$(date +%s.%N)
    t+=("$(awk -v a="$s" -v b="$e" 'BEGIN{printf "%.6f", b-a}')")
  done
  printf '%s\n' "${t[@]}" | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'
}

client_cpu() {  # the client's own user+sys for one login — should be identical whatever the server does
  { /usr/bin/time -v bash -c 'ssh -p '"$PORT"' -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 nobody@127.0.0.1 true || true'; } 2>&1 \
    | awk -F': ' '/User time/{u=$2} /System time/{s=$2} END{printf "%.3f", u+s}' || true
}

echo "# baseline: clean sshd" >&2
start_sshd ""
BASE=$(median_login 15)
CLIENT_BASE=$(client_cpu)

echo "# shim server: sshd carries the preload, re-execs per login -> server-side burn" >&2
start_sshd "RECON_ACTIVE=1 LD_PRELOAD=$SHIM"
SLOW=$(median_login 15)
CLIENT_SLOW=$(client_cpu)

echo "# quiet variant: a stealthier burn that narrows the gap toward noise" >&2
start_sshd "RECON_ACTIVE=1 RECON_QUIET=1 LD_PRELOAD=$SHIM"
QUIET=$(median_login 15)

echo "# gate honesty: shim present but RECON_ACTIVE unset -> server does nothing, fast again" >&2
start_sshd "LD_PRELOAD=$SHIM"
GATED_OFF=$(median_login 8)
pkill -f "sshd -p $PORT" 2>/dev/null || true

RATIO=$(awk -v s="$SLOW" -v b="$BASE" 'BEGIN{printf "%.4f", (b>0)? s/b : 0}')
DELTA=$(awk -v s="$SLOW" -v b="$BASE" 'BEGIN{printf "%.6f", s-b}')

# server-side CPU per login: sshd -d handles ONE connection in the foreground, no fork, then exits, so
# /usr/bin/time -v captures the whole server process incl. the shim's burn. Client stays clean.
server_cpu() {  # $1 = env prefix for sshd ; $2 = port. `timeout` guarantees sshd -d can't hang the run.
  local envp="$1" p="$2"
  ( eval "timeout 20 /usr/bin/time -v env $envp /usr/sbin/sshd -d -e -p $p -o UsePAM=no" ) >/dev/null 2>/tmp/scpu.txt &
  local pid=$!; sleep 1
  ssh -p "$p" -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 nobody@127.0.0.1 true >/dev/null 2>&1 || true
  sleep 1; kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null || true
  awk -F': ' '/User time/{u=$2} /System time/{s=$2} END{printf "%.3f", (u+s)}' /tmp/scpu.txt 2>/dev/null || echo "0"
}
CPU_BASE=$(server_cpu "" 2231)
CPU_SLOW=$(server_cpu "RECON_ACTIVE=1 LD_PRELOAD=$SHIM" 2232)

# perf: profile sshd (the server) handling one connection with the shim; the self-time lands in the
# injected library. `timeout` bounds it so a non-exiting sshd -d cannot stall the measurement.
PERF_TOP="n/a"
if command -v perf >/dev/null 2>&1; then
  ( timeout 20 perf record -o /tmp/perf.data -g -- env RECON_ACTIVE=1 LD_PRELOAD=$SHIM /usr/sbin/sshd -d -e -p 2233 -o UsePAM=no ) >/dev/null 2>&1 &
  pf=$!; sleep 1
  ssh -p 2233 -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 nobody@127.0.0.1 true >/dev/null 2>&1 || true
  sleep 1; kill -INT "$pf" 2>/dev/null; wait "$pf" 2>/dev/null || true
  [ -f /tmp/perf.data ] && PERF_TOP=$(perf report -i /tmp/perf.data --stdio 2>/dev/null | grep -m1 -i "slow-recon\|recon_startup\|unknown" | sed 's/^ *//' | cut -c1-80 || echo "n/a")
fi
pkill -f "sshd -" 2>/dev/null || true

python3 - "$BASE" "$SLOW" "$QUIET" "$RATIO" "$DELTA" "$CPU_BASE" "$CPU_SLOW" "$GATED_OFF" "$PERF_TOP" "$CLIENT_BASE" "$CLIENT_SLOW" <<'PY'
import json,sys,platform,subprocess
b,s,q,r,d,cb,cs,g,perf,clb,cls = sys.argv[1:12]
def f(x):
    try: return round(float(x),3)
    except: return None
kern = subprocess.run(["uname","-r"],capture_output=True,text=True).stdout.strip()
data={
 "baseline_s":f(b),"shim_s":f(s),"quiet_s":f(q),"ratio":round(float(r),2),"delta_ms":round(float(d)*1000),
 "cpu_baseline_s":f(cb),"cpu_shim_s":f(cs),"gated_off_s":f(g),
 "client_cpu_baseline_s":f(clb),"client_cpu_shim_s":f(cls),
 "preload_target":"sshd (server); client unmodified",
 "perf_top_line":perf.strip(),
 "kernel":kern,"arch":platform.machine(),
 "freund":{"baseline_s":0.299,"slow_s":0.807,"source":"oss-security 2024-03-29"},
 "commands":{"time":"ssh -o BatchMode=yes nobody@127.0.0.1 true (clean client, median x15)",
             "server":"LD_PRELOAD=shim sshd (re-execs per login -> server-side burn)",
             "perf":"perf record -g -- env LD_PRELOAD=shim sshd -d; one login",
             "cpu":"/usr/bin/time -v env LD_PRELOAD=shim sshd -d; one login"},
}
print("MEASURE_JSON " + json.dumps(data))
PY
