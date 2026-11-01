#!/usr/bin/env bash
# challenge.sh — the safe, 30-second version a viewer can run on their own machine.
# It only READS: it times a failed SSH login a few times and shows what's linked into a binary.
# It sends nothing, installs nothing, and reaches no network beyond localhost.
set -euo pipefail

echo "# how long does a failed local login take? (3 runs)"
for i in 1 2 3; do
  /usr/bin/time -p ssh -o BatchMode=yes -o ConnectTimeout=5 nobody@127.0.0.1 true 2>&1 \
    | awk '/^real/{print "  run '"$i"': " $2 "s"}' || true
done

# Inspect SSHD (the login server), NOT the ssh client: the xz attack rode in through
# sshd -> libsystemd -> liblzma (Freund A9, Arch advisory), so the ssh client's ldd would be empty.
# This is Arch's own documented check. On a healthy box the grep prints nothing.
echo "# what shared libraries load into sshd, your login server? (nothing exotic should be here)"
ldd "$(command -v sshd 2>/dev/null || echo /usr/sbin/sshd)" | grep -E 'lzma|systemd' | sed 's/^/  /' \
  || echo "  (none linked — good: sshd here does not pull in lzma or systemd)"
