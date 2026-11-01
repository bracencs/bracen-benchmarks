#!/usr/bin/env bash
# lab.sh — run one view of the reconstruction inside the throwaway Linux container.
#
#   ./lab.sh time | cpu | perf | gate | quiet | challenge
#
# Everything happens inside the container (localhost only): a benign CPU-burning library preloaded
# into sshd. No xz, no backdoor, no bypass. Build it once with:
#   docker build -t bracen-xz-recon xz-500ms
set -euo pipefail
MODE="${1:-time}"
IMG="${XZ_IMAGE:-bracen-xz-recon:latest}"
DOCKER="${DOCKER_BIN:-docker}"
exec "$DOCKER" run --rm --privileged "$IMG" bash /lab/demo.sh "$MODE"
