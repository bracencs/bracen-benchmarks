#!/bin/bash
# Verify + unpack the owner-approved source film (2026-09-24): Tears of Steel 1080p, CC-BY 3.0,
# (c) Blender Foundation | mango.blender.org. Downloaded once from the Blender demo mirror
# https://ftp.nluug.nl/pub/graphics/blender/demo/movies/ToS/tears_of_steel_1080p.mov.zip — this script never
# downloads; it refuses unless the zip is exactly the approved size. media/ is gitignored.
set -euo pipefail
cd "$(dirname "$0")/media"
ZIP=tears_of_steel_1080p.mov.zip; WANT=583774281
[ -f "$ZIP" ] || { echo "missing $ZIP — download is owner-gated (see header)"; exit 1; }
GOT=$(stat -f %z "$ZIP"); [ "$GOT" = "$WANT" ] || { echo "size $GOT != approved $WANT — refusing"; exit 1; }
shasum -a 256 "$ZIP" | tee "$ZIP.sha256"
[ -f tears_of_steel_1080p.mov ] || unzip -o -q "$ZIP"
ls -l *.mov
