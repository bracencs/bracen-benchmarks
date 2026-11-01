#!/bin/bash
# Build the motion-vector extractor against Homebrew's FFmpeg. NOTE: `cc` is a zsh alias on this Mac (not the
# compiler) — always call /usr/bin/clang explicitly.
set -euo pipefail
cd "$(dirname "$0")"
/usr/bin/clang -O2 -Wall extract_mvs.c $(pkg-config --cflags --libs libavformat libavcodec libavutil) -o extract_mvs
echo "built $(pwd)/extract_mvs"
