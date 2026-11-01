#!/bin/bash
# count_frames.sh <video> [start end] — how many frames in a stretch of video are complete pictures?
#   ./count_frames.sh media/tears_of_steel_1080p.mov 25 35
# Reads frame types and sizes with ffprobe (decodes nothing you did not ask for), then counts.
# I = a complete picture · P = built from an earlier frame · B = built from frames on both sides.
set -euo pipefail
f="${1:?usage: count_frames.sh <video> [start end]}"; a="${2:-0}"; b="${3:-999999}"
# read 2 s PAST the end: a B-frame shown before <end> can be STORED after a frame shown later (stored order is
# not display order), so stopping the read at <end> would miss it. The awk filter below cuts on display time.
ffprobe -v error -select_streams v:0 -read_intervals "${a}%$(awk "BEGIN{print $b + 2}")" \
  -show_entries frame=pts_time,pkt_size,pict_type -of compact=p=0 "$f" |
awk -F'|' -v a="$a" -v b="$b" '
  { for (i = 1; i <= NF; i++) { split($i, kv, "="); v[kv[1]] = kv[2] }
    t = v["pts_time"] + 0; if (t < a || t >= b) next
    n++; c[v["pict_type"]]++; if (n <= 8) first = first sprintf("%s %s   ", v["pict_type"], v["pkt_size"]) }
  END {
    printf "%d frames\n\n", n
    printf "%4d  I  complete pictures\n", c["I"]
    printf "%4d  P  built from an earlier frame\n", c["P"]
    printf "%4d  B  built from frames on both sides\n\n", c["B"]
    printf "first 8 frames, in bytes:\n%s\n", first }'
