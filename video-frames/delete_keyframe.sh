#!/bin/bash
# delete_keyframe.sh <in.mp4> [out.mp4] — delete every keyframe except the very first one. Every other frame is left untouched.
#   ./delete_keyframe.sh media/cut_into_pan.mp4
# The noise bitstream filter drops packets without re-encoding anything: every other frame keeps its exact bytes (the MP4 container around them is rewritten).
set -euo pipefail
in="${1:?usage: delete_keyframe.sh <in.mp4> [out.mp4]}"; out="${2:-media/deleted.mp4}"
count() { ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 "$1"; }
keys()  { ffprobe -v error -select_streams v:0 -show_entries packet=flags -of csv=p=0 "$1" | grep -c K; }
echo "before:  $(count "$in") frames, $(keys "$in") keyframes"
ffmpeg -v error -y -i "$in" -c copy -bsf:v "noise=drop=key*gt(pts\,0)" "$out"
echo "after:   $(count "$out") frames, $(keys "$out") keyframe"
