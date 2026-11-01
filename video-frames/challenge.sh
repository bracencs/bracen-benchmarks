#!/bin/bash
# challenge.sh <any video you own> — how many of its frames are complete pictures?
# The one line below is the whole challenge (it is also what the video shows on screen):
# COMMAND: ffprobe -v error -select_streams v:0 -show_entries frame=pict_type -of default=nw=1:nk=1 your_video.mp4 | sort | uniq -c
# (default=nw=1:nk=1, not csv: the csv writer prints the first frame as "I," — a trailing comma from its side
#  data — and `uniq` then counts it as a separate kind of frame.)
set -euo pipefail
f="${1:?usage: challenge.sh <video>}"
ffprobe -v error -select_streams v:0 -show_entries frame=pict_type -of default=nw=1:nk=1 "$f" | sort | uniq -c
