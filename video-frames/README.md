# video-frames — most of a video isn't pictures

Everything the bracen video measures, as scripts you can run on your own files. You need `ffmpeg`/`ffprobe`
(any recent build). To build the motion-vector tool you also need its headers (`brew install ffmpeg pkg-config`).

| script | what it shows |
|---|---|
| `challenge.sh <video>` | how many frames in ANY video are complete pictures (I) vs instructions (P, B) |
| `count_frames.sh <video> <start> <end>` | the same count for one stretch, plus the first frames' sizes in bytes |
| `delete_keyframe.sh <in.mp4>` | deletes every keyframe except the first, leaving every other frame untouched; play the output to watch the smear |
| `extract_mvs` (`build_c.sh`) | prints every motion vector in a file, one JSON line per frame (`--summary` for per-frame stats) |
| `measure.py` | reproduces every number in the video from the source film |
| `pixels.py` | splits 64 real pixels of one frame into brightness and colour: why a raw frame is 1.5 bytes per pixel |

The film is **Tears of Steel** (2012), (c) Blender Foundation | mango.blender.org, licensed CC-BY 3.0. It is not
included here. `fetch.sh` verifies the 1080p release file (`tears_of_steel_1080p.mov.zip`, 583,774,281 bytes) from
the Blender demo mirror and unpacks it into `media/`. The film's soundtrack is CC-BY-ND and is not used anywhere.

The expected outputs are in `../recordings/video-frames/measure.json`.

The mosh source clip (`media/cut_into_pan.mp4`) is 10 s of the film around the cut at 4:17.9, re-encoded so the
encoder's own scene-cut detection puts ONE keyframe exactly on the first frame of the new shot (see measure.py).
