#!/usr/bin/env python3
"""
measure.py — every number and clip the `video-frames` long uses, measured from the real file.

Source: Tears of Steel (2012), 1080p release file, CC-BY 3.0 (c) Blender Foundation | mango.blender.org
(verified + unpacked by fetch.sh). The film's AUDIO is never used — the soundtrack is CC-BY-ND.

Outputs (all relative to the repo root):
  recordings/video-frames/measure.json   — raw measurements + the exact commands that produced them
  public/video-frames/*.mp4, *.png                  — muted clips and stills the film shows
Run from anywhere:  /usr/bin/python3 video-frames/measure.py

Shot choice is itself measured (shots.json, from motion_summary.csv + scene_cuts.txt): the pan is the
longest run of coherent horizontal motion, the quiet shot the stillest dialogue, the cut for the keyframe
deletion the one that enters that pan. Nothing here is typed from memory.
"""
import json, shutil, subprocess, os, statistics as st, math
from fractions import Fraction

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
SRC = "video-frames/media/tears_of_steel_1080p.mov"
REC = "recordings/video-frames"
PUB = "public/video-frames"
MVS = "video-frames/extract_mvs"
FF, FP = shutil.which("ffmpeg"), shutil.which("ffprobe")
os.makedirs(PUB, exist_ok=True)

# ---- the shots (seconds in the release file), chosen by the measurements in shots.json -------------------
SHOTS = {
    "quiet":     (25.0, 35.0),    # bridge dialogue, 10 s = 240 frames; 5.5 KB/frame in the phone-style film encode (stillest dialogue)
    "pan":       (272.0, 280.0),  # the strongest stretch of the 257.9-289.5 tracking shot: 99% of blocks moving, one direction
    "action":    (470.0, 475.0),  # robot battle in the cathedral: among the heaviest shots in the phone-style film encode
    "particles": (216.0, 221.0),  # holographic dome of glittering particles: the "snow/confetti" case, also among the heaviest
}
CUT = 257.917                   # scene cut (scene_cuts.txt) from the street scene into the pan
MOSH = (253.5, 263.5)           # 4.4 s before the cut, 5.6 s after

# The "phone-style" analysis encode: constant QUALITY (CRF), one keyframe every 2 s, B-frames on, no extra
# keyframes at scene cuts. Documented verbatim in measure.json and on screen.
PHONE = ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-g", "48", "-keyint_min", "48", "-sc_threshold", "0",
         "-bf", "2", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart"]

def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout

def probe_frames(path):
    out = json.loads(run([FP, "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "frame=pict_type,pkt_size,pts_time,pkt_dts_time,key_frame", "-of", "json", path]))
    return [dict(t=f["pict_type"], b=int(f["pkt_size"]), pts=float(f["pts_time"]),
                 dts=float(f.get("pkt_dts_time", f["pts_time"])), key=int(f["key_frame"])) for f in out["frames"]]

def packets(path):
    """Packets in STORED (decode) order — this is the order the bytes sit in the file."""
    out = json.loads(run([FP, "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "packet=pts_time,dts_time,size,flags", "-of", "json", path]))
    return [dict(pts=float(p["pts_time"]), dts=float(p["dts_time"]), b=int(p["size"]), key="K" in p["flags"]) for p in out["packets"]]

def type_stats(frames):
    out = {}
    for t in "IPB":
        s = [f["b"] for f in frames if f["t"] == t]
        out[t] = dict(n=len(s), mean=round(st.mean(s)) if s else 0, median=round(st.median(s)) if s else 0,
                      max=max(s) if s else 0, min=min(s) if s else 0)
    return out

M = {"source": {"file": os.path.basename(SRC), "license": "CC-BY 3.0",
                "credit": "Tears of Steel (2012) (c) Blender Foundation | mango.blender.org",
                "url": "https://mango.blender.org", "mirror": "https://ftp.nluug.nl/pub/graphics/blender/demo/movies/ToS/"},
     "commands": {}, "phone_encode": " ".join(PHONE)}

# ---- 1. the whole release file (already probed: original_frames.json) ------------------------------------
orig = json.load(open(f"{REC}/original_frames.json"))["frames"]
of = [dict(t=f["pict_type"], b=int(f["pkt_size"]), pts=float(f["pts_time"])) for f in orig]
meta = json.loads(run([FP, "-v", "error", "-select_streams", "v:0", "-show_entries",
                       "stream=width,height,avg_frame_rate,bit_rate,nb_frames,codec_name,profile:format=duration", "-of", "json", SRC]))
s0 = meta["streams"][0]; W, H = s0["width"], s0["height"]; fps = float(Fraction(s0["avg_frame_rate"]))   # "24/1" → 24.0 (parsed, never evaluated)
raw_frame = W * H * 3 // 2                           # 8-bit 4:2:0: one luma byte per pixel + two quarter-size chroma planes
raw_bps = raw_frame * fps * 8
file_bps = int(s0["bit_rate"])
ikeys = [i for i, f in enumerate(of) if f["t"] == "I"]
gaps = [b - a for a, b in zip(ikeys, ikeys[1:])]
M["release"] = dict(codec=s0["codec_name"], profile=s0["profile"], width=W, height=H, fps=fps, frames=len(of),
                    duration=round(float(meta["format"]["duration"]), 2), video_bitrate=file_bps,
                    types=type_stats(of), i_share=round(len(ikeys) / len(of), 4),
                    gop_median=st.median(gaps), gop_max=max(gaps), raw_frame_bytes=raw_frame, raw_bps=raw_bps,
                    compression_ratio=round(raw_bps / file_bps, 1))
# the release file's own encoder settings, from the x264 SEI string it carries (a primary source inside the file)
import re as _re
with open(SRC, "rb") as fh: blob = fh.read(20_000_000)
sei = _re.search(rb"x264 - core \d+[^\x00]*", blob).group(0).decode("latin-1")
opts = dict(_re.findall(r"(\w+)=(\S+)", sei))
M["release"]["x264"] = {k: opts.get(k) for k in ("rc", "bitrate", "vbv_maxrate", "vbv_bufsize", "keyint", "keyint_min",
                                                  "scenecut", "bframes", "pb_ratio", "ref", "constrained_intra")}
M["release"]["x264_core"] = sei.split(" - ")[1].split()[1]
M["commands"]["release_frames"] = f"ffprobe -v error -select_streams v:0 -show_entries frame=pict_type,pkt_size,pts_time -of json {os.path.basename(SRC)}"

# release-file frames inside the quiet shot (the cold open counts THESE — a real release file, not our encode)
a, b = SHOTS["quiet"]
q = [f for f in of if a <= f["pts"] < b]
M["release_quiet"] = dict(start=a, end=b, frames=len(q), types={t: sum(1 for f in q if f["t"] == t) for t in "IPB"},
                          sizes_first12=[(f["t"], f["b"]) for f in q[:12]])

# ---- 2. the phone-style clips ----------------------------------------------------------------------------
clips = {}
for name, (a, b) in SHOTS.items():
    out = f"{PUB}/shot_{name}.mp4"
    run([FF, "-v", "error", "-y", "-ss", str(a), "-to", str(b), "-i", SRC, *PHONE, out])
    fr = probe_frames(out)
    clips[name] = dict(file=out, start=a, end=b, frames=len(fr), types=type_stats(fr),
                       bytes_per_frame=round(sum(f["b"] for f in fr) / len(fr)), sizes=[f["b"] for f in fr],
                       kinds=[f["t"] for f in fr])
M["clips"] = clips
M["commands"]["phone_clip"] = f"ffmpeg -ss <start> -to <end> -i {os.path.basename(SRC)} " + " ".join(PHONE) + " shot.mp4"

# stored order vs display order, from the quiet clip's packets (first 12)
pk = packets(f"{PUB}/shot_quiet.mp4")[:12]
disp = sorted(pk, key=lambda p: p["pts"])
M["decode_order"] = dict(stored=[dict(pts=round(p["pts"], 4), dts=round(p["dts"], 4), b=p["b"], key=p["key"]) for p in pk],
                         display_rank=[disp.index(p) for p in pk])

# ---- 3. motion vectors on the pan (one representative P-frame), + the dominant direction over the clip ---
lines = run([MVS, f"{PUB}/shot_pan.mp4"]).splitlines()
frames = [json.loads(l) for l in lines]
def disp_of(mv): return (mv[2] - mv[0], mv[3] - mv[1])   # displacement = dst - src (content motion, px/frame-ref)
dom = []
for fr in frames:
    past = [mv for mv in fr["mvs"] if mv[6] == -1]
    if len(past) > 50: dom.append(st.median(disp_of(mv)[0] for mv in past))
pick = max((fr for fr in frames if fr["type"] == "P"), key=lambda fr: sum(1 for mv in fr["mvs"] if mv[6] == -1 and disp_of(mv) != (0, 0)))
pidx = pick["n"] - 1                                        # 0-based display index in the clip
run([FF, "-v", "error", "-y", "-i", f"{PUB}/shot_pan.mp4", "-vf", f"select=eq(n\\,{pidx})", "-frames:v", "1", f"{PUB}/pan_frame.png"])
run([FF, "-v", "error", "-y", "-flags2", "+export_mvs", "-i", f"{PUB}/shot_pan.mp4", "-vf",
     f"select=eq(n\\,{pidx}),codecview=mv=pf+bf+bb", "-frames:v", "1", f"{PUB}/pan_codecview.png"])
past = [mv for mv in pick["mvs"] if mv[6] == -1]
M["pan_mvs"] = dict(frame_index=pidx, frame_type=pick["type"], vectors=len(pick["mvs"]), past_vectors=len(past),
                    moving=sum(1 for mv in past if disp_of(mv) != (0, 0)),
                    median_dx=st.median(disp_of(mv)[0] for mv in past), median_dy=st.median(disp_of(mv)[1] for mv in past),
                    clip_median_dx=st.median(dom) if dom else 0, convention="displacement = dst - src: where the block IS now minus where it was copied FROM",
                    mvs=[mv for mv in past if disp_of(mv) != (0, 0)])
# how far back is "the frame it copies from"? the previous anchor (I or P) in display order
kinds = clips["pan"]["kinds"]
prev_anchor = max(i for i in range(pidx) if kinds[i] in "IP")
M["pan_mvs"]["ref_gap_frames"] = pidx - prev_anchor
# the film's arrow field: the real vectors aggregated onto a coarse grid (median displacement per cell, block count)
CELL = 96
cells = {}
for mv in past:
    cells.setdefault((mv[2] // CELL, mv[3] // CELL), []).append(disp_of(mv))
M["pan_mvs"]["grid"] = dict(cell=CELL, cols=1920 // CELL, rows=-(-800 // CELL), cells=[
    dict(c=c, r=r, dx=st.median(v[0] for v in vs), dy=st.median(v[1] for v in vs), n=len(vs)) for (c, r), vs in sorted(cells.items())])
# the tracked subject vs the background — found FROM THE DATA, not from a hand-picked column range (review
# 2026-09-24: columns 5-7 pooled background cells and reported 24 px for the man; his own cells say ~18).
# A cell is "subject" when its median displacement is under half the frame's overall median: the camera follows it.
all_dx = st.median(disp_of(mv)[0] for mv in past)
cell_dx = {k: st.median(v[0] for v in vs) for k, vs in cells.items() if len(vs) >= 4}
subj = {k for k, dx in cell_dx.items() if abs(dx) < 0.5 * abs(all_dx)}
bg = [v[0] for k, vs in cells.items() if k not in subj for v in vs]
fg = [v[0] for k, vs in cells.items() if k in subj for v in vs]
M["pan_mvs"]["background_median_dx"] = st.median(bg); M["pan_mvs"]["subject_median_dx"] = st.median(fg)
M["pan_mvs"]["subject_cells"] = sorted([list(k) for k in subj]); M["pan_mvs"]["subject_blocks"] = len(fg)
M["commands"]["mvs"] = "extract_mvs shot_pan.mp4   (FFmpeg doc/examples/extract_mvs.c, flags2 +export_mvs)"

# ---- 4. delete one keyframe: the cut into the pan --------------------------------------------------------
a, b = MOSH
cut_rel = round(CUT - a, 3)
mosh_src = f"{PUB}/mosh_source.mp4"
# The encoder's own scene-cut detection places the keyframe EXACTLY on the first frame of the new shot. A forced
# keyframe (-force_key_frames at the cut time) landed one frame late: x264 had already painted the new shot into the
# P-frames before it with intra blocks (75 KB and 83 KB where a normal P-frame is ~30 KB), so deleting that key did
# nothing visible. That is itself true and worth knowing — a P-frame CAN carry picture — and it is kept in the data.
run([FF, "-v", "error", "-y", "-ss", str(a), "-to", str(b), "-i", SRC, "-c:v", "libx264", "-preset", "medium", "-crf", "20",
     "-g", "9999", "-keyint_min", "1", "-sc_threshold", "40", "-bf", "0", "-pix_fmt", "yuv420p", "-an", mosh_src])
import shutil; shutil.copyfile(mosh_src, "video-frames/media/cut_into_pan.mp4")  # what delete_keyframe.sh is filmed on
mf = probe_frames(mosh_src)
keys = [round(f["pts"], 3) for f in mf if f["t"] == "I"]
broken = f"{REC}/mosh_broken.mp4"
run([FF, "-v", "error", "-y", "-i", mosh_src, "-c", "copy", "-bsf:v", "noise=drop=key*gt(pts\\,0)", broken])
# decode the broken stream with FFmpeg and RE-ENCODE it clean, so the browser that renders the film never
# decodes a broken stream itself (Chrome conceals errors differently) — what is shown is what FFmpeg produced
run([FF, "-v", "error", "-y", "-i", broken, "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-an",
     f"{PUB}/mosh_result.mp4"])
bf = probe_frames(broken)
kcut = [round(f["pts"], 3) for f in mf if f["t"] == "I" and f["pts"] > 0.1]
cut_rel = kcut[0] if kcut else cut_rel        # the encoder's keyframe IS the cut
for k, t in [("before", cut_rel - 0.5), ("after05", cut_rel + 0.5), ("after2", cut_rel + 2.0), ("after4", cut_rel + 4.0)]:
    run([FF, "-v", "error", "-y", "-ss", str(t), "-i", f"{PUB}/mosh_result.mp4", "-frames:v", "1", f"{PUB}/mosh_{k}.png"])
    run([FF, "-v", "error", "-y", "-ss", str(t), "-i", mosh_src, "-frames:v", "1", f"{PUB}/clean_{k}.png"])
pk_src, pk_broken = packets(mosh_src), packets(broken)
# what FFmpeg's decoder actually DOES with the broken stream (review 2026-09-24): it does not blindly follow every
# instruction — it DISCARDS the P-frames that point straight at the missing keyframe, then carries on from the
# last picture it has. Count them, with the display-time gap they leave.
dec_pts = sorted(f["pts"] for f in bf)
gap_after_cut = max((b2 - a2, a2, b2) for a2, b2 in zip(dec_pts, dec_pts[1:]))
M["mosh"] = dict(packets_before=len(pk_src), packets_after=len(pk_broken), packets_removed=len(pk_src) - len(pk_broken),
                 decoder_output_frames=len(bf), decoder_discarded=len(pk_broken) - len(bf),
                 gap_seconds=round(gap_after_cut[0], 3), gap_from=round(gap_after_cut[1], 3), gap_to=round(gap_after_cut[2], 3), source_start=a, source_end=b, cut_at=cut_rel, keyframes_before=keys, frames=len(mf),
                 frames_after_drop=len(bf), keyframes_after=[round(f["pts"], 3) for f in bf if f["t"] == "I"],
                 dropped=len(mf) - len(bf), cut_frame_bytes=next((f["b"] for f in mf if f["t"] == "I" and f["pts"] > 0.1), None))
M["commands"]["mosh"] = "ffmpeg -i mosh_source.mp4 -c copy -bsf:v \"noise=drop=key*gt(pts\\,0)\" broken.mp4"

# the FIRST attempt, kept because it is true and on screen: a keyframe FORCED at the cut time. x264 placed it one
# frame late, so the new shot's first frame was already built inside a P-frame from blocks that copy nothing.
forced = f"{REC}/forced_key.mp4"
run([FF, "-v", "error", "-y", "-ss", str(a), "-to", str(b), "-i", SRC, "-c:v", "libx264", "-preset", "medium", "-crf", "20",
     "-g", "9999", "-keyint_min", "1", "-sc_threshold", "0", "-force_key_frames", str(cut_rel), "-bf", "0",
     "-pix_fmt", "yuv420p", "-an", forced])
ff = probe_frames(forced)
cut_i = min(range(len(ff)), key=lambda i: abs(ff[i]["pts"] - cut_rel))          # first frame of the new shot
near = [f["b"] for i, f in enumerate(ff) if f["t"] == "P" and i != cut_i and abs(f["pts"] - cut_rel) <= 1.0]
M["forced_key"] = dict(cut_at=cut_rel, cut_frame_type=ff[cut_i]["t"], cut_frame_bytes=ff[cut_i]["b"],
                       key_at=round(next(f["pts"] for f in ff if f["t"] == "I" and f["pts"] > 0.1), 3),
                       neighbour_p_median=round(st.median(near)), neighbour_window_s=1.0)
# ...and what that first try LOOKED like: the same one-filter deletion on it, half a second after the cut. The new shot
# is already there (its first frame was built from scratch inside a P-frame), so there is no smear to see.
run([FF, "-v", "error", "-y", "-i", forced, "-c", "copy", "-bsf:v", "noise=drop=key*gt(pts\\,0)", f"{REC}/forced_key_broken.mp4"])
run([FF, "-v", "error", "-y", "-ss", str(cut_rel + 0.5), "-i", f"{REC}/forced_key_broken.mp4", "-frames:v", "1", f"{PUB}/first_try_after05.png"])

# ---- 4b. what a P-frame's bytes are spent on: x264's own per-frame accounting ------------------------------
# The stats file x264 writes for a first pass, on the pan with the phone-style settings (slow-firstpass keeps the
# analysis identical; the packet sizes are checked equal to shot_pan.mp4). x264 splits each frame's bits into tex
# (transform coefficients: the corrections, and the pixels of blocks built from scratch), mv (motion vectors AND
# macroblock headers/modes) and misc.
import re
a, b = SHOTS["pan"]
statf = f"{REC}/pan_x264.stats"
run([FF, "-v", "error", "-y", "-ss", str(a), "-to", str(b), "-i", SRC, *PHONE, "-x264-params",
     f"stats={statf}:pass=1:slow-firstpass=1", f"{REC}/pan_stats.mp4"])
same = [p["b"] for p in packets(f"{REC}/pan_stats.mp4")] == [p["b"] for p in packets(f"{PUB}/shot_pan.mp4")]
bits = {}
for l in open(statf):
    if not l.startswith("in:"): continue
    d = dict(re.findall(r"(\w+):(\S+)", l)); t = d["type"].upper()
    agg = bits.setdefault(t, dict(n=0, tex=0, mv=0, misc=0)); agg["n"] += 1
    for k in ("tex", "mv", "misc"): agg[k] += int(d[k])
for t, agg in bits.items():
    tot = agg["tex"] + agg["mv"] + agg["misc"]
    agg.update(tex_share=round(agg["tex"] / tot, 3), mv_share=round(agg["mv"] / tot, 3))
M["pan_bits"] = dict(by_type=bits, identical_to_shot_pan=same, note="x264 pass-1 stats; mv bucket includes macroblock headers")
M["commands"]["pan_bits"] = "ffmpeg ... (phone-style settings) -x264-params stats=pan_x264.stats:pass=1:slow-firstpass=1"

# ---- 5. the whole-film landscape, phone-style (constant quality) vs the release file (near-constant rate) -
land = f"{REC}/film_phone.mp4"
if not os.path.exists(land):
    run([FF, "-v", "error", "-y", "-i", SRC, "-t", "640", "-vf", "scale=960:-2", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "20", "-g", "48", "-keyint_min", "48", "-sc_threshold", "0", "-bf", "2", "-an", land])
lf = probe_frames(land)
# PER SECOND, relative to each file's own average (max-pooling per frame only ever caught the I-frame spikes and hid
# the real difference). Seconds 0-12 are the black title card and fade, excluded from the averages.
T = 640
def per_sec(frames):
    s = [0] * T
    for f in frames:
        if f["pts"] < T: s[int(f["pts"])] += f["b"]
    return s
Rs, Ps = per_sec(of), per_sec(lf)
mR, mP = st.mean(Rs[13:]), st.mean(Ps[13:])
shot_rel = {k: dict(release=round(st.mean(Rs[int(a):int(b)]) / mR, 2), phone=round(st.mean(Ps[int(a):int(b)]) / mP, 2))
            for k, (a, b) in SHOTS.items()}
M["landscape"] = dict(note="bytes per SECOND relative to the file's own average (x1.0 = average); first 640 s, credits excluded",
                      release_rel=[round(v / mR, 3) for v in Rs], phone_rel=[round(v / mP, 3) for v in Ps],
                      release_mbps=round(mR * 8 / 1e6, 2), phone_mbps=round(mP * 8 / 1e6, 2),
                      release_cv=round(st.pstdev(Rs[13:]) / mR, 3), phone_cv=round(st.pstdev(Ps[13:]) / mP, 3),
                      release_max=round(max(Rs[13:]) / mR, 2), phone_max=round(max(Ps[13:]) / mP, 2), shots=shot_rel,
                      phone_encode="scale=960 wide, libx264 veryfast crf 20, 2 s GOP (landscape only)")
# ---- 6. data for the film's remaining evidence scenes (2026-09-25) -------------------------------------------
from PIL import Image, ImageChops

# (a) the residual, for real: one background block of the analysed pan P-frame, copied from where its motion vector
#     says (in the reference frame 3 frames earlier), next to the real block, and their difference.
ref_idx = pidx - M["pan_mvs"]["ref_gap_frames"]
run([FF, "-v", "error", "-y", "-i", f"{PUB}/shot_pan.mp4", "-vf", f"select=eq(n\\,{ref_idx})", "-frames:v", "1", f"{REC}/pan_ref.png"])
cur_img, ref_img = Image.open(f"{PUB}/pan_frame.png").convert("RGB"), Image.open(f"{REC}/pan_ref.png").convert("RGB")
bgmv = [mv for mv in past if mv[4] == 16 and mv[5] == 16 and abs(disp_of(mv)[0] - M["pan_mvs"]["background_median_dx"]) <= 2
        and 400 < mv[2] < 1500 and 200 < mv[3] < 600]
mv = max(bgmv, key=lambda m: m[3])          # a mid-frame background block with the typical displacement
dx0, dy0 = mv[2] - 8, mv[3] - 8; sx0, sy0 = mv[0] - 8, mv[1] - 8
real = cur_img.crop((dx0, dy0, dx0 + 16, dy0 + 16)); copied = ref_img.crop((sx0, sy0, sx0 + 16, sy0 + 16))
diff = ImageChops.difference(real, copied)
amp = diff.point(lambda v: min(255, v * 4))
for name, im in (("res_copied", copied), ("res_real", real), ("res_diff", amp)):
    im.resize((256, 256), Image.NEAREST).save(f"{PUB}/{name}.png")
dstat = [sum(px) / 3 for px in diff.getdata()]
M["residual"] = dict(block=mv, dst=[dx0, dy0], src=[sx0, sy0], mean_abs_diff=round(st.mean(dstat), 2), max_abs_diff=round(max(dstat), 1),
                     note="16x16 block; diff amplified x4 for display only; mean/max are raw 0-255 levels")

# (b) the particle dome: the real vectors of its busiest P-frame, in a 320x320 crop, vs the pan's
pl = [json.loads(l) for l in run([MVS, f"{PUB}/shot_particles.mp4"]).splitlines()]
pp = max((fr for fr in pl if fr["type"] == "P"), key=lambda fr: sum(1 for m in fr["mvs"] if m[6] == -1 and disp_of(m) != (0, 0)))
ppast = [m for m in pp["mvs"] if m[6] == -1]
run([FF, "-v", "error", "-y", "-i", f"{PUB}/shot_particles.mp4", "-vf", f"select=eq(n\\,{pp['n'] - 1})", "-frames:v", "1", f"{PUB}/particles_frame.png"])
# the crop: the 320x320 window whose vectors disagree most (spread of direction), measured
def spread(ms):
    import math
    angs = [math.atan2(disp_of(m)[1], disp_of(m)[0]) for m in ms if disp_of(m) != (0, 0)]
    if len(angs) < 12: return 0
    c = sum(math.cos(a) for a in angs) / len(angs); s_ = sum(math.sin(a) for a in angs) / len(angs)
    return 1 - math.hypot(c, s_)          # 0 = all one way, 1 = every way
best = max(((x, y) for x in range(0, 1600, 80) for y in range(0, 480, 80)),
           key=lambda xy: spread([m for m in ppast if xy[0] <= m[2] < xy[0] + 320 and xy[1] <= m[3] < xy[1] + 320]))
crop = [m for m in ppast if best[0] <= m[2] < best[0] + 320 and best[1] <= m[3] < best[1] + 320]
pan_crop = [m for m in past if 800 <= m[2] < 1120 and 200 <= m[3] < 520]
M["particles_mvs"] = dict(frame_index=pp["n"] - 1, crop=[best[0], best[1], 320, 320], vectors=crop, spread=round(spread(crop), 3),
                          pan_crop=[800, 200, 320, 320], pan_vectors=pan_crop, pan_spread=round(spread(pan_crop), 3))

# (c) what healed the smear: blocks of the BROKEN stream that carry no motion vector (built from scratch), sampled
#     every 6th frame from +1.5 s to +4.5 s after the cut. 16x16 macroblock grid of the 1920x800 frame.
bl = [json.loads(l) for l in run([MVS, broken]).splitlines()]
# pts timebase of the broken mp4: read it rather than assume
tb = run([FP, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=time_base", "-of", "csv=p=0", broken]).strip()
tbf = float(Fraction(tb))
heal = []
for fr in bl:
    t = (fr["pts"] or 0) * tbf
    if not (cut_rel + 1.5 <= t <= cut_rel + 4.5): continue
    covered = set()
    for m in fr["mvs"]:
        for yy in range(m[3] - m[5] // 2, m[3] + m[5] // 2, 16):
            for xx in range(m[2] - m[4] // 2, m[2] + m[4] // 2, 16):
                covered.add((xx // 16, yy // 16))
    allmb = {(x, y) for x in range(120) for y in range(50)}
    heal.append(dict(t=round(t - cut_rel, 3), intra=sorted([list(c) for c in allmb - covered])))
M["heal"] = dict(samples=heal[::6], note="16x16 macroblocks with no motion vector in the broken stream = built from scratch (intra)",
                 mean_intra_share=round(st.mean(len(h["intra"]) / 6000 for h in heal), 3) if heal else None)

# (d) frame-type ribbons, real: the mosh source (I at 0 and at the cut), and keyframe thumbnails of the quiet clip
M["mosh"]["kinds"] = [f["t"] for f in mf]
qk = [i for i, t in enumerate(clips["quiet"]["kinds"]) if t == "I"]
for j, i in enumerate(qk):
    run([FF, "-v", "error", "-y", "-i", f"{PUB}/shot_quiet.mp4", "-vf", f"select=eq(n\\,{i}),scale=320:-2", "-frames:v", "1", f"{PUB}/quiet_key{j}.png"])
M["quiet_keys"] = qk
# full-width frames for the scrubbing scene: every keyframe, plus one frame between two keyframes (the frame a seek
# asks for; the player has to start at the keyframe before it and decode forward)
QUIET_SEEK = qk[2] + (qk[3] - qk[2]) * 2 // 3
for i in qk + [QUIET_SEEK]:
    run([FF, "-v", "error", "-y", "-i", f"{PUB}/shot_quiet.mp4", "-vf", f"select=eq(n\\,{i})", "-frames:v", "1", f"{PUB}/quiet_full_{i}.png"])
M["quiet_seek"] = QUIET_SEEK

json.dump(M, open(f"{REC}/measure.json", "w"), indent=1)
print(json.dumps({k: v for k, v in M.items() if k not in ("landscape", "pan_mvs", "clips")}, indent=1)[:4000])
print("clips:", {k: dict(frames=v["frames"], bpf=v["bytes_per_frame"], types=v["types"]) for k, v in clips.items()})
print("pan:", {k: v for k, v in M["pan_mvs"].items() if k != "mvs"})
print("landscape:", {k: v for k, v in M["landscape"].items() if not k.endswith("_rel")})
