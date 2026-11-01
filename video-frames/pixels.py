#!/usr/bin/env python3
"""
pixels.py — why one raw frame is 1.5 bytes per pixel, shown on 64 real pixels of the film.

Takes an 8x8 patch of the pan frame (public/video-frames/pan_frame.png, a decoded frame of the release file,
1920x800) and splits it the way a raw 4:2:0 frame stores it:
  px_patch.png   the 64 pixels as decoded
  px_luma.png    brightness (Y): one byte for EVERY pixel                     -> 64 bytes
  px_chroma.png  colour (Cb, Cr) averaged over each 2x2 block, brightness held flat so only colour shows:
                 one Cb byte + one Cr byte per 4 pixels = half a byte per pixel -> 32 bytes
  64 + 32 = 96 bytes for 64 pixels = 1.5 bytes per pixel.
The patch sits on the yellow print of the jacket, where colour changes inside the patch, so the 2x2 blocks show.
Run from anywhere after measure.py:  /usr/bin/python3 video-frames/pixels.py
"""
import json, os
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
PUB, REC = "public/video-frames", "recordings/video-frames"
X, Y, N = 620, 560, 8          # even x/y, so the patch lines up with the 2x2 colour grid

src = Image.open(f"{PUB}/pan_frame.png").convert("RGB")
patch = src.crop((X, Y, X + N, Y + N))
patch.save(f"{PUB}/px_patch.png")

ycc = patch.convert("YCbCr")
yy, cb, cr = ycc.split()
yy.save(f"{PUB}/px_luma.png")

cbp, crp = cb.load(), cr.load()
for by in range(0, N, 2):
    for bx in range(0, N, 2):
        cells = [(bx + i, by + j) for j in (0, 1) for i in (0, 1)]
        mb = round(sum(cbp[c] for c in cells) / 4)
        mr = round(sum(crp[c] for c in cells) / 4)
        for c in cells:
            cbp[c], crp[c] = mb, mr
flat = Image.new("L", (N, N), 150)
Image.merge("YCbCr", (flat, cb, cr)).convert("RGB").save(f"{PUB}/px_chroma.png")

luma_bytes, chroma_bytes = N * N, 2 * (N // 2) * (N // 2)
out = dict(frame="pan_frame.png", x=X, y=Y, size=N, luma_bytes=luma_bytes, chroma_bytes=chroma_bytes,
           bytes_per_pixel=(luma_bytes + chroma_bytes) / (N * N))
json.dump(out, open(f"{REC}/pixels.json", "w"), indent=1)
print(out)
