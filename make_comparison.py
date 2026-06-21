#!/usr/bin/env python3
"""Compose before/after comparison figures for the Wave 4 report.

Each row, for one test frame:
    [ synthetic input | Wave 3 output (old) | Wave 4 output (new) | real target ]

Panels are included only if the file exists, so it degrades gracefully.

Sources (defaults):
  synthetic input : datasets/chess_paired_v2/test/<file>.png   (left half)
  real target     : datasets/chess_paired_v2/test/<file>.png   (right half)
  Wave 3 output   : results/wave3_qa/<file>_fake_B.png          (256px, old model)
  Wave 4 output   : results/chess_hd_test/<file>_fake_B.png     (512px, this run)

Usage:
  python3 make_comparison.py [out.png] [frame1 frame2 ...]
  (frames like game2_frame_031664_black; default = an auto sparse->dense spread)
"""
import glob
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.abspath(__file__))
V2_TEST = os.path.join(ROOT, "datasets", "chess_paired_v2", "test")
WAVE3 = os.path.join(ROOT, "results", "wave3_qa")
WAVE4 = os.path.join(ROOT, "results", "chess_hd_test")
TILE = 256
PAD = 6
LABEL_H = 18
HEADERS = ["synthetic input", "Wave 3 (old)", "Wave 4 (new)", "real target"]


def load_half(path, which):
    if not os.path.exists(path):
        return None
    im = Image.open(path).convert("RGB")
    w, h = im.size
    half = im.crop((0, 0, w // 2, h)) if which == "L" else im.crop((w // 2, 0, w, h))
    return half.resize((TILE, TILE), Image.BICUBIC)


def load_full(path):
    if not os.path.exists(path):
        return None
    return Image.open(path).convert("RGB").resize((TILE, TILE), Image.BICUBIC)


def panels_for(frame):
    pair = os.path.join(V2_TEST, frame + ".png")
    syn = load_half(pair, "L")
    tgt = load_half(pair, "R")
    # Wave 3 QA files have suffix _fake_B.png; Wave 4 test files are plain .png
    w3 = load_full(os.path.join(WAVE3, frame + "_fake_B.png"))
    w4 = load_full(os.path.join(WAVE4, frame + ".png"))
    return [syn, w3, w4, tgt]


def auto_frames(n=8):
    # Use wave4 test frames (all 140) for primary spread; wave3 overlap shown where available
    files = sorted(glob.glob(os.path.join(WAVE4, "*.png")))
    frames = [os.path.basename(f).replace(".png", "") for f in files]
    if not frames:
        # fallback: wave3 QA frames
        files = sorted(glob.glob(os.path.join(WAVE3, "*_fake_B.png")))
        frames = [os.path.basename(f).replace("_fake_B.png", "") for f in files]
    frames = [fr for fr in frames if os.path.exists(os.path.join(V2_TEST, fr + ".png"))]
    if len(frames) <= n:
        return frames
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    return [frames[i] for i in idx]


def main():
    args = sys.argv[1:]
    out = args[0] if args and args[0].endswith(".png") else os.path.join(ROOT, "results", "wave4_comparison.png")
    frames = [a for a in args if not a.endswith(".png")] or auto_frames()
    if not frames:
        print("no frames found"); return

    cols = len(HEADERS)
    rows = len(frames)
    cw = TILE + PAD
    ch = TILE + PAD
    W = cols * cw + PAD
    H = rows * ch + LABEL_H + PAD
    canvas = Image.new("RGB", (W, H), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    for c, hd in enumerate(HEADERS):
        draw.text((PAD + c * cw + 4, 4), hd, fill=(240, 240, 240))
    for r, frame in enumerate(frames):
        ps = panels_for(frame)
        y = LABEL_H + PAD + r * ch
        for c, p in enumerate(ps):
            x = PAD + c * cw
            if p is None:
                draw.rectangle([x, y, x + TILE, y + TILE], fill=(40, 40, 40))
                draw.text((x + 6, y + 6), "(missing)", fill=(150, 150, 150))
            else:
                canvas.paste(p, (x, y))
        draw.text((PAD + 2, y + 2), frame.replace("game2_frame_", "g2-"), fill=(255, 230, 120))
    canvas.save(out)
    print("wrote", out, "with", rows, "frames")


if __name__ == "__main__":
    main()
