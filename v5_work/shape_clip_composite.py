"""Shape-clipped raw-generator composite.

This local probe is deliberately different from geom_locked_composite.py:
it does not paste synthetic-render texture into the output. It keeps the raw
translated image and only uses the synthetic segmentation as a matte to remove
piece-like spill outside the true silhouette, filling that spill with a simple
board-color estimate from the same generated image.

The goal is to test whether we can reduce side lobes/double heads without the
synthetic-looking geometry-lock texture.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

import sys

sys.path.insert(0, os.path.dirname(__file__))
from detect_head_artifacts import sids  # noqa: E402


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            out = np.maximum(out, np.roll(np.roll(mask, dy, 0), dx, 1))
    return out


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    return ~dilate(~mask, radius)


def soft(mask: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        return mask.astype(np.float32)
    img = Image.fromarray((mask.astype(np.uint8) * 255), "L")
    return np.asarray(img.filter(ImageFilter.GaussianBlur(radius=radius)), np.float32) / 255.0


def cell_board_estimate(fake: np.ndarray, piece: np.ndarray, exclude: np.ndarray) -> np.ndarray:
    """Piece-free per-cell median board estimate, expanded to full image."""
    h, w, _ = fake.shape
    cell = h // 8
    bg = np.zeros_like(fake, dtype=np.float32)
    global_keep = ~exclude
    global_med = np.median(fake[global_keep] if global_keep.any() else fake.reshape(-1, 3), axis=0)
    for r in range(8):
        for c in range(8):
            y0, y1 = r * cell, (r + 1) * cell
            x0, x1 = c * cell, (c + 1) * cell
            keep = ~exclude[y0:y1, x0:x1]
            # Prefer pixels far from pieces. If a crowded square has too few, use
            # non-silhouette pixels; if still too few, fall back to global board color.
            if keep.sum() < 64:
                keep = ~piece[y0:y1, x0:x1]
            if keep.sum() >= 16:
                med = np.median(fake[y0:y1, x0:x1][keep], axis=0)
            else:
                med = global_med
            bg[y0:y1, x0:x1] = med
    # Smooth block boundaries; the composite only uses this near pieces.
    return np.asarray(
        Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=3.0)),
        np.float32,
    )


def process_one(
    fake_path: Path,
    ds_dir: Path,
    labels: dict,
    out_path: Path,
    clip_radius: int,
    protect_radius: int,
    feather: float,
    restore_inner: float,
) -> None:
    name = fake_path.stem
    meta = labels[name]
    fake = np.asarray(Image.open(fake_path).convert("RGB"), np.float32)
    seg = sids(Image.open(ds_dir / meta["split"] / f"{name}_seg.png"))

    piece = seg >= 3
    outer = dilate(piece, clip_radius)
    protected = erode(piece, protect_radius)
    exclude = dilate(piece, max(clip_radius + 4, 8))
    board = cell_board_estimate(fake, piece, exclude)

    # Replace near-piece pixels outside the true silhouette with board estimate.
    clip_zone = outer & (~piece)
    a_clip = soft(clip_zone, feather)[..., None]
    comp = fake * (1.0 - a_clip) + board * a_clip

    # Optionally protect the inner silhouette against feather spill, preserving
    # raw generator color/texture where it is least likely to be a side lobe.
    if restore_inner > 0:
        a_inner = (soft(protected, max(0.5, feather * 0.5)) * restore_inner)[..., None]
        comp = comp * (1.0 - a_inner) + fake * a_inner

    Image.fromarray(np.clip(comp, 0, 255).astype(np.uint8)).save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", required=True, type=Path)
    parser.add_argument("--ds", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--clip-radius", type=int, default=14)
    parser.add_argument("--protect-radius", type=int, default=1)
    parser.add_argument("--feather", type=float, default=1.5)
    parser.add_argument("--restore-inner", type=float, default=1.0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    labels = json.load(open(args.ds / "labels.json"))
    files = sorted(args.fake.glob("*.png"))
    n = 0
    for i, fake_path in enumerate(files):
        if fake_path.stem not in labels:
            continue
        process_one(
            fake_path,
            args.ds,
            labels,
            args.out / fake_path.name,
            args.clip_radius,
            args.protect_radius,
            args.feather,
            args.restore_inner,
        )
        n += 1
        if i % 25 == 0:
            print(f"processed {i}/{len(files)} {fake_path.name}")
    print(f"saved {n} composites -> {args.out}")


if __name__ == "__main__":
    main()
