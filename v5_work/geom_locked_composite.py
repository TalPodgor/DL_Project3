"""Geometry-locked local composite baseline.

This is not a trainable model. It answers a narrow question: if piece geometry is
hard-gated by the synthetic silhouette, do the visible double-head / side-lobe
failures disappear enough to justify a decoupled foreground/background model?

Output = generated board/background outside the synthetic piece mask + a
soft-feathered, color-matched synthetic piece render inside the piece mask.

Usage:
  python3 geom_locked_composite.py \
    --fake v5_work/eval_noC_silAB/fake_B \
    --ds datasets/chess_v5_oblique_aligned_bright \
    --out v5_work/eval_geom_locked_silAB/fake_B
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
from detect_head_artifacts import lum, sids  # noqa: E402


def robust_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = x.reshape(-1, x.shape[-1]).astype(np.float32)
    med = np.median(flat, axis=0)
    lo = np.percentile(flat, 15, axis=0)
    hi = np.percentile(flat, 85, axis=0)
    scale = np.maximum(hi - lo, 8.0)
    return med, scale


def color_match_piece(
    syn: np.ndarray,
    fake: np.ndarray,
    mask: np.ndarray,
    contrast: float,
    shading_strength: float,
) -> np.ndarray:
    out = syn.astype(np.float32).copy()
    if mask.sum() < 16:
        return out
    src_med, src_scale = robust_stats(syn[mask])
    dst_med, dst_scale = robust_stats(fake[mask])
    matched = (syn.astype(np.float32) - src_med.reshape(1, 1, 3)) / src_scale.reshape(1, 1, 3)
    matched = matched * (contrast * dst_scale.reshape(1, 1, 3)) + dst_med.reshape(1, 1, 3)
    # Keep source shading but soften it toward the translated palette.
    src_l = lum(syn.astype(np.float32))
    ml = float(np.median(src_l[mask]))
    shading = np.clip((src_l - ml) / 80.0, -0.55, 0.55)[..., None]
    matched = matched + shading * shading_strength
    out[mask] = matched[mask]
    return np.clip(out, 0, 255)


def process_one(
    fake_path: Path,
    ds_dir: Path,
    labels: dict,
    out_path: Path,
    feather: float,
    piece_blur: float,
    contrast: float,
    shading_strength: float,
) -> None:
    name = fake_path.stem
    meta = labels[name]
    ab = np.asarray(Image.open(ds_dir / meta["split"] / f"{name}.png").convert("RGB"))
    syn = ab[:, : ab.shape[1] // 2]
    fake = np.asarray(Image.open(fake_path).convert("RGB"))
    seg = sids(Image.open(ds_dir / meta["split"] / f"{name}_seg.png"))

    piece = seg >= 3
    styled = syn.astype(np.float32).copy()
    for pid in sorted(int(x) for x in np.unique(seg) if x >= 3):
        pm = seg == pid
        styled = color_match_piece(styled.astype(np.uint8), fake, pm, contrast, shading_strength)
    if piece_blur > 0:
        styled = np.asarray(
            Image.fromarray(np.clip(styled, 0, 255).astype(np.uint8)).filter(
                ImageFilter.GaussianBlur(radius=piece_blur)
            ),
            dtype=np.float32,
        )

    alpha = Image.fromarray((piece.astype(np.uint8) * 255), "L")
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=feather))
    a = np.asarray(alpha, dtype=np.float32)[..., None] / 255.0
    comp = styled * a + fake.astype(np.float32) * (1.0 - a)
    Image.fromarray(np.clip(comp, 0, 255).astype(np.uint8)).save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", required=True, type=Path)
    parser.add_argument("--ds", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--feather", type=float, default=1.2)
    parser.add_argument("--piece-blur", type=float, default=0.0)
    parser.add_argument("--contrast", type=float, default=0.72)
    parser.add_argument("--shading-strength", type=float, default=34.0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    labels = json.load(open(args.ds / "labels.json"))
    files = sorted(args.fake.glob("*.png"))
    for i, fake_path in enumerate(files):
        if fake_path.stem not in labels:
            continue
        process_one(
            fake_path,
            args.ds,
            labels,
            args.out / fake_path.name,
            args.feather,
            args.piece_blur,
            args.contrast,
            args.shading_strength,
        )
        if i % 25 == 0:
            print(f"processed {i}/{len(files)} {fake_path.name}")
    print(f"saved {len(list(args.out.glob('*.png')))} composites -> {args.out}")


if __name__ == "__main__":
    main()
