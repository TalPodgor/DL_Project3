"""Create a piece-parallax boosted V5 dataset.

The aligned-bright V5 data already fixed the worst global camera angle issue,
but the rendered piece bodies can still be too short/upright after board-plane
rectification. This script applies a conservative, deterministic source-side
geometry correction:

  * detect each rendered piece from the Blender semantic pass
  * erase its old source pixels from A/seg/depth
  * stretch that piece upward while keeping its base fixed
  * write the transformed A, *_seg.png and *_depth.png alongside the same B target

Only the synthetic source side is modified. The real target and labels stay the
same, so this can be used as a focused camera/parallax ablation.
"""
import argparse
import json
import os
import shutil
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


PIECE_TO_ID = {
    "P": 3, "N": 4, "B": 5, "R": 6, "Q": 7, "K": 8,
    "p": 9, "n": 10, "b": 11, "r": 12, "q": 13, "k": 14,
}

CLASS_COLORS_LINEAR = {
    "P": (0.92, 0.88, 0.72),
    "N": (0.82, 0.82, 0.52),
    "B": (0.92, 0.72, 0.44),
    "R": (0.72, 0.92, 0.72),
    "Q": (0.72, 0.82, 0.98),
    "K": (0.90, 0.70, 0.98),
    "p": (0.18, 0.12, 0.08),
    "n": (0.32, 0.18, 0.08),
    "b": (0.20, 0.32, 0.12),
    "r": (0.10, 0.24, 0.34),
    "q": (0.34, 0.12, 0.30),
    "k": (0.34, 0.10, 0.10),
}


def linear_to_srgb_u8(v):
    v = np.asarray(v, dtype=np.float32)
    srgb = np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1.0 / 2.4) - 0.055)
    return np.clip(np.round(srgb * 255.0), 0, 255).astype(np.int16)


PALETTE = np.stack([linear_to_srgb_u8(CLASS_COLORS_LINEAR[p]) for p in PIECE_TO_ID], axis=0)
PALETTE_IDS = np.asarray([PIECE_TO_ID[p] for p in PIECE_TO_ID], dtype=np.uint8)


def semantic_ids(seg_rgb, threshold=75, bright_gate=30):
    arr = np.asarray(seg_rgb.convert("RGB"), dtype=np.int16)
    flat = arr.reshape(-1, 3)
    diff = flat[:, None, :] - PALETTE[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    nearest = np.argmin(dist2, axis=1)
    best = dist2[np.arange(flat.shape[0]), nearest]
    ids = np.zeros(flat.shape[0], dtype=np.uint8)
    bright_enough = np.max(flat, axis=1) >= bright_gate
    keep = (best <= threshold * threshold) & bright_enough
    ids[keep] = PALETTE_IDS[nearest[keep]]
    return ids.reshape(arr.shape[:2])


def components_for_class(mask, min_pixels):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    ys, xs = np.nonzero(mask)
    for sy, sx in zip(ys, xs):
        if seen[sy, sx]:
            continue
        q = deque([(int(sy), int(sx))])
        seen[sy, sx] = True
        pts = []
        while q:
            y, x = q.popleft()
            pts.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        if len(pts) >= min_pixels:
            yy = [p[0] for p in pts]
            xx = [p[1] for p in pts]
            comps.append({
                "points": pts,
                "bbox": (min(xx), min(yy), max(xx) + 1, max(yy) + 1),
                "area": len(pts),
            })
    return comps


def all_piece_components(ids, min_pixels):
    comps = []
    for cid in range(3, 15):
        comps.extend(components_for_class(ids == cid, min_pixels))
    comps.sort(key=lambda c: (c["bbox"][1], c["bbox"][0]))
    return comps


def dilate_mask(mask, radius):
    if radius <= 0:
        return mask
    pil = Image.fromarray((mask.astype(np.uint8) * 255), "L")
    pil = pil.filter(ImageFilter.MaxFilter(radius * 2 + 1))
    return np.asarray(pil) > 0


def erase_rgb_with_blur(img, erase_mask, radius=11):
    arr = np.asarray(img.convert("RGB")).copy()
    blurred = np.asarray(img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=radius)))
    arr[erase_mask] = blurred[erase_mask]
    return Image.fromarray(arr, "RGB")


def erase_gray_with_blur(img, erase_mask, radius=11):
    arr = np.asarray(img.convert("L")).copy()
    blurred = np.asarray(img.convert("L").filter(ImageFilter.GaussianBlur(radius=radius)))
    arr[erase_mask] = blurred[erase_mask]
    return Image.fromarray(arr, "L")


def alpha_from_points(points, bbox, size):
    x0, y0, x1, y1 = bbox
    alpha = Image.new("L", size, 0)
    draw = ImageDraw.Draw(alpha)
    for y, x in points:
        draw.point((x - x0, y - y0), fill=255)
    alpha = alpha.filter(ImageFilter.MaxFilter(3))
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=0.45))
    return alpha


def extra_for_component(height, max_extra, min_extra, power):
    if max_extra <= 0:
        return 0
    # Short pawns get a mild correction; tall officers get the full correction.
    t = np.clip(height / 120.0, 0.35, 1.0) ** power
    return int(round(min_extra + (max_extra - min_extra) * t))


def paste_stretched(base, src_img, comp, extra, resample, canvas_mode):
    if extra <= 0:
        return base
    x0, y0, x1, y1 = comp["bbox"]
    w, h = x1 - x0, y1 - y0
    new_h = h + extra
    crop = src_img.crop((x0, y0, x1, y1))
    alpha = alpha_from_points(comp["points"], comp["bbox"], (w, h))
    if crop.mode != "RGBA":
        crop = crop.convert("RGBA")
    crop.putalpha(alpha)
    crop = crop.resize((w, new_h), resample=resample)
    top = y1 - new_h
    if top < 0:
        crop = crop.crop((0, -top, w, new_h))
        top = 0
    if canvas_mode == "L":
        layer = Image.new("L", base.size, 0)
        layer.paste(crop.convert("L"), (x0, top), crop.getchannel("A"))
        mask = Image.new("L", base.size, 0)
        mask.paste(crop.getchannel("A"), (x0, top))
        base.paste(layer, (0, 0), mask)
    else:
        base.paste(crop.convert(canvas_mode), (x0, top), crop.getchannel("A"))
    return base


def transform_sample(pair_path, out_pair_path, max_extra, min_extra, min_pixels, erase_radius):
    pair = Image.open(pair_path).convert("RGB")
    w2 = pair.width // 2
    src_a = pair.crop((0, 0, w2, pair.height))
    real_b = pair.crop((w2, 0, pair.width, pair.height))
    seg_path = pair_path.with_name(pair_path.stem + "_seg.png")
    depth_path = pair_path.with_name(pair_path.stem + "_depth.png")
    seg = Image.open(seg_path).convert("RGB")
    depth = Image.open(depth_path).convert("L")

    ids = semantic_ids(seg)
    comps = all_piece_components(ids, min_pixels=min_pixels)
    piece_mask = ids >= 3
    erase_mask = dilate_mask(piece_mask, radius=2)

    out_a = erase_rgb_with_blur(src_a, erase_mask, radius=erase_radius)
    out_seg = Image.fromarray(np.zeros_like(np.asarray(seg)), "RGB")
    out_depth = erase_gray_with_blur(depth, erase_mask, radius=erase_radius)

    for comp in comps:
        x0, y0, x1, y1 = comp["bbox"]
        extra = extra_for_component(y1 - y0, max_extra=max_extra, min_extra=min_extra, power=1.0)
        out_a = paste_stretched(out_a, src_a, comp, extra, Image.Resampling.BICUBIC, "RGB")
        out_seg = paste_stretched(out_seg, seg, comp, extra, Image.Resampling.NEAREST, "RGB")
        out_depth = paste_stretched(out_depth, depth, comp, extra, Image.Resampling.BILINEAR, "L")

    out_pair = Image.new("RGB", pair.size)
    out_pair.paste(out_a, (0, 0))
    out_pair.paste(real_b, (w2, 0))
    out_pair_path.parent.mkdir(parents=True, exist_ok=True)
    out_pair.save(out_pair_path)
    out_seg.save(out_pair_path.with_name(out_pair_path.stem + "_seg.png"))
    out_depth.save(out_pair_path.with_name(out_pair_path.stem + "_depth.png"))
    return len(comps)


def iter_pair_paths(src, split):
    return sorted(
        p for p in (src / split).glob("*.png")
        if not p.name.endswith("_seg.png") and not p.name.endswith("_depth.png")
    )


def build_sheet(src, out, sheet_path, names):
    cells = []
    for name in names:
        in_p = src / "test" / f"{name}.png"
        out_p = out / "test" / f"{name}.png"
        if not in_p.exists() or not out_p.exists():
            continue
        original = Image.open(in_p).convert("RGB")
        boosted = Image.open(out_p).convert("RGB")
        # Show only the synthetic A side, where the intervention happens.
        w = original.width // 2
        cells.append((original.crop((0, 0, w, original.height)), boosted.crop((0, 0, w, boosted.height)), name))
    if not cells:
        return
    thumb = 256
    row_h = thumb + 28
    sheet = Image.new("RGB", (thumb * 2, row_h * len(cells)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (orig, boosted, name) in enumerate(cells):
        y = i * row_h
        sheet.paste(orig.resize((thumb, thumb), Image.Resampling.BICUBIC), (0, y + 28))
        sheet.paste(boosted.resize((thumb, thumb), Image.Resampling.BICUBIC), (thumb, y + 28))
        draw.text((4, y + 6), f"{name} | original A", fill=(0, 0, 0))
        draw.text((thumb + 4, y + 6), "parallax-boosted A", fill=(0, 0, 0))
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(sheet_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--max-extra", type=int, default=8)
    ap.add_argument("--min-extra", type=int, default=2)
    ap.add_argument("--min-pixels", type=int, default=25)
    ap.add_argument("--erase-radius", type=int, default=11)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sheet", type=Path, default=None)
    args = ap.parse_args()

    if not args.src.exists():
        raise FileNotFoundError(args.src)
    args.out.mkdir(parents=True, exist_ok=True)
    label_src = args.src / "labels.json"
    if label_src.exists():
        labels = json.loads(label_src.read_text())
        for meta in labels.values():
            meta.setdefault("postprocess", {})
            meta["postprocess"]["parallax_boost"] = {
                "max_extra": args.max_extra,
                "min_extra": args.min_extra,
                "note": "source-side vertical piece stretch anchored at base",
            }
        (args.out / "labels.json").write_text(json.dumps(labels, indent=2))
    for extra in ["stats.json"]:
        p = args.src / extra
        if p.exists():
            shutil.copy2(p, args.out / extra)

    total = 0
    total_comps = 0
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        paths = iter_pair_paths(args.src, split)
        if args.limit:
            paths = paths[:args.limit]
        for idx, p in enumerate(paths, 1):
            rel = p.relative_to(args.src)
            out_p = args.out / rel
            if out_p.exists() and not args.force:
                continue
            comps = transform_sample(
                p, out_p,
                max_extra=args.max_extra,
                min_extra=args.min_extra,
                min_pixels=args.min_pixels,
                erase_radius=args.erase_radius,
            )
            total += 1
            total_comps += comps
            if idx == 1 or idx % 100 == 0:
                print(f"[{split}] {idx}/{len(paths)} {p.name} comps={comps}")
    print(f"[done] wrote {total} samples; transformed {total_comps} components -> {args.out}")

    if args.sheet:
        names = [
            "game2_frame_000200_black_middle",
            "game2_frame_003100_white_middle",
            "game2_frame_013896_black_middle",
            "game2_frame_031332_white_middle",
        ]
        build_sheet(args.src, args.out, args.sheet, names)
        print(f"[sheet] {args.sheet}")


if __name__ == "__main__":
    main()
