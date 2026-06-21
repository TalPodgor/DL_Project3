"""Temporal real-sprite compositor probe.

Build a bank of real chess-piece sprites from train frames by subtracting an
empty-frame background from the same game/view/square, then composite those
real sprites onto generated test boards. Synthetic geometry is used only as a
matte/placement guardrail; synthetic RGB is never pasted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

import sys

sys.path.insert(0, os.path.dirname(__file__))
from detect_head_artifacts import ID2P, PIECE_TO_ID, WIN, fen_grid, sids  # noqa: E402


@dataclass
class Item:
    name: str
    split: str
    game: int
    viewpoint: str
    grid: np.ndarray


@dataclass
class Sprite:
    patch: np.ndarray
    alpha: np.ndarray
    center_y: float
    center_x: float
    quality: float
    source: str
    bg_source: str
    area: float


def real_half(ds_dir: Path, split: str, name: str) -> np.ndarray:
    ab = np.asarray(Image.open(ds_dir / split / f"{name}.png").convert("RGB"), np.uint8)
    return ab[:, ab.shape[1] // 2:]


def seg_ids(ds_dir: Path, split: str, name: str) -> np.ndarray:
    return sids(Image.open(ds_dir / split / f"{name}_seg.png"))


def crop_with_pad(arr: np.ndarray, cy: int, cx: int, win: int, fill: float = 0) -> np.ndarray:
    h, w = arr.shape[:2]
    half = win // 2
    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half
    shape = (win, win) + arr.shape[2:]
    out = np.empty(shape, dtype=arr.dtype)
    out[...] = fill
    yy0, yy1 = max(0, y0), min(h, y1)
    xx0, xx1 = max(0, x0), min(w, x1)
    if yy1 <= yy0 or xx1 <= xx0:
        return out
    oy0, ox0 = yy0 - y0, xx0 - x0
    out[oy0:oy0 + (yy1 - yy0), ox0:ox0 + (xx1 - xx0)] = arr[yy0:yy1, xx0:xx1]
    return out


def paste_with_crop(dst: np.ndarray, patch: np.ndarray, alpha: np.ndarray, cy: int, cx: int) -> None:
    h, w = dst.shape[:2]
    win = patch.shape[0]
    half = win // 2
    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half
    yy0, yy1 = max(0, y0), min(h, y1)
    xx0, xx1 = max(0, x0), min(w, x1)
    if yy1 <= yy0 or xx1 <= xx0:
        return
    py0, px0 = yy0 - y0, xx0 - x0
    p = patch[py0:py0 + (yy1 - yy0), px0:px0 + (xx1 - xx0)].astype(np.float32)
    a = alpha[py0:py0 + (yy1 - yy0), px0:px0 + (xx1 - xx0), None].astype(np.float32)
    dst[yy0:yy1, xx0:xx1] = p * a + dst[yy0:yy1, xx0:xx1] * (1.0 - a)


def mask_center(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return (mask.shape[0] * 0.5, mask.shape[1] * 0.5)
    return ((float(ys.min()) + float(ys.max())) * 0.5, (float(xs.min()) + float(xs.max())) * 0.5)


def shift_array(arr: np.ndarray, dy: int, dx: int, fill) -> np.ndarray:
    out = np.empty_like(arr)
    out[...] = fill
    h, w = arr.shape[:2]
    sy0 = max(0, -dy)
    sy1 = min(h, h - dy)
    sx0 = max(0, -dx)
    sx1 = min(w, w - dx)
    dy0 = max(0, dy)
    dx0 = max(0, dx)
    if sy1 > sy0 and sx1 > sx0:
        out[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = arr[sy0:sy1, sx0:sx1]
    return out


def shift_patch_and_alpha(patch: np.ndarray, alpha: np.ndarray, dy: int, dx: int) -> tuple[np.ndarray, np.ndarray]:
    b = max(3, patch.shape[0] // 16)
    border = np.concatenate([
        patch[:b].reshape(-1, 3),
        patch[-b:].reshape(-1, 3),
        patch[:, :b].reshape(-1, 3),
        patch[:, -b:].reshape(-1, 3),
    ], axis=0)
    fill = np.median(border, axis=0).astype(patch.dtype)
    return shift_array(patch, dy, dx, fill), shift_array(alpha, dy, dx, 0.0)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    img = Image.fromarray(mask.astype(np.uint8) * 255, "L")
    return np.asarray(img.filter(ImageFilter.MaxFilter(radius * 2 + 1))) > 0


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    img = Image.fromarray(mask.astype(np.uint8) * 255, "L")
    return np.asarray(img.filter(ImageFilter.MinFilter(radius * 2 + 1))) > 0


def soft(mask: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        return mask.astype(np.float32)
    img = Image.fromarray(mask.astype(np.uint8) * 255, "L")
    return np.asarray(img.filter(ImageFilter.GaussianBlur(radius=radius)), np.float32) / 255.0


def connected_components(mask: np.ndarray, min_area: int = 8) -> list[tuple[int, np.ndarray]]:
    mask = mask.astype(bool)
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[tuple[int, np.ndarray]] = []
    h, w = mask.shape
    for y in range(h):
        xs = np.where(mask[y] & ~seen[y])[0]
        for x0 in xs:
            if seen[y, x0] or not mask[y, x0]:
                continue
            stack = [(y, int(x0))]
            seen[y, x0] = True
            coords = []
            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))
                for ny in range(max(0, cy - 1), min(h, cy + 2)):
                    for nx in range(max(0, cx - 1), min(w, cx + 2)):
                        if not seen[ny, nx] and mask[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            if len(coords) >= min_area:
                cm = np.zeros(mask.shape, dtype=bool)
                yy, xx = zip(*coords)
                cm[np.asarray(yy), np.asarray(xx)] = True
                comps.append((len(coords), cm))
    comps.sort(key=lambda x: x[0], reverse=True)
    return comps


def cells_for_window(row: int, col: int, win: int, tile: int = 512) -> list[tuple[int, int]]:
    cell = tile // 8
    half = win // 2
    cy = row * cell + cell // 2
    cx = col * cell + cell // 2
    y0, y1 = max(0, cy - half), min(tile, cy + half)
    x0, x1 = max(0, cx - half), min(tile, cx + half)
    return [
        (rr, cc)
        for rr in range(y0 // cell, (y1 - 1) // cell + 1)
        for cc in range(x0 // cell, (x1 - 1) // cell + 1)
    ]


def load_items(labels: dict) -> list[Item]:
    items = []
    for name, meta in labels.items():
        items.append(
            Item(
                name=name,
                split=meta["split"],
                game=int(meta["game"]),
                viewpoint=meta["viewpoint"],
                grid=fen_grid(meta["fen"], meta["viewpoint"]),
            )
        )
    return items


def choose_bg_frame(candidates: list[Item], row: int, col: int, win: int) -> Item | None:
    cells = cells_for_window(row, col, win)
    best = None
    best_score = -1
    for item in candidates:
        if int(item.grid[row, col]) != 0:
            continue
        score = 0
        for rr, cc in cells:
            score += 1 if int(item.grid[rr, cc]) == 0 else 0
        # Central/direct-neighbour emptiness matters more than diagonal cells.
        for rr, cc in cells:
            if abs(rr - row) + abs(cc - col) <= 1 and int(item.grid[rr, cc]) == 0:
                score += 2
        if score > best_score:
            best = item
            best_score = score
    return best


def make_sprite(
    item: Item,
    bg_item: Item,
    piece_id: int,
    row: int,
    col: int,
    ds_dir: Path,
    image_cache,
    seg_cache,
    args,
) -> Sprite | None:
    piece = ID2P[piece_id]
    win = WIN[piece]
    cell = 64
    cy = row * cell + cell // 2
    cx = col * cell + cell // 2
    real = image_cache(item.split, item.name)
    bg = image_cache(bg_item.split, bg_item.name)
    seg = seg_cache(item.split, item.name)

    patch = crop_with_pad(real, cy, cx, win)
    bg_patch = crop_with_pad(bg, cy, cx, win)
    class_mask = crop_with_pad((seg == piece_id).astype(np.uint8), cy, cx, win).astype(bool)
    if class_mask.sum() < args.min_sil_pixels:
        return None

    guide = dilate(class_mask, args.extract_sil_dilate)
    diff = np.sqrt(((patch.astype(np.float32) - bg_patch.astype(np.float32)) ** 2).sum(axis=2))
    outside = diff[~guide]
    noise = float(np.percentile(outside, 92)) if outside.size else 12.0
    threshold = max(args.diff_thresh, noise + args.noise_margin)
    raw_hi = (diff > threshold) & guide
    loose_guide = dilate(class_mask, args.extract_sil_dilate + args.loose_guide_extra)
    low_threshold = max(args.diff_thresh * args.low_thresh_ratio, noise + args.low_noise_margin)
    raw_lo = (diff > low_threshold) & loose_guide
    if args.grabcut:
        try:
            import cv2
        except Exception as exc:  # pragma: no cover - depends on local optional package
            raise RuntimeError("--grabcut requires opencv-python-headless") from exc
        gc = np.full(raw_lo.shape, cv2.GC_PR_BGD, dtype=np.uint8)
        gc[~loose_guide] = cv2.GC_BGD
        gc[raw_lo] = cv2.GC_PR_FGD
        sure_fg = raw_hi & dilate(core := class_mask, 4)
        gc[sure_fg] = cv2.GC_FGD
        if (gc == cv2.GC_FGD).sum() >= 8 and (gc != cv2.GC_BGD).sum() >= 32:
            bg_model = np.zeros((1, 65), np.float64)
            fg_model = np.zeros((1, 65), np.float64)
            cv2.grabCut(
                patch[..., ::-1].copy(),
                gc,
                None,
                bg_model,
                fg_model,
                args.grabcut_iters,
                cv2.GC_INIT_WITH_MASK,
            )
            raw_lo = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)) & loose_guide

    # Keep low-threshold components only when they are seeded by a high-confidence
    # temporal-difference region and overlap the broad synthetic guide. This gives a
    # fuller real matte without letting board texture flood the crop.
    comps = connected_components(raw_lo, min_area=args.min_component_area)
    if not comps:
        return None
    keep = np.zeros(raw_lo.shape, dtype=bool)
    core = dilate(class_mask, 2)
    for area, comp in comps[: args.max_components_keep]:
        if (comp & raw_hi).sum() >= max(3, int(area * 0.04)) and (comp & core).sum() >= 3:
            keep |= comp
    if keep.sum() == 0:
        return None
    if args.close_grow > 0:
        keep = dilate(keep, args.close_grow)
    if args.close_shrink > 0:
        keep = erode(keep, args.close_shrink)
    keep = dilate(keep, args.alpha_grow)
    diff_alpha = soft(keep, args.alpha_blur)
    # Temporal difference is good for rejecting bad sprites, but as the only matte
    # it can fragment highlights or dark crowns. Use the rendered class silhouette
    # as a solid shape prior for opacity while still using diff-derived alpha near
    # boundaries and for quality filtering.
    if args.shape_alpha_weight > 0:
        shape_alpha = soft(dilate(class_mask, args.shape_alpha_dilate), args.shape_alpha_blur)
        alpha = np.maximum(diff_alpha, shape_alpha * args.shape_alpha_weight)
    else:
        alpha = diff_alpha
    alpha = np.clip(alpha * args.alpha_gain, 0.0, args.max_alpha)
    area_ratio = float((alpha > 0.25).mean())
    if not (args.min_area <= area_ratio <= args.max_area):
        return None

    strong_comps = connected_components(alpha > 0.45, min_area=args.min_component_area)
    if len(strong_comps) > args.max_strong_components:
        return None

    mean_diff = float(diff[alpha > 0.35].mean()) if (alpha > 0.35).any() else 0.0
    quality = mean_diff + 35.0 * area_ratio - 4.0 * max(0, len(strong_comps) - 1)
    return Sprite(
        patch=patch,
        alpha=alpha.astype(np.float32),
        center_y=mask_center(class_mask)[0],
        center_x=mask_center(class_mask)[1],
        quality=quality,
        source=item.name,
        bg_source=bg_item.name,
        area=area_ratio,
    )


def build_bank(ds_dir: Path, labels: dict, args) -> dict:
    items = load_items(labels)
    train = [item for item in items if item.split == "train"]
    by_gv = defaultdict(list)
    for item in train:
        by_gv[(item.game, item.viewpoint)].append(item)

    @lru_cache(maxsize=None)
    def image_cache(split: str, name: str) -> np.ndarray:
        return real_half(ds_dir, split, name)

    @lru_cache(maxsize=None)
    def seg_cache(split: str, name: str) -> np.ndarray:
        return seg_ids(ds_dir, split, name)

    bank = defaultdict(list)
    attempted = Counter()
    accepted = Counter()
    for item in train:
        candidates = by_gv[(item.game, item.viewpoint)]
        for row in range(8):
            for col in range(8):
                piece_id = int(item.grid[row, col])
                if piece_id < 3:
                    continue
                piece = ID2P[piece_id]
                key_exact = (piece, item.viewpoint, row, (row + col) & 1)
                key_loose = (piece, item.viewpoint, None, (row + col) & 1)
                if len(bank[key_exact]) >= args.max_per_exact and len(bank[key_loose]) >= args.max_per_loose:
                    continue
                if args.max_attempts_per_piece > 0 and attempted[piece] >= args.max_attempts_per_piece:
                    continue
                attempted[piece] += 1
                bg_item = choose_bg_frame(candidates, row, col, WIN[piece])
                if bg_item is None:
                    continue
                sprite = make_sprite(item, bg_item, piece_id, row, col, ds_dir, image_cache, seg_cache, args)
                if sprite is None:
                    continue
                bank[key_exact].append(sprite)
                bank[key_loose].append(sprite)
                bank[(piece, item.viewpoint, None, None)].append(sprite)
                accepted[piece] += 1

    for key, vals in bank.items():
        vals.sort(key=lambda s: s.quality, reverse=True)
        limit = args.max_per_exact if key[2] is not None else args.max_per_loose
        del vals[limit:]

    print("sprite extraction attempted:", dict(attempted))
    print("sprite extraction accepted:", dict(accepted))
    print("bank keys:", len(bank), "sprites:", sum(len(v) for v in bank.values()))
    return bank


def choose_sprite(bank: dict, piece: str, viewpoint: str, row: int, parity: int, salt: str) -> Sprite | None:
    keys = [
        (piece, viewpoint, row, parity),
        (piece, viewpoint, row, 1 - parity),
        (piece, viewpoint, None, parity),
        (piece, viewpoint, None, None),
    ]
    for key in keys:
        vals = bank.get(key)
        if vals:
            top = vals[: min(24, len(vals))]
            idx = int(hashlib.md5(salt.encode("utf-8")).hexdigest(), 16) % len(top)
            return top[idx]
    return None


def board_estimate(fake: np.ndarray, seg: np.ndarray, grid: np.ndarray, args) -> np.ndarray:
    tile = fake.shape[0]
    cell = tile // 8
    piece = seg >= 3
    remove = dilate(piece, args.bg_remove_radius)
    # Optional wider erasure. Keep off by default because full-window erasure
    # creates flat square patches when many pieces are present.
    if args.bg_window_extra > 0:
        for row in range(8):
            for col in range(8):
                pid = int(grid[row, col])
                if pid < 3:
                    continue
                win = WIN[ID2P[pid]]
                half = win // 2 + args.bg_window_extra
                cy = row * cell + cell // 2
                cx = col * cell + cell // 2
                y0, y1 = max(0, cy - half), min(tile, cy + half)
                x0, x1 = max(0, cx - half), min(tile, cx + half)
                remove[y0:y1, x0:x1] = True

    bg = np.zeros_like(fake, np.float32)
    global_keep = ~remove
    global_med = np.median(fake[global_keep] if global_keep.any() else fake.reshape(-1, 3), axis=0)
    parity_meds = {}
    for parity in (0, 1):
        keep_cells = np.zeros((tile, tile), dtype=bool)
        for row in range(8):
            for col in range(8):
                if ((row + col) & 1) == parity and int(grid[row, col]) == 0:
                    keep_cells[row * cell:(row + 1) * cell, col * cell:(col + 1) * cell] = True
        keep = keep_cells & ~remove
        parity_meds[parity] = np.median(fake[keep], axis=0) if keep.sum() >= 128 else global_med

    for row in range(8):
        for col in range(8):
            y0, y1 = row * cell, (row + 1) * cell
            x0, x1 = col * cell, (col + 1) * cell
            keep = ~remove[y0:y1, x0:x1]
            if keep.sum() >= 64:
                med = np.median(fake[y0:y1, x0:x1][keep], axis=0)
            else:
                med = parity_meds[(row + col) & 1]
            bg[y0:y1, x0:x1] = med
    return np.asarray(
        Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=args.bg_blur)
        ),
        np.float32,
    )


def process_one(fake_path: Path, ds_dir: Path, labels: dict, bank: dict, out_path: Path, args) -> bool:
    name = fake_path.stem
    if name not in labels:
        return False
    meta = labels[name]
    fake = np.asarray(Image.open(fake_path).convert("RGB"), np.float32)
    grid = fen_grid(meta["fen"], meta["viewpoint"])
    seg = seg_ids(ds_dir, meta["split"], name)
    bg = board_estimate(fake, seg, grid, args)
    remove = dilate(seg >= 3, args.composite_remove_radius)
    comp = fake * (1.0 - soft(remove, args.composite_remove_blur)[..., None]) + bg * soft(
        remove, args.composite_remove_blur
    )[..., None]

    cell = fake.shape[0] // 8
    # Far-to-near in screen space, so nearer lower pieces can cover farther ones.
    for row in range(8):
        for col in range(8):
            piece_id = int(grid[row, col])
            if piece_id < 3:
                continue
            piece = ID2P[piece_id]
            sprite = choose_sprite(bank, piece, meta["viewpoint"], row, (row + col) & 1, f"{name}:{row}:{col}")
            if sprite is None:
                continue
            win = WIN[piece]
            cy = row * cell + cell // 2
            cx = col * cell + cell // 2
            target_mask = crop_with_pad((seg == piece_id).astype(np.uint8), cy, cx, win).astype(bool)
            clip = soft(dilate(target_mask, args.target_clip_radius), args.target_clip_blur)
            ty, tx = mask_center(target_mask)
            dy = int(round(ty - sprite.center_y))
            dx = int(round(tx - sprite.center_x))
            patch, sprite_alpha = shift_patch_and_alpha(sprite.patch, sprite.alpha, dy, dx)
            alpha = sprite_alpha * clip
            if args.opacity != 1.0:
                alpha = np.clip(alpha * args.opacity, 0.0, 1.0)
            paste_with_crop(comp, patch, alpha, cy, cx)

    Image.fromarray(np.clip(comp, 0, 255).astype(np.uint8)).save(out_path)
    return True


def write_bank_debug(out_dir: Path, bank: dict) -> None:
    summary = {}
    for key, vals in bank.items():
        k = "|".join("" if x is None else str(x) for x in key)
        summary[k] = {
            "n": len(vals),
            "quality_mean": float(np.mean([v.quality for v in vals])) if vals else 0.0,
            "area_mean": float(np.mean([v.area for v in vals])) if vals else 0.0,
            "top_sources": [v.source for v in vals[:5]],
        }
    with open(out_dir / "bank_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", required=True, type=Path)
    parser.add_argument("--ds", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-per-exact", type=int, default=28)
    parser.add_argument("--max-per-loose", type=int, default=96)
    parser.add_argument("--max-attempts-per-piece", type=int, default=0)
    parser.add_argument("--diff-thresh", type=float, default=18.0)
    parser.add_argument("--noise-margin", type=float, default=8.0)
    parser.add_argument("--extract-sil-dilate", type=int, default=14)
    parser.add_argument("--loose-guide-extra", type=int, default=10)
    parser.add_argument("--low-thresh-ratio", type=float, default=0.58)
    parser.add_argument("--low-noise-margin", type=float, default=2.0)
    parser.add_argument("--close-grow", type=int, default=3)
    parser.add_argument("--close-shrink", type=int, default=2)
    parser.add_argument("--alpha-grow", type=int, default=1)
    parser.add_argument("--alpha-blur", type=float, default=1.2)
    parser.add_argument("--alpha-gain", type=float, default=1.15)
    parser.add_argument("--max-alpha", type=float, default=0.96)
    parser.add_argument("--shape-alpha-dilate", type=int, default=1)
    parser.add_argument("--shape-alpha-blur", type=float, default=1.0)
    parser.add_argument("--shape-alpha-weight", type=float, default=0.0)
    parser.add_argument("--min-area", type=float, default=0.018)
    parser.add_argument("--max-area", type=float, default=0.52)
    parser.add_argument("--min-sil-pixels", type=int, default=24)
    parser.add_argument("--min-component-area", type=int, default=10)
    parser.add_argument("--max-components-keep", type=int, default=5)
    parser.add_argument("--max-strong-components", type=int, default=5)
    parser.add_argument("--grabcut", action="store_true")
    parser.add_argument("--grabcut-iters", type=int, default=3)
    parser.add_argument("--bg-remove-radius", type=int, default=12)
    parser.add_argument("--bg-window-extra", type=int, default=0)
    parser.add_argument("--bg-blur", type=float, default=3.0)
    parser.add_argument("--composite-remove-radius", type=int, default=12)
    parser.add_argument("--composite-remove-blur", type=float, default=1.4)
    parser.add_argument("--target-clip-radius", type=int, default=22)
    parser.add_argument("--target-clip-blur", type=float, default=2.2)
    parser.add_argument("--opacity", type=float, default=1.0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    labels = json.load(open(args.ds / "labels.json"))
    bank = build_bank(args.ds, labels, args)
    write_bank_debug(args.out, bank)

    files = sorted(args.fake.glob("*.png"))
    n = 0
    for i, fake_path in enumerate(files):
        if process_one(fake_path, args.ds, labels, bank, args.out / fake_path.name, args):
            n += 1
        if i % 25 == 0:
            print(f"processed {i}/{len(files)} {fake_path.name}")
    print(f"saved {n} composites -> {args.out}")


if __name__ == "__main__":
    main()
