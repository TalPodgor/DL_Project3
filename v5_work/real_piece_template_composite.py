"""Real-piece template composite probe.

This is a deliberately pragmatic fallback experiment. It builds a small bank of
real piece crops from the TRAIN split only, estimates an alpha matte for each
piece, removes generated pieces from a fake_B board with a crude board estimate,
and pastes real-looking piece templates into occupied test squares.

It is not meant to be the final architecture. It tests whether a real-piece prior
solves the remaining visual problem better than synthetic geometry locking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


PIECE_TO_ID = {"P": 3, "N": 4, "B": 5, "R": 6, "Q": 7, "K": 8,
               "p": 9, "n": 10, "b": 11, "r": 12, "q": 13, "k": 14}
ID_TO_PIECE = {v: k for k, v in PIECE_TO_ID.items()}
PIECES = set(PIECE_TO_ID)
WIN_BY_CLASS = {"P": 92, "N": 104, "B": 112, "R": 100, "Q": 116, "K": 116,
                "p": 92, "n": 104, "b": 112, "r": 100, "q": 116, "k": 116}


def fen_grid(fen: str, viewpoint: str) -> np.ndarray:
    ids = np.zeros((8, 8), np.uint8)
    for r, row in enumerate(fen.split()[0].split("/")):
        c = 0
        for ch in row:
            if ch.isdigit():
                c += int(ch)
            else:
                ids[r, c] = PIECE_TO_ID[ch]
                c += 1
    if viewpoint == "black":
        ids = np.rot90(ids, 2)
    return ids


def crop_with_pad(img: np.ndarray, cy: int, cx: int, win: int) -> np.ndarray:
    h, w = img.shape[:2]
    half = win // 2
    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half
    out = np.zeros((win, win, img.shape[2]), np.float32)
    yy0, yy1 = max(0, y0), min(h, y1)
    xx0, xx1 = max(0, x0), min(w, x1)
    oy0, ox0 = yy0 - y0, xx0 - x0
    out[oy0:oy0 + (yy1 - yy0), ox0:ox0 + (xx1 - xx0)] = img[yy0:yy1, xx0:xx1]
    if yy0 > y0:
        out[:oy0] = out[oy0:oy0 + 1]
    if yy1 < y1:
        out[oy0 + (yy1 - yy0):] = out[oy0 + (yy1 - yy0) - 1:oy0 + (yy1 - yy0)]
    if xx0 > x0:
        out[:, :ox0] = out[:, ox0:ox0 + 1]
    if xx1 < x1:
        out[:, ox0 + (xx1 - xx0):] = out[:, ox0 + (xx1 - xx0) - 1:ox0 + (xx1 - xx0)]
    return out


def paste_with_crop(dst: np.ndarray, patch: np.ndarray, alpha: np.ndarray, cy: int, cx: int) -> None:
    h, w = dst.shape[:2]
    win = patch.shape[0]
    half = win // 2
    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half
    yy0, yy1 = max(0, y0), min(h, y1)
    xx0, xx1 = max(0, x0), min(w, x1)
    py0, px0 = yy0 - y0, xx0 - x0
    p = patch[py0:py0 + (yy1 - yy0), px0:px0 + (xx1 - xx0)]
    a = alpha[py0:py0 + (yy1 - yy0), px0:px0 + (xx1 - xx0), None]
    dst[yy0:yy1, xx0:xx1] = p * a + dst[yy0:yy1, xx0:xx1] * (1.0 - a)


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


def soft(mask: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        return mask.astype(np.float32)
    img = Image.fromarray((mask.astype(np.uint8) * 255), "L")
    return np.asarray(img.filter(ImageFilter.GaussianBlur(radius=radius)), np.float32) / 255.0


def estimate_alpha(patch: np.ndarray, threshold: float) -> np.ndarray:
    win = patch.shape[0]
    b = max(5, win // 14)
    border = np.concatenate([
        patch[:b].reshape(-1, 3),
        patch[-b:].reshape(-1, 3),
        patch[:, :b].reshape(-1, 3),
        patch[:, -b:].reshape(-1, 3),
    ], axis=0)
    bg = np.median(border, axis=0)
    diff = np.linalg.norm(patch - bg.reshape(1, 1, 3), axis=2)
    lum = 0.299 * patch[..., 0] + 0.587 * patch[..., 1] + 0.114 * patch[..., 2]
    bg_lum = float(0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2])
    # Do not use raw gradient as a foreground cue here: the real board texture is
    # high-gradient and otherwise the mask becomes almost the whole crop.
    mask = (diff > threshold) | (np.abs(lum - bg_lum) > threshold * 0.45)
    # Suppress border noise; real pieces should live near the centre.
    yy, xx = np.mgrid[:win, :win]
    centre = ((yy - win / 2) ** 2 + (xx - win / 2) ** 2) < (win * 0.47) ** 2
    mask &= centre
    mask = dilate(mask, 2)
    alpha = soft(mask, 1.5)
    return np.clip(alpha, 0.0, 0.95)


def board_estimate(fake: np.ndarray, grid: np.ndarray, remove_radius: int) -> np.ndarray:
    h, _, _ = fake.shape
    cell = h // 8
    occ = np.zeros((h, h), bool)
    for r in range(8):
        for c in range(8):
            if grid[r, c] >= 3:
                y0, y1 = r * cell, (r + 1) * cell
                x0, x1 = c * cell, (c + 1) * cell
                occ[y0:y1, x0:x1] = True
    exclude = dilate(occ, remove_radius)
    bg = np.zeros_like(fake, np.float32)
    global_keep = ~exclude
    global_med = np.median(fake[global_keep] if global_keep.any() else fake.reshape(-1, 3), axis=0)
    for r in range(8):
        for c in range(8):
            y0, y1 = r * cell, (r + 1) * cell
            x0, x1 = c * cell, (c + 1) * cell
            keep = ~exclude[y0:y1, x0:x1]
            med = np.median(fake[y0:y1, x0:x1][keep], axis=0) if keep.sum() >= 16 else global_med
            bg[y0:y1, x0:x1] = med
    return np.asarray(
        Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=4.0)),
        np.float32,
    )


def real_half(ds_dir: Path, split: str, name: str) -> np.ndarray:
    ab = np.asarray(Image.open(ds_dir / split / f"{name}.png").convert("RGB"), np.float32)
    return ab[:, ab.shape[1] // 2:]


def build_bank(ds_dir: Path, labels: dict, max_per_key: int, alpha_threshold: float) -> dict:
    bank = defaultdict(list)
    for name, meta in sorted(labels.items()):
        if meta["split"] != "train":
            continue
        img = real_half(ds_dir, "train", name)
        grid = fen_grid(meta["fen"], meta["viewpoint"])
        cell = img.shape[0] // 8
        for r in range(8):
            for c in range(8):
                pid = int(grid[r, c])
                if pid < 3:
                    continue
                piece = ID_TO_PIECE[pid]
                parity = (r + c) & 1
                key = (piece, meta["viewpoint"], parity)
                if len(bank[key]) >= max_per_key:
                    continue
                cy, cx = r * cell + cell // 2, c * cell + cell // 2
                win = WIN_BY_CLASS[piece]
                patch = crop_with_pad(img, cy, cx, win)
                alpha = estimate_alpha(patch, alpha_threshold)
                area = float((alpha > 0.2).mean())
                if 0.03 <= area <= 0.55:
                    bank[key].append((patch, alpha))
    return bank


def choose(bank: dict, piece: str, viewpoint: str, parity: int, salt: str):
    keys = [(piece, viewpoint, parity), (piece, viewpoint, 1 - parity)]
    for key in keys:
        vals = bank.get(key)
        if vals:
            idx = int(hashlib.md5(salt.encode("utf-8")).hexdigest(), 16) % len(vals)
            return vals[idx]
    return None


def process_one(fake_path: Path, ds_dir: Path, labels: dict, bank: dict, out_path: Path,
                remove_radius: int, remove_feather: float) -> None:
    name = fake_path.stem
    meta = labels[name]
    fake = np.asarray(Image.open(fake_path).convert("RGB"), np.float32)
    grid = fen_grid(meta["fen"], meta["viewpoint"])
    cell = fake.shape[0] // 8
    bg = board_estimate(fake, grid, remove_radius)
    remove = np.zeros((fake.shape[0], fake.shape[1]), bool)
    for r in range(8):
        for c in range(8):
            if grid[r, c] >= 3:
                y0, y1 = r * cell, (r + 1) * cell
                x0, x1 = c * cell, (c + 1) * cell
                remove[y0:y1, x0:x1] = True
    a_remove = soft(remove, remove_feather)[..., None]
    comp = fake * (1.0 - a_remove) + bg * a_remove
    for r in range(8):
        for c in range(8):
            pid = int(grid[r, c])
            if pid < 3:
                continue
            piece = ID_TO_PIECE[pid]
            tpl = choose(bank, piece, meta["viewpoint"], (r + c) & 1, f"{name}:{r}:{c}:{piece}")
            if tpl is None:
                continue
            patch, alpha = tpl
            cy, cx = r * cell + cell // 2, c * cell + cell // 2
            paste_with_crop(comp, patch, alpha, cy, cx)
    Image.fromarray(np.clip(comp, 0, 255).astype(np.uint8)).save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", required=True, type=Path)
    parser.add_argument("--ds", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-per-key", type=int, default=80)
    parser.add_argument("--alpha-threshold", type=float, default=20.0)
    parser.add_argument("--remove-radius", type=int, default=2)
    parser.add_argument("--remove-feather", type=float, default=1.2)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    labels = json.load(open(args.ds / "labels.json"))
    bank = build_bank(args.ds, labels, args.max_per_key, args.alpha_threshold)
    print("bank sizes:")
    for key in sorted(bank):
        print(key, len(bank[key]))
    files = sorted(args.fake.glob("*.png"))
    n = 0
    for i, fake_path in enumerate(files):
        if fake_path.stem not in labels:
            continue
        process_one(fake_path, args.ds, labels, bank, args.out / fake_path.name,
                    args.remove_radius, args.remove_feather)
        n += 1
        if i % 25 == 0:
            print(f"processed {i}/{len(files)} {fake_path.name}")
    print(f"saved {n} composites -> {args.out}")


if __name__ == "__main__":
    main()
