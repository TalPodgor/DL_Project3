"""Single-piece inserter probe.

This is a standalone alternative to whole-board GAN translation. It trains on
one occupied square at a time:

  input  = clean real board background crop + synthetic visible silhouette/depth
           + piece class/view/parity conditioning
  output = real crop, but composited through the synthetic silhouette alpha

At render time, it removes generated pieces from an existing fake board, then
inserts one learned real-style piece per occupied square. The shape is locked by
the synthetic visible silhouette, but RGB/texture are learned from real crops.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

import sys

sys.path.insert(0, os.path.dirname(__file__))
from detect_head_artifacts import ID2P, PIECE_TO_ID, WIN, fen_grid, sids  # noqa: E402


N_CLASSES = 12
PIECE_INDEX = {p: i for i, p in enumerate("PNBRQKpnbrqk")}
RESAMPLE_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR")


@dataclass
class Item:
    name: str
    split: str
    game: int
    viewpoint: str
    grid: np.ndarray


@dataclass
class Sample:
    item: Item
    bg_item: Item
    row: int
    col: int
    piece_id: int


def real_half(ds_dir: Path, split: str, name: str) -> np.ndarray:
    ab = np.asarray(Image.open(ds_dir / split / f"{name}.png").convert("RGB"), np.uint8)
    return ab[:, ab.shape[1] // 2:]


def syn_half(ds_dir: Path, split: str, name: str) -> np.ndarray:
    ab = np.asarray(Image.open(ds_dir / split / f"{name}.png").convert("RGB"), np.uint8)
    return ab[:, : ab.shape[1] // 2]


def depth_img(ds_dir: Path, split: str, name: str) -> np.ndarray:
    return np.asarray(Image.open(ds_dir / split / f"{name}_depth.png").convert("L"), np.uint8)


def seg_ids(ds_dir: Path, split: str, name: str) -> np.ndarray:
    return sids(Image.open(ds_dir / split / f"{name}_seg.png"))


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
        for rr, cc in cells:
            if abs(rr - row) + abs(cc - col) <= 1 and int(item.grid[rr, cc]) == 0:
                score += 3
        if score > best_score:
            best = item
            best_score = score
    return best


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


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    img = Image.fromarray(mask.astype(np.uint8) * 255, "L")
    return np.asarray(img.filter(ImageFilter.MaxFilter(radius * 2 + 1))) > 0


def soft(mask: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        return mask.astype(np.float32)
    img = Image.fromarray(mask.astype(np.uint8) * 255, "L")
    return np.asarray(img.filter(ImageFilter.GaussianBlur(radius=radius)), np.float32) / 255.0


def resize_arr(arr: np.ndarray, size: int, resample=RESAMPLE_BILINEAR) -> np.ndarray:
    if arr.ndim == 2:
        return np.asarray(Image.fromarray(arr).resize((size, size), resample))
    return np.asarray(Image.fromarray(arr).resize((size, size), resample))


def to_chw_rgb(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32).transpose(2, 0, 1) / 127.5 - 1.0)


def mask_to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32)[None])


class PieceInsertDataset(Dataset):
    def __init__(self, ds_dir: Path, split: str, crop_size: int, max_samples: int = 0):
        self.ds_dir = ds_dir
        self.crop_size = crop_size
        labels = json.load(open(ds_dir / "labels.json"))
        items = [item for item in load_items(labels) if item.split == split]
        by_gv = defaultdict(list)
        for item in items:
            by_gv[(item.game, item.viewpoint)].append(item)

        samples: list[Sample] = []
        for item in items:
            candidates = by_gv[(item.game, item.viewpoint)]
            for row in range(8):
                for col in range(8):
                    piece_id = int(item.grid[row, col])
                    if piece_id < 3:
                        continue
                    bg_item = choose_bg_frame(candidates, row, col, WIN[ID2P[piece_id]])
                    if bg_item is None:
                        continue
                    samples.append(Sample(item, bg_item, row, col, piece_id))
        if max_samples and len(samples) > max_samples:
            random.Random(123).shuffle(samples)
            samples = samples[:max_samples]
        self.samples = samples
        print(f"PieceInsertDataset split={split} samples={len(self.samples)}")

    @lru_cache(maxsize=2048)
    def _real(self, split: str, name: str) -> np.ndarray:
        return real_half(self.ds_dir, split, name)

    @lru_cache(maxsize=2048)
    def _seg(self, split: str, name: str) -> np.ndarray:
        return seg_ids(self.ds_dir, split, name)

    @lru_cache(maxsize=2048)
    def _depth(self, split: str, name: str) -> np.ndarray:
        return depth_img(self.ds_dir, split, name)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        piece = ID2P[sample.piece_id]
        win = WIN[piece]
        cell = 64
        cy = sample.row * cell + cell // 2
        cx = sample.col * cell + cell // 2

        bg = crop_with_pad(self._real(sample.bg_item.split, sample.bg_item.name), cy, cx, win)
        target = crop_with_pad(self._real(sample.item.split, sample.item.name), cy, cx, win)
        seg = self._seg(sample.item.split, sample.item.name)
        sil = crop_with_pad((seg == sample.piece_id).astype(np.uint8) * 255, cy, cx, win)
        dep = crop_with_pad(self._depth(sample.item.split, sample.item.name), cy, cx, win)

        bg = resize_arr(bg, self.crop_size)
        target = resize_arr(target, self.crop_size)
        sil = resize_arr(sil, self.crop_size, RESAMPLE_BILINEAR).astype(np.float32) / 255.0
        dep = resize_arr(dep, self.crop_size, RESAMPLE_BILINEAR).astype(np.float32) / 127.5 - 1.0
        alpha = soft(dilate(sil > 0.08, 2), 1.2)
        alpha = np.clip(alpha, 0.0, 1.0)

        cond = np.zeros((N_CLASSES + 2, self.crop_size, self.crop_size), np.float32)
        cond[PIECE_INDEX[piece]] = 1.0
        cond[N_CLASSES] = 1.0 if sample.item.viewpoint == "black" else 0.0
        cond[N_CLASSES + 1] = float((sample.row + sample.col) & 1)
        x = torch.cat([
            to_chw_rgb(bg),
            mask_to_tensor(sil * 2.0 - 1.0),
            mask_to_tensor(dep),
            torch.from_numpy(cond),
        ], dim=0)
        return {
            "x": x,
            "bg": to_chw_rgb(bg),
            "target": to_chw_rgb(target),
            "alpha": mask_to_tensor(alpha),
        }


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch: int, base: int = 48):
        super().__init__()
        self.c1 = ConvBlock(in_ch, base)
        self.c2 = ConvBlock(base, base * 2)
        self.c3 = ConvBlock(base * 2, base * 4)
        self.mid = ConvBlock(base * 4, base * 4)
        self.u3 = ConvBlock(base * 8, base * 2)
        self.u2 = ConvBlock(base * 4, base)
        self.u1 = nn.Sequential(
            nn.Conv2d(base * 2, base, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        e1 = self.c1(x)
        e2 = self.c2(F.avg_pool2d(e1, 2))
        e3 = self.c3(F.avg_pool2d(e2, 2))
        m = self.mid(F.avg_pool2d(e3, 2))
        d3 = F.interpolate(m, scale_factor=2, mode="bilinear", align_corners=False)
        d3 = self.u3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.u2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        return self.u1(torch.cat([d1, e1], dim=1))


def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ds = PieceInsertDataset(args.ds, "train", args.crop_size, args.max_samples)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True)
    in_ch = 3 + 1 + 1 + N_CLASSES + 2
    net = UNetSmall(in_ch, args.base).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.5, 0.999))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        net.train()
        sums = defaultdict(float)
        n = 0
        for batch in loader:
            x = batch["x"].to(device)
            bg = batch["bg"].to(device)
            target = batch["target"].to(device)
            alpha = batch["alpha"].to(device)
            pred_piece = net(x)
            comp = pred_piece * alpha + bg * (1.0 - alpha)
            piece_loss = ((comp - target).abs() * (0.2 + alpha * args.piece_weight)).mean()
            bg_loss = ((comp - bg).abs() * (1.0 - alpha)).mean() * args.bg_weight
            tv = (pred_piece[:, :, 1:] - pred_piece[:, :, :-1]).abs().mean()
            tv = tv + (pred_piece[:, :, :, 1:] - pred_piece[:, :, :, :-1]).abs().mean()
            loss = piece_loss + bg_loss + tv * args.tv_weight
            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = x.shape[0]
            sums["loss"] += float(loss.item()) * bs
            sums["piece"] += float(piece_loss.item()) * bs
            sums["bg"] += float(bg_loss.item()) * bs
            n += bs
        print(
            f"epoch {epoch}/{args.epochs} "
            f"loss={sums['loss']/n:.4f} piece={sums['piece']/n:.4f} bg={sums['bg']/n:.4f}",
            flush=True,
        )
        torch.save({"net": net.state_dict(), "args": vars(args)}, args.out)
    print(f"saved {args.out}")


def board_estimate(fake: np.ndarray, seg: np.ndarray, grid: np.ndarray, radius: int, blur: float) -> np.ndarray:
    tile = fake.shape[0]
    cell = tile // 8
    remove = dilate(seg >= 3, radius)
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
        Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=blur)),
        np.float32,
    )


def infer_patch(net, device, bg_crop, sil_crop, dep_crop, piece, viewpoint, parity, crop_size):
    bg_rs = resize_arr(bg_crop.astype(np.uint8), crop_size)
    sil_rs = resize_arr((sil_crop.astype(np.uint8) * 255), crop_size, RESAMPLE_BILINEAR).astype(np.float32) / 255.0
    dep_rs = resize_arr(dep_crop.astype(np.uint8), crop_size, RESAMPLE_BILINEAR).astype(np.float32) / 127.5 - 1.0
    cond = np.zeros((N_CLASSES + 2, crop_size, crop_size), np.float32)
    cond[PIECE_INDEX[piece]] = 1.0
    cond[N_CLASSES] = 1.0 if viewpoint == "black" else 0.0
    cond[N_CLASSES + 1] = float(parity)
    x = torch.cat([
        to_chw_rgb(bg_rs),
        mask_to_tensor(sil_rs * 2.0 - 1.0),
        mask_to_tensor(dep_rs),
        torch.from_numpy(cond),
    ], dim=0)[None].to(device)
    with torch.no_grad():
        pred = net(x)[0].cpu().numpy().transpose(1, 2, 0)
    pred = np.clip((pred + 1.0) * 127.5, 0, 255).astype(np.uint8)
    alpha = soft(dilate(sil_rs > 0.08, 2), 1.2)
    return pred, alpha


def render(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    train_args = ckpt.get("args", {})
    crop_size = int(train_args.get("crop_size", args.crop_size))
    base = int(train_args.get("base", args.base))
    in_ch = 3 + 1 + 1 + N_CLASSES + 2
    net = UNetSmall(in_ch, base).to(device)
    net.load_state_dict(ckpt["net"])
    net.eval()
    args.out.mkdir(parents=True, exist_ok=True)
    labels = json.load(open(args.ds / "labels.json"))
    files = sorted(args.fake.glob("*.png"))
    n = 0
    for i, fake_path in enumerate(files):
        name = fake_path.stem
        if name not in labels:
            continue
        meta = labels[name]
        grid = fen_grid(meta["fen"], meta["viewpoint"])
        fake = np.asarray(Image.open(fake_path).convert("RGB"), np.float32)
        seg = seg_ids(args.ds, meta["split"], name)
        dep = depth_img(args.ds, meta["split"], name)
        comp = board_estimate(fake, seg, grid, args.bg_remove_radius, args.bg_blur)
        cell = fake.shape[0] // 8
        for row in range(8):
            for col in range(8):
                piece_id = int(grid[row, col])
                if piece_id < 3:
                    continue
                piece = ID2P[piece_id]
                win = WIN[piece]
                cy = row * cell + cell // 2
                cx = col * cell + cell // 2
                bg_crop = crop_with_pad(comp.astype(np.uint8), cy, cx, win)
                sil_crop = crop_with_pad((seg == piece_id).astype(np.uint8), cy, cx, win).astype(bool)
                dep_crop = crop_with_pad(dep, cy, cx, win)
                pred, alpha = infer_patch(net, device, bg_crop, sil_crop, dep_crop, piece, meta["viewpoint"], (row + col) & 1, crop_size)
                pred = resize_arr(pred, win)
                alpha = resize_arr((alpha * 255).astype(np.uint8), win, RESAMPLE_BILINEAR).astype(np.float32) / 255.0
                paste_with_crop(comp, pred, np.clip(alpha * args.opacity, 0.0, 1.0), cy, cx)
        Image.fromarray(np.clip(comp, 0, 255).astype(np.uint8)).save(args.out / fake_path.name)
        n += 1
        if i % 25 == 0:
            print(f"rendered {i}/{len(files)} {fake_path.name}", flush=True)
    print(f"saved {n} rendered boards -> {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--ds", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--base", type=int, default=48)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--piece-weight", type=float, default=5.0)
    p.add_argument("--bg-weight", type=float, default=0.25)
    p.add_argument("--tv-weight", type=float, default=0.01)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=train)

    p = sub.add_parser("render")
    p.add_argument("--ds", required=True, type=Path)
    p.add_argument("--fake", required=True, type=Path)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--base", type=int, default=48)
    p.add_argument("--bg-remove-radius", type=int, default=12)
    p.add_argument("--bg-blur", type=float, default=3.0)
    p.add_argument("--opacity", type=float, default=1.0)
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=render)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
