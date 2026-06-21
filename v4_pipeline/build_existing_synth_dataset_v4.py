"""
build_existing_synth_dataset_v4.py

Build a geometry-conditioned paired dataset from the already generated Blender
images in data from drive/dataset/{trainA,testA}. This deliberately avoids new
Blender rendering: the existing synthetic images are visually cleaner and have
recognisable piece types.

Outputs:
  datasets/chess_existing_geom_v4/{train,test}/{name}.png
      [A_existing_synthetic | B_real_target]
  datasets/chess_existing_geom_v4/{train,test}/{name}_seg.png
      FEN semantic ids, full-cell labels, ids 1..14
  datasets/chess_existing_geom_v4/{train,test}/{name}_geom.png
      geometry hints from FEN + synthetic image:
        R: synthetic foreground/silhouette estimate
        G: relative piece height by FEN class
        B: synthetic edge map inside occupied cells
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from build_paired_dataset import (
    REAL_RE,
    COLOR_WEIGHT,
    LOWCONF_CORR,
    fen_to_grids,
    orient_grid,
    load_fen_index,
    lookup_fen,
    cell_grids,
    choose_orientation,
    occupancy_match_rate,
    load_gray_array,
    load_rgb,
    maybe_rot180,
    resolve_data_dir,
)
from build_paired_dataset_v2 import canonicalize_color, compute_reference


TILE = 512
SYN_RE = re.compile(r"^game(\d+)_frame_(\d+)_(left|middle|right)_(white|black)\.png$", re.I)

CLASS_ID = {"empty_light": 1, "empty_dark": 2}
for i, t in enumerate(["p", "n", "b", "r", "q", "k"]):
    CLASS_ID["w" + t] = 3 + i
    CLASS_ID["b" + t] = 9 + i

HEIGHT_BY_KIND = {
    "p": 0.46,
    "n": 0.64,
    "b": 0.78,
    "r": 0.55,
    "q": 0.86,
    "k": 0.95,
}


def fen_piece_grid(fen, viewpoint):
    ids = np.zeros((8, 8), np.uint8)
    rows = fen.split()[0].split("/")
    for r, row in enumerate(rows):
        c = 0
        for ch in row:
            if ch.isdigit():
                c += int(ch)
                continue
            color = "w" if ch.isupper() else "b"
            ids[r, c] = CLASS_ID[color + ch.lower()]
            c += 1
    if viewpoint == "black":
        ids = np.rot90(ids, 2)
    return ids


def seg_from_fen(fen, viewpoint, tile=TILE):
    grid = fen_piece_grid(fen, viewpoint)
    seg = np.zeros((tile, tile), np.uint8)
    cell = tile // 8
    for r in range(8):
        for c in range(8):
            cid = int(grid[r, c])
            if cid == 0:
                cid = CLASS_ID["empty_light"] if ((r + c) % 2 == 0) else CLASS_ID["empty_dark"]
            seg[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = cid
    return seg


def height_from_seg(seg):
    h = np.zeros(seg.shape, np.float32)
    for cid in range(3, 15):
        kind_idx = (cid - 3) % 6
        kind = ["p", "n", "b", "r", "q", "k"][kind_idx]
        h[seg == cid] = HEIGHT_BY_KIND[kind]
    return h


def foreground_from_synthetic(syn_img, seg):
    gray = np.asarray(syn_img.convert("L"), np.float32)
    h, w = gray.shape
    cell = h // 8
    fg = np.zeros((h, w), np.float32)
    occ = seg >= 3
    for r in range(8):
        for c in range(8):
            y0, y1 = r * cell, (r + 1) * cell
            x0, x1 = c * cell, (c + 1) * cell
            if not occ[y0:y1, x0:x1].any():
                continue
            patch = gray[y0:y1, x0:x1]
            border = max(2, cell // 10)
            ring = np.concatenate([
                patch[:border, :].reshape(-1),
                patch[-border:, :].reshape(-1),
                patch[:, :border].reshape(-1),
                patch[:, -border:].reshape(-1),
            ])
            bg = float(np.median(ring))
            diff = np.abs(patch - bg)
            thr = max(8.0, float(np.percentile(diff, 62)))
            fg[y0:y1, x0:x1] = np.where(diff >= thr, diff, 0.0)
    if fg.max() > 0:
        fg = np.clip(fg / max(float(np.percentile(fg[fg > 0], 95)), 1.0), 0, 1)
    return fg


def edge_from_gray(syn_img, occ):
    gray = np.asarray(syn_img.convert("L"), np.float32) / 255.0
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = np.abs(gray[:, 2:] - gray[:, :-2])
    gy[1:-1, :] = np.abs(gray[2:, :] - gray[:-2, :])
    edge = (gx + gy) * occ.astype(np.float32)
    if edge.max() > 0:
        edge = np.clip(edge / max(float(np.percentile(edge[edge > 0], 96)), 1e-6), 0, 1)
    return edge


def geom_from_synthetic(syn_img, seg):
    fg = foreground_from_synthetic(syn_img, seg)
    height = height_from_seg(seg)
    edge = edge_from_gray(syn_img, seg >= 3)
    geom = np.stack([fg, height, edge], axis=-1)
    return np.clip(geom * 255.0 + 0.5, 0, 255).astype(np.uint8)


def count_pieces(fen):
    return sum(1 for ch in fen.split()[0] if ch.isalpha())


def build_synthetic_index(syn_dir):
    idx = {}
    for path in sorted(syn_dir.iterdir()):
        m = SYN_RE.match(path.name)
        if not m:
            continue
        game, frame, view, viewpoint = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4).lower()
        idx[(game, frame, viewpoint, view)] = path
    return idx


def make_montage(samples, qa_dir, qa_samples):
    if not samples:
        return
    samples = sorted(samples, key=lambda x: x[2])
    k = min(qa_samples, len(samples))
    chosen = [samples[i] for i in np.linspace(0, len(samples) - 1, k).astype(int)]
    cols = 4
    rows = (k + cols - 1) // cols
    tw, th = 256, 128
    pad, label_h = 6, 16
    canvas = Image.new("RGB", (cols * (tw + pad) + pad, rows * (th + label_h + pad) + pad), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    ex_dir = qa_dir / "examples"
    ex_dir.mkdir(parents=True, exist_ok=True)
    for i, (split, name, n_pieces, paired) in enumerate(chosen):
        r, c = divmod(i, cols)
        x = pad + c * (tw + pad)
        y = pad + r * (th + label_h + pad)
        canvas.paste(paired.resize((tw, th), Image.BICUBIC), (x, y + label_h))
        draw.text((x + 2, y + 2), f"{split} {name[:30]} p{n_pieces}", fill=(235, 235, 235))
        if i < 8:
            paired.save(ex_dir / f"example_{i:02d}_{name}")
    canvas.save(qa_dir / "montage.png")


def build(project_dir, data_dir, out_dir, train_views, test_views, tile, color_canon, qa_samples):
    fen_index = load_fen_index(project_dir)
    if not fen_index:
        raise RuntimeError("No game*.csv files found")

    ref = None
    if color_canon:
        ref, ref_n = compute_reference(data_dir, {5}, fen_index)
        print(f"[color] canonical real target from game5 n={ref_n}: "
              f"a={ref[0]:.2f}/{ref[1]:.2f} b={ref[2]:.2f}/{ref[3]:.2f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    qa_dir = out_dir / "_qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    labels = {}
    rows = []
    stats = {
        "pairs": 0,
        "skipped": 0,
        "syn_rot180": 0,
        "real_rot180": 0,
        "lowconf": 0,
        "match_syn_sum": 0.0,
        "match_real_sum": 0.0,
    }
    skipped = []
    qa_pool = []

    split_cfg = {
        "train": ("trainA", "trainB", train_views),
        "test": ("testA", "testB", test_views),
    }
    for split, (syn_sub, real_sub, views) in split_cfg.items():
        syn_idx = build_synthetic_index(data_dir / syn_sub)
        real_dir = data_dir / real_sub
        split_out = out_dir / split
        split_out.mkdir(parents=True, exist_ok=True)
        real_files = sorted(p for p in real_dir.iterdir() if REAL_RE.match(p.name))
        for real_path in tqdm(real_files, desc=split, unit="real"):
            m = REAL_RE.match(real_path.name)
            game, frame, viewpoint = int(m.group(1)), int(m.group(2)), m.group(3).lower()
            fen = lookup_fen(fen_index, game, frame)
            if fen is None:
                skipped.append(("no_fen", real_path.name))
                stats["skipped"] += 1
                continue

            occ_std, color_std = fen_to_grids(fen)
            fen_occ = orient_grid(occ_std, viewpoint)
            fen_color = orient_grid(color_std, viewpoint)
            real_gray = load_gray_array(real_path)
            r_occ, r_lum = cell_grids(real_gray)
            r_rot, r_oc, r_cc = choose_orientation(r_occ, r_lum, fen_occ, fen_color)
            real_oriented = maybe_rot180(load_rgb(real_path), r_rot)
            if color_canon:
                real_arr = canonicalize_color(np.asarray(real_oriented), ref)
                real_img = Image.fromarray(real_arr)
            else:
                real_img = real_oriented
            real_img = real_img.resize((tile, tile), Image.BICUBIC)
            r_occ_o = np.rot90(r_occ, 2) if r_rot else r_occ
            r_match = occupancy_match_rate(r_occ_o, fen_occ)

            seg = seg_from_fen(fen, viewpoint, tile)
            n_pieces = count_pieces(fen)

            for view in views:
                syn_path = syn_idx.get((game, frame, viewpoint, view))
                if syn_path is None:
                    skipped.append(("no_synthetic", f"{real_path.name}:{view}"))
                    stats["skipped"] += 1
                    continue
                syn_gray = load_gray_array(syn_path)
                s_occ, s_lum = cell_grids(syn_gray)
                s_rot, s_oc, s_cc = choose_orientation(s_occ, s_lum, fen_occ, fen_color)
                syn_img = maybe_rot180(load_rgb(syn_path), s_rot).resize((tile, tile), Image.BICUBIC)
                s_occ_o = np.rot90(s_occ, 2) if s_rot else s_occ
                s_match = occupancy_match_rate(s_occ_o, fen_occ)
                lowconf = min(s_oc + COLOR_WEIGHT * s_cc, r_oc + COLOR_WEIGHT * r_cc) < LOWCONF_CORR

                name = f"game{game}_frame_{frame:06d}_{viewpoint}_{view}"
                paired = Image.new("RGB", (2 * tile, tile))
                paired.paste(syn_img, (0, 0))
                paired.paste(real_img, (tile, 0))
                paired.save(split_out / f"{name}.png")
                Image.fromarray(seg, mode="L").save(split_out / f"{name}_seg.png")
                Image.fromarray(geom_from_synthetic(syn_img, seg), mode="RGB").save(split_out / f"{name}_geom.png")

                labels[name] = {
                    "fen": fen,
                    "viewpoint": viewpoint,
                    "synthetic_view": view,
                    "game": game,
                    "frame": frame,
                    "split": split,
                    "real_source": real_path.name,
                    "synthetic_source": syn_path.name,
                    "color_canonicalized": bool(color_canon),
                }
                rows.append([split, name, game, frame, viewpoint, view, n_pieces,
                             int(s_rot), int(r_rot), round(s_match, 3), round(r_match, 3),
                             round(s_oc, 3), round(s_cc, 3), round(r_oc, 3), round(r_cc, 3),
                             int(lowconf)])
                stats["pairs"] += 1
                stats["syn_rot180"] += int(s_rot)
                stats["real_rot180"] += int(r_rot)
                stats["lowconf"] += int(lowconf)
                stats["match_syn_sum"] += s_match
                stats["match_real_sum"] += r_match
                if len(qa_pool) < 5000:
                    qa_pool.append((split, f"{name}.png", n_pieces, paired.copy()))

    with open(out_dir / "labels.json", "w") as f:
        json.dump(labels, f, indent=2)
    with open(out_dir / "stats.json", "w") as f:
        json.dump({
            **stats,
            "train_views": train_views,
            "test_views": test_views,
            "tile": tile,
            "classes": CLASS_ID,
            "color_canonicalized": bool(color_canon),
        }, f, indent=2)
    with open(qa_dir / "pairs.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["split", "name", "game", "frame", "viewpoint", "synthetic_view", "pieces",
                     "syn_rot180", "real_rot180", "syn_occ_match", "real_occ_match",
                     "syn_occ_corr", "syn_color_corr", "real_occ_corr", "real_color_corr",
                     "lowconf"])
        wr.writerows(rows)
    with open(qa_dir / "skipped.txt", "w") as f:
        for reason, name in skipped:
            f.write(f"{reason}\t{name}\n")
    make_montage(qa_pool, qa_dir, qa_samples)

    n = max(stats["pairs"], 1)
    print("[done]", out_dir)
    print(f"pairs={stats['pairs']} skipped={stats['skipped']} "
          f"syn_rot={stats['syn_rot180']/n:.2%} real_rot={stats['real_rot180']/n:.2%} "
          f"syn_match={stats['match_syn_sum']/n:.3f} real_match={stats['match_real_sum']/n:.3f} "
          f"lowconf={stats['lowconf']}")


def parse_views(s):
    allowed = {"left", "middle", "right"}
    views = [x.strip() for x in s.split(",") if x.strip()]
    bad = [x for x in views if x not in allowed]
    if bad:
        raise ValueError(f"Bad views {bad}; allowed {sorted(allowed)}")
    return views


def main():
    project_dir = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-dir", default=str(project_dir / "datasets" / "chess_existing_geom_v4"))
    ap.add_argument("--train-views", default="left,middle,right")
    ap.add_argument("--test-views", default="middle")
    ap.add_argument("--tile", type=int, default=TILE)
    ap.add_argument("--no-color-canon", action="store_true")
    ap.add_argument("--qa-samples", type=int, default=32)
    args = ap.parse_args()
    data_dir = resolve_data_dir(project_dir, args.data_dir)
    build(project_dir=project_dir,
          data_dir=data_dir,
          out_dir=Path(args.out_dir),
          train_views=parse_views(args.train_views),
          test_views=parse_views(args.test_views),
          tile=args.tile,
          color_canon=not args.no_color_canon,
          qa_samples=args.qa_samples)


if __name__ == "__main__":
    main()
