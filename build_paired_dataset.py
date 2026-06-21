#!/usr/bin/env python3
"""
build_paired_dataset.py  -  Wave 1 of the "unpaired CUT -> paired supervised" refactor.

WHY THIS EXISTS
---------------
The chess synth->real model was trained UNPAIRED (CUT). The dominant defect is
"phantom pieces": the generator invents pieces on empty squares because the
unpaired objective never supervises per-square occupancy. But the data is in fact
PAIRED: every real photo has a synthetic render of the identical FEN + viewpoint.
This script rebuilds the data in pix2pix "aligned" format so the model can be
fine-tuned with a supervised L1 anchor (Wave 2+).

THE 180-DEGREE CATCH
--------------------
For the SAME (game, frame, viewpoint) label, the synthetic render is rotated 180
degrees relative to the real photo (verified on game7 frame 031396 white: real has
black pieces at top / white at bottom; synthetic-middle has the opposite). Naively
pairing by filename would feed misaligned pairs into an L1 loss and wreck training.
So we re-orient each image to match the FEN (ground truth) before pairing.

WHAT IT DOES
------------
For each real image  game{N}_frame_{ID}_{white|black}.jpg  we:
  1. Look up the FEN for (game, frame) from game{N}.csv.
  2. Pair it with the synthetic MIDDLE crop  game{N}_frame_{ID}_middle_{viewpoint}.png
     (middle crop best matches the real photo's framing).
  3. Build the expected 8x8 occupancy/color grid from the FEN in the orientation
     implied by the viewpoint (white: rank8 at top; black: 180-rotated).
  4. Independently re-orient the real and synthetic images (0 or 180 deg) to best
     match that FEN grid, using an occupancy correlation + a piece-color tie-break.
  5. Resize both to TILE x TILE and write a side-by-side [synthetic | real] image
     (A=left=input, B=right=target) for `--dataset_mode aligned --direction AtoB`.

OUTPUT
------
datasets/chess_paired/
    train/  game{N}_frame_{ID}_{viewpoint}.png   (512x256: synthetic | real)
    test/   ...
    _qa/    montage.png, examples/, stats.txt, pairs.csv, skipped.txt

Runs locally on CPU.  Deps: numpy, Pillow, tqdm.
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# ----------------------------------------------------------------------------- #
# Config / constants
# ----------------------------------------------------------------------------- #
TILE = 256                 # each half is TILE x TILE; combined = 2*TILE x TILE
CROP = "middle"            # which synthetic crop to pair with the real photo
TIE_EPS = 0.05             # |occ_corr(0) - occ_corr(180)| below this -> use color
COLOR_WEIGHT = 0.5         # weight of color correlation in the combined score
LOWCONF_CORR = 0.25        # combined score below this -> flag pair as low-confidence

REAL_RE = re.compile(r"^game(\d+)_frame_(\d+)_(white|black)\.(?:jpg|jpeg|png)$", re.I)

SPLITS = {
    # split_name: (synthetic_subdir, real_subdir)
    "train": ("trainA", "trainB"),
    "test": ("testA", "testB"),
}


# ----------------------------------------------------------------------------- #
# FEN parsing
# ----------------------------------------------------------------------------- #
def fen_placement(fen: str) -> str:
    """Return just the piece-placement field of a FEN."""
    return fen.strip().split()[0]


def fen_to_grids(fen: str):
    """
    Parse a FEN into two 8x8 numpy grids in STANDARD orientation (row 0 = rank 8,
    i.e. the top of a normal board diagram; col 0 = file a):
        occ   : 1 if a piece is on the square else 0
        color : +1 white piece, -1 black piece, 0 empty
    """
    occ = np.zeros((8, 8), dtype=np.float32)
    color = np.zeros((8, 8), dtype=np.float32)
    ranks = fen_placement(fen).split("/")
    if len(ranks) != 8:
        raise ValueError(f"FEN does not have 8 ranks: {fen!r}")
    for r, rank in enumerate(ranks):
        c = 0
        for ch in rank:
            if ch.isdigit():
                c += int(ch)
            else:
                if c > 7:
                    raise ValueError(f"FEN rank overflows 8 files: {rank!r}")
                occ[r, c] = 1.0
                color[r, c] = 1.0 if ch.isupper() else -1.0
                c += 1
    return occ, color


def orient_grid(grid: np.ndarray, viewpoint: str) -> np.ndarray:
    """
    Rotate a standard-orientation FEN grid into the orientation a photo from the
    given viewpoint should have. 'white' viewpoint => white pieces nearest camera
    (bottom) => standard orientation. 'black' viewpoint => 180-deg rotation.
    """
    return grid if viewpoint == "white" else np.rot90(grid, 2)


# ----------------------------------------------------------------------------- #
# Frame -> FEN lookup from the per-game CSVs
# ----------------------------------------------------------------------------- #
def load_fen_index(project_dir: Path):
    """
    Build {game:int -> sorted list of (from_frame, to_frame, fen)} from game*.csv.
    Frame numbers in the CSV are plain ints (not zero-padded); filenames are
    zero-padded to 6 digits, so callers must match via int().
    """
    index = defaultdict(list)
    for csv_path in sorted(project_dir.glob("game*.csv")):
        m = re.match(r"game(\d+)\.csv$", csv_path.name)
        if not m:
            continue
        game = int(m.group(1))
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    f0 = int(row["from_frame"])
                    f1 = int(row["to_frame"])
                    fen = row["fen"].strip()
                except (KeyError, ValueError):
                    continue
                if fen:
                    index[game].append((f0, f1, fen))
        index[game].sort()
    return index


def lookup_fen(index, game: int, frame: int):
    """Find the FEN whose [from_frame, to_frame] interval contains `frame`."""
    rows = index.get(game)
    if not rows:
        return None
    # exact-start fast path
    for f0, f1, fen in rows:
        if f0 <= frame <= f1:
            return fen
    return None


# ----------------------------------------------------------------------------- #
# Image -> per-cell occupancy & luminance grids
# ----------------------------------------------------------------------------- #
def cell_grids(gray: np.ndarray):
    """
    Split a board-filling grayscale image into 8x8 cells and return:
        occ_score : per-cell "piece-ness" = mean |central - cell_background|
        lum       : per-cell mean central luminance (for piece-color tie-break)
    The board is assumed to roughly fill the frame and be axis-aligned (true after
    the board-filling perspective crop used to build both domains). Central-region
    sampling tolerates mild perspective residual.
    """
    H, W = gray.shape
    occ = np.zeros((8, 8), dtype=np.float32)
    lum = np.zeros((8, 8), dtype=np.float32)
    ys = np.linspace(0, H, 9).astype(int)
    xs = np.linspace(0, W, 9).astype(int)
    for i in range(8):
        for j in range(8):
            cell = gray[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            if cell.size == 0:
                continue
            ch, cw = cell.shape
            cy0, cy1 = int(ch * 0.20), max(int(ch * 0.80), int(ch * 0.20) + 1)
            cx0, cx1 = int(cw * 0.20), max(int(cw * 0.80), int(cw * 0.20) + 1)
            central = cell[cy0:cy1, cx0:cx1]
            # background estimate from a thin border ring of the cell
            by = max(1, int(ch * 0.10))
            bx = max(1, int(cw * 0.10))
            ring = np.concatenate([
                cell[:by, :].ravel(), cell[-by:, :].ravel(),
                cell[:, :bx].ravel(), cell[:, -bx:].ravel(),
            ])
            bg = float(np.median(ring)) if ring.size else float(cell.mean())
            occ[i, j] = float(np.mean(np.abs(central.astype(np.float32) - bg)))
            lum[i, j] = float(central.mean())
    return occ, lum


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def choose_orientation(occ_score, lum, fen_occ, fen_color):
    """
    Decide whether an image needs a 180-deg rotation to match the FEN grids.
    Returns (rotate180: bool, occ_corr: float, color_corr: float).

    Primary signal: correlation of detected occupancy with FEN occupancy.
    Tie-break / reinforcement: correlation of piece luminance with FEN color
    (white pieces are bright, black pieces dark) -> resolves occupancy-symmetric
    positions (e.g. the opening) where occupancy alone is ambiguous.
    """
    # luminance signal restricted to occupied squares, mean-centered
    occ_mask = (np.abs(fen_color) > 0).astype(np.float32)
    if occ_mask.sum() > 0:
        lum_c = (lum - lum[occ_mask > 0].mean()) * occ_mask
    else:
        lum_c = lum - lum.mean()

    def score(rot):
        o = np.rot90(occ_score, 2) if rot else occ_score
        l = np.rot90(lum_c, 2) if rot else lum_c
        oc = _corr(o, fen_occ)
        cc = _corr(l, fen_color)
        return oc, cc

    oc0, cc0 = score(False)
    oc180, cc180 = score(True)

    if abs(oc0 - oc180) < TIE_EPS:
        # occupancy ambiguous -> let color decide
        rotate = cc180 > cc0
    else:
        # combined score, occupancy dominates with a color nudge
        s0 = oc0 + COLOR_WEIGHT * cc0
        s180 = oc180 + COLOR_WEIGHT * cc180
        rotate = s180 > s0

    if rotate:
        return True, oc180, cc180
    return False, oc0, cc0


def otsu_threshold(vals: np.ndarray) -> float:
    v = np.sort(vals.ravel().astype(np.float64))
    if v[-1] - v[0] < 1e-9:
        return v[-1] + 1.0  # all equal -> treat all as empty
    best_t, best_var = v[0], -1.0
    for k in range(1, len(v)):
        t = 0.5 * (v[k - 1] + v[k])
        lo, hi = v[v <= t], v[v > t]
        if len(lo) == 0 or len(hi) == 0:
            continue
        w0, w1 = len(lo) / len(v), len(hi) / len(v)
        var = w0 * w1 * (lo.mean() - hi.mean()) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return best_t


def occupancy_match_rate(occ_score, fen_occ) -> float:
    """After orientation, how well does detected occupancy match the FEN (0..1)."""
    thr = otsu_threshold(occ_score)
    detected = (occ_score > thr).astype(np.float32)
    return float((detected == fen_occ).mean())


# ----------------------------------------------------------------------------- #
# Image IO helpers
# ----------------------------------------------------------------------------- #
def load_gray_array(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.float32)


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB").copy()


def maybe_rot180(im: Image.Image, rotate: bool) -> Image.Image:
    return im.transpose(Image.Transpose.ROTATE_180) if rotate else im


# ----------------------------------------------------------------------------- #
# Main build
# ----------------------------------------------------------------------------- #
def resolve_data_dir(project_dir: Path, override: str | None) -> Path:
    if override:
        return Path(override)
    for cand in [project_dir / "data from drive" / "dataset", project_dir / "dataset"]:
        if (cand / "trainB").is_dir():
            return cand
    raise FileNotFoundError("Could not locate dataset dir with trainA/trainB. Use --data-dir.")


def build(project_dir: Path, data_dir: Path, out_dir: Path, qa_samples: int):
    fen_index = load_fen_index(project_dir)
    if not fen_index:
        sys.exit("No game*.csv FEN files found - cannot anchor orientation.")

    out_dir.mkdir(parents=True, exist_ok=True)
    qa_dir = out_dir / "_qa"
    qa_ex_dir = qa_dir / "examples"
    qa_ex_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "pairs": 0,
        "syn_rot180": 0,
        "real_rot180": 0,
        "lowconf": 0,
        "match_syn_sum": 0.0,
        "match_real_sum": 0.0,
    }
    skipped = []          # (reason, filename)
    pair_records = []     # rows for pairs.csv
    qa_pool = []          # (split, name, combined_pil) candidates for the montage

    for split, (syn_sub, real_sub) in SPLITS.items():
        syn_dir = data_dir / syn_sub
        real_dir = data_dir / real_sub
        (out_dir / split).mkdir(parents=True, exist_ok=True)

        real_files = sorted(p for p in real_dir.iterdir() if REAL_RE.match(p.name))
        for real_path in tqdm(real_files, desc=f"{split:5s}", unit="pair"):
            m = REAL_RE.match(real_path.name)
            game, frame, viewpoint = int(m.group(1)), int(m.group(2)), m.group(3).lower()

            fen = lookup_fen(fen_index, game, frame)
            if fen is None:
                skipped.append(("no_fen", real_path.name))
                continue

            syn_name = f"game{game}_frame_{frame:06d}_{CROP}_{viewpoint}.png"
            syn_path = syn_dir / syn_name
            if not syn_path.exists():
                skipped.append(("no_synthetic", real_path.name))
                continue

            try:
                occ_std, color_std = fen_to_grids(fen)
            except ValueError as e:
                skipped.append((f"bad_fen:{e}", real_path.name))
                continue
            fen_occ = orient_grid(occ_std, viewpoint)
            fen_color = orient_grid(color_std, viewpoint)

            # decide orientation for each domain independently, anchored to the FEN
            syn_gray = load_gray_array(syn_path)
            real_gray = load_gray_array(real_path)
            s_occ, s_lum = cell_grids(syn_gray)
            r_occ, r_lum = cell_grids(real_gray)
            s_rot, s_oc, s_cc = choose_orientation(s_occ, s_lum, fen_occ, fen_color)
            r_rot, r_oc, r_cc = choose_orientation(r_occ, r_lum, fen_occ, fen_color)

            # match rates (using oriented occupancy) for QA
            s_occ_o = np.rot90(s_occ, 2) if s_rot else s_occ
            r_occ_o = np.rot90(r_occ, 2) if r_rot else r_occ
            s_match = occupancy_match_rate(s_occ_o, fen_occ)
            r_match = occupancy_match_rate(r_occ_o, fen_occ)

            combined_low = min(s_oc + COLOR_WEIGHT * s_cc, r_oc + COLOR_WEIGHT * r_cc)
            lowconf = combined_low < LOWCONF_CORR

            # build the side-by-side [synthetic | real] image
            syn_im = maybe_rot180(load_rgb(syn_path), s_rot).resize((TILE, TILE), Image.BICUBIC)
            real_im = maybe_rot180(load_rgb(real_path), r_rot).resize((TILE, TILE), Image.BICUBIC)
            combined = Image.new("RGB", (2 * TILE, TILE))
            combined.paste(syn_im, (0, 0))
            combined.paste(real_im, (TILE, 0))

            out_name = f"game{game}_frame_{frame:06d}_{viewpoint}.png"
            combined.save(out_dir / split / out_name)

            # stats
            stats["pairs"] += 1
            stats["syn_rot180"] += int(s_rot)
            stats["real_rot180"] += int(r_rot)
            stats["lowconf"] += int(lowconf)
            stats["match_syn_sum"] += s_match
            stats["match_real_sum"] += r_match
            n_pieces = int(occ_std.sum())
            pair_records.append([
                split, out_name, game, frame, viewpoint, n_pieces,
                int(s_rot), int(r_rot),
                round(s_oc, 3), round(s_cc, 3), round(r_oc, 3), round(r_cc, 3),
                round(s_match, 3), round(r_match, 3), int(lowconf),
            ])
            if len(qa_pool) < 4000:
                qa_pool.append((split, out_name, n_pieces, combined.copy()))

    write_qa(qa_dir, qa_ex_dir, stats, skipped, pair_records, qa_pool, qa_samples)
    print_summary(stats, skipped, out_dir)
    return stats


def write_qa(qa_dir, qa_ex_dir, stats, skipped, pair_records, qa_pool, qa_samples):
    # pairs.csv
    with open(qa_dir / "pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "file", "game", "frame", "viewpoint", "n_pieces",
                    "syn_rot180", "real_rot180", "syn_occ_corr", "syn_color_corr",
                    "real_occ_corr", "real_color_corr", "syn_match", "real_match",
                    "lowconf"])
        w.writerows(pair_records)

    # skipped.txt
    with open(qa_dir / "skipped.txt", "w") as f:
        for reason, name in skipped:
            f.write(f"{reason}\t{name}\n")

    # stats.txt
    n = max(stats["pairs"], 1)
    with open(qa_dir / "stats.txt", "w") as f:
        f.write(f"pairs_built          : {stats['pairs']}\n")
        f.write(f"synthetic_rot180     : {stats['syn_rot180']} "
                f"({100*stats['syn_rot180']/n:.1f}%)\n")
        f.write(f"real_rot180          : {stats['real_rot180']} "
                f"({100*stats['real_rot180']/n:.1f}%)\n")
        f.write(f"mean_occ_match_syn   : {stats['match_syn_sum']/n:.3f}\n")
        f.write(f"mean_occ_match_real  : {stats['match_real_sum']/n:.3f}\n")
        f.write(f"low_confidence_pairs : {stats['lowconf']}\n")
        f.write(f"skipped              : {len(skipped)}\n")

    # montage: prefer a spread across piece counts (sparse..dense)
    if qa_pool:
        qa_pool_sorted = sorted(qa_pool, key=lambda t: t[2])
        k = min(qa_samples, len(qa_pool_sorted))
        idx = np.linspace(0, len(qa_pool_sorted) - 1, k).astype(int)
        chosen = [qa_pool_sorted[i] for i in idx]

        cols = 4
        rows = (k + cols - 1) // cols
        thumb_w, thumb_h = 2 * 128, 128  # combined thumb 256x128
        pad, label_h = 6, 14
        cell_w, cell_h = thumb_w + pad, thumb_h + label_h + pad
        canvas = Image.new("RGB", (cols * cell_w + pad, rows * cell_h + pad), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)
        for n_i, (split, name, n_pieces, comb) in enumerate(chosen):
            r, c = divmod(n_i, cols)
            x = pad + c * cell_w
            y = pad + r * cell_h
            thumb = comb.resize((thumb_w, thumb_h), Image.BICUBIC)
            canvas.paste(thumb, (x, y + label_h))
            label = f"{split} {name.replace('game','g').replace('_frame_','-')[:26]} p{n_pieces}"
            draw.text((x + 2, y + 2), label, fill=(230, 230, 230))
            # save a few full-size examples too
            if n_i < 8:
                comb.save(qa_ex_dir / f"example_{n_i:02d}_{name}")
        canvas.save(qa_dir / "montage.png")


def print_summary(stats, skipped, out_dir):
    n = max(stats["pairs"], 1)
    print("\n" + "=" * 64)
    print("PAIRED DATASET BUILD COMPLETE")
    print("=" * 64)
    print(f"  pairs built            : {stats['pairs']}")
    print(f"  synthetic rotated 180  : {stats['syn_rot180']} ({100*stats['syn_rot180']/n:.1f}%)")
    print(f"  real rotated 180       : {stats['real_rot180']} ({100*stats['real_rot180']/n:.1f}%)")
    print(f"  mean occ-match (syn)   : {stats['match_syn_sum']/n:.3f}")
    print(f"  mean occ-match (real)  : {stats['match_real_sum']/n:.3f}")
    print(f"  low-confidence pairs   : {stats['lowconf']}")
    print(f"  skipped                : {len(skipped)}")
    print(f"\n  output : {out_dir}")
    print(f"  QA     : {out_dir / '_qa' / 'montage.png'}")
    print("=" * 64)


def main():
    project_dir = Path(__file__).parent.resolve()
    ap = argparse.ArgumentParser(description="Build paired (aligned) chess dataset with FEN-anchored orientation fix.")
    ap.add_argument("--data-dir", default=None, help="dir containing trainA/trainB/testA/testB")
    ap.add_argument("--out-dir", default=str(project_dir / "datasets" / "chess_paired"))
    ap.add_argument("--qa-samples", type=int, default=24)
    args = ap.parse_args()

    data_dir = resolve_data_dir(project_dir, args.data_dir)
    out_dir = Path(args.out_dir)
    print(f"project : {project_dir}")
    print(f"data    : {data_dir}")
    print(f"out     : {out_dir}\n")
    build(project_dir, data_dir, out_dir, args.qa_samples)


if __name__ == "__main__":
    main()
