#!/usr/bin/env python3
"""
build_paired_dataset_v2.py  -  Wave 4 dataset rebuild.

WHY THIS EXISTS (vs build_paired_dataset.py / Wave 1)
-----------------------------------------------------
Two evidence-backed defects of the v1 dataset were limiting the model:

1. RESOLUTION.  v1 emitted 256x256 halves, but the REAL target photos are natively
   480x480 and the synthetic renders 512x512 -- so 256px threw away ~2x of the
   available detail (the model's blur is partly just downscaling).  v2 emits 512x512
   halves (1024x512 combined).

2. ILL-POSED COLOR.  The real photos' colour temperature is a PER-RECORDING-SESSION
   camera white-balance constant, NOT a function of the board:

        game   mean Lab b* (warm<->cool)   within-game std
        4      14.0                         1.1
        5      34.6                         0.9
        6      28.4                         1.0
        7      13.8                         0.9
        2(te)  32.4                         0.7

   i.e. a 21-unit b* spread ACROSS games but ~1-unit spread WITHIN a game.  The
   synthetic input is colour-identical regardless of game, so any deterministic
   G(synthetic)->real cannot predict the target's white balance; an L1/regression
   objective collapses to the cross-game average (a muddy b*~23), which is exactly
   the "gray/cold instead of warm wood" failure.  This is unfixable by more training
   or a bigger model -- the information is not in the input.

   FIX: canonicalise the colour of the REAL target half -- a Reinhard transfer of the
   chroma channels (Lab a*,b*) to a single fixed "warm wood" reference -- so every
   target shares one consistent colour temperature.  The mapping becomes well-posed:
   the model learns one warm-wood look it can reliably reproduce.  This is a REAL fix
   (it removes a nuisance variable that makes the task ambiguous), not a cosmetic
   patch.  Lightness (L*) is left untouched so per-position exposure/contrast and
   piece darkness are preserved (L* is content-dependent via piece count).

Everything else (FEN-anchored 180deg orientation fix, pairing, QA) is reused verbatim
from build_paired_dataset.py.

OUTPUT
------
datasets/chess_paired_v2/
    train/  game{N}_frame_{ID}_{viewpoint}.png   (1024x512: synthetic | real_canon)
    test/   ...
    _qa/    montage.png, color_canon_qa.png, stats.txt, pairs.csv, ref.txt

Runs locally on CPU.  Deps: numpy, Pillow, tqdm  (NO skimage/cv2 -- Lab is implemented
in numpy here and round-trips to ~1e-15).
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# Reuse the proven Wave-1 helpers (FEN parsing, orientation fix, IO, QA writers).
from build_paired_dataset import (
    REAL_RE, SPLITS, CROP, COLOR_WEIGHT, LOWCONF_CORR,
    fen_to_grids, orient_grid, load_fen_index, lookup_fen,
    cell_grids, choose_orientation, occupancy_match_rate,
    load_gray_array, load_rgb, maybe_rot180, resolve_data_dir,
)

# ----------------------------------------------------------------------------- #
# sRGB <-> CIELab (D65), numpy-only. Validated round-trip err ~1e-15.
# ----------------------------------------------------------------------------- #
_M = np.array([[0.4124, 0.3576, 0.1805],
               [0.2126, 0.7152, 0.0722],
               [0.0193, 0.1192, 0.9505]])
_Minv = np.linalg.inv(_M)
_Xn, _Yn, _Zn = 0.95047, 1.0, 1.08883
_d = 6.0 / 29.0


def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def _f(t):
    return np.where(t > _d ** 3, np.cbrt(t), t / (3 * _d * _d) + 4.0 / 29.0)


def _finv(t):
    return np.where(t > _d, t ** 3, 3 * _d * _d * (t - 4.0 / 29.0))


def rgb2lab(rgb):
    """rgb float[0,1] HxWx3 -> Lab."""
    lin = _srgb_to_linear(rgb)
    xyz = lin @ _M.T
    fx = _f(xyz[..., 0] / _Xn)
    fy = _f(xyz[..., 1] / _Yn)
    fz = _f(xyz[..., 2] / _Zn)
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], -1)


def lab2rgb(lab):
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    xyz = np.stack([_finv(fx) * _Xn, _finv(fy) * _Yn, _finv(fz) * _Zn], -1)
    lin = xyz @ _Minv.T
    return _linear_to_srgb(lin)


# ----------------------------------------------------------------------------- #
# Colour canonicalisation (Reinhard on chroma a*,b*; L* preserved)
# ----------------------------------------------------------------------------- #
def lab_chroma_stats(im_rgb_u8: np.ndarray):
    """Per-image (a_mean,a_std,b_mean,b_std) in Lab."""
    lab = rgb2lab(im_rgb_u8.astype(np.float32) / 255.0)
    a, b = lab[..., 1], lab[..., 2]
    return float(a.mean()), float(a.std()), float(b.mean()), float(b.std())


def canonicalize_color(im_rgb_u8: np.ndarray, ref, std_clip=(0.6, 1.6)) -> np.ndarray:
    """Shift/scale a*,b* of an image to the reference (a_mean,a_std,b_mean,b_std).
    L* is left unchanged so exposure/contrast and piece darkness are preserved."""
    a_rm, a_rs, b_rm, b_rs = ref
    lab = rgb2lab(im_rgb_u8.astype(np.float32) / 255.0)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    a_m, a_s = a.mean(), a.std() + 1e-6
    b_m, b_s = b.mean(), b.std() + 1e-6
    a_scale = np.clip(a_rs / a_s, *std_clip)
    b_scale = np.clip(b_rs / b_s, *std_clip)
    a2 = (a - a_m) * a_scale + a_rm
    b2 = (b - b_m) * b_scale + b_rm
    out = lab2rgb(np.stack([L, a2, b2], -1))
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def compute_reference(data_dir: Path, ref_games, fen_index):
    """Robust (median) a*,b* mean/std over the chosen warm reference games' REAL
    targets (training split)."""
    real_dir = data_dir / SPLITS["train"][1]
    means_a, stds_a, means_b, stds_b = [], [], [], []
    for real_path in sorted(real_dir.iterdir()):
        m = REAL_RE.match(real_path.name)
        if not m:
            continue
        if int(m.group(1)) not in ref_games:
            continue
        im = np.asarray(load_rgb(real_path))
        am, as_, bm, bs = lab_chroma_stats(im)
        means_a.append(am); stds_a.append(as_); means_b.append(bm); stds_b.append(bs)
    if not means_a:
        raise RuntimeError(f"No reference images found for games {ref_games}")
    ref = (float(np.median(means_a)), float(np.median(stds_a)),
           float(np.median(means_b)), float(np.median(stds_b)))
    return ref, len(means_a)


# ----------------------------------------------------------------------------- #
# Build
# ----------------------------------------------------------------------------- #
def build(project_dir, data_dir, out_dir, tile, ref_games, do_color, qa_samples):
    fen_index = load_fen_index(project_dir)
    if not fen_index:
        sys.exit("No game*.csv FEN files found - cannot anchor orientation.")

    ref, ref_n = (None, 0)
    if do_color:
        ref, ref_n = compute_reference(data_dir, ref_games, fen_index)
        print(f"colour reference (games {sorted(ref_games)}, n={ref_n}): "
              f"a*~N({ref[0]:.2f},{ref[1]:.2f})  b*~N({ref[2]:.2f},{ref[3]:.2f})")

    out_dir.mkdir(parents=True, exist_ok=True)
    qa_dir = out_dir / "_qa"
    (qa_dir / "examples").mkdir(parents=True, exist_ok=True)

    stats = {"pairs": 0, "syn_rot180": 0, "real_rot180": 0, "lowconf": 0,
             "match_syn_sum": 0.0, "match_real_sum": 0.0}
    skipped, pair_records, qa_pool, canon_qa = [], [], [], []
    b_before, b_after = [], []  # b* spread tracking (per game)

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
                skipped.append(("no_fen", real_path.name)); continue
            syn_name = f"game{game}_frame_{frame:06d}_{CROP}_{viewpoint}.png"
            syn_path = syn_dir / syn_name
            if not syn_path.exists():
                skipped.append(("no_synthetic", real_path.name)); continue
            try:
                occ_std, color_std = fen_to_grids(fen)
            except ValueError as e:
                skipped.append((f"bad_fen:{e}", real_path.name)); continue

            fen_occ = orient_grid(occ_std, viewpoint)
            fen_color = orient_grid(color_std, viewpoint)

            syn_gray = load_gray_array(syn_path)
            real_gray = load_gray_array(real_path)
            s_occ, s_lum = cell_grids(syn_gray)
            r_occ, r_lum = cell_grids(real_gray)
            s_rot, s_oc, s_cc = choose_orientation(s_occ, s_lum, fen_occ, fen_color)
            r_rot, r_oc, r_cc = choose_orientation(r_occ, r_lum, fen_occ, fen_color)

            s_occ_o = np.rot90(s_occ, 2) if s_rot else s_occ
            r_occ_o = np.rot90(r_occ, 2) if r_rot else r_occ
            s_match = occupancy_match_rate(s_occ_o, fen_occ)
            r_match = occupancy_match_rate(r_occ_o, fen_occ)
            lowconf = min(s_oc + COLOR_WEIGHT * s_cc, r_oc + COLOR_WEIGHT * r_cc) < LOWCONF_CORR

            # synthetic input half (oriented, resized; no colour change)
            syn_im = maybe_rot180(load_rgb(syn_path), s_rot).resize((tile, tile), Image.BICUBIC)

            # real target half (oriented), colour-canonicalised, then resized
            real_oriented = np.asarray(maybe_rot180(load_rgb(real_path), r_rot))
            _, _, b_pre, _ = lab_chroma_stats(real_oriented)
            if do_color:
                real_canon = canonicalize_color(real_oriented, ref)
            else:
                real_canon = real_oriented
            _, _, b_post, _ = lab_chroma_stats(real_canon)
            b_before.append((game, b_pre)); b_after.append((game, b_post))
            real_im = Image.fromarray(real_canon).resize((tile, tile), Image.BICUBIC)

            combined = Image.new("RGB", (2 * tile, tile))
            combined.paste(syn_im, (0, 0))
            combined.paste(real_im, (tile, 0))
            out_name = f"game{game}_frame_{frame:06d}_{viewpoint}.png"
            combined.save(out_dir / split / out_name)

            stats["pairs"] += 1
            stats["syn_rot180"] += int(s_rot); stats["real_rot180"] += int(r_rot)
            stats["lowconf"] += int(lowconf)
            stats["match_syn_sum"] += s_match; stats["match_real_sum"] += r_match
            n_pieces = int(occ_std.sum())
            pair_records.append([split, out_name, game, frame, viewpoint, n_pieces,
                                 int(s_rot), int(r_rot), round(s_oc, 3), round(s_cc, 3),
                                 round(r_oc, 3), round(r_cc, 3), round(s_match, 3),
                                 round(r_match, 3), int(lowconf), round(b_pre, 2), round(b_post, 2)])
            if len(qa_pool) < 4000:
                qa_pool.append((split, out_name, n_pieces, combined.copy()))
            # collect a few before/after colour QA strips (real only)
            if do_color and split == "test" and len(canon_qa) < 6:
                before_t = Image.fromarray(real_oriented).resize((tile, tile), Image.BICUBIC)
                canon_qa.append((out_name, before_t, real_im.copy()))

    write_qa_v2(qa_dir, stats, skipped, pair_records, qa_pool, canon_qa,
                qa_samples, ref, ref_games, b_before, b_after, do_color, tile)
    n = max(stats["pairs"], 1)
    print("\n" + "=" * 64)
    print("PAIRED DATASET v2 BUILD COMPLETE")
    print("=" * 64)
    print(f"  pairs built            : {stats['pairs']}  (tile={tile}, color_canon={do_color})")
    print(f"  synthetic rotated 180  : {stats['syn_rot180']} ({100*stats['syn_rot180']/n:.1f}%)")
    print(f"  real rotated 180       : {stats['real_rot180']} ({100*stats['real_rot180']/n:.1f}%)")
    print(f"  mean occ-match (syn)   : {stats['match_syn_sum']/n:.3f}")
    print(f"  mean occ-match (real)  : {stats['match_real_sum']/n:.3f}")
    print(f"  low-confidence pairs   : {stats['lowconf']}")
    print(f"  skipped                : {len(skipped)}")
    print(f"\n  output : {out_dir}")
    print("=" * 64)
    return stats


def _b_spread_by_game(rows):
    g = defaultdict(list)
    for game, b in rows:
        g[game].append(b)
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in sorted(g.items())}


def write_qa_v2(qa_dir, stats, skipped, pair_records, qa_pool, canon_qa,
                qa_samples, ref, ref_games, b_before, b_after, do_color, tile):
    with open(qa_dir / "pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "file", "game", "frame", "viewpoint", "n_pieces",
                    "syn_rot180", "real_rot180", "syn_occ_corr", "syn_color_corr",
                    "real_occ_corr", "real_color_corr", "syn_match", "real_match",
                    "lowconf", "b_before", "b_after"])
        w.writerows(pair_records)
    with open(qa_dir / "skipped.txt", "w") as f:
        for reason, name in skipped:
            f.write(f"{reason}\t{name}\n")

    bb = _b_spread_by_game(b_before)
    ba = _b_spread_by_game(b_after)
    with open(qa_dir / "stats.txt", "w") as f:
        n = max(stats["pairs"], 1)
        f.write(f"pairs_built          : {stats['pairs']}\n")
        f.write(f"tile                 : {tile}\n")
        f.write(f"color_canon          : {do_color}\n")
        if do_color:
            f.write(f"ref_games            : {sorted(ref_games)}\n")
            f.write(f"ref a*               : mean={ref[0]:.2f} std={ref[1]:.2f}\n")
            f.write(f"ref b*               : mean={ref[2]:.2f} std={ref[3]:.2f}\n")
        f.write(f"synthetic_rot180     : {stats['syn_rot180']} ({100*stats['syn_rot180']/n:.1f}%)\n")
        f.write(f"real_rot180          : {stats['real_rot180']} ({100*stats['real_rot180']/n:.1f}%)\n")
        f.write(f"mean_occ_match_syn   : {stats['match_syn_sum']/n:.3f}\n")
        f.write(f"mean_occ_match_real  : {stats['match_real_sum']/n:.3f}\n")
        f.write(f"low_confidence_pairs : {stats['lowconf']}\n")
        f.write(f"skipped              : {len(skipped)}\n")
        f.write("\n--- b* (warm<->cool) spread by game: BEFORE -> AFTER canon ---\n")
        all_before = [b for _, b in b_before]
        all_after = [b for _, b in b_after]
        for g in sorted(bb):
            f.write(f"  game{g}: {bb[g][0]:6.2f}(+-{bb[g][1]:.2f})  ->  {ba[g][0]:6.2f}(+-{ba[g][1]:.2f})\n")
        f.write(f"  ALL  : mean={np.mean(all_before):.2f} std={np.std(all_before):.2f}"
                f"  ->  mean={np.mean(all_after):.2f} std={np.std(all_after):.2f}\n")

    # colour before/after QA strip
    if canon_qa:
        pad = 6
        cols = 2
        rows = len(canon_qa)
        cw, ch = tile + pad, tile + 18 + pad
        canvas = Image.new("RGB", (cols * cw + pad, rows * ch + pad), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)
        for i, (name, before_im, after_im) in enumerate(canon_qa):
            y = pad + i * ch
            draw.text((pad + 2, y + 2), f"{name}  BEFORE", fill=(230, 230, 230))
            draw.text((pad + cw + 2, y + 2), "AFTER canon", fill=(230, 230, 230))
            canvas.paste(before_im, (pad, y + 18))
            canvas.paste(after_im, (pad + cw, y + 18))
        canvas.save(qa_dir / "color_canon_qa.png")

    # montage spread sparse..dense
    if qa_pool:
        qa_pool_sorted = sorted(qa_pool, key=lambda t: t[2])
        k = min(qa_samples, len(qa_pool_sorted))
        idx = np.linspace(0, len(qa_pool_sorted) - 1, k).astype(int)
        chosen = [qa_pool_sorted[i] for i in idx]
        cols = 4
        rows = (k + cols - 1) // cols
        tw, th = 2 * 128, 128
        pad, lh = 6, 14
        cw, ch = tw + pad, th + lh + pad
        canvas = Image.new("RGB", (cols * cw + pad, rows * ch + pad), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)
        for n_i, (split, name, n_pieces, comb) in enumerate(chosen):
            r, c = divmod(n_i, cols)
            x, y = pad + c * cw, pad + r * ch
            canvas.paste(comb.resize((tw, th), Image.BICUBIC), (x, y + lh))
            draw.text((x + 2, y + 2), f"{split} {name.replace('game','g').replace('_frame_','-')[:26]} p{n_pieces}", fill=(230, 230, 230))
            if n_i < 8:
                comb.save(qa_dir / "examples" / f"example_{n_i:02d}_{name}")
        canvas.save(qa_dir / "montage.png")


def main():
    project_dir = Path(__file__).parent.resolve()
    ap = argparse.ArgumentParser(description="Build paired chess dataset v2 (512px + colour canon).")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-dir", default=str(project_dir / "datasets" / "chess_paired_v2"))
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--ref-games", default="5,6", help="warm games used to define the colour reference")
    ap.add_argument("--no-color-canon", action="store_true", help="ablation: skip colour canonicalisation")
    ap.add_argument("--qa-samples", type=int, default=24)
    args = ap.parse_args()

    data_dir = resolve_data_dir(project_dir, args.data_dir)
    ref_games = {int(x) for x in args.ref_games.split(",") if x.strip()}
    out_dir = Path(args.out_dir)
    print(f"project : {project_dir}\ndata    : {data_dir}\nout     : {out_dir}\n")
    build(project_dir, data_dir, out_dir, args.tile, ref_games,
          not args.no_color_canon, args.qa_samples)


if __name__ == "__main__":
    main()
