"""Audit generated chess pieces for double-head / multi-lobe head artifacts.

This is intentionally a visual-audit helper, not a final oracle. It scores each
occupied square with several simple cues, writes per-piece CSV/JSON, and saves
worst-case crop sheets sorted by score so the metric can be checked by eye.

Usage:
  python3 double_head_audit.py \
    --ds datasets/chess_v5_oblique_aligned_bright \
    --real v5_work/audit_baseline_pieceD/real_B \
    --variant bright_silAB=v5_work/eval_noC_silAB/fake_B \
    --variant bright_silABC=v5_work/eval_bright/fake_B \
    --out-dir v5_work/double_head_audit
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sys

sys.path.insert(0, os.path.dirname(__file__))
from detect_head_artifacts import ID2P, WIN, fen_grid, lobes, lum, sids  # noqa: E402


PADDING = (235, 235, 235)
DISP = 132


@dataclass
class PieceScore:
    variant: str
    file: str
    piece: str
    row: int
    col: int
    neighbours: int
    score: float
    vertical_extra: int
    horizontal_extra: int
    component_extra: int
    fake_vlobes: int
    real_vlobes: int
    fake_hlobes: int
    real_hlobes: int
    fake_components: int
    real_components: int
    fake_head_width: float
    real_head_width: float
    crop_win: int


def parse_variant(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--variant must be NAME=DIR")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("variant name is empty")
    return name, Path(path)


def crop_with_pad(img: np.ndarray, cy: int, cx: int, win: int, fill=PADDING) -> np.ndarray:
    """Fixed-size crop centered at (cy,cx), filling off-image area neutrally."""
    h = win // 2
    out = np.empty((win, win, 3), dtype=np.uint8)
    out[:, :] = np.asarray(fill, dtype=np.uint8)
    y0, y1 = cy - h, cy + h
    x0, x1 = cx - h, cx + h
    sy0, sy1 = max(0, y0), min(img.shape[0], y1)
    sx0, sx1 = max(0, x0), min(img.shape[1], x1)
    if sy1 <= sy0 or sx1 <= sx0:
        return out
    dy0, dx0 = sy0 - y0, sx0 - x0
    out[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = img[sy0:sy1, sx0:sx1]
    return out


def crop_mask_with_pad(mask: np.ndarray, cy: int, cx: int, win: int) -> np.ndarray:
    h = win // 2
    out = np.zeros((win, win), dtype=mask.dtype)
    y0, y1 = cy - h, cy + h
    x0, x1 = cx - h, cx + h
    sy0, sy1 = max(0, y0), min(mask.shape[0], y1)
    sx0, sx1 = max(0, x0), min(mask.shape[1], x1)
    if sy1 <= sy0 or sx1 <= sx0:
        return out
    dy0, dx0 = sy0 - y0, sx0 - x0
    out[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = mask[sy0:sy1, sx0:sx1]
    return out


def resize(img: np.ndarray, size: int = DISP) -> Image.Image:
    return Image.fromarray(img).resize((size, size), Image.Resampling.BILINEAR)


def run_count(mask: np.ndarray, min_len: int = 3) -> int:
    runs = 0
    cur = 0
    for v in mask.astype(bool):
        if v:
            cur += 1
        else:
            if cur >= min_len:
                runs += 1
            cur = 0
    if cur >= min_len:
        runs += 1
    return runs


def largest_runs_width(mask: np.ndarray, min_len: int = 3) -> int:
    best = 0
    cur = 0
    for v in mask.astype(bool):
        if v:
            cur += 1
        else:
            if cur >= min_len:
                best = max(best, cur)
            cur = 0
    if cur >= min_len:
        best = max(best, cur)
    return best


def components(mask: np.ndarray, min_area: int = 10) -> int:
    """8-connected component count without scipy."""
    mask = mask.astype(bool)
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    count = 0
    for y in range(h):
        xs = np.where(mask[y] & ~seen[y])[0]
        for x0 in xs:
            if seen[y, x0] or not mask[y, x0]:
                continue
            stack = [(y, int(x0))]
            seen[y, x0] = True
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                for ny in range(max(0, cy - 1), min(h, cy + 2)):
                    for nx in range(max(0, cx - 1), min(w, cx + 2)):
                        if not seen[ny, nx] and mask[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            if area >= min_area:
                count += 1
    return count


def board_rgb(crop: np.ndarray, piece_mask: np.ndarray) -> np.ndarray:
    board = crop[~piece_mask]
    if board.size == 0:
        board = crop.reshape(-1, 3)
    return np.median(board.reshape(-1, 3), axis=0)


def foreground_mask(crop: np.ndarray, piece_mask: np.ndarray, board: np.ndarray) -> np.ndarray:
    diff = crop.astype(np.float32) - board.reshape(1, 1, 3).astype(np.float32)
    dist = np.sqrt((diff * diff).sum(axis=2))
    bg = dist[~piece_mask]
    # Board texture can be busy, so set the threshold relative to the local board.
    thresh = max(18.0, float(np.percentile(bg, 88)) + 5.0) if bg.size else 18.0
    return dist > thresh


def head_roi(mask: np.ndarray, piece: str) -> tuple[slice, slice, int, int, int, int]:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        h, w = mask.shape
        return slice(0, h), slice(0, w), 0, h - 1, 0, w - 1
    top, bot = int(ys.min()), int(ys.max())
    left, right = int(xs.min()), int(xs.max())
    height = bot - top + 1
    # Pawns' head is a large fraction of the visible silhouette; officers need
    # enough room for mitres/crowns but not the base stack.
    frac = 0.50 if piece in ("P", "p") else 0.42
    y1 = min(mask.shape[0], top + max(16, int(height * frac)))
    pad_x = max(6, int((right - left + 1) * 0.18))
    x0 = max(0, left - pad_x)
    x1 = min(mask.shape[1], right + pad_x + 1)
    return slice(top, y1), slice(x0, x1), top, bot, left, right


def vertical_lobes(crop: np.ndarray, mask: np.ndarray, board_luma: float) -> int:
    L = lum(crop.astype(np.float32))
    if not mask.any():
        return 0
    cols = np.where(mask.any(axis=0))[0]
    rows = np.where(mask.any(axis=1))[0]
    band = slice(int(cols.min()), int(cols.max()) + 1)
    prof = np.array(
        [L[i, band][mask[i, band]].mean() if mask[i, band].any() else np.nan for i in range(mask.shape[0])]
    )
    top, bot = int(rows.min()), int(rows.max())
    p = np.where(np.isnan(prof[top:bot + 1]), board_luma, prof[top:bot + 1])
    runs, _ = lobes(p, board_luma)
    return runs


def split_metrics(crop: np.ndarray, mask: np.ndarray, piece: str) -> tuple[int, int, float]:
    if not mask.any():
        return 0, 0, 0.0
    board = board_rgb(crop, mask)
    fg = foreground_mask(crop, mask, board)
    ysl, xsl, *_ = head_roi(mask, piece)
    head_fg = fg[ysl, xsl]
    head_mask = mask[ysl, xsl]
    if head_fg.size == 0:
        return 0, 0, 0.0

    # Restrict the strongest split cues to the expected head neighbourhood, but
    # allow small excursions outside the exact silhouette to catch side ghosts.
    col_signal = (head_fg & np.pad(head_mask[:, 1:-1] if head_mask.shape[1] > 2 else head_mask,
                                   ((0, 0), (1, 1)), mode="edge") if head_mask.shape[1] > 2 else head_fg)
    profile = col_signal.mean(axis=0)
    if profile.max() <= 0:
        h_lobes = 0
        width = 0.0
    else:
        active = profile >= max(0.10, profile.max() * 0.42)
        h_lobes = run_count(active, min_len=3)
        width = float(largest_runs_width(active, min_len=3)) / max(1, active.size)

    comp_mask = head_fg & np.pad(head_mask[:, 1:-1] if head_mask.shape[1] > 2 else head_mask,
                                 ((0, 0), (1, 1)), mode="edge") if head_mask.shape[1] > 2 else head_fg
    comp = components(comp_mask, min_area=max(8, head_fg.size // 180))
    return h_lobes, comp, width


def piece_neighbours(grid: np.ndarray, row: int, col: int) -> int:
    count = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        rr, cc = row + dr, col + dc
        if 0 <= rr < 8 and 0 <= cc < 8 and grid[rr, cc] >= 3:
            count += 1
    return count


def score_piece(
    variant: str,
    filename: str,
    piece: str,
    row: int,
    col: int,
    neighbours: int,
    fake_crop: np.ndarray,
    real_crop: np.ndarray,
    mask_crop: np.ndarray,
    win: int,
) -> PieceScore | None:
    piece_mask = mask_crop >= 3
    target_mask = mask_crop == next(k for k, v in ID2P.items() if v == piece)
    if target_mask.sum() >= 25:
        piece_mask = target_mask
    if piece_mask.sum() < 25:
        return None

    fake_board = board_rgb(fake_crop, piece_mask)
    real_board = board_rgb(real_crop, piece_mask)
    fake_v = vertical_lobes(fake_crop, piece_mask, float(lum(fake_board.reshape(1, 1, 3))[0, 0]))
    real_v = vertical_lobes(real_crop, piece_mask, float(lum(real_board.reshape(1, 1, 3))[0, 0]))
    fake_h, fake_comp, fake_width = split_metrics(fake_crop, piece_mask, piece)
    real_h, real_comp, real_width = split_metrics(real_crop, piece_mask, piece)

    vertical_extra = max(0, fake_v - real_v)
    horizontal_extra = max(0, fake_h - real_h)
    component_extra = max(0, fake_comp - real_comp)
    spread_extra = max(0.0, fake_width - real_width)
    # Weighted for ranking inspectable cases, not for claiming absolute truth.
    score = (
        2.0 * vertical_extra
        + 2.25 * horizontal_extra
        + 1.5 * component_extra
        + 1.25 * spread_extra
        + 0.15 * neighbours
    )
    return PieceScore(
        variant=variant,
        file=filename,
        piece=piece,
        row=row,
        col=col,
        neighbours=neighbours,
        score=round(float(score), 4),
        vertical_extra=int(vertical_extra),
        horizontal_extra=int(horizontal_extra),
        component_extra=int(component_extra),
        fake_vlobes=int(fake_v),
        real_vlobes=int(real_v),
        fake_hlobes=int(fake_h),
        real_hlobes=int(real_h),
        fake_components=int(fake_comp),
        real_components=int(real_comp),
        fake_head_width=round(float(fake_width), 4),
        real_head_width=round(float(real_width), 4),
        crop_win=int(win),
    )


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill=(0, 0, 0)) -> None:
    draw.text(xy, text, fill=fill, font=ImageFont.load_default())


def make_panel(
    rows: list[PieceScore],
    variant_dir: Path,
    real_dir: Path,
    ds_dir: Path,
    labels: dict,
    out_path: Path,
    max_rows: int,
) -> None:
    if not rows:
        return
    cells = []
    header_h = 32
    gap = 4
    cell_w = DISP * 3 + gap * 2
    for rank, row in enumerate(rows[:max_rows], 1):
        name = row.file[:-4]
        meta = labels[name]
        ds_img = np.asarray(Image.open(ds_dir / meta["split"] / f"{name}.png").convert("RGB"))
        syn = ds_img[:, : ds_img.shape[1] // 2]
        fake = np.asarray(Image.open(variant_dir / row.file).convert("RGB"))
        real = np.asarray(Image.open(real_dir / row.file).convert("RGB"))
        cell = fake.shape[0] // 8
        cy = row.row * cell + cell // 2
        cx = row.col * cell + cell // 2
        syn_c = resize(crop_with_pad(syn, cy, cx, row.crop_win))
        fake_c = resize(crop_with_pad(fake, cy, cx, row.crop_win))
        real_c = resize(crop_with_pad(real, cy, cx, row.crop_win))
        panel = Image.new("RGB", (cell_w, header_h + DISP), "white")
        d = ImageDraw.Draw(panel)
        draw_label(d, (4, 2), f"#{rank} {row.piece}@{row.row},{row.col} s={row.score:.2f} n={row.neighbours}")
        draw_label(
            d,
            (4, 16),
            f"v {row.fake_vlobes}>{row.real_vlobes}  h {row.fake_hlobes}>{row.real_hlobes}  c {row.fake_components}>{row.real_components}",
        )
        for x, title, im in [(0, "syn", syn_c), (DISP + gap, "gen", fake_c), ((DISP + gap) * 2, "real", real_c)]:
            panel.paste(im, (x, header_h))
            draw_label(d, (x + 3, header_h + 3), title, fill=(20, 20, 20))
        cells.append(panel)

    per_row = 2
    rows_img = []
    for i in range(0, len(cells), per_row):
        chunk = cells[i:i + per_row]
        row_img = Image.new("RGB", (per_row * cell_w + (per_row - 1) * 12, header_h + DISP), "black")
        for j, panel in enumerate(chunk):
            row_img.paste(panel, (j * (cell_w + 12), 0))
        rows_img.append(row_img)
    out = Image.new("RGB", (rows_img[0].width, len(rows_img) * rows_img[0].height + (len(rows_img) - 1) * 10), "black")
    y = 0
    for row_img in rows_img:
        out.paste(row_img, (0, y))
        y += row_img.height + 10
    out.save(out_path)


def select_diverse(rows: list[PieceScore], max_rows: int, *, max_per_grid_key: int = 1) -> list[PieceScore]:
    selected: list[PieceScore] = []
    grid_counts: dict[tuple[str, int, int], int] = defaultdict(int)
    file_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        grid_key = (row.piece, row.row, row.col)
        if grid_counts[grid_key] >= max_per_grid_key:
            continue
        if file_counts[row.file] >= 2:
            continue
        selected.append(row)
        grid_counts[grid_key] += 1
        file_counts[row.file] += 1
        if len(selected) >= max_rows:
            break
    return selected


def summarize(rows: list[PieceScore]) -> dict:
    if not rows:
        return {}
    out = {"pieces": len(rows)}
    for label, pred in [
        ("all", lambda r: True),
        ("pawns", lambda r: r.piece in ("P", "p")),
        ("officers", lambda r: r.piece not in ("P", "p")),
        ("white", lambda r: r.piece.isupper()),
        ("black", lambda r: r.piece.islower()),
        ("isolated", lambda r: r.neighbours == 0),
        ("crowded", lambda r: r.neighbours > 0),
    ]:
        subset = [r for r in rows if pred(r)]
        if not subset:
            continue
        out[label] = {
            "n": len(subset),
            "score_mean": round(float(np.mean([r.score for r in subset])), 4),
            "score_p90": round(float(np.percentile([r.score for r in subset], 90)), 4),
            "vertical_extra_rate": round(float(np.mean([r.vertical_extra > 0 for r in subset])), 4),
            "horizontal_extra_rate": round(float(np.mean([r.horizontal_extra > 0 for r in subset])), 4),
            "component_extra_rate": round(float(np.mean([r.component_extra > 0 for r in subset])), 4),
            "any_extra_rate": round(float(np.mean([
                r.vertical_extra > 0 or r.horizontal_extra > 0 or r.component_extra > 0 for r in subset
            ])), 4),
        }
    by_piece = {}
    for piece in "PNBRQKpnbrqk":
        subset = [r for r in rows if r.piece == piece]
        if subset:
            by_piece[piece] = {
                "n": len(subset),
                "score_mean": round(float(np.mean([r.score for r in subset])), 4),
                "any_extra_rate": round(float(np.mean([
                    r.vertical_extra > 0 or r.horizontal_extra > 0 or r.component_extra > 0 for r in subset
                ])), 4),
            }
    out["by_piece"] = by_piece
    return out


def audit_variant(variant: str, fake_dir: Path, real_dir: Path, ds_dir: Path, labels: dict) -> list[PieceScore]:
    files = sorted(p.name for p in fake_dir.glob("*.png"))
    rows: list[PieceScore] = []
    id_for_piece = {v: k for k, v in ID2P.items()}
    for file in files:
        name = file[:-4]
        if name not in labels:
            continue
        meta = labels[name]
        seg_path = ds_dir / meta["split"] / f"{name}_seg.png"
        if not seg_path.exists():
            continue
        fake = np.asarray(Image.open(fake_dir / file).convert("RGB"))
        real = np.asarray(Image.open(real_dir / file).convert("RGB"))
        seg = sids(Image.open(seg_path))
        grid = fen_grid(meta["fen"], meta["viewpoint"])
        cell = fake.shape[0] // 8
        for r in range(8):
            for c in range(8):
                pid = int(grid[r, c])
                if pid < 3:
                    continue
                piece = ID2P[pid]
                win = WIN[piece]
                cy = r * cell + cell // 2
                cx = c * cell + cell // 2
                mask = crop_mask_with_pad(seg == id_for_piece[piece], cy, cx, win)
                if mask.sum() < 25:
                    mask = crop_mask_with_pad(seg >= 3, cy, cx, win)
                scored = score_piece(
                    variant=variant,
                    filename=file,
                    piece=piece,
                    row=r,
                    col=c,
                    neighbours=piece_neighbours(grid, r, c),
                    fake_crop=crop_with_pad(fake, cy, cx, win),
                    real_crop=crop_with_pad(real, cy, cx, win),
                    mask_crop=mask.astype(np.uint8) * pid,
                    win=win,
                )
                if scored:
                    rows.append(scored)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", required=True, type=Path)
    parser.add_argument("--real", required=True, type=Path)
    parser.add_argument("--variant", action="append", required=True, type=parse_variant)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-panel", type=int, default=32)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels = json.load(open(args.ds / "labels.json"))
    summary = {}
    for name, fake_dir in args.variant:
        print(f"auditing {name}: {fake_dir}")
        rows = audit_variant(name, fake_dir, args.real, args.ds, labels)
        rows_sorted = sorted(rows, key=lambda r: (r.score, r.horizontal_extra, r.vertical_extra), reverse=True)
        summary[name] = summarize(rows)

        csv_path = args.out_dir / f"{name}_piece_scores.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(rows_sorted[0]).keys()))
            writer.writeheader()
            for row in rows_sorted:
                writer.writerow(asdict(row))
        with open(args.out_dir / f"{name}_piece_scores.json", "w") as f:
            json.dump([asdict(r) for r in rows_sorted], f, indent=2)
        make_panel(rows_sorted, fake_dir, args.real, args.ds, labels, args.out_dir / f"{name}_worst_heads.png", args.max_panel)
        make_panel(
            select_diverse(rows_sorted, args.max_panel),
            fake_dir,
            args.real,
            args.ds,
            labels,
            args.out_dir / f"{name}_worst_heads_diverse.png",
            args.max_panel,
        )
        for suffix, pred in [
            ("pawns_diverse", lambda r: r.piece in ("P", "p")),
            ("officers_diverse", lambda r: r.piece not in ("P", "p")),
            ("isolated_diverse", lambda r: r.neighbours == 0),
        ]:
            subset = [r for r in rows_sorted if pred(r)]
            make_panel(
                select_diverse(subset, args.max_panel),
                fake_dir,
                args.real,
                args.ds,
                labels,
                args.out_dir / f"{name}_{suffix}.png",
                args.max_panel,
            )
        print(json.dumps(summary[name], indent=2))

    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved audit to {args.out_dir}")


if __name__ == "__main__":
    main()
