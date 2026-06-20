"""
piece_crop_sheet.py -- per-piece-type legibility montages.

For each of the 12 piece types, tile occupied-square crops as
[ synthetic | generated | real ] triplets so a human can judge whether the
GENERATED piece is recognizable as the correct type (the honest legibility judge
the diagnosis asks for -- the per-square classifier score overstates quality).

Crops use the SAME 112px window + regular cell grid as square_eval.py, so the
sheets line up with the scored crops. PIL/numpy only (no torch needed).

Usage:
  python piece_crop_sheet.py \
    --data ./datasets/chess_v5_oblique \
    --images ./results/<NAME>/test_latest/images/fake_B \
    --out ./piece_sheets_<NAME> [--split test --per-type 30 --cols 6]
"""
import os, re, json, glob, argparse
from PIL import Image, ImageDraw

CLASSES = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
NAMES = {'P': 'white_pawn', 'N': 'white_knight', 'B': 'white_bishop', 'R': 'white_rook',
         'Q': 'white_queen', 'K': 'white_king', 'p': 'black_pawn', 'n': 'black_knight',
         'b': 'black_bishop', 'r': 'black_rook', 'q': 'black_queen', 'k': 'black_king'}
CROP = 112  # window around a cell centre (matches square_eval.py)


def cell_center(f, rank, vp, S=512):
    sq = S / 8.0
    if vp == 'white':
        return (f + 0.5) * sq, (8 - rank + 0.5) * sq
    return (7 - f + 0.5) * sq, (rank - 1 + 0.5) * sq


def fen_grid(fen):
    g = {}
    for ri, row in enumerate(fen.split()[0].split('/')):
        rank = 8 - ri; f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
            else:
                g[(f, rank)] = ch; f += 1
    return g


def crop_square(img, f, rank, vp, out=96):
    cx, cy = cell_center(f, rank, vp, img.width)
    half = CROP // 2
    return img.crop((cx - half, cy - half, cx + half, cy + half)).resize((out, out), Image.BILINEAR)


def load_halves(data, split, name):
    p = os.path.join(data, split, name + '.png')
    if not os.path.exists(p):
        return None, None
    comb = Image.open(p).convert('RGB'); W = comb.width // 2
    return comb.crop((0, 0, W, comb.height)), comb.crop((W, 0, 2 * W, comb.height))  # synthetic A, real B


def build_sheet(triplets, title, cols, out_path, cell=96, gap=4, label_h=18):
    n = len(triplets)
    if n == 0:
        return False
    rows = (n + cols - 1) // cols
    trip_w = cell * 3 + gap * 2          # syn | gen | real
    cw = trip_w + gap
    ch = cell + label_h + gap
    W = cols * cw + gap
    H = rows * ch + label_h + gap
    canvas = Image.new('RGB', (W, H), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 3), f'{title}   (synthetic | GENERATED | real)   n={n}', fill=(0, 0, 0))
    for i, (syn, gen, real) in enumerate(triplets):
        r, c = divmod(i, cols)
        x = gap + c * cw
        y = label_h + gap + r * ch
        for k, im in enumerate((syn, gen, real)):
            canvas.paste(im, (x + k * (cell + gap), y + label_h))
        draw.text((x, y + 2), 'syn   gen   real', fill=(40, 40, 40))
    canvas.save(out_path)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--images', required=True, help='generated fake_B dir')
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--per-type', type=int, default=30)
    ap.add_argument('--cols', type=int, default=6)
    ap.add_argument('--cell', type=int, default=96)
    a = ap.parse_args()

    labels = json.load(open(os.path.join(a.data, 'labels.json')))
    os.makedirs(a.out, exist_ok=True)
    buckets = {c: [] for c in CLASSES}

    gen_files = sorted(glob.glob(os.path.join(a.images, '*.png')) + glob.glob(os.path.join(a.images, '*.jpg')))
    for gp in gen_files:
        nm = re.sub(r'_(fake_B|real_B|real_A)$', '', os.path.basename(gp)[:-4])
        if nm not in labels:
            continue
        vp = labels[nm]['viewpoint']; grid = fen_grid(labels[nm]['fen'])
        syn_img, real_img = load_halves(a.data, a.split, nm)
        if syn_img is None:
            continue
        gen_img = Image.open(gp).convert('RGB')
        for (f, rank), ch in grid.items():
            if all(len(buckets[c]) >= a.per_type for c in CLASSES):
                break
            if len(buckets[ch]) >= a.per_type:
                continue
            buckets[ch].append((
                crop_square(syn_img, f, rank, vp, a.cell),
                crop_square(gen_img, f, rank, vp, a.cell),
                crop_square(real_img, f, rank, vp, a.cell),
            ))

    made = []
    for c in CLASSES:
        out_path = os.path.join(a.out, f'piece_sheet_{NAMES[c]}.png')
        if build_sheet(buckets[c], f'{c}  {NAMES[c]}', a.cols, out_path, cell=a.cell):
            made.append(out_path)
            print(f'[{c}] {NAMES[c]:14s} n={len(buckets[c]):3d} -> {out_path}')
        else:
            print(f'[{c}] {NAMES[c]:14s} n=0 (none found)')
    print(f'[done] wrote {len(made)} sheets into {a.out}')


if __name__ == '__main__':
    main()
