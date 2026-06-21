"""
build_paired_dataset_v3.py -- geometry-aligned paired dataset.

For every real photo (already board-rectified to a canonical 512^2 square), render
a view-aligned synthetic (oblique Blender render -> rectified to the same canonical
square) + a semantic class-id mask, all registered to the real target.

Outputs datasets/chess_paired_v3/{train,test}/:
  {name}.png      = [A_synthetic | B_real]  (1024x512)
  {name}_seg.png  = class-id mask (512, mode L, ids 1..14)

Run with system python3 (PIL/numpy). Invokes Blender for rendering.
"""
import os, re, json, glob, hashlib, subprocess, argparse
import numpy as np
from PIL import Image
from rectify_and_pack import rectify, canonical_targets  # reuse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
RAW = os.path.join(ROOT, "data from drive", "dataset")
CACHE = os.path.join(ROOT, "v3_pipeline", "cache")
OUT = os.path.join(ROOT, "datasets", "chess_paired_v3")
BLEND = os.path.join(ROOT, "v3_pipeline", "assets", "ChessScene.blend")
RENDER = os.path.join(ROOT, "v3_pipeline", "render_aligned.py")
ELEV, DIST, LENS, RES = 35.0, 9.0, 35.0, 768
W = 512

CLASS_ID = {"empty_light": 1, "empty_dark": 2}
for i, t in enumerate(["p", "n", "b", "r", "q", "k"]):
    CLASS_ID["w" + t] = 3 + i; CLASS_ID["b" + t] = 9 + i
def palette(cid):
    return np.array([(cid*17) % 256, (cid*53) % 256, (cid*97) % 256], float)
PIECE_IDS = list(range(3, 15))
PIECE_PAL = np.stack([palette(c) for c in PIECE_IDS])  # (12,3)

# ---------- FEN index ----------
def load_fen_index():
    idx = {}
    for csv in glob.glob(os.path.join(ROOT, "game*.csv")):
        g = int(re.search(r"game(\d+)", os.path.basename(csv)).group(1))
        rows = []
        for ln in open(csv).read().splitlines()[1:]:
            p = ln.split(",")
            if len(p) >= 3 and p[0].strip().isdigit():
                rows.append((int(p[0]), int(p[1]), p[2].strip()))
        idx[g] = sorted(rows)
    return idx
FEN_IDX = load_fen_index()
def lookup_fen(game, frame):
    rows = FEN_IDX.get(game, [])
    fen = None
    for fr, to, f in rows:
        if fr <= frame: fen = f
        else: break
    return fen or (rows[0][2] if rows else None)

# ---------- mask -> class-id (snap pieces, paint empty parity) ----------
def mask_to_ids(maskR_path):
    arr = np.asarray(Image.open(maskR_path).convert("RGB"), float)  # 512x512x3
    h, w, _ = arr.shape
    flat = arr.reshape(-1, 3)
    d = np.linalg.norm(flat[:, None, :] - PIECE_PAL[None, :, :], axis=2)  # (N,12)
    nn = d.argmin(1); nndist = d.min(1)
    is_piece = nndist < 40.0
    ids = np.zeros(h * w, np.uint8)
    ids[is_piece] = np.array(PIECE_IDS, np.uint8)[nn[is_piece]]
    ids = ids.reshape(h, w)
    # empty squares -> parity by canonical 64px grid
    yy, xx = np.mgrid[0:h, 0:w]
    parity = ((xx // (w // 8) + yy // (h // 8)) % 2)
    empty = ids == 0
    ids[empty & (parity == 0)] = CLASS_ID["empty_light"]
    ids[empty & (parity == 1)] = CLASS_ID["empty_dark"]
    return Image.fromarray(ids, mode="L")

# ---------- main ----------
def main(limit=None, render=True):
    os.makedirs(CACHE, exist_ok=True)
    for s in ("train", "test"):
        os.makedirs(os.path.join(OUT, s), exist_ok=True)
    files = []
    for split, sub in (("train", "trainB"), ("test", "testB")):
        for p in sorted(glob.glob(os.path.join(RAW, sub, "*.jpg"))):
            m = re.search(r"game(\d+)_frame_(\d+)_(white|black)", os.path.basename(p))
            if not m: continue
            g, fr, vp = int(m.group(1)), int(m.group(2)), m.group(3)
            fen = lookup_fen(g, fr)
            if fen is None: continue
            files.append(dict(split=split, path=p, game=g, frame=fr, vp=vp,
                              fen=fen, name=f"game{g}_frame_{fr:06d}_{vp}"))
    if limit: files = files[:limit]
    print(f"[files] {len(files)}")

    # unique (fen,vp) -> cache key + job
    uniq = {}
    for fr in files:
        key = hashlib.md5(f"{fr['fen']}|{fr['vp']}|{ELEV}|{DIST}|{LENS}".encode()).hexdigest()[:16]
        fr["key"] = key
        if key not in uniq:
            uniq[key] = dict(fen=fr["fen"], viewpoint=fr["vp"], out=os.path.join(CACHE, key),
                             elev=ELEV, dist=DIST, lens=LENS, res=RES)
    jobs = [j for j in uniq.values() if not os.path.exists(j["out"] + "_rgb.png")]
    print(f"[unique renders] {len(uniq)}  to_render={len(jobs)}")

    if render and jobs:
        jf = os.path.join(CACHE, "jobs.json"); json.dump(jobs, open(jf, "w"))
        subprocess.run(["blender", "-b", BLEND, "--python", RENDER, "--", "--jobs", jf], check=True)

    # rectify unique
    for key, j in uniq.items():
        if not os.path.exists(j["out"] + "_rgbR.png"):
            rectify(j["out"], j["viewpoint"])

    # assemble per-frame
    n = 0
    for fr in files:
        pre = os.path.join(CACHE, fr["key"])
        A = Image.open(pre + "_rgbR.png").convert("RGB").resize((W, W))
        B = Image.open(fr["path"]).convert("RGB").resize((W, W))
        comb = Image.new("RGB", (2 * W, W)); comb.paste(A, (0, 0)); comb.paste(B, (W, 0))
        comb.save(os.path.join(OUT, fr["split"], fr["name"] + ".png"))
        mask_to_ids(pre + "_maskR.png").save(os.path.join(OUT, fr["split"], fr["name"] + "_seg.png"))
        n += 1
        if n % 100 == 0: print(f"[assemble] {n}/{len(files)}")
    json.dump({"n": n, "elev": ELEV, "dist": DIST, "lens": LENS,
               "classes": CLASS_ID}, open(os.path.join(OUT, "stats.json"), "w"))
    print(f"[done] {n} pairs -> {OUT}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    main(limit=a.limit, render=not a.no_render)
