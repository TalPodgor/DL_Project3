"""
rectify_and_pack.py -- warp an oblique Blender render so the board's playing
surface maps onto the canonical 512x512 square (matching the board-rectified real
photos). RGB -> bilinear, mask -> nearest. Run with system python3 (needs PIL).

  python3 rectify_and_pack.py --prefix renders/start_white --viewpoint white
-> writes renders/start_white_rgbR.png, renders/start_white_maskR.png
"""
import json, argparse
import numpy as np
from PIL import Image

W = 512

def canonical_targets(viewpoint):
    # playing-surface corner (by file/rank name) -> canonical pixel (x,y)
    if viewpoint == "white":   # a8 top-left, rank8 at top, file a left
        return {"a8": (0, 0), "h8": (W, 0), "h1": (W, W), "a1": (0, W)}
    else:                       # black: 180-degree rotation of white layout
        return {"a8": (W, W), "h8": (0, W), "h1": (0, 0), "a1": (W, 0)}

def solve_homography(src, dst):
    # returns 3x3 H s.t. dst ~ H @ src  (src,dst: list of (x,y))
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.asarray(A, float)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]

def rectify(prefix, viewpoint):
    meta = json.load(open(prefix + "_corners.json"))
    corners = meta["corners"]
    tgt = canonical_targets(viewpoint)
    names = ["a1", "h1", "a8", "h8"]
    src = [tuple(corners[n]) for n in names]       # oblique-render pixels
    dst = [tgt[n] for n in names]                   # canonical pixels
    # PIL needs the map output->input, i.e. H mapping dst -> src
    Hds = solve_homography(dst, src)
    coeffs = (Hds.flatten() / Hds[2, 2])[:8].tolist()
    out = {}
    for kind, resample in [("rgb", Image.BILINEAR), ("mask", Image.NEAREST)]:
        im = Image.open(f"{prefix}_{kind}.png").convert("RGB")
        warp = im.transform((W, W), Image.PERSPECTIVE, coeffs, resample)
        op = f"{prefix}_{kind}R.png"
        warp.save(op); out[kind] = op
    print("[rectify]", out)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--viewpoint", required=True, choices=["white", "black"])
    a = ap.parse_args()
    rectify(a.prefix, a.viewpoint)
