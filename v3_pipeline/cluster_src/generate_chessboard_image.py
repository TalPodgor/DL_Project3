"""
generate_chessboard_image.py  -- Wave 5 deliverable (geometry-first).

    generate_chessboard_image(fen, viewpoint) -> synthetic.png, realistic.png, side_by_side.png

Pipeline:
  1. Blender renders an oblique synthetic chess scene from the FEN at the canonical
     camera pose (render_aligned.py + ChessScene.blend).
  2. Rectify the board to the canonical 512^2 square (numpy homography).
  3. Build the FEN semantic mask (class ids) from the rectified mask render.
  4. The trained seg-conditioned generator G( concat(synthetic, one-hot mask) ) -> photo.

Requires (same env): Blender (for step 1) + torch + this CUT repo (for step 4).
Place in the repo root with assets/{ChessScene.blend, render_aligned.py}.

CLI:
  conda run -n pytorch python generate_chessboard_image.py \
      --fen "<FEN>" --viewpoint white --out ./out --ckpt ./checkpoints/chess_segv3/latest_net_G.pth
  # --no-render : reuse an existing rendered prefix (for stage-split testing)
"""
import os, sys, json, math, argparse, subprocess
import numpy as np
from PIL import Image
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
BLEND = os.path.join(ASSETS, "ChessScene.blend")
RENDER = os.path.join(ASSETS, "render_aligned.py")
ELEV, DIST, LENS, RES, W, K = 35.0, 9.0, 35.0, 768, 512, 15

CLASS_ID = {"empty_light": 1, "empty_dark": 2}
for i, t in enumerate(["p", "n", "b", "r", "q", "k"]):
    CLASS_ID["w" + t] = 3 + i; CLASS_ID["b" + t] = 9 + i
def _pal(cid): return np.array([(cid*17) % 256, (cid*53) % 256, (cid*97) % 256], float)
PIECE_IDS = list(range(3, 15)); PIECE_PAL = np.stack([_pal(c) for c in PIECE_IDS])

def _solve_h(src, dst):
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A += [[x, y, 1, 0, 0, 0, -u*x, -u*y, -u], [0, 0, 0, x, y, 1, -v*x, -v*y, -v]]
    _, _, Vt = np.linalg.svd(np.asarray(A, float)); H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]

def _canon(vp):
    return ({"a8": (0, 0), "h8": (W, 0), "h1": (W, W), "a1": (0, W)} if vp == "white"
            else {"a8": (W, W), "h8": (0, W), "h1": (0, 0), "a1": (W, 0)})

def rectify(prefix, vp):
    meta = json.load(open(prefix + "_corners.json")); c = meta["corners"]; tgt = _canon(vp)
    names = ["a1", "h1", "a8", "h8"]
    Hds = _solve_h([tgt[n] for n in names], [tuple(c[n]) for n in names])
    coeffs = (Hds.flatten() / Hds[2, 2])[:8].tolist()
    rgb = Image.open(prefix + "_rgb.png").convert("RGB").transform((W, W), Image.PERSPECTIVE, coeffs, Image.BILINEAR)
    msk = Image.open(prefix + "_mask.png").convert("RGB").transform((W, W), Image.PERSPECTIVE, coeffs, Image.NEAREST)
    return rgb, msk

def mask_to_ids(maskR):
    arr = np.asarray(maskR, float).reshape(-1, 3)
    d = np.linalg.norm(arr[:, None, :] - PIECE_PAL[None], axis=2)
    nn, nnd = d.argmin(1), d.min(1); is_p = nnd < 40.0
    ids = np.zeros(W*W, np.uint8); ids[is_p] = np.array(PIECE_IDS, np.uint8)[nn[is_p]]
    ids = ids.reshape(W, W)
    yy, xx = np.mgrid[0:W, 0:W]; par = ((xx//(W//8) + yy//(W//8)) % 2)
    e = ids == 0
    ids[e & (par == 0)] = CLASS_ID["empty_light"]; ids[e & (par == 1)] = CLASS_ID["empty_dark"]
    return ids

def load_G(ckpt, device):
    sys.path.insert(0, os.getcwd())
    from models import networks
    from types import SimpleNamespace
    opt = SimpleNamespace(no_antialias=False, no_antialias_up=False)
    G = networks.define_G(3 + K, 3, 64, "resnet_9blocks", "instance", True,
                          "xavier", 0.02, False, False, [], opt)
    sd = torch.load(ckpt, map_location=device)
    if hasattr(sd, "_metadata"): del sd._metadata
    (G.module if hasattr(G, "module") else G).load_state_dict(sd)
    return G.to(device).eval()

def _to_tensor(pil):
    a = np.asarray(pil.convert("RGB"), np.float32) / 127.5 - 1.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)

def _save(t, path):
    a = ((t.squeeze(0).permute(1, 2, 0).clamp(-1, 1).cpu().numpy() + 1) * 127.5).astype(np.uint8)
    Image.fromarray(a).save(path)

def generate_chessboard_image(fen, viewpoint, out="./out", ckpt="./checkpoints/chess_segv3/latest_net_G.pth",
                              render=True, prefix=None):
    os.makedirs(out, exist_ok=True)
    prefix = prefix or os.path.join(out, "_render")
    if render:
        subprocess.run(["blender", "-b", BLEND, "--python", RENDER, "--",
                        "--fen", fen, "--viewpoint", viewpoint, "--out", prefix,
                        "--elev", str(ELEV), "--dist", str(DIST), "--lens", str(LENS), "--res", str(RES)],
                       check=True)
    rgb, msk = rectify(prefix, viewpoint)
    ids = mask_to_ids(msk)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = load_G(ckpt, device)
    A = _to_tensor(rgb).to(device)
    seg = torch.from_numpy(ids.astype(np.int64))[None].to(device)
    onehot = torch.nn.functional.one_hot(seg.clamp(0, K-1), K).permute(0, 3, 1, 2).float()
    with torch.no_grad():
        fake = G(torch.cat([A, onehot], 1))
    syn_p = os.path.join(out, "synthetic.png"); real_p = os.path.join(out, "realistic.png")
    rgb.save(syn_p); _save(fake, real_p)
    sbs = Image.new("RGB", (2*W, W)); sbs.paste(rgb, (0, 0)); sbs.paste(Image.open(real_p), (W, 0))
    sbs.save(os.path.join(out, "side_by_side.png"))
    print("[generate]", syn_p, real_p)
    return syn_p, real_p

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen", required=True); ap.add_argument("--viewpoint", default="white", choices=["white", "black"])
    ap.add_argument("--out", default="./out"); ap.add_argument("--ckpt", default="./checkpoints/chess_segv3/latest_net_G.pth")
    ap.add_argument("--no-render", action="store_true"); ap.add_argument("--prefix")
    a = ap.parse_args()
    generate_chessboard_image(a.fen, a.viewpoint, a.out, a.ckpt, render=not a.no_render, prefix=a.prefix)
