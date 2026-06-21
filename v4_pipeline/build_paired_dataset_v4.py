"""
build_paired_dataset_v4.py -- paired dataset using the original chess-set asset.

Outputs:
  datasets/chess_paired_v4/{train,test}/{name}.png       = [A_synthetic | B_real]
  datasets/chess_paired_v4/{train,test}/{name}_seg.png   = class ids, 1..14
  datasets/chess_paired_v4/{train,test}/{name}_geom.png  = geometry hints:
      R: rendered piece silhouette
      G: relative piece height by class
      B: rendered silhouette/class edge

Rendering is local-only because the cluster has no Blender. Training consumes the
saved PNGs on the cluster.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "v3_pipeline"))
from rectify_and_pack import rectify


RAW = os.path.join(ROOT, "data from drive", "dataset")
CACHE = os.path.join(ROOT, "v4_pipeline", "cache")
OUT = os.path.join(ROOT, "datasets", "chess_paired_v4")
BLEND = os.path.join(ROOT, "chess-set.blend")
RENDER = os.path.join(ROOT, "v4_pipeline", "render_realset_aligned.py")
CAMERA_CONFIG = os.path.join(ROOT, "v4_pipeline", "camera_v4.json")
W = 512
RES = 768

CLASS_ID = {"empty_light": 1, "empty_dark": 2}
for i, t in enumerate(["p", "n", "b", "r", "q", "k"]):
    CLASS_ID["w" + t] = 3 + i
    CLASS_ID["b" + t] = 9 + i

PIECE_IDS = list(range(3, 15))


def palette(cid):
    return np.array([(cid * 17) % 256, (cid * 53) % 256, (cid * 97) % 256], float)


PIECE_PAL = np.stack([palette(c) for c in PIECE_IDS])

HEIGHT_BY_KIND = {
    "p": 0.46,
    "n": 0.64,
    "b": 0.78,
    "r": 0.55,
    "q": 0.86,
    "k": 0.95,
}
CID_TO_HEIGHT = {}
for color_offset in (3, 9):
    for i, kind in enumerate(["p", "n", "b", "r", "q", "k"]):
        CID_TO_HEIGHT[color_offset + i] = HEIGHT_BY_KIND[kind]


def load_camera_config(path=CAMERA_CONFIG):
    if not os.path.exists(path):
        return {"default": {"elev": 35.0, "dist": 9.0, "lens": 35.0, "yaw": 0.0}}
    return json.load(open(path))


def camera_for(config, game, viewpoint):
    for key in (f"game{game}_{viewpoint}", f"game{game}", viewpoint, "default"):
        if key in config:
            cam = dict(config[key])
            break
    else:
        cam = {}
    return {
        "elev": float(cam.get("elev", 35.0)),
        "dist": float(cam.get("dist", 9.0)),
        "lens": float(cam.get("lens", 35.0)),
        "yaw": float(cam.get("yaw", 0.0)),
    }


def load_fen_index():
    idx = {}
    for csv in glob.glob(os.path.join(ROOT, "game*.csv")):
        match = re.search(r"game(\d+)", os.path.basename(csv))
        if not match:
            continue
        game = int(match.group(1))
        rows = []
        for line in open(csv).read().splitlines()[1:]:
            parts = line.split(",")
            if len(parts) >= 3 and parts[0].strip().isdigit():
                rows.append((int(parts[0]), int(parts[1]), parts[2].strip()))
        idx[game] = sorted(rows)
    return idx


FEN_IDX = load_fen_index()


def lookup_fen(game, frame):
    rows = FEN_IDX.get(game, [])
    fen = None
    for fr, _, row_fen in rows:
        if fr <= frame:
            fen = row_fen
        else:
            break
    return fen or (rows[0][2] if rows else None)


def mask_to_ids(mask_path):
    arr = np.asarray(Image.open(mask_path).convert("RGB"), float)
    h, w, _ = arr.shape
    flat = arr.reshape(-1, 3)
    d = np.linalg.norm(flat[:, None, :] - PIECE_PAL[None, :, :], axis=2)
    nn = d.argmin(1)
    nndist = d.min(1)
    is_piece = nndist < 40.0
    ids = np.zeros(h * w, np.uint8)
    ids[is_piece] = np.array(PIECE_IDS, np.uint8)[nn[is_piece]]
    ids = ids.reshape(h, w)

    yy, xx = np.mgrid[0:h, 0:w]
    parity = ((xx // (w // 8) + yy // (h // 8)) % 2)
    empty = ids == 0
    ids[empty & (parity == 0)] = CLASS_ID["empty_light"]
    ids[empty & (parity == 1)] = CLASS_ID["empty_dark"]
    return ids


def edge_from_ids(ids):
    edge = np.zeros(ids.shape, bool)
    edge[:, 1:] |= ids[:, 1:] != ids[:, :-1]
    edge[:, :-1] |= ids[:, 1:] != ids[:, :-1]
    edge[1:, :] |= ids[1:, :] != ids[:-1, :]
    edge[:-1, :] |= ids[1:, :] != ids[:-1, :]
    piece = ids >= 3
    edge &= piece
    # one-pixel dilation without scipy/cv2
    dil = edge.copy()
    dil[1:, :] |= edge[:-1, :]
    dil[:-1, :] |= edge[1:, :]
    dil[:, 1:] |= edge[:, :-1]
    dil[:, :-1] |= edge[:, 1:]
    return dil


def geom_from_ids(ids):
    piece = ids >= 3
    height = np.zeros(ids.shape, np.float32)
    for cid, val in CID_TO_HEIGHT.items():
        height[ids == cid] = val
    geom = np.zeros((ids.shape[0], ids.shape[1], 3), np.uint8)
    geom[:, :, 0] = piece.astype(np.uint8) * 255
    geom[:, :, 1] = np.clip(height * 255.0, 0, 255).astype(np.uint8)
    geom[:, :, 2] = edge_from_ids(ids).astype(np.uint8) * 255
    return geom


def collect_files(splits):
    files = []
    wanted = set(splits)
    for split, sub in (("train", "trainB"), ("test", "testB")):
        if split not in wanted:
            continue
        for path in sorted(glob.glob(os.path.join(RAW, sub, "*.jpg"))):
            match = re.search(r"game(\d+)_frame_(\d+)_(white|black)", os.path.basename(path))
            if not match:
                continue
            game, frame, viewpoint = int(match.group(1)), int(match.group(2)), match.group(3)
            fen = lookup_fen(game, frame)
            if fen is None:
                continue
            files.append({
                "split": split,
                "path": path,
                "game": game,
                "frame": frame,
                "viewpoint": viewpoint,
                "fen": fen,
                "name": f"game{game}_frame_{frame:06d}_{viewpoint}",
            })
    return files


def key_for(frame, cam):
    bits = [
        "realset-v4",
        frame["fen"],
        frame["viewpoint"],
        str(cam["elev"]),
        str(cam["dist"]),
        str(cam["lens"]),
        str(cam["yaw"]),
    ]
    return hashlib.md5("|".join(bits).encode()).hexdigest()[:16]


def main(limit=None, splits=("train", "test"), render=True):
    os.makedirs(CACHE, exist_ok=True)
    for split in ("train", "test"):
        os.makedirs(os.path.join(OUT, split), exist_ok=True)

    camera_config = load_camera_config()
    files = collect_files(splits)
    if limit:
        files = files[:limit]
    print(f"[files] {len(files)} splits={','.join(splits)}")

    uniq = {}
    for frame in files:
        cam = camera_for(camera_config, frame["game"], frame["viewpoint"])
        frame["camera"] = cam
        key = key_for(frame, cam)
        frame["key"] = key
        if key not in uniq:
            uniq[key] = {
                "fen": frame["fen"],
                "viewpoint": frame["viewpoint"],
                "out": os.path.join(CACHE, key),
                "res": RES,
                **cam,
            }

    jobs = [job for job in uniq.values() if not os.path.exists(job["out"] + "_rgb.png")]
    print(f"[unique renders] {len(uniq)} to_render={len(jobs)}")
    if render and jobs:
        jobs_path = os.path.join(CACHE, "jobs.json")
        with open(jobs_path, "w") as f:
            json.dump(jobs, f)
        subprocess.run(["blender", "-b", BLEND, "--python", RENDER, "--", "--jobs", jobs_path], check=True)

    for key, job in uniq.items():
        if not os.path.exists(job["out"] + "_rgbR.png") or not os.path.exists(job["out"] + "_maskR.png"):
            rectify(job["out"], job["viewpoint"])

    labels = {}
    for i, frame in enumerate(files, 1):
        pre = os.path.join(CACHE, frame["key"])
        rgb = Image.open(pre + "_rgbR.png").convert("RGB").resize((W, W), Image.BILINEAR)
        real = Image.open(frame["path"]).convert("RGB").resize((W, W), Image.BILINEAR)
        ids = mask_to_ids(pre + "_maskR.png")

        paired = Image.new("RGB", (2 * W, W))
        paired.paste(rgb, (0, 0))
        paired.paste(real, (W, 0))

        out_dir = os.path.join(OUT, frame["split"])
        paired.save(os.path.join(out_dir, frame["name"] + ".png"))
        Image.fromarray(ids, mode="L").save(os.path.join(out_dir, frame["name"] + "_seg.png"))
        Image.fromarray(geom_from_ids(ids), mode="RGB").save(os.path.join(out_dir, frame["name"] + "_geom.png"))

        labels[frame["name"]] = {
            "fen": frame["fen"],
            "viewpoint": frame["viewpoint"],
            "game": frame["game"],
            "frame": frame["frame"],
            "split": frame["split"],
            "camera": frame["camera"],
            "render_key": frame["key"],
        }
        if i % 100 == 0:
            print(f"[assemble] {i}/{len(files)}")

    with open(os.path.join(OUT, "labels.json"), "w") as f:
        json.dump(labels, f, indent=2)
    with open(os.path.join(OUT, "stats.json"), "w") as f:
        json.dump({"n": len(files), "classes": CLASS_ID, "camera_config": camera_config}, f, indent=2)
    print(f"[done] {len(files)} pairs -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--splits", default="train,test", help="comma list: train,test")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    main(limit=args.limit,
         splits=tuple(s.strip() for s in args.splits.split(",") if s.strip()),
         render=not args.no_render)
