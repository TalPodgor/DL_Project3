"""
Per-session camera calibration for V5.

Renders each game's starting position at an elevation sweep, board-rectifies each,
and builds an overlay montage (synthetic alpha-blended on the real target) so the
best-matching camera elevation can be picked per game. The V5 build used a single
global camera (camera-y 0.90, camera-z 1.28 -> ~55deg, near top-down) for every
game; the real footage is far more oblique (~25-45deg). Matching obliquity per
session aligns the synthetic piece shear with the real one (the root cause of blur).
"""
import argparse
import json
import math
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path("/Users/rnpqlr/Desktop/empty/dl project")
BLENDER = "/opt/homebrew/bin/blender"
BLEND = ROOT / "chess-set.blend"
RENDER = ROOT / "v5_pipeline" / "render_oblique_blender.py"
CROP = ROOT / "v5_pipeline" / "pil_perspective_crop.py"
DATA = ROOT / "datasets" / "chess_v5_oblique"
OUT = ROOT / "v5_work" / "calib"

REPS = {
    2: "game2_frame_000200_white_middle",
    4: "game4_frame_000028_white_middle",
    5: "game5_frame_000044_white_middle",
    6: "game6_frame_000036_white_middle",
    7: "game7_frame_000172_white_middle",
}


def elev_deg(z, y):
    return round(math.degrees(math.atan2(z, y)))


def real_target(name, split):
    im = Image.open(DATA / split / f"{name}.png").convert("RGB")
    w = im.width // 2
    return im.crop((w, 0, 2 * w, im.height)).resize((512, 512))


def render_and_crop(name, fen, vp, y, z, lens, res, samples):
    rname = f"{name}_y{int(y*100):03d}_z{int(z*100):03d}"
    rdir = OUT / "raw" / rname
    cdir = OUT / "crop" / rname
    rdir.mkdir(parents=True, exist_ok=True)
    cdir.mkdir(parents=True, exist_ok=True)
    crop_rgb = cdir / f"{rname}_rgb.png"
    if not crop_rgb.exists():
        subprocess.run(
            [BLENDER, str(BLEND), "--background", "--python", str(RENDER), "--",
             "--fen", fen, "--viewpoint", vp, "--out-dir", str(rdir), "--name", rname,
             "--resolution", str(res), "--samples", str(samples),
             "--camera-y", str(y), "--camera-z", str(z), "--lens", str(lens)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(
            ["python3", str(CROP), "--metadata", str(rdir / f"{rname}_metadata.json"),
             "--out-dir", str(cdir), "--size", "512"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return Image.open(crop_rgb).convert("RGB").resize((512, 512))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-y", type=float, default=1.05)
    ap.add_argument("--lens", type=float, default=48.0)
    ap.add_argument("--res", type=int, default=900)
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--elevs", type=str, default="28,32,36,40,45,50")
    ap.add_argument("--games", type=str, default="2,4,5,6,7")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    y = args.camera_y
    elevs = [float(e) for e in args.elevs.split(",")]
    games = [int(g) for g in args.games.split(",")]
    labels = json.loads((DATA / "labels.json").read_text())

    for g in games:
        name = REPS[g]
        meta = labels[name]
        fen, vp, split = meta["fen"], meta["viewpoint"], meta["split"]
        real = real_target(name, split)
        tiles = [("REAL", real)]
        for e in elevs:
            z = round(y * math.tan(math.radians(e)), 3)
            syn = render_and_crop(name, fen, vp, y, z, args.lens, args.res, args.samples)
            ov = Image.blend(syn, real, 0.5)
            tiles.append((f"e{int(e)} z{z}", ov))
            print(f"game{g} elev={e} z={z}")
        tw, lab = 300, 18
        canvas = Image.new("RGB", (tw * len(tiles), tw + lab), (245, 245, 245))
        d = ImageDraw.Draw(canvas)
        for i, (t, im) in enumerate(tiles):
            canvas.paste(im.resize((tw, tw)), (i * tw, lab))
            d.text((i * tw + 3, 3), t, fill=(220, 0, 0))
        out = OUT / f"calib_game{g}_y{int(y*100):03d}.png"
        canvas.save(out)
        print("saved", out)


if __name__ == "__main__":
    main()
