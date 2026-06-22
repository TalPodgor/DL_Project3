# Project 3 — Synthetic-to-Real Chessboard Image Generation from FEN

BGU *Intro to Deep Learning* — Final Project 3.

**Authors:** Tal Podgor, Ilay Vanunu, Ran Packler.

Given a chess position (FEN) and a viewpoint, the system renders a clean
**synthetic** board with Blender and then translates it into a **realistic**
photo-like image while preserving piece positions and camera orientation.

- **Repository:** https://github.com/TalPodgor/DL_Project3
- **Datasets & large artifacts (shared drive):** https://drive.google.com/drive/folders/1oJpGFstyY0AmyeIcL4Cjk1B03UtL3_cN
- **Final report:** [`output/pdf/project3_final_report.pdf`](output/pdf/project3_final_report.pdf)

## Task

Implement the required evaluation function:

```python
def generate_chessboard_image(fen: str, viewpoint: str) -> None
```

`viewpoint ∈ {"white", "black"}` (which color is closest to the camera). It saves
three PNGs to `./results/` (creating it if needed) and returns nothing:

```
./results/synthetic.png       clean synthetic render of the position
./results/realistic.png       synthetic -> real translation (final GAN)
./results/side_by_side.png    [ synthetic | realistic ]
```

## Final model — `chess_v5_bright_silABC`

A **paired, geometry-conditioned, pix2pixHD-style** translator (`paired_geom_hd`).
It is **not** CycleGAN and **not** the earlier CUT baseline.

The generator is an antialiased `resnet_9blocks` network (instance norm) that
takes a **21-channel** input and outputs realistic RGB:

| Channels | Source |
|---|---|
| 3 | synthetic oblique RGB render |
| 15 | one-hot **silhouette segmentation** (empty light/dark + 12 piece classes), built from the FEN and the rendered piece silhouette (`fen_silhouette`) |
| 3 | **geometry** tensor: depth render, piece-silhouette mask, silhouette edge |

The **"silABC"** name = the loss recipe layered on the paired GAN + feature-matching
+ VGG + L1 objectives:

- **A** — silhouette-shaped semantic conditioning (kills square halos / wood bleed).
- **B** — silhouette **edge** (Sobel) loss + piece-weighted L1 (sharp, anti-halo piece outlines).
- **C** — local class-conditional **piece-crop** objectives (crop GAN + feature matching + VGG + a frozen square-classifier loss) for legible small pieces.

Geometry/semantic conditioning is **inference-safe**: at test time it is derived
from the FEN + synthetic render alone (no real/target image is used).

Full config: [`v5_work/final_config/bright_silABC_train_opt.txt`](v5_work/final_config/bright_silABC_train_opt.txt).
Model + dataset code: [`v5_work/v5_cluster_src/`](v5_work/v5_cluster_src/).

## Setup

```bash
git clone https://github.com/TalPodgor/DL_Project3.git
cd DL_Project3
pip install -r requirements.txt          # numpy, Pillow, torch, torchvision, python-chess, ...
```

Also required:

1. **Blender** (https://www.blender.org/download/) — for the synthetic render.
   Put `blender` on your `PATH` or set the `BLENDER` env var. (`chess-set.blend`
   ships in the repo.)
2. **Final checkpoint** — already **bundled in the repo** at
   `checkpoints/chess_v5_bright_silABC/latest_net_G.pth` (~44 MB). No download
   needed. See [`submission/CHECKPOINT_MANIFEST.md`](submission/CHECKPOINT_MANIFEST.md)
   for the architecture/load contract.

The realistic stage runs on CPU or GPU; a GPU makes it near-instant.

## Run the demo

```bash
# standard starting position, white closest to camera (defaults)
python run_project3_demo.py

# any FEN + viewpoint
python run_project3_demo.py \
  --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" \
  --viewpoint black
```

Or call the function directly:

```python
from generate_chessboard_image import generate_chessboard_image
generate_chessboard_image("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "white")
```

### Expected outputs

```
./results/synthetic.png       512x512 RGB
./results/realistic.png       512x512 RGB
./results/side_by_side.png    1024x512 RGB  ([ synthetic | realistic ])
```

If PyTorch or the checkpoint is missing, the demo still writes
`./results/synthetic.png`, then **fails with a precise message** explaining what
to install/place — it never fabricates a realistic image. Sample synthetic
outputs for both viewpoints are in
[`submission/sample_outputs/`](submission/sample_outputs/).

## How it works (pipeline)

1. **Render** the synthetic board with Blender:
   `v5_pipeline/render_oblique_blender.py` (oblique 44°, "bright" lighting) →
   raw rgb / semantic / depth, then `v5_pipeline/pil_perspective_crop.py`
   rectifies them to 512×512.
2. **Build conditioning** (`chess_v5_infer/preprocess.py`): RGB + one-hot
   `fen_silhouette` segmentation + geometry tensor (a framework-free port of the
   training dataset `v5_work/v5_cluster_src/v5_oblique_dataset.py`).
3. **Translate** with the generator (`chess_v5_infer/networks.py`, loading
   `latest_net_G.pth`): `netG(concat(rgb, onehot, geom))` → realistic RGB.
4. **Save** the three PNGs.

## Repository layout

```
generate_chessboard_image.py     REQUIRED API (entry point)
run_project3_demo.py             CLI wrapper (--fen / --viewpoint)
chess_v5_infer/                  framework-free inference helpers
  ├── preprocess.py              21-channel conditioning from FEN + render
  └── networks.py                antialiased resnet_9blocks generator (loads latest_net_G.pth)
chess-set.blend                  Blender asset (3D pieces + board)
v5_pipeline/                     final synthetic-render pipeline
  ├── render_oblique_blender.py  FEN -> oblique rgb/seg/depth (Blender)
  ├── pil_perspective_crop.py    rectify renders to 512x512
  └── build_v5_dataset.py        builds the paired training dataset
v5_work/
  ├── v5_cluster_src/            final model + dataset code (paired_geom_hd, v5_oblique)
  ├── final_config/              bright_silABC_train_opt.txt (exact final config)
  └── make_final_report_pdf.py   regenerates the report PDF
output/pdf/project3_final_report.pdf   final report (6 pp)
submission/                      guidelines summary, manifests, sample outputs
checkpoints/chess_v5_bright_silABC/    place latest_net_G.pth here (from Drive)
trained_model/                   LEGACY CUT baseline (history only — not used)
requirements.txt
```

Heavy data (`datasets/`, `synthetic_game*/`, `test_output/`, `results/`, …) lives
on the shared drive / is git-ignored; it is not part of the repo.

## Training (reproduction)

Trained on the BGU cluster (GPU) on the paired oblique "bright" dataset
`chess_v5_oblique_aligned_bright` (built by `v5_pipeline/build_v5_dataset.py`).
The model is `paired_geom_hd` with `dataset_mode=v5_oblique`,
`v5_semantic_source=fen_silhouette`, `load_size=crop_size=512`. Launch/eval
scripts and the full option dump:

```
v5_work/v5_cluster_src/train_v5_oblique_hd.sbatch
v5_work/v5_cluster_src/test_v5_oblique_hd.sbatch
v5_work/final_config/bright_silABC_train_opt.txt
```

Do not retrain to use the model — only the checkpoint is needed for inference.

## Known limitations

- The realistic stage requires PyTorch + the checkpoint (not bundled in Git; on
  the shared drive). Without them only the synthetic stage runs.
- Tall pieces lean across square boundaries under the oblique camera; the
  silhouette + base-patch conditioning mitigates but does not fully remove
  occasional small/dark-piece legibility loss.
- A later re-evaluation suggested a sibling variant `chess_v5_bright_silAB`
  (dropping loss "C") may slightly improve blind realism; `silABC` is kept as the
  final model and `silAB` is noted as future work in the report.

## Project history (earlier approach)

The first iteration used **CUT** (Contrastive Unpaired Translation, Park et al.
ECCV 2020); those weights remain in `trained_model/` for history only. The final
system replaced it with the paired, geometry-conditioned `paired_geom_hd`
described above. Older notes: `PROJECT_CONTEXT_FOR_CLAUDE.txt`,
`BGU_CLUSTER_TRAINING_GUIDE.txt`.
