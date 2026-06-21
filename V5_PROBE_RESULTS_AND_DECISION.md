# V5 Probe Results And Decision

Date: 2026-06-07

## Goal

Test whether the main V4 failure was caused by poor synthetic geometry, before
committing to a heavier architecture such as SPADE or diffusion.

## Probe Dataset

- Dataset: `datasets/chess_v5_oblique_probe`
- Train: 32 middle-only samples
- Test: 16 middle-only samples
- Source render: oblique Blender render from `chess-set.blend`
- Controls: Blender semantic silhouette, depth-like render, silhouette edge

## Experiment A: Silhouette Semantic One-Hot

Model input:

- RGB: V5 oblique render
- One-hot semantic: Blender silhouette ids only
- Geom: depth + silhouette + edge

Result:

```json
{
  "n_boards": 16,
  "square_acc": 0.87109375,
  "occupancy_acc": 0.8974609375,
  "phantom_rate(empty->piece)": 0.205078125,
  "missing_rate(piece->empty)": 0.0,
  "type_acc(both occupied)": 0.947265625,
  "color_acc(both occupied)": 1.0,
  "whole_board_occupancy_exact": 0.0,
  "whole_board_full_exact": 0.0
}
```

Visual diagnosis:

- Piece identity improved, but empty squares gained many ghost pieces.
- The model lacked a full-cell "this square is empty/occupied" semantic layout.
- The V4 warm-start expected full-cell semantics, so a silhouette-only map was
  structurally mismatched.

## Experiment B: Full-Cell FEN Semantic One-Hot + V5 Geometry

Model input:

- RGB: V5 oblique render
- One-hot semantic: full-cell FEN labels from `labels.json`
- Geom: depth + Blender silhouette + silhouette edge

Result:

```json
{
  "n_boards": 16,
  "square_acc": 0.990234375,
  "occupancy_acc": 1.0,
  "phantom_rate(empty->piece)": 0.0,
  "missing_rate(piece->empty)": 0.0,
  "type_acc(both occupied)": 0.98046875,
  "color_acc(both occupied)": 1.0,
  "whole_board_occupancy_exact": 1.0,
  "whole_board_full_exact": 0.5
}
```

Visual diagnosis:

- The empty-square phantoms are mostly removed.
- Piece positions and colors are much more stable.
- Remaining weakness: pieces are still somewhat flat/smeared, and edge ranks can
  look like pasted horizontal bands. This is not solved by the probe.

## Decision

Use Experiment B as the V5 baseline:

- Keep middle-only pairing.
- Use existing 3D Blender geometry.
- Use full-cell FEN semantic maps for the generator one-hot input.
- Use Blender silhouette/depth/edge as geometry channels.
- Do not use silhouette-only semantic conditioning for the generator.

## Next Run

Build the full dataset:

```bash
python3 v5_pipeline/build_v5_dataset.py \
  --split train \
  --out-dir datasets/chess_v5_oblique \
  --samples 8 --resolution 900 --quiet

python3 v5_pipeline/build_v5_dataset.py \
  --split test \
  --out-dir datasets/chess_v5_oblique \
  --samples 8 --resolution 900 --quiet
```

Train:

```bash
DATAROOT=./datasets/chess_v5_oblique \
NAME=chess_v5_oblique_fenseg_full \
EP=40 EPD=20 BS=2 PIECEW=0.35 PCLS=1.0 \
sbatch ~/chess_cut_project/v5_cluster_src/train_v5_oblique_hd.sbatch
```

If the full run still looks smeared, the next real step is not another data
patch. It is adding a local piece/crop discriminator or moving to SPADE-style
semantic image synthesis while keeping the same V5 dataset.
