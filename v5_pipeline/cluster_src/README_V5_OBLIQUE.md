# V5 Oblique Training Bundle

V5A is a decision experiment, not the final architecture. It tests whether the
main failure in V4 came from bad synthetic geometry rather than from the GAN
alone.

## What Changed From V4

- Training data is `middle` view only. No left/right synthetic views are mapped
  to the same real target.
- Source `A` is rendered from `chess-set.blend` with an oblique camera.
- Each sample has:
  - `{name}.png`: `[A_v5_oblique_rgb | B_real]`
  - `{name}_seg.png`: Blender semantic silhouette RGB
  - `{name}_depth.png`: depth-like render
- `v5_oblique_dataset.py` uses full-cell FEN semantics from `labels.json` for
  the generator one-hot input, and uses Blender semantic RGB only to build the
  silhouette/edge geometry channels.

## Local Dataset Build

Probe dataset:

```bash
python3 v5_pipeline/build_v5_dataset.py \
  --split train --limit 32 \
  --out-dir datasets/chess_v5_oblique_probe \
  --samples 4 --resolution 760 --quiet

python3 v5_pipeline/build_v5_dataset.py \
  --split test --limit 16 \
  --out-dir datasets/chess_v5_oblique_probe \
  --samples 4 --resolution 760 --quiet
```

Full dataset:

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

## Cluster Install

Copy these to the cluster:

```bash
scp -r datasets/chess_v5_oblique_probe USER@slurm.bgu.ac.il:~/chess_cut_project/contrastive-unpaired-translation/datasets/
scp -r v5_pipeline/cluster_src USER@slurm.bgu.ac.il:~/chess_cut_project/v5_cluster_src
```

On the cluster:

```bash
cd ~/chess_cut_project/contrastive-unpaired-translation
bash ~/chess_cut_project/v5_cluster_src/install_v5_oblique_files.sh
```

## Probe Train/Test/Score

```bash
cd ~/chess_cut_project/contrastive-unpaired-translation

DATAROOT=./datasets/chess_v5_oblique_probe \
NAME=chess_v5_oblique_probe \
EP=20 EPD=0 BS=2 \
sbatch ~/chess_cut_project/v5_cluster_src/train_v5_oblique_hd.sbatch

DATAROOT=./datasets/chess_v5_oblique_probe \
NUM_TEST=16 \
sbatch ~/chess_cut_project/v5_cluster_src/test_v5_oblique_hd.sbatch latest chess_v5_oblique_probe

DATAROOT=./datasets/chess_v5_oblique_probe \
sbatch ~/chess_cut_project/v5_cluster_src/score_v5_oblique.sbatch latest chess_v5_oblique_probe
```

Visual pass/fail matters more than the probe score because only 16 test boards
are used. The first visual gate is whether non-pawn pieces stop melting into
generic blobs.

If you want to reproduce the earlier silhouette-only ablation, add:

```bash
--v5_semantic_source silhouette
```

to both train and test commands. The default `fen` mode is preferred because the
silhouette-only probe produced many phantom pieces on empty squares.
