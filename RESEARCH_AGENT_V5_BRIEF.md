# Research Brief: Synthetic-to-Real Chessboard Translation Failure Analysis and V5 Proposal

## Task Context

We are working on a deep-learning project for synthetic-to-real chessboard image translation.

The required deliverable is a trained image-to-image generation model:

```text
generate_chessboard_image(fen, viewpoint)
```

It should generate:

- `synthetic.png`
- `realistic.png`
- `side_by_side.png`

The model input at inference must be a single static synthetic image of a chessboard. The output must look realistic while preserving the exact geometry: board layout, camera pose, piece locations, and piece identity.

Important assignment constraints:

- This must be a model-based image translation solution, not a manual post-processing/compositing patch.
- Temporal/video information may be used only to create labels or training data.
- The generation model itself must not depend on previous/next frames at inference.
- We are allowed to change Blender code, generate more synthetic data, adapt camera/lighting/materials, and download/use additional Blender chess assets if useful.
- The project explicitly invites methods such as GANs, CycleGAN, diffusion-based models, etc.
- Main priorities: geometry correctness first, realism second, style/beauty third.

## Current Project Structure

Project root:

```text
/Users/rnpqlr/Desktop/empty/dl project
```

Cluster repo:

```text
~/chess_cut_project/contrastive-unpaired-translation
```

Cluster environment:

```text
ssh bgu
conda env: pytorch
SLURM course partition
```

Important local files:

```text
chess-set.blend
synthetic_chess_generator.py
build_paired_dataset.py
build_paired_dataset_v2.py
v4_pipeline/build_existing_synth_dataset_v4.py
v4_pipeline/cluster_src/geom_aligned_dataset.py
v4_pipeline/cluster_src/paired_geom_hd_model.py
v4_pipeline/cluster_src/train_geom_hd.sbatch
v4_pipeline/cluster_src/test_geom_hd.sbatch
v4_pipeline/cluster_src/score_geom_v4.sbatch
V5_ROOT_CAUSE_AND_DECISION.md
```

Current dataset:

```text
datasets/chess_existing_geom_v4/
  train/
  test/
  labels.json
  stats.json
  _qa/
```

Current generated result snapshot:

```text
cluster_results/chess_geomv4_smoke5/
  montage.png
  worst_piece_crops.png
  report_chess_geomv4_smoke5_latest.json
  test_latest/images/fake_B/
  test_latest/images/real_A/
  test_latest/images/real_B/
```

## What Has Already Been Tried

### Earlier unpaired/CUT-style approach

Unpaired translation failed fundamentally: it produced phantom pieces, missing pieces, and generic piece-like texture on empty squares. The reason appears structural: unpaired objectives do not enforce exact per-square fidelity.

### Paired pix2pix-style / pix2pixHD-style approach

We moved to paired training because the dataset has FEN labels and aligned real/synthetic frame identities.

This fixed most phantom/occupancy failures, but the pieces still look visually bad.

### V3

Used a Blender scene asset:

```text
v3_pipeline/assets/ChessScene.blend
```

The generated Blender images looked very poor. The model learned occupancy but produced smeared/blobby pieces.

### V4

We stopped using the bad new Blender renderer and instead used already-existing synthetic images from:

```text
data from drive/dataset/trainA
data from drive/dataset/testA
```

These existing synthetic images look cleaner than our failed v3 Blender outputs.

V4 dataset input:

- paired image `[A_existing_synthetic | B_real_target]`
- FEN semantic segmentation map
- geometry map from synthetic/FEN:
  - synthetic foreground/silhouette estimate
  - FEN piece height prior
  - synthetic edge map inside occupied cells

V4 model:

- `paired_geom_hd`
- generator input:

```text
RGB synthetic + one-hot FEN segmentation + geom maps
```

- losses:
  - GAN
  - feature matching
  - VGG perceptual
  - masked L1
  - optional frozen square-classifier loss

Smoke training run:

```text
chess_geomv4_smoke5
5 epochs
```

V4 score:

```text
n_boards: 140
square_acc: 0.9732142857142857
occupancy_acc: 0.9993303571428571
phantom_rate(empty->piece): 0.0006968641114982578
missing_rate(piece->empty): 0.0006211180124223603
type_acc(both occupied): 0.9272840273461778
color_acc(both occupied): 0.9987569919204475
whole_board_occupancy_exact: 0.9571428571428572
whole_board_full_exact: 0.24285714285714285
```

This is a major geometry improvement but still visually unsatisfactory.

## Current Failure Modes

The remaining problem is not mainly occupancy. Occupancy is almost solved.

The real problem is that pieces look bad:

- pieces are blurry, melted, smeared, or watercolor-like
- many pieces look like brown/beige blobs rather than 3D chess pieces
- piece identity is ambiguous, especially for non-pawns
- shadows are weak or physically inconsistent
- piece bases and bodies do not have convincing 3D volume
- edge/crop artifacts remain near board boundaries
- the board texture is acceptable but not fully realistic

Independent harsh visual QA concluded:

- the model places something in the right square, but it often does not look like a real chess piece
- classifier metrics do not punish visual realism enough
- `whole_board_full_exact` is still low, which means most boards have at least one piece-type error

## Quantitative Clues

Per-class recall from current V4 result:

```text
white bishop: 81.0%
white king: 84.3%
black rook: 66.1%
white queen: 86.8%
```

These are exactly the classes where shape and 3D detail matter.

Sharpness analysis:

```text
fake occupied-cell sharpness / real occupied-cell sharpness ~= 0.79
synthetic occupied-cell sharpness / real occupied-cell sharpness ~= 0.51
```

This means the model improves sharpness compared to the synthetic source, but it still does not reach real piece detail.

## Suspected Root Causes

### 1. The synthetic input does not contain enough 3D geometry

The current synthetic images are close to top-down. Pieces appear like icons/discs/symbols more than real oblique 3D objects.

The real photos, after board rectification, still contain tall 3D pieces, oblique silhouettes, shadows, bases, and overlapping vertical shape.

So the model is effectively being asked to solve:

```text
top-down token/icon -> oblique 3D wooden chess piece
```

This may be too much for a small paired GAN trained on only hundreds of real frames.

### 2. Board alignment is not the same as piece alignment

The board grid is roughly aligned after perspective rectification. But the real pieces are tall. Their bodies and shadows extend across square boundaries after rectification.

Therefore pixel L1 between synthetic and real is partially contradictory around pieces:

- if L1 is strong, it encourages blur/averaging
- if L1 is weak, the model is free to generate vague blobs

In V4 we used:

```text
l1_piece_w = 0.1
```

This protects against misaligned-piece blur but also means piece interiors are weakly supervised.

### 3. V4 training data may be internally contradictory

The V4 train dataset uses:

```text
left, middle, right synthetic views
```

all mapped to the same real target.

This gives the model more samples, but it may confuse geometry supervision: three different synthetic silhouettes/edges are asked to map to one identical real image.

The test set uses only `middle`.

### 4. The original Blender asset may not have been used properly

The reduced scene used in V3 looked bad:

```text
v3_pipeline/assets/ChessScene.blend
```

But the original assignment asset exists:

```text
chess-set.blend
```

It is larger and contains real 3D meshes with meaningful piece heights:

- king about 4.37
- queen about 3.99
- bishop about 3.6
- rook about 2.66
- pawns about 2.65

So Blender itself may not be the problem. The bad renderer/camera/material setup may be the problem.

### 5. Existing renderer camera is likely wrong for this task

In `synthetic_chess_generator.py`, the camera logic is essentially top-down:

```text
All cameras look STRAIGHT DOWN
```

This contradicts the real images, where the rectified board still shows oblique 3D piece height.

This is probably a central cause.

## Current Working Hypothesis

The current model has learned board occupancy and rough color transfer. It fails because the input representation lacks the 3D information needed for sharp piece synthesis.

This is not primarily a hyperparameter problem.

The likely highest-leverage change is to build a V5 pipeline where synthetic inputs are camera-matched 3D renders from the original `chess-set.blend`, with useful control maps, and then train a stronger conditional synthesis model.

## Proposed Direction So Far

V5 should start from data/source representation, not from another small loss tweak.

Potential V5 plan:

1. Use original `chess-set.blend`, not the reduced v3 scene.
2. Fix Blender camera to mimic the real footage before perspective rectification:
   - oblique camera
   - visible piece height
   - realistic shadows
   - board fills the frame after perspective crop
3. Render only the paired synthetic view corresponding to the real target, initially `middle` only.
4. Export additional control maps from Blender:
   - semantic class map
   - depth map
   - object/silhouette mask
   - normal map if feasible
   - edge map
5. Train a conditional image synthesis model:
   - pix2pixHD baseline
   - SPADE-style semantic synthesis if needed
   - crop/piece discriminator only after source geometry is fixed
6. Evaluate with both:
   - classifier-based geometry score
   - visual crop montage
   - fake-vs-real occupied-cell sharpness/edge density

## What We Want From the Research Agent

Please investigate this project with full academic freedom under the assignment constraints.

We do not want small patches unless they are part of a coherent plan.

Please answer:

1. What is the most likely root cause of the bad piece appearance?
2. Is the proposed V5 direction correct, or is there a better framing?
3. Should this be treated as:
   - paired image-to-image translation,
   - semantic image synthesis,
   - conditional diffusion / ControlNet-style finetuning,
   - 3D-aware rendering + style transfer,
   - or a hybrid?
4. What model architecture should we use given the small dataset and course constraints?
5. What synthetic data should we generate next?
6. Should we use the original Blender asset, download better assets, or abandon Blender realism and use control maps only?
7. How should we avoid piece blur caused by imperfect real/synthetic alignment?
8. How should we enforce exact piece identity without relying on non-differentiable postprocessing?
9. What ablations would make the final report academically convincing?
10. What is the shortest experiment sequence that can falsify bad ideas quickly?

Hard requirement:

The final inference model must take one static synthetic board image as input and produce one realistic image. It must not require real target images, temporal context, or manual postprocessing at inference.

Please produce:

- a root-cause diagnosis
- a ranked solution plan
- concrete architecture/loss/data recommendations
- estimated training cost
- risk assessment
- a minimal experiment schedule
- what metrics and visual checks to use for go/no-go decisions
