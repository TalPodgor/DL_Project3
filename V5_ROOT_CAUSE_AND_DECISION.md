# V5 Root Cause Diagnosis and Decision

## Verdict

Do not run a long v4 training as-is. The current failure is not a small tuning
problem. The model has mostly solved square occupancy, but it is not receiving a
strong enough geometric signal to synthesize sharp 3D chess pieces.

The real solution is to change the source representation and the training
objective together:

1. Render or generate synthetic inputs that contain realistic 3D piece geometry
   under an oblique camera, not top-down icon-like pieces.
2. Train a semantic/geometry-conditioned image synthesis model that preserves the
   layout while learning real board texture, lighting, and piece appearance.
3. Stop using contradictory left/middle/right synthetic views mapped to one
   identical real target for paired supervision.

## Evidence

- v4 score after 5 epochs:
  - square_acc: 97.32%
  - occupancy_acc: 99.93%
  - type_acc: 92.73%
  - whole_board_full_exact: 24.29%
- Occupancy is nearly saturated, so the remaining failure is piece identity and
  visual form, not phantom/missing pieces.
- Fake occupied-cell sharpness is only about 79% of the real occupied-cell
  sharpness. Synthetic occupied cells are only about 51% of real sharpness.
- Per-class recall exposes shape-sensitive failures:
  - white bishop: 81%
  - white king: 84.3%
  - black rook: 66.1%
- The current generator deliberately reduces L1 pressure inside piece cells:
  `l1_piece_w=0.1`. This avoids punishing unavoidable misalignment, but it also
  means piece interiors are weakly supervised.
- The current Blender camera in `synthetic_chess_generator.py` is effectively
  top-down. It cannot provide the piece height, body silhouette, and shadows
  that appear in the rectified real photos.
- The paired v4 train set maps `left`, `middle`, and `right` synthetic views to
  the same real target. That makes synthetic foreground/edge channels noisy
  supervision rather than reliable geometry.

## Root Causes

1. The synthetic input does not contain enough 3D information.
   The model is asked to infer a real oblique piece from a mostly top-down icon.

2. The paired target is not pixel-aligned at the piece level.
   Board squares align, but tall real pieces and shadows extend across cells
   after rectification.

3. The current objective protects the board more than the pieces.
   Lowering L1 inside occupied cells helps avoid blur from misalignment but
   leaves the actual piece shapes to GAN/VGG/classifier losses.

4. The square classifier is too forgiving.
   It measures whether a crop is classifiable, not whether it is a convincing
   wooden 3D chess piece.

5. Data augmentation is partly contradictory.
   Left/middle/right synthetic variants paired to the same real image confuse
   synthetic-derived geometry hints.

## V5 Decision

Build V5 around camera-matched 3D synthetic renders, not around patches on v4.

Primary experiment:

- Use the original `chess-set.blend` asset, not the reduced v3 scene.
- Fix Blender camera to mimic the real rectified footage before perspective crop:
  oblique camera, visible piece height, shadows, and consistent board filling.
- Render only the synthetic view that corresponds to the real paired target for
  paired training, initially `middle` only.
- Export extra control maps from Blender:
  - semantic class map
  - depth or height map
  - object/silhouette mask
  - optional Canny/edge map
- Train a pix2pixHD/SPADE-style conditional model:
  RGB synthetic render + semantic/depth/control maps -> real target.
- Add a local piece discriminator or crop realism loss only after the source
  geometry is fixed.

Minimal ablations:

1. V5A: middle-only paired dataset, current model, `l1_piece_w=1.0`.
2. V5B: camera-matched 3D renders, current geometry-conditioned pix2pixHD.
3. V5C: V5B plus piece-crop discriminator.
4. V5D: V5B plus SPADE/semantic normalization if V5B still washes out classes.

Go/no-go metric:

- Do not accept a model based only on occupancy.
- Require visual crop montage to show recognizable non-pawn pieces.
- Require fake occupied-cell sharpness to approach real occupied-cell sharpness,
  not merely improve classifier score.
