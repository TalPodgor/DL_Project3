# Advisory-Based Action Plan, 2026-06-14

Goal: materially reduce visible double-head / multi-lobe / side-lobe chess-piece
defects. Relative ranking is not enough; the user rejects the current absolute
visual quality.

## Critical Read Of External Report

Accepted:

- The strongest current hypothesis is not `pieceD` itself. It is the interaction
  of weak alignment, reconstruction/perceptual losses, and geometry/semantic
  conditioning that lets the generator average or duplicate small piece shapes.
- Classifier metrics are deceptive; `noPieceD` improved `type_acc` while visual
  defects stayed bad.
- Explicit geometry preservation or foreground/background decoupling is worth
  testing because it targets the actual failure mode.

Rejected or downweighted:

- SPADE-specific diagnosis is not directly applicable: this model concatenates
  one-hot semantic maps into a ResNet generator; it is not a SPADE generator.
- RegGAN/flow registration and wavelet architectures are too heavy for first
  response. They may be useful later, but first we need cheaper falsification.
- Local VGG restriction is not a strong next test because disabling all local
  crop losses already failed. Global VGG/reconstruction on piece regions remains
  suspect and is being tested.

## Experiment 1: No Piece Reconstruction Pressure

Hypothesis: L1/VGG/edge pressure on misaligned pieces is creating shape
averaging/ghost heads. Remove direct reconstruction/perceptual pressure from
piece silhouettes while preserving board/background losses.

Cluster chain:

- Train: `18162283`
- Test: `18162284`
- Score: `18162285`

Config:

```bash
NAME=chess_v5_bright_noPieceRecon
DATAROOT=./datasets/chess_v5_oblique_aligned_bright
INIT=./checkpoints/chess_v5_aligned_silAB/latest_net_G.pth
SEMSRC=fen_silhouette
BS=1 EP=40 EPD=20
PIECEW=0.0
VGG=0.0
EDGE=0.0
CX=0
PCLS=0
PGAN=0
PFM=0
PVGG=0
PCX=0
```

Result:

- Train `18162283`: `COMPLETED`, elapsed `04:00:18`.
- Test `18162284`: `COMPLETED`, elapsed `00:01:36`.
- Score `18162285`: `COMPLETED`, elapsed `00:00:11`.
- Pulled images to `v5_work/eval_noPieceRecon/fake_B`.
- Pulled report to
  `v5_work/reports_noC/report_chess_v5_bright_noPieceRecon_latest.json`.

Metrics:

- Classifier-like metrics: `square_acc=0.9253`, `type_acc=0.8126`,
  `whole_board_occupancy_exact=0.6857`. These are not the main arbiter.
- Existing vertical double detector:
  - `bright_noPieceRecon`: 28.7% all, 30.0% pawns.
  - Baseline `bright_silAB`: 27.7% all, 29.5% pawns.
- Split-cue audit:
  - `bright_noPieceRecon`: 53.6% all, 61.6% pawns.
  - Baseline `bright_silAB`: 53.4% all, 59.3% pawns.
- Legacy defect audit:
  - `merge_delta_mean=8.09`, slightly better than `bright_silAB` 8.21.
  - `edge_halo_flag_rate=0.352`, worse than `bright_silAB` 0.272.

Conclusion: removing direct piece reconstruction/perceptual pressure is not
sufficient. It slightly changes secondary metrics but does not materially reduce
the visible double-head / multi-lobe defect. Visual pawn crops
(`v5_work/head_crops_bright_noPieceRecon_Pp.png`) still show the same flattened
blobs and side-lobed heads.

## Experiment 2: Full-Cell FEN Conditioning, No Piece Reconstruction

Hypothesis: `fen_silhouette` hybrid conditioning may inject silhouette/overhang
boundary cues that become side lobes. Test simpler full-cell FEN semantics while
keeping piece reconstruction pressure removed.

Cluster chain:

- Train: `18162385` after `18162285`
- Test: `18162386`
- Score: `18162387`

Config:

```bash
NAME=chess_v5_bright_fen_noPieceRecon
DATAROOT=./datasets/chess_v5_oblique_aligned_bright
INIT=./checkpoints/chess_v5_aligned_pieceD/latest_net_G.pth
SEMSRC=fen
BS=1 EP=40 EPD=20
PIECEW=0.0
VGG=0.0
EDGE=0.0
CX=0
PCLS=0
PGAN=0
PFM=0
PVGG=0
PCX=0
```

Rationale for warm-start: `chess_v5_aligned_pieceD` was trained with
`v5_semantic_source: fen`, so it is less contaminated by `fen_silhouette` than
`aligned_silAB`.

Result:

- Train `18162385`: `COMPLETED`, elapsed `03:58:29`.
- Test `18162386`: `COMPLETED`, elapsed `00:01:29`.
- Score `18162387`: `COMPLETED`, elapsed `00:00:10`.
- Pulled images to `v5_work/eval_fen_noPieceRecon/fake_B`.
- Pulled report to
  `v5_work/reports_noC/report_chess_v5_bright_fen_noPieceRecon_latest.json`.

Metrics:

- Classifier-like metrics degraded: `square_acc=0.8883`, `type_acc=0.7809`,
  `whole_board_occupancy_exact=0.55`, with high phantom rate `0.0474`.
- Existing vertical double detector:
  - `bright_fen_noPieceRecon`: 28.9% all, 31.8% pawns.
  - Baseline `bright_silAB`: 27.7% all, 29.5% pawns.
- Split-cue audit:
  - `bright_fen_noPieceRecon`: 57.6% all, 64.8% pawns.
  - Baseline `bright_silAB`: 53.4% all, 59.3% pawns.
- Legacy defect audit:
  - `merge_delta_mean=7.19`, improved vs `bright_silAB` 8.21.
  - `edge_halo_flag_rate=0.474`, much worse than `bright_silAB` 0.272.

Conclusion: switching from `fen_silhouette` to full-cell `fen` while removing
piece reconstruction pressure does not solve the structural defects. It improves
one merge-like metric but worsens split-cue, halo, classifier-like occupancy, and
visual artifacts. Visual pawn crops
(`v5_work/head_crops_bright_fen_noPieceRecon_Pp.png`) show persistent blobs plus
new color/phantom artifacts.

## Local Geometry-Locked Composite Probe

Purpose: quickly test whether hard geometry preservation can reduce the defect
before implementing a trainable decoupled foreground/background model.

Added script:

```bash
v5_work/geom_locked_composite.py
```

Harder first composite was too synthetic and high-halo. A softer version is more
informative:

```bash
/opt/homebrew/opt/python@3.14/bin/python3.14 v5_work/geom_locked_composite.py \
  --fake v5_work/eval_noC_silAB/fake_B \
  --ds datasets/chess_v5_oblique_aligned_bright \
  --out v5_work/eval_geom_locked_soft_silAB/fake_B \
  --feather 2.0 \
  --piece-blur 1.6 \
  --contrast 0.38 \
  --shading-strength 14
```

Results vs `bright_silAB`:

- Existing vertical detector:
  - `bright_silAB`: 27.7% all, 29.5% pawns.
  - `geom_locked_soft_silAB`: 18.1% all, 20.3% pawns.
- Split-cue audit:
  - `bright_silAB`: 53.4% all, 59.3% pawns.
  - `geom_locked_soft_silAB`: 24.0% all, 26.2% pawns.

Visual read: not final quality. Pieces look too milky/synthetic and blending
needs work, but double-head/side-lobe failures are materially reduced. This is
strong evidence that a trainable geometry-gated or decoupled foreground approach
is worth implementing if the pure training ablations do not solve it.

Artifacts:

- `v5_work/eval_geom_locked_soft_silAB/fake_B`
- `v5_work/head_crops_geom_locked_soft_silAB_Pp.png`
- `v5_work/double_head_audit_geom_locked/`

## Next Decision Rule

Both pure training ablations failed:

- `noPieceRecon` did not improve double-head/split-cue defects.
- `fen_noPieceRecon` got worse in split-cue, halo, and visual artifacts.

Next step: move to a trainable geometry-gated foreground/background approach.
The local soft composite is the only intervention so far that materially reduced
the structural defect (`53.4% -> 24.0%` all split-cue, `59.3% -> 26.2%` pawns),
even though it is not visually final. The implementation should preserve or
strongly constrain piece geometry while learning better foreground style and
boundary blending.

## Experiment 3: Model-Integrated Geometry Lock, Test-Only

Implemented a model option in `v5_work/v5_cluster_src/paired_geom_hd_model.py`:

- `--geom_lock_alpha`
- `--geom_lock_feather`
- `--geom_lock_blur`
- `--geom_lock_contrast`

This blends piece regions toward a class-wise AdaIN-style synthetic source
stylization inside the true synthetic silhouette. The generator still controls
background and piece color statistics, but high-contrast foreground structure is
locked to the source geometry.

Updated cluster scripts:

- `v5_work/v5_cluster_src/train_v5_oblique_hd.sbatch`
- `v5_work/v5_cluster_src/test_v5_oblique_hd.sbatch`

Uploaded and installed to the cluster repo. Backup on cluster:
`~/chess_cut_project/v5_cluster_src/backup_20260615_105024`.

Test-only chain on existing `bright_silAB` generator:

- Checkpoint copy:
  `checkpoints/chess_v5_bright_silAB_glock_t1/latest_net_G.pth`
- Test: `18172155`
- Score: `18172156`

Config:

```bash
NAME=chess_v5_bright_silAB_glock_t1
DATAROOT=./datasets/chess_v5_oblique_aligned_bright
SEMSRC=fen_silhouette
GLOCK=1.0
GLOCKFEATHER=2
GLOCKBLUR=2
GLOCKCONTRAST=0.38
```

Decision rule: if test-only geometry lock cuts split-cue defects comparably to
the local soft composite without unacceptable color/halo artifacts, submit a
short train run with the same lock so the generator can adapt colors and
background under the locked foreground geometry.

### `glock_t1` result

Config:

```bash
GLOCK=1.0
GLOCKFEATHER=2
GLOCKBLUR=2
GLOCKCONTRAST=0.38
```

Result: structurally strong but visually too washed/transparent.

- Split-cue audit:
  - baseline `bright_silAB`: 53.4% all, 59.3% pawns.
  - `glock_t1`: 12.2% all, 14.0% pawns.
- Legacy vertical detector:
  - `glock_t1`: 8.9% all, 11.8% pawns.
- Defect audit:
  - `transparent_head_flag_rate=0.917`
  - `detail_ratio_mean=0.456`

Conclusion: geometry lock can make double heads/side lobes mostly disappear,
but this parameterization is too milky for final output.

### `glock_t2` result

Config:

```bash
GLOCK=1.0
GLOCKFEATHER=1
GLOCKBLUR=1
GLOCKCONTRAST=0.85
```

Result: better contrast/detail but too many artifacts returned.

- Split-cue audit:
  - `glock_t2`: 35.3% all, 35.2% pawns.
  - worse than `glock_t1` and worse than local `geom_locked_soft_silAB` (24.0% all).
- Legacy vertical detector:
  - `glock_t2`: 17.3% all, 20.9% pawns.
- Defect audit:
  - `transparent_head_flag_rate=0.405`
  - `detail_ratio_mean=0.896`
  - `edge_halo_flag_rate=0.275`

Conclusion: higher retained contrast and lower smoothing solve much of the
washed-piece issue, but leak back structural defects. Do not train `t2` as-is.

### Next run: `glock_t1p5`

Decision: run a test-only midpoint before committing to training:

```bash
GLOCK=1.0
GLOCKFEATHER=2
GLOCKBLUR=2
GLOCKCONTRAST=0.60
```

Rationale: keep the stronger smoothing/mask behavior from `t1`, but raise
contrast only moderately. If it stays near `t1` structurally while improving
transparency/detail, use it for a short train run. If it remains washed, train
the cleaner `t1`-style lock so the raw generator can learn compensating color
statistics under the geometry-locked output.

Result:

- Legacy vertical detector:
  - `glock_t1p5`: 11.9% all, 15.2% pawns.
- Split-cue audit:
  - `glock_t1p5`: 19.2% all, 20.8% pawns.
  - better than local `geom_locked_soft_silAB` (24.0% all) and much better than
    `glock_t2` (35.3% all), but not as clean as `glock_t1` (12.2% all).
- Defect audit:
  - `transparent_head_flag_rate=0.816`
  - `detail_ratio_mean=0.566`
  - still too washed for final output, though slightly better than `glock_t1`.

Conclusion: `t1p5` is the best geometry-lock compromise so far, but the remaining
failure is style/opacity, not head topology. Move to a short train run so the raw
generator can adapt its foreground statistics while the output remains geometry
locked.

## Experiment 4: Short Geometry-Locked Training Probe

Submitted:

- Train: `18172249`
- Test: `18172250`
- Score: `18172251`

Config:

```bash
NAME=chess_v5_bright_silAB_glock_t1p5_train12
INIT=./checkpoints/chess_v5_bright_silAB/latest_net_G.pth
DATAROOT=./datasets/chess_v5_oblique_aligned_bright
SEMSRC=fen_silhouette
GLOCK=1.0
GLOCKFEATHER=2
GLOCKBLUR=2
GLOCKCONTRAST=0.60
BS=1
EP=12
EPD=6
PCLS=0
PGAN=1.0
PFM=10.0
PVGG=5.0
PIECEW=0.35
VGG=3.0
EDGE=0.0
CX=0.0
PCX=0.0
```

Reason for `PCLS=0`: the frozen square classifier is badly miscalibrated on
geometry-locked outputs and should not steer the training objective in this
probe. Keep adversarial/perceptual local piece pressure to improve opacity and
style, but do not use classifier reward as a proxy for human visual quality.

Result:

- Jobs completed successfully:
  - Train `18172249`: 49m39s.
  - Test `18172250`: 1m28s.
  - Score `18172251`: 9s.
- Important caveat: `test latest` evaluated the epoch-15/latest checkpoint,
  because `latest_net_G.pth` was last saved at epoch 15. Epochs 16-18 completed
  but were not saved by the current `save_epoch_freq=5` / `save_latest_freq`
  setup.
- Legacy vertical detector:
  - test-only `glock_t1p5`: 11.9% all, 15.2% pawns.
  - trained `glock_t1p5_train12`: 11.0% all, 15.2% pawns.
- Split-cue audit:
  - test-only `glock_t1p5`: 19.2% all, 20.8% pawns.
  - trained `glock_t1p5_train12`: 17.5% all, 18.8% pawns.
- Defect/style audit:
  - `transparent_head_flag_rate`: 0.816 -> 0.792, only a small improvement.
  - `detail_ratio_mean`: 0.566 -> 0.490, worse.
  - `edge_halo_flag_rate`: 0.241 -> 0.443, much worse.

Conclusion: training did not collapse geometry and did slightly improve the
split-cue metrics, but it did not solve the visual problem. It leaves the pieces
too soft/transparent and adds boundary/color halo. Do not scale this exact
training recipe. The next useful change should be in the geometry-lock rendering
formula or save/eval schedule, not just more epochs with the same objective.

## Experiment 5: Stop Using Synthetic Texture

User visual assessment: the geometry-locked outputs look unacceptable and
synthetic. This is consistent with the implementation: it copies source render
structure/texture into the final output, so it can remove extra lobes while also
making the pieces look like Blender pieces rather than real pieces.

Local probes:

- `shape_clip_composite.py`: keeps raw generator output and only clips near-piece
  spill outside the synthetic silhouette.
  - Style/defect audit for `shape_clip_r16` looked much better
    (`transparent_head_flag_rate=0.006`, `detail_ratio_mean=0.920`, halo 0),
    but split/head artifacts got worse (`any_extra_rate=65.4%` all). It also
    visually produced top-view blobs/patches, not acceptable pieces.
- `real_piece_template_composite.py`: attempted a train-only real-piece template
  bank. First alpha extraction was unstable because board texture and square
  boundaries dominated the masks. This path is not ready as a quick fix.

Conclusion: no more synthetic geometry-lock as a final path. Clipping alone
preserves real-ish style but does not fix topology. The next probe should make
the generator itself learn real-looking crop structure, not paste source render
texture.

## Experiment 6: Real-Style Crop Contextual Probe

Submitted a non-locking training probe:

- Train: `18177965`
- Test: `18177966`
- Score: `18177967`

Config:

```bash
NAME=chess_v5_bright_silAB_pcx_train15
INIT=./checkpoints/chess_v5_bright_silAB/latest_net_G.pth
DATAROOT=./datasets/chess_v5_oblique_aligned_bright
SEMSRC=fen_silhouette
GLOCK=0.0
BS=1
EP=10
EPD=5
PCLS=0
PGAN=0.5
PFM=5.0
PVGG=8.0
PCX=3.0
PIECEW=0.15
VGG=1.5
EDGE=0.0
CX=0.0
```

Rationale: target real piece-crop structure using crop-level Contextual Loss and
local perceptual pressure, while avoiding the frozen classifier and avoiding any
synthetic texture lock. Total epoch count is 15 so the default save schedule
evaluates a saved latest checkpoint at epoch 15.

Result:

- Jobs completed successfully:
  - Train `18177965`: 1h09m46s.
  - Test `18177966`: 1m27s.
  - Score `18177967`: 10s.
- The epoch-15/latest checkpoint was saved and evaluated.
- Style/defect audit:
  - `transparent_head_flag_rate=0.0047` (good; no synthetic-lock milkiness).
  - `edge_halo_flag_rate=0.161` (acceptable relative to geometry-lock train).
  - `detail_ratio_mean=0.807` (not collapsed, but softer than real).
- Legacy vertical detector:
  - `pcx_train15`: 26.8% all, 28.2% pawns.
  - baseline `bright_silAB`: ~27.7% all, ~29.5% pawns.
- Split-cue audit:
  - `pcx_train15`: 53.1% all, 59.8% pawns.
  - baseline `bright_silAB`: 53.4% all, 59.3% pawns.

Conclusion: PCX/local real-crop pressure preserves non-synthetic style, but it
does not fix the double-head/side-lobe topology. It is essentially baseline on
the primary artifact. The core issue is not solved by contextual/perceptual crop
matching alone.

## Experiment 7: Two-Stream Piece Composite Probe

Implemented an opt-in structural mode in `paired_geom_hd_model.py`:

- `--piece_comp_alpha`
- `--piece_comp_feather`

When enabled, `netG` outputs 6 channels instead of 3:

- first RGB stream: background
- second RGB stream: foreground/piece appearance
- final output: `foreground * alpha + background * (1-alpha)`

`alpha` is the synthetic silhouette mask only. No synthetic RGB texture is copied
into the image. The old 3-channel checkpoint head is expanded into both streams
on warm-start so the initial composite starts close to `bright_silAB`, not random
noise.

Submitted:

- Train: `18178817`
- Test: `18178818`
- Score: `18178819`

Config:

```bash
NAME=chess_v5_bright_silAB_pcomp_train10
INIT=./checkpoints/chess_v5_bright_silAB/latest_net_G.pth
DATAROOT=./datasets/chess_v5_oblique_aligned_bright
SEMSRC=fen_silhouette
GLOCK=0.0
PCOMP=1.0
PCOMPFEATHER=1
BS=1
EP=5
EPD=5
PCLS=0
PGAN=1.0
PFM=10.0
PVGG=5.0
PCX=0.0
PIECEW=0.7
VGG=3.0
EDGE=0.0
CX=0.0
```

Rationale: test the proposed structural compromise directly: constrain visible
piece shape with a matte, but make the foreground texture fully generated and
trained against real targets, avoiding the synthetic RGB lock that the user
rightly rejected.

Result:

- Jobs completed successfully:
  - Train `18178817`: 28m07s.
  - Test `18178818`: 1m30s.
  - Score `18178819`: 10s.
- Warm-start behaved as intended: the old 3-channel G head was expanded into the
  6-channel two-stream head (`expanded=2; missing=0`).
- Style/defect audit:
  - `transparent_head_flag_rate=0.0056` (good; no synthetic-lock milkiness).
  - `edge_halo_flag_rate=0.195` (acceptable, but not better than PCX).
  - `detail_ratio_mean=0.837` (style not collapsed).
- Legacy vertical detector:
  - `pcomp_train10`: 28.3% all, 30.1% pawns.
  - baseline `bright_silAB`: ~27.7% all, ~29.5% pawns.
- Split-cue audit:
  - `pcomp_train10`: 53.45% all, 60.0% pawns.
  - baseline `bright_silAB`: 53.4% all, 59.3% pawns.

Conclusion: the two-stream composite preserves non-synthetic style, but it does
not solve topology. A fixed silhouette alpha is not enough; the foreground stream
still learns blob-like real texture inside the matte, and the head split remains
baseline-level. The remaining effective constraint so far is the rejected
synthetic RGB geometry lock, so the next viable direction must constrain internal
piece structure/topology, not only outer alpha.

## Experiment 8: Two-Stream Composite + Real Pseudo-Mask Shape Loss

Implemented an opt-in crop-level soft shape loss:

- `--lambda_piece_shape`
- `--piece_shape_size`
- `--piece_shape_thresh`
- `--piece_shape_temp`
- `--piece_shape_out_w`

For each sampled occupied crop, the model estimates a foreground saliency mask
from fake and real crops by comparing crop pixels to their own border-estimated
background. It then matches the fake soft mask to the real soft mask and adds an
extra penalty for fake foreground outside the real foreground. This is intended
to constrain internal piece topology without copying synthetic RGB.

Submitted:

- Train: `18179263`
- Test: `18179264`
- Score: `18179265`

Config:

```bash
NAME=chess_v5_bright_silAB_pcomp_shape_train10
INIT=./checkpoints/chess_v5_bright_silAB/latest_net_G.pth
DATAROOT=./datasets/chess_v5_oblique_aligned_bright
SEMSRC=fen_silhouette
GLOCK=0.0
PCOMP=1.0
PCOMPFEATHER=1
BS=1
EP=5
EPD=5
PCLS=0
PGAN=1.0
PFM=10.0
PVGG=5.0
PCX=0.0
PSHAPE=8.0
PSHAPESZ=48
PSHAPETH=0.10
PSHAPETEMP=0.035
PSHAPEOUT=2.0
PIECEW=0.7
VGG=3.0
EDGE=0.0
CX=0.0
```

Rationale: pcomp alone constrained only the outer alpha and failed. This probe
adds an internal real-derived shape constraint while preserving the non-synthetic
foreground/background generation path.

Result:

- Jobs completed successfully:
  - Train `18179263`: 28m12s.
  - Test `18179264`: 1m31s.
  - Score `18179265`: 10s.
- Style/defect audit:
  - `transparent_head_flag_rate=0.0062`, so it avoids the milky geometry-lock
    failure.
  - `edge_halo_flag_rate=0.151`, acceptable.
  - `detail_ratio_mean=0.802`, similar to other non-locking probes.
- Legacy vertical detector:
  - `pcomp_shape_train10`: 27.1% all, 29.1% pawns.
- Split-cue audit:
  - `pcomp_shape_train10`: 51.2% all, 57.3% pawns.
  - `pcomp_train10`: 53.45% all, 60.0% pawns.
  - baseline `bright_silAB`: 53.4% all, 59.3% pawns.

Conclusion: real-derived pseudo-mask shape loss gives only a small improvement.
It preserves non-synthetic style, but it does not materially solve the double
head / side-lobe topology. Visual crops still show flattened blobs and side
lobes. The likely reason is that the real target crops are not a stable aligned
shape target; the model can satisfy the soft saliency objective while still
placing a blob in the wrong location. The next test should use the synthetic
silhouette as a topology target only, without copying synthetic RGB.

## Experiment 9: Two-Stream Composite + Source-Silhouette Shape Guardrail

Implemented an opt-in crop-level source-shape loss:

- `--lambda_piece_src_shape`
- `--piece_src_shape_in_w`
- `--piece_src_shape_out_w`
- `--piece_src_shape_blur`

For each sampled occupied crop, the model extracts a differentiable foreground
saliency mask from the fake crop, as in Experiment 8, but compares it to the
rendered source silhouette crop instead of the real crop. This is deliberately
not a geometry-lock: it does not paste source RGB or source texture into the
output. It only penalizes generated salient structure outside the rendered
piece silhouette, with a weak coverage term inside the silhouette.

Submitted:

- Train: `18179419`
- Test: `18179420`
- Score: `18179421`

Config:

```bash
NAME=chess_v5_bright_silAB_pcomp_srcshape_train10
INIT=./checkpoints/chess_v5_bright_silAB/latest_net_G.pth
DATAROOT=./datasets/chess_v5_oblique_aligned_bright
SEMSRC=fen_silhouette
GLOCK=0.0
PCOMP=1.0
PCOMPFEATHER=1
BS=1
EP=5
EPD=5
PCLS=0
PGAN=1.0
PFM=10.0
PVGG=3.0
PCX=0.0
PSHAPE=0.0
PSRCSHAPE=12.0
PSRCSHAPEIN=0.15
PSRCSHAPEOUT=4.0
PSRCSHAPEBLUR=1
PIECEW=0.25
VGG=1.5
EDGE=0.0
CX=0.0
```

Initial smoke status:

- Active train job `18179419` started on `ise-pheno-05`.
- Options show `lambda_piece_src_shape=12.0` and `piece_comp_alpha=1.0`.
- Warm-start expansion is correct: `expanded=2; missing=0`.
- First printed loss includes `G_PSRC: 27.686`, confirming the new topology
  guardrail participates in backward.

Result:

- Jobs completed successfully:
  - Train `18179419`: 59m55s.
  - Test `18179420`: 1m28s.
  - Score `18179421`: 9s.
- Style/defect audit:
  - `transparent_head_flag_rate=0.0043`, so style opacity is not the failure.
  - `edge_halo_flag_rate=0.217`, acceptable but not best.
  - `detail_ratio_mean=0.880`, better texture/detail than `pcomp_shape_train10`.
- Legacy vertical detector:
  - `pcomp_srcshape_train10`: 31.3% all, 34.7% pawns.
  - This is worse than `pcomp_shape_train10` (27.1% all, 29.1% pawns) and
    worse than baseline-level non-locking probes.
- Split-cue audit:
  - `pcomp_srcshape_train10`: 56.65% all, 64.66% pawns.
  - Baseline `bright_silAB`: 53.42% all, 59.32% pawns.
  - `pcomp_shape_train10`: 51.24% all, 57.28% pawns.
  - `glock_t1p5_train12`: 17.48% all, 18.80% pawns, but visually rejected as
    washed/synthetic.

Conclusion: source-silhouette shape as a loss, without RGB locking, did not
constrain the generator into the desired topology. It improved neither the
visible artifact nor the quantitative split-cue metric; it worsened pawns in
particular. The current evidence says small objective changes inside this
unpaired GAN family are not likely to deliver the required visual bar. The only
intervention that reliably removed double heads was hard geometry locking, and
that produced unacceptable synthetic/milky pieces. The next useful work should
be a strategy change rather than another minor loss-weight ablation.
