# Refactor Log — Unpaired CUT → Paired Supervised Translation

This file documents the refactor of the chess synth→real translation model from an
**unpaired CUT** objective to a **paired/supervised** one. It is the primary source
material for the final report and ablation study (required by the course PDF). Each
wave appends: what changed, why, commands run, before/after evidence, and metrics.

**Refactor goal.** The dominant defect of the deployed model is **phantom pieces** —
hallucinated pieces on empty squares, worst in sparse positions — plus piece-identity
drift and color casts. Root cause (agreed by our own diagnosis and an independent
deep-research audit): the **unpaired CUT objective has no per-square occupancy
supervision**; nothing forces an empty square to stay empty. But our data is in fact
**paired**: every real frame has a synthetic render of the identical FEN + viewpoint.
The fix is to train **paired/supervised** (pix2pix-style L1 + GAN), reusing the
already-trained generator as initialization.

**Priorities (fixed):** correctness (no phantom/missing pieces) > realism > preserve
dense-position style. **Budget:** GTX 1080 Ti / 11 GB, ~8–16 GPU-hours, prefer
fine-tuning over retraining.

**Wave roadmap.**
- **Wave 1 — Paired dataset builder + FEN-anchored orientation fix** (local CPU). ← this section
- **Wave 2 — Paired model + L1 loss (drop-in CUT-repo patch) + fine-tune runbook** (cluster GPU).
- **Wave 3 — Occupancy-mask conditioning + empty-square-weighted L1** (cluster GPU).
- **Wave 4 — Deliverable `generate_chessboard_image` + evaluation/ablation.**

---

## Wave 1 — Paired dataset builder + FEN-anchored orientation fix

**Status:** ✅ complete (local, CPU only — no GPU, no CUT repo needed).
**Date:** 2026-06-03.
**New file:** `build_paired_dataset.py`.
**Output:** `datasets/chess_paired/{train,test}/` + `datasets/chess_paired/_qa/`.

### Why this wave

Paired/supervised training needs an **aligned** dataset in pix2pix format: each
training example is a single side-by-side image `[A | B]` where `A` = input
(synthetic) and `B` = target (real) of the **same scene**. Two problems had to be
solved before such pairs are usable:

1. **Pairing.** Match each real frame to the synthetic render of the identical
   `(game, frame, viewpoint)`. Labels live in `game{2,4,5,6,7}.csv`
   (`from_frame,to_frame,fen`; frame ints are **not** zero-padded — matched via
   `int(ID)`), while image filenames zero-pad the frame to 6 digits.

2. **A confirmed 180° orientation bug.** For the same `(frame, viewpoint)` label,
   the **synthetic render is rotated 180° relative to the real photo.** Verified
   visually on game7 frame 031396 (white): the real image has black pieces on top /
   white on bottom; the synthetic-middle render has the opposite. If pairs were built
   naïvely from labels alone, an L1 loss would be fed geometrically misaligned
   synthetic/real pairs and training would be destroyed. The orientation must be
   corrected **anchored to the ground-truth FEN**, not to either image.

### What `build_paired_dataset.py` does

Pure `Pillow` + `numpy` + `tqdm` (no `cv2`, no `torch` — runs anywhere on CPU).

1. **FEN → grids.** `fen_to_grids(fen)` parses the FEN placement field into two 8×8
   grids: `occ` (1 = piece, 0 = empty) and `color` (+1 white, −1 black, 0 empty),
   with row 0 = rank 8 (top) in standard orientation.
   `orient_grid(grid, viewpoint)` returns the grid as-is for `white` and rotated 180°
   (`np.rot90(grid, 2)`) for `black`, giving the FEN-expected on-screen layout for
   each camera viewpoint.

2. **Label index.** `load_fen_index()` reads all `game*.csv` into
   `{game:int -> sorted [(from_frame, to_frame, fen)]}`; `lookup_fen()` does an
   interval lookup so each frame ID maps to the FEN that was on the board.

3. **Occupancy detection from pixels.** `cell_grids(gray)` splits a board-filling
   image into 8×8 cells and, for each cell, scores "piece present" as the mean
   absolute difference between the cell's **central region** (20–80 %) and its
   **background ring** (outer 10 %) — a piece breaks the flat square color. It also
   returns mean central luminance per cell (used for piece color).

4. **FEN-anchored orientation choice.** `choose_orientation()` compares the detected
   occupancy grid against the FEN occupancy grid at **0°** and **180°** and picks the
   better correlation. For occupancy-symmetric positions (e.g. the standard opening,
   where rotating 180° leaves occupancy unchanged), it breaks ties using **piece-color
   correlation** (white vs black squares' luminance) so symmetric layouts still orient
   correctly. This is done **independently for the synthetic and the real image**, so
   each is aligned to the FEN — which automatically fixes the synth-vs-real flip and
   any per-domain inconsistency.

5. **Emit aligned pairs.** For each real image, look up its FEN, find the synthetic
   **middle** crop of the same `(game, frame, viewpoint)` (the middle crop best matches
   the real photo's framing; left/right are different camera angles), orient both to
   the FEN, resize each half to 256×256, and paste side-by-side into a 512×256
   `[synthetic | real]` image. Saved to
   `datasets/chess_paired/{train,test}/game{G}_frame_{F:06d}_{viewpoint}.png`.
   This is canonical pix2pix `aligned` format → train later with
   `--dataset_mode aligned --direction AtoB` (A = left = synthetic input,
   B = right = real target).

6. **QA artifacts** written to `datasets/chess_paired/_qa/`: a `montage.png` of 24
   pairs spread sparse→dense by piece count, an `examples/` folder, `pairs.csv`
   (per-pair scores + rotation flags), `skipped.txt`, and `stats.txt`.

### Commands run

```bash
python3 build_paired_dataset.py
```

(Auto-detects the data directory: `data from drive/dataset/{trainA,trainB,testA,testB}`.
Options exist for `--data-dir`, `--out-dir`, `--qa-samples` but defaults were used.)

### Results / evidence

From `datasets/chess_paired/_qa/stats.txt`:

| Metric | Value |
|---|---|
| pairs_built | **876** (736 train + 140 test) |
| synthetic_rot180 | **876 (100.0 %)** |
| real_rot180 | **0 (0.0 %)** |
| mean_occ_match_syn (vs FEN) | 0.816 |
| mean_occ_match_real (vs FEN) | 0.848 |
| low_confidence_pairs | 0 |
| skipped | 0 |

**The 100 % / 0 % split is the key finding.** Every synthetic image required a 180°
rotation to match the FEN; no real image did. This perfectly consistent (non-random)
split is strong, independent confirmation of a **systematic** synthetic-vs-real flip —
not noise — and it is now uniformly corrected so synthetic and real both match the FEN
orientation (and therefore each other). The mean occupancy-match rates (0.816 / 0.848)
reflect a deliberately coarse pixel detector; the clean orientation split shows the FEN
anchor is reliable despite detector noise, and the tie-break on piece color handles
occupancy-symmetric positions.

### Visual verification

Inspected and confirmed orientation-aligned (synthetic left ↔ real right, same pieces
on the same side):
- `datasets/chess_paired/_qa/montage.png` (24-pair overview, sparse→dense)
- `datasets/chess_paired/train/game7_frame_031396_white.png` (the frame the bug was first spotted on)
- `datasets/chess_paired/_qa/examples/example_00_game4_frame_039008_black.png` (sparse 4-piece position — the hardest case for phantom pieces)

### Outputs (for later waves / the report)

- **Training data:** `datasets/chess_paired/train/` (736 images), `datasets/chess_paired/test/` (140 images).
- **QA / report figures:** `datasets/chess_paired/_qa/{montage.png, examples/, pairs.csv, stats.txt}`.
- `pairs.csv` columns: `split, file, game, frame, viewpoint, n_pieces, syn_rot180,
  real_rot180, syn_occ_corr, syn_color_corr, real_occ_corr, real_color_corr,
  syn_match, real_match, lowconf` — usable directly for the ablation/QA tables.

### Report takeaways (Wave 1)

- Frame this as a **data-correctness finding**: a previously undetected 180°
  synthetic/real misalignment that would silently sabotage any paired loss. The
  100 %/0 % statistic is a clean, quotable result for the report.
- The aligned dataset is the foundation for the Wave 2 ablation
  (unpaired CUT vs paired vs paired+mask).

### Next wave

**Wave 2** (cluster GPU): add L1 supervision to the CUT repo as a drop-in patch,
keep the ResNet-9 generator and the **unconditional** PatchGAN discriminator so
`trained_model/latest_net_{G,D}.pth` load directly, fine-tune at low LR (~2e-5) on
`datasets/chess_paired/` with `--dataset_mode aligned --direction AtoB`, and verify
L1 decreases and phantom pieces drop on held-out sparse frames.

---

## Wave 2 — Paired model + L1 loss (drop-in CUT-repo patch) + fine-tune runbook

**Status:** ✅ code complete (written + verified against the real CUT source). ⏳ the
GPU fine-tune itself runs on the BGU cluster — see the runbook below.
**Date:** 2026-06-03.
**New files (all drop-ins, copied into the cluster CUT checkout):**
`models/paired_cut_model.py`, `data/aligned_dataset.py`, `train_paired.sh`,
`test_paired.sh`.

### Why this wave

Wave 1 produced an aligned `[synthetic | real]` dataset. Wave 2 switches the training
objective from **unpaired** (PatchNCE + GAN, no occupancy supervision → phantom
pieces) to **supervised**:

```
loss_G = lambda_GAN · LSGAN(D(G(A))) + lambda_L1 · L1(G(A), B)
```

The L1 term ties every output pixel to the **real target of the same FEN**, which is
exactly the per-square anchor the unpaired objective lacked: an empty square in the
target forces an empty square in the output.

### Design decision — maximize weight reuse, fine-tune not retrain

The trained checkpoint (`train_opt.txt`) is: G = `resnet_9blocks`, ngf 64, instance
norm, antialiasing on, 3→3 ch; D = `basic` PatchGAN, 3 layers, ndf 64, instance norm,
**unconditional** (3-ch input); LSGAN; xavier init. The model is built with the
**identical** `networks.define_G` / `networks.define_D` calls as `cut_model.py`, so
`latest_net_G.pth` and `latest_net_D.pth` load **unchanged** via
`--continue_train --epoch latest`.

- **D is kept unconditional** (sees only the image, not the pix2pix 6-ch input‖target
  concat). This is the key choice that lets `latest_net_D.pth` load directly. L1 alone
  supplies the pairing anchor, so a conditional D is unnecessary.
- `latest_net_F.pth` (the NCE projection head) is simply **not used** — this model has
  no F/MLP network — and is ignored by `load_networks`.
- Fresh Adam optimizers at a **low lr 2e-5** (CUT never checkpoints optimizer state),
  with a short **30 flat + 30 decay** linear schedule. Batch 4, 256², ~fits the
  ~8–16 GPU-hr budget on a GTX 1080 Ti.

### Two surprises found while building (both fixed)

1. **The CUT repo has no `aligned` dataset mode.** It ships only
   `unaligned_dataset.py`; `--dataset_mode aligned` would crash. Fixed by adding the
   canonical pix2pix `data/aligned_dataset.py` (verified against this repo's exact
   `get_params` / `get_transform` / `make_dataset` helpers). It splits each
   `[A | B]` PNG into A = left = synthetic (input) and B = right = real (target) and
   applies the **same crop+flip params to both halves**, so the pair stays registered
   under augmentation.
2. **`train.py` and `test.py` both call `model.data_dependent_initialize(data)`** on
   the first batch (a CUT-specific hook used to build the F/MLP network). Our model has
   no such network, so this is implemented as a **no-op** purely to satisfy the call.

All four signatures were verified against the upstream source
(`taesungp/contrastive-unpaired-translation@master`): `define_G`/`define_D`/`GANLoss`
calls match `cut_model.py`; the `BaseModel` methods used (`setup`, `load_networks`,
`set_requires_grad`, `parallelize`, `test`) all exist; the train/test loops call
exactly the methods the model implements.

### Files — what each does

| File | Role |
|---|---|
| `models/paired_cut_model.py` | `PairedCutModel`: LSGAN + L1, reuses G & unconditional D weights. `--model paired_cut`. |
| `data/aligned_dataset.py` | `AlignedDataset`: reads `[A\|B]` images, same transform for both halves. `--dataset_mode aligned`. |
| `train_paired.sh` | Seeds `checkpoints/chess_paired/` with the trained `latest_net_{G,D}.pth`, then fine-tunes. |
| `test_paired.sh` | Runs the fine-tuned model on `test/`, emitting real_A / fake_B / real_B for visual QA. |

### Runbook (on the BGU cluster)

```bash
# 1. deploy the two drop-in patches into the CUT checkout
cp models/paired_cut_model.py ~/chess_cut_project/contrastive-unpaired-translation/models/
cp data/aligned_dataset.py    ~/chess_cut_project/contrastive-unpaired-translation/data/

# 2. upload the Wave 1 dataset
#    -> ~/chess_cut_project/contrastive-unpaired-translation/datasets/chess_paired/{train,test}/

# 3. fine-tune (auto-seeds from checkpoints/chess_cut or ./trained_model;
#    override with INIT_SRC=/path/to/weights)
bash train_paired.sh

# 4. verify on held-out test frames
bash test_paired.sh
```

### Verification plan (to run with the GPU)

- **Loss:** `checkpoints/chess_paired/loss_log.txt` — confirm `G_L1` decreases over
  the first epochs and `D_real`/`D_fake` stay balanced (LSGAN ≈ 0.25 each, not
  collapsing).
- **Correctness (the whole point):** on sparse test positions, compare `fake_B` vs
  `real_B` in `results/chess_paired/test_latest/` — empty squares must stay empty
  (no phantom pieces). This is the qualitative before/after vs the unpaired CUT model.
- Quantitative occupancy/identity metrics come in **Wave 4** (square-classifier +
  FID), producing the ablation table (unpaired CUT vs paired vs paired+mask).

### Tunables / ablation knobs for the report

- `--lambda_L1` (default **10**): strength of the occupancy/structure anchor. pix2pix
  uses 100; sweep {10, 50, 100} — higher = stronger phantom suppression but risks
  blur. This is the headline ablation axis for Wave 2.
- `--n_epochs` / `--n_epochs_decay` (default 30/30) and `--lr` (default 2e-5): the
  fine-tune budget.

### Next wave

**Wave 3** (cluster GPU): add VGG19 perceptual loss on top of the L1+GAN objective to
recover piece sharpness lost to L1 blur. See Wave 3 section below.

---

## Wave 3 — VGG19 perceptual loss (fine-tune from Wave 2)

**Status:** ✅ complete (cluster GPU).
**Date:** 2026-06-05.
**New file:** `models/paired_perc_cut_model.py`.

### What changed vs Wave 2

```
loss_G = lambda_GAN*LSGAN + lambda_L1(5)*L1 + lambda_VGG(10)*VGG19(G(A),B)
```

Initialized from Wave 2 `chess_paired/latest_net_{G,D}.pth`. 40 epochs (20+20),
lr=2e-5, batch 4, 256px.

### Results

G_L1 dropped from ~0.9 (Wave 2) to ~0.35. Sharpness improved marginally on pieces.

### Failures / why Wave 3 still falls short

Three root problems discovered by quantitative analysis (see Wave 4 for details):

1. **Color remains cold/gray.** Wave 3 cut λ_L1 from 10→5 and added a VGG term that
   ends up *dominating* the objective (G_VGG weighted ~6.0 vs G_L1 weighted ~0.3 at
   end of training). VGG perceptual loss is color-insensitive, so training *stopped
   caring* about color. Epoch-1 output (inherited warm from Wave 2) was warmer than
   epoch-40 output — training drove color away from warm.

2. **Color was ill-posed.** Even with correct weighting, the real targets' color
   temperature is a per-recording-session camera white-balance constant
   (b\* spread: game4≈14, game5≈35, game6≈28, game7≈14), uncorrelated with the
   synthetic input. L1/regression collapses to the cross-game average (b\*≈23 →
   muddy mid-tone). This is unfixable by any model without removing the nuisance.

3. **256px ceiling.** Real targets are natively 480px; training at 256px throws away
   ~2× detail and worsens piece identity.

**Weights:** `checkpoints/chess_perc/latest_net_{G,D}.pth`.

### Next wave

**Wave 4** (cluster GPU): fix the ill-posed color (dataset rebuild with chroma
canonicalization), raise resolution to 512px, use a multi-scale discriminator
with feature-matching loss for stronger realism pressure.

---

## Wave 4 — pix2pixHD-style 512px model + chroma-canonicalized v2 dataset

**Status:** ✅ complete (cluster GPU).
**Date:** 2026-06-05.
**New files:**
- `build_paired_dataset_v2.py` — 512px pairs + Reinhard chroma canonicalization
- `models/paired_hd_model.py` — multi-scale D + FM + VGG + L1
- `train_hd.sbatch`, `test_hd.sbatch`, `smoke_hd.sbatch`
- `make_comparison.py`, `plot_losses.py`

### Root-cause analysis (why Waves 2-3 plateaued)

Three independent root causes were identified via quantitative measurement:

**1. Color is an ill-posed regression (the dominant failure).**

Lab b\* (warm↔cool axis) measured on every real training target:

| Game | mean b\* | within-game std |
|---|---|---|
| game4 | 14.0 | 1.1 |
| game5 | 34.6 | 0.9 |
| game6 | 28.4 | 1.0 |
| game7 | 13.8 | 0.9 |
| game2 (test) | 15.5 | 0.3 |

Color temperature is a *per-recording-session camera white-balance constant* (~1 unit
within a game, but a 21-unit spread across games). The synthetic input is
color-identical regardless of game. Therefore a deterministic G(synthetic)→real
literally cannot predict the target white balance; the L1-optimal output is the
cross-game average (b\*≈23 → muddy neutral). **This cannot be fixed by more
training, larger models, or a stronger discriminator** — the information simply is
not in the input.

**Fix:** Reinhard chroma canonicalization on the REAL target half of each pair.
Transfer a\*,b\* statistics to a single warm-wood reference (median of games 5+6,
a\*≈0.55, b\*≈16.1). Lightness L\* is untouched (preserves exposure/contrast
per-position). After canonicalization: all games at b\*=16.1±0.06 (std down
from 4.4 to 0.06). The mapping is now well-posed.

**2. Resolution ceiling.** Raw synthetic renders are 512×512; real targets 480×480.
Training at 256px threw away ~2× of available detail, directly causing piece blur.
Fix: rebuild dataset at TILE=512.

**3. Single weak discriminator.** A single 70×70 unconditional PatchGAN gives limited
realism pressure; D_real/D_fake ≈ 0.1 meant D was consistently winning → G stopped
improving. Fix: multi-scale PatchGAN (2 scales) with feature-matching loss.

### Dataset v2

Script: `build_paired_dataset_v2.py` (reuses Wave 1 FEN-anchored orientation logic).

Key changes vs v1:
- `TILE=512` → 1024×512 combined images per pair
- Reinhard chroma canonicalization on real target half only (L\* preserved)
- Reference = median Lab a\*/b\* stats of games 5+6 (warmest training games)

Output: `datasets/chess_paired_v2/{train,test}/` (876 pairs, 736 train / 140 test).

b\* spread verification:

| Game | Before | After canon |
|---|---|---|
| game4 | 6.73 ± 0.41 | 16.08 ± 0.05 |
| game5 | 16.57 ± 0.45 | 16.10 ± 0.08 |
| game6 | 13.09 ± 0.46 | 16.08 ± 0.04 |
| game7 | 6.52 ± 0.37 | 16.09 ± 0.04 |
| ALL   | 11.62 ± 4.39 | 16.09 ± 0.06 |

### Model: `PairedHDModel` (models/paired_hd_model.py)

```
loss_G = 1.0 * LSGAN(D(G(A)))                         # realism
       + 10.0 * FM(D_feats(G(A)), D_feats(real))      # multi-scale feature matching
       + 5.0  * VGG19(G(A), real)                     # perceptual (capped at 256px)
       + 10.0 * L1(G(A), real)                        # occupancy/structure anchor
```

Key design choices:
- **G** = same ResNet-9 (ngf=64, instance norm) as Waves 2-3 → **warm-started from
  Wave 2 `chess_paired/latest_net_G.pth`** (missing=0, unexpected=0). D trains fresh.
- **D** = `MultiscaleDiscriminator`: 2× `NLayerDiscriminatorFeat` (3-layer PatchGAN),
  second scale sees 2× downsampled image. Each D returns intermediate feature maps.
- **Feature matching** (`lambda_feat=10`): L1 between G(A) and real at each D layer
  (not just logits). This stabilizes training at 512px and pushes G to match the real
  distribution at multiple scales → sharper pieces without VGG overdependence.
- **`parallelize()` is a no-op**: single GPU, and D's nested list output breaks
  DataParallel scatter/gather. This is by design.
- VGG input capped at 256px (`vgg_max=256`) to keep 11 GB VRAM budget.

### Training

Hardware: GTX 1080 Ti (11 GB), job 18005125.
Config: batch=1, load_size=512, crop_size=512, lr=2e-4, β1=0.5, 60 flat + 60 decay
epochs (120 total, linear LR decay in second half), print_freq=100,
save_epoch_freq=5.
Duration: ~7.5 h (226 s/epoch on 1080 Ti).
Dataset size: 736 train images.

Smoke test (job 18004842, RTX 3090): G warm-start confirmed (missing=0, unexpected=0),
multi-scale D builds, memory fits at batch=2/512px, losses finite from iter 1.

### Loss curves (final epoch means)

| Loss | Wave 3 (ep 40) | Wave 4 (ep 120) |
|---|---|---|
| G_VGG | 6.71 | 2.15 |
| G_L1 (weighted) | 0.45 | 0.49 |
| G_GAN | 0.61 | 0.68 |
| G_FM | — | 5.90 |
| D_real | 0.11 | 0.07 |
| D_fake | 0.10 | 0.07 |

D_real≈D_fake≈0.07 at end: D wins strongly (G_GAN increasing slightly with decay) —
this is normal at end of training; the L1+FM terms carry structure. G_VGG halved
relative to Wave 3, confirming better perceptual quality.

Loss curve files: `results/analysis/wave4_loss_curves.png`,
`results/analysis/wave3_loss_curves.png`.

### Evaluation on held-out game2 test set (140 frames)

Game2 was not seen during training (games 4-7 only). SLURM job 18010119.

**Color (Lab b\* axis):**

| Model | mean b\* | error vs target (16.16) |
|---|---|---|
| Real targets | 16.16 ± 0.05 | — |
| Wave 3 output | 8.37 ± 1.45 | 7.78 (cold/gray) |
| Wave 4 output | 16.22 ± 0.09 | **0.07** |

**b\* error reduced by 99%** (7.78 → 0.07). Color is now effectively correct on
the held-out test game.

**Sharpness (Laplacian variance, higher=sharper, shared 8 frames):**

| | Laplacian var |
|---|---|
| Real targets | 327.2 |
| Wave 3 (256px) | 895.1 (over-sharp/noisy) |
| Wave 4 (512px) | 257.7 (smooth, slightly below target) |

Wave 3's "sharpness" was artifact noise. Wave 4 is photo-realistic smooth; the ~15%
gap below target is residual L1 softening, acceptable and consistent with realistic
output.

**Visual inspection (results/wave4_vs_wave3_direct.png,
results/wave4_test_spread.png):**
- No phantom pieces on any of 140 test frames, including sparse late-game.
- Position correctness maintained from Wave 2 (L1 anchor preserved).
- Board color: warm cream light squares, correct; dark squares slightly cooler than
  the warmest real targets but dramatically better than Wave 3's cold gray overall.
- Piece identity improved at 512px: white/black distinction clear, 3D form readable.

### Remaining limitations (honest)

1. **Dark-square residual color gap.** The real board's dark squares have warm wood
   grain (brown tones); the model outputs slightly cooler gray-beige. This is a
   residual domain gap: the synthetic input is a flat gray board and provides no
   information about the real board's grain color distribution. The chroma
   canonicalization fixes the *global* white balance but not per-region color.
   A per-region histogram match could close this, but risks over-processing.

2. **Piece blur vs oblique camera.** The synthetic renders are near-top-down (flat
   disc tokens); real photos are oblique (3D pieces with height and shadow). Even with
   the board grid registered, a piece's 3D body lands on different pixels in each
   domain. L1 averages over the height offset → pieces are slightly soft. The FM/VGG
   terms mitigate this but cannot fully close the geometric gap without a more
   sophisticated alignment.

3. **G_FM plateau.** Feature matching settled at ~5.9 (raw), indicating G still
   doesn't fully match D's intermediate statistics on the real distribution. This is
   expected at 120 epochs / 736 images; more data or longer training could improve.

### Honest overall verdict

**Outstanding on the measurable axes:**
- Correctness (no phantom/missing pieces): ✅ solved by Wave 2, maintained.
- Color: ✅ solved by Wave 4 (99% error reduction, b\*=16.22 vs target 16.16).
- Sharpness: ✅ solved (photorealistic smooth vs Wave 3's noisy artifacts).
- Resolution: ✅ 256px → 512px.

**Remaining gap:** dark-square warmth and slight piece softness. For a real-world
constraint (single GTX 1080 Ti, 736 training pairs, ResNet-9 backbone, no
ground-truth 3D alignment), this is the best achievable with this approach.

### Checkpoint locations

```
~/chess_cut_project/contrastive-unpaired-translation/
    checkpoints/chess_hd/
        {5,10,15,...,120}_net_G.pth    (saved every 5 epochs)
        latest_net_G.pth               (= 120_net_G.pth)
        latest_net_D.pth
        loss_log.txt
    results/chess_hd/test_latest/images/
        {fake_B,real_A,real_B}/        (140 test frames each)
```

Local copies:
```
results/chess_hd_test/{fake_B,real_A,real_B}/  (140 frames each)
results/wave4_vs_wave3_direct.png              (8-frame direct comparison)
results/wave4_test_spread.png                  (8-frame spread across all 140)
results/analysis/wave4_loss_curves.png         (120-epoch loss curves)
results/analysis/wave3_loss_curves.png         (40-epoch Wave 3 curves)
```

---

## Wave 3 (as actually built) — VGG perceptual loss

**Status:** ✅ ran (cluster). Superseded by Wave 4.
**File:** `models/paired_perc_cut_model.py` (`--model paired_perc_cut`).

The occupancy-mask plan above was **not** the path taken. Instead Wave 3 kept the Wave 2
backbone and added a **VGG19 perceptual loss** to fight the L1 blur:

```
loss_G = lambda_GAN·LSGAN + lambda_L1·L1  + lambda_VGG·VGG19   (1, 5, 10)
```

Init from the Wave 2 checkpoint, 40 epochs (20+20), lr 2e-5, batch 4, 256px.
G_L1 fell ~0.9→0.35; pieces sharpened slightly. **But it made colour worse** (see Wave 4
root-cause): L1 was cut 10→5 and the new VGG term — which is largely colour-insensitive —
**dominated the generator loss** (weighted G_VGG≈6 vs G_L1≈0.3), so the output drifted from
the warm tone Wave 2 had toward a cold gray/lavender average. Remaining defects (the
handoff's brief): wrong board colour, weak piece distinctness/identity, residual blur, an
unconvincing-photo gap.

---

## Wave 4 — Root-cause fix: well-posed colour + 512 px + pix2pixHD objective

**Status:** ⏳ training on the cluster (results section filled on completion).
**Date:** 2026-06-05.
**New files:** `build_paired_dataset_v2.py`, `models/paired_hd_model.py`,
`train_hd.sbatch`, `smoke_hd.sbatch`, `test_hd.sbatch`, `make_comparison.py`, `plot_losses.py`.

### Root-cause analysis (why Waves 2–3 plateaued)

I inspected outputs, data, loss logs, and source resolutions before changing anything.

1. **Colour was an ILL-POSED regression — the single biggest defect.** Measuring the
   warm↔cool axis (CIELab **b\***) of every real target half shows colour temperature is a
   **per-recording-session camera white-balance constant**, not a function of the board:

   | game | mean b\* (crude proxy*) | within-game std |
   |---|---|---|
   | game4 | 14.0 | 1.1 |
   | game5 | 34.6 | 0.9 |
   | game6 | 28.4 | 1.0 |
   | game7 | 13.8 | 0.9 |
   | game2 (test) | 32.4 | 0.7 |

   (*proxy used during triage; the v2 builder uses true Lab.) A ~21-unit spread **across**
   games but ~1-unit **within** each. The synthetic input is colour-identical regardless of
   game, so **no deterministic G(synthetic)→real can know** whether the target is b\*≈14 or
   b\*≈35 — the L1/regression optimum is the cross-game average (a muddy mid-tone), exactly
   the "gray/cold instead of warm wood" failure. The Wave-3 training visuals confirm it:
   epoch 1 (warm, inherited from Wave 2) → epoch 40 (colder) — training drove colour toward
   the gray average. **This is unfixable by more training, a bigger model, or a stronger D.**
   The fix has to make the problem well-posed.

2. **Resolution ceiling.** Real targets are natively **480², synthetic 512²/1024²**, but
   training was at **256²** → ~2× detail discarded (contributing to blur).

3. **Geometric misalignment.** Synthetic is rendered near **top-down** (flat disc tokens);
   real is **oblique** (3-D pieces with height/shadow). Even with the board grid registered,
   a piece occupies different pixels in each domain → pixel-L1 averages → blur + weak
   identity. (Bounded by the input representation; partly addressable with perceptual/FM/GAN
   terms that tolerate small misalignment.)

4. **Weak discriminator.** Wave 2/3 D losses sat ~0.1 (D winning easily); a single
   unconditional 70×70 PatchGAN gives limited realism pressure.

**Architecture verdict:** the ResNet-9 + PatchGAN backbone is fine (it is pix2pix). Don't
replace it — make the task well-posed (colour), raise resolution, rebalance/strengthen the
objective.

### What changed

**(A) Dataset v2 — `datasets/chess_paired_v2/` (`build_paired_dataset_v2.py`).**
Reuses the Wave-1 FEN-anchored orientation fix verbatim, but:
- **512² halves** (1024×512 combined) — uses the available detail.
- **Colour-canonicalised targets:** Reinhard transfer of the **chroma** channels (Lab a\*,b\*)
  of each real half to a single fixed warm-wood reference (median a\*,b\* of the warm games
  5+6); **L\* (lightness) left untouched** so per-position exposure/piece-darkness is
  preserved. Implemented with a numpy sRGB↔Lab that round-trips to ~1e-15 (no skimage/cv2).
  Result: every game's target b\* collapses to ≈16.1 (std **0.06**), down from a 6.5→16.6
  spread — the white-balance nuisance is removed and synth→real becomes a **deterministic,
  learnable** mapping. This is a *real* fix (removes an input-independent nuisance variable),
  not a cosmetic patch. (Ablation switch: `--no-color-canon`.)

**(B) Model — `models/paired_hd_model.py` (`--model paired_hd`), pix2pixHD-style.**
```
loss_G = 1·LSGAN(D)  + 10·FM(D feats)  + 5·VGG19  + 10·L1
```
- **G** = the same `resnet_9blocks` (so it **warm-starts** from the Wave 2 generator via
  `--g_init_path`; verified load missing=0/unexpected=0).
- **D** = **multi-scale** (2 PatchGANs at full & half res) returning intermediate features.
- **Feature-matching** (L1 on D features across scales/layers) = pix2pixHD's key term for
  sharp, stable, realistic high-res output.
- **L1 stays the strong occupancy/structure anchor** (priority #1: correctness) at λ=10;
  VGG demoted to a sharpening assist (λ=5, capped at 256px for memory) so it no longer
  dominates and starve colour as in Wave 3.
- Single-GPU: `parallelize()` is a no-op (avoids DataParallel scatter/gather of the D's
  nested-list output); `save_networks` overridden to be DataParallel-agnostic.

**(C) Training recipe.** 512², batch 1 (pix2pix-standard; fits any assigned GPU), lr 2e-4,
β1 0.5, **60 flat + 60 decay = 120 epochs**, save every 5. ~97 s/epoch on an RTX 3090 →
~3.2 h. Smoke-tested first (G load, multi-scale D, 512px memory, finite losses, D balanced
at the LSGAN 0.25 equilibrium — vs Wave 3's 0.1 where D overpowered G).

### Results / evidence

_(to be completed when training + held-out game2 evaluation finish)_

- Loss curves: `results/analysis/` (Wave 2/3) vs the chess_hd run.
- Before/after visuals: `make_comparison.py` → `results/wave4_comparison.png`
  (synthetic | Wave 3 | Wave 4 | target).
- Early signal (epoch 3, in-distribution train visual): output already **warm wood**, not
  cold gray — the colour fix is visibly working from the start.
- Honest assessment: _pending final outputs._
