# Synthetic→Real Chessboard Translation: Diagnosis & Improvement Plan

> Deep-dive analysis of the CUT model outputs, root-cause diagnosis of all failure
> modes, architecture verdict, and a concrete improvement plan within an 8–16 h
> fine-tuning budget on a GTX 1080 Ti. Inference deliverable
> (`generate_chessboard_image`) is unchanged by the recommended plan.

## TL;DR verdict

The phantom-piece problem is **not a CUT tuning bug — it is the expected behavior of any *unpaired* translator on a task that requires exact structural fidelity.** CUT's only content-preservation mechanism (PatchNCE) is a soft, statistical, patch-level constraint with no concept of "this specific square must stay empty," so it *cannot* guarantee correctness. **However, the trained ResNet-9 generator and its learned style are good and worth keeping.** The decisive fact is that **the data is fully paired and the pairing was thrown away.** Every real frame has a known FEN + view, and both domains are perspective-normalized to a board-filling frame, so square *e4* sits at the same place in both images. The fix is to **convert to *paired* (pix2pix-style) supervision, fine-tuned from the existing CUT generator** — this directly supervises occupancy and kills phantom pieces while preserving the style already learned. This fits the 8–16 h budget easily.

---

## 1. Findings (what was observed)

### Code & data pipeline
- **Training was unpaired** (`dataset_mode: unaligned`, `model: cut`) — confirmed in [trained_model/train_opt.txt](trained_model/train_opt.txt): LSGAN + PatchNCE (layers 0,4,8,12,16), λ_GAN=λ_NCE=1.0, `nce_idt=True`, ResNet-9, instance norm, 256², 400 epochs. Losses converged cleanly (NCE 4.1→1.37, G_GAN≈0.45) — **training is healthy; the problem is the objective, not optimization.**
- **The data is intrinsically paired.** Real frames live at `raw_data/games_with_csv/gameX/tagged_images/frame_NNNNNN.jpg`; synthetic renders at `synthetic_dataset/gameX/images/frame_NNNNNN.png`; `gt.csv` ties each to `(fen, view)`. The assignment PDF states it explicitly: *"for every distinct board state… there is a specific frame that corresponds to that state."* Same frame number ⇒ same position/viewpoint in both domains.
- **Both domains are board-filling.** Synthetic is perspective-warped so the playing surface fills the frame ([postprocess_crop.py](postprocess_crop.py)); the real tagged images are likewise cropped to the board (confirmed visually). ⇒ the 8×8 grid is approximately pixel-registered across domains — the precondition for paired training.
- **Dataset imbalance & framing mismatch:** trainA=2,664 (444 frames × *left/middle/right* × 2 views) vs trainB=736 (368 frames × 2 views). Synthetic has 3 horizontal crops; real has 1. The 3:1 count gap is from the extra synthetic crops, not a position-coverage difference.
- **Density is NOT strongly biased** (quantified from the FENs): train games 4–7 are 36% dense / 37% mid / **28% sparse (<12 pieces)**. So "the model never saw empty boards" is **false** — sparse positions are well represented. This rules out the simplest density-bias explanation and points at the method.
- **A real data-prep bug: synthetic and real are 180°-rotated relative to each other for the same label.** game4 frame 28 is the start position (`view=white`): the *real* image correctly shows black at top / white at bottom; the *synthetic* shows the inverse. Invisible to unpaired CUT (both domains get a 180° aug, so each distribution is rotation-symmetric) but **fatal to naive pairing** if not corrected. Fully under our control (we own the renderer).

### Output failure modes (from the result images)
Dense, mid, and sparse positions were read in both viewpoints ([test_output/comparisons/](test_output/comparisons)). Catalogue:

1. **Phantom pieces (primary).** Sparse late-game (`frame_031800`): synthetic has ~6 pieces, output is littered with extra pieces on empty squares. Present in *every* regime; **visibility scales with the number of empty squares**, which is why late-game looks worst and dense looks "fine."
2. **Piece-identity collapse.** Mid/late pieces become generic blobs — knight/bishop/queen are not distinguishable. Violates the assignment's "preserve piece identity."
3. **Color inconsistency.** White-view outputs take a purple/lavender cast; black-view and mid-game take yellow/green. Partly *faithful* (the real video genuinely has a lavender tint on dark squares + per-frame white-balance drift), partly the generator keying color to spurious viewpoint cues.
4. **Board-geometry artifacts.** Wavy/warped grid lines, non-square squares, mild perspective wobble.
5. **Edge/boundary artifacts.** Pieces smear or spill off the top edge (PatchGAN border effects + board-fill puts pieces at the frame edge).
6. **Texture over-application.** Real-domain clutter/blur/shadow is applied uniformly, depositing piece-like smudges onto clean empty squares — this is the *mechanism* feeding (1).
7. **Resolution ceiling.** 256² caps achievable detail and worsens (2).

### Baseline comparison (CycleGAN try-1)
[cyclegan_try1](cyclegan_try1) (unpaired, cycle-consistency λ=10, game 2 only, **un-cropped** inputs) shows the *opposite* failure: **content collapse** — pieces melt away into a near-empty wood board. So both unpaired methods fail at exact fidelity, in opposite directions (CUT *adds*, CycleGAN *deletes*). Strong evidence that the failure is intrinsic to unpaired translation, not to one hyperparameter set.

---

## 2. Root-cause analysis

| Failure | Root cause | Category |
|---|---|---|
| **Phantom pieces** | Unpaired objective: NCE only maximizes patch-level mutual information between input/output features. Nothing forbids mapping an empty-square patch → a piece patch if it lowers the adversarial loss. The real manifold is "cluttered everywhere" (shadows, piece bases, blur), so the discriminator *rewards* adding piece-like texture to empties. No per-square supervision exists. | **Method (unpaired) — fundamental** |
| **Identity collapse** | NCE preserves coarse structure, not fine shape; large Blender-vs-wood shape gap; 256² resolution; mild piece-height misalignment. | Method + data + resolution |
| **Color casts/inconsistency** | Real domain's own white-balance variance + generator latching color onto viewpoint. No paired target to anchor per-image color. | Data + method |
| **Geometry/edge artifacts** | Generator free-form hallucination; PatchGAN border effects; warp interpolation; low res. | Architecture detail + resolution |
| **(Latent) orientation/color inversion** | Renderer's `view` convention (`base_z_rotation=180° if view=='white'`) is flipped vs the real frames. | **Data-prep bug** |

The unifying root cause: **the task demands exact geometric fidelity, but the training objective provides only weak, statistical, alignment-free content preservation.** Density imbalance is *not* the driver (28% of training positions are sparse).

---

## 3. Architecture verdict

**CUT-as-configured cannot reach "perfect," but we should not start over.** Decompose it:

- **Unsuitable part — the *unpaired objective*.** PatchNCE is designed for tasks (horse→zebra) where exact content is *not* required. On a fidelity-critical task it has no mechanism to guarantee occupancy, so phantom/missing pieces are an *expected* failure, not a bug that can be tuned away. Pushing λ_NCE, identity loss, occlusion-free renders, etc. would *reduce* but never *eliminate* it.
- **Reusable part — the *generator and its style*.** The ResNet-9 generator (and discriminator) are exactly the pix2pix architecture, the learned wood/lighting style is convincing on dense boards, and the checkpoint is a strong initialization.

**Verdict: keep the network, change the objective.** Convert the problem from unpaired to **paired/supervised** by exploiting the pairing already available. This is the single highest-leverage change and it preserves what works. CUT was the wrong *training regime* for this task, not the wrong *backbone*. (Note: paired supervision is used only at *training* time; at inference the model still takes a single static synthetic image, so the assignment's "no temporal/extra input" rule is respected, and `generate_chessboard_image` is unchanged.)

---

## 4. Recommended improvement plan (primary)

**Paired fine-tuning (pix2pix-style) initialized from the existing CUT generator**, implemented *in-place in the CUT codebase* so the checkpoint loads exactly (identical G/D config — avoids any architecture-mismatch when reusing weights).

**A. Build the paired set (the enabling step).**
- For each real tagged frame in games 4–7 (~368 frames × 2 views ≈ **736 pairs**), render the **board-filling "middle" synthetic** for its `(fen, view)`, **fixing orientation** so synthetic matches real (correct the 180° `view` convention, verify against the start position). Existing `trainA` middle crops + an orientation fix can be reused rather than re-rendering, for speed.
- Output an *aligned* dataset (`A_i`=synthetic, `B_i`=real). Augment with H-flip + small color jitter; optionally add 90°/180° rotations (apply to both). Optionally render extra **PGN-derived** FENs as *unpaired* realism data for a small NCE side-loss (see ablation).

**B. Model & init.**
- Generator `G` = ResNet-9 (instance norm), **init from `trained_model/latest_net_G.pth`**. Discriminator = basic PatchGAN, **init from `latest_net_D.pth`**. Keep `netF` available only if an NCE term is retained.

**C. Loss = realism + structure (this is what kills phantom pieces).**
- `L = λ_GAN·L_LSGAN(G(A),real) + λ_L1·‖G(A)−B‖₁ + λ_perc·LPIPS/VGG(G(A),B)`
- Suggested: **λ_GAN=1, λ_L1=10, λ_perc=1** (pix2pix defaults; drop λ_L1 toward 5 if pieces blur from height-misalignment).
- *Why each term*: **L1 against the real target** supervises occupancy per pixel → empty target squares penalize any hallucinated piece (**fixes phantom pieces**) and occupied squares penalize drops (**fixes missing pieces**). **Perceptual/LPIPS** is tolerant to the small piece-height misalignment and restores **piece identity/sharpness** (failure 2). **GAN** keeps photographic **realism** (failure 6, priority 2) so the output isn't a blurry regression. **Init-from-CUT** preserves the working **style** (priority 3) and makes it converge in a fraction of the epochs.

**D. Training budget.**
- Start at **256²**, batch 4 — ~184 iters/epoch at ~0.1 s/iter ⇒ an epoch in ~20 s; **80–120 epochs in ≈1–2 h.** Then a **512² polish pass** (batch 1–2 on the 1080 Ti) for a few-dozen epochs to lift identity/detail — still well inside 8–16 h. Use `lr=2e-4`, β₁=0.5, linear decay.

**E. Deliverable & ablation (required by the rubric).**
- `generate_chessboard_image(fen, viewpoint)` is unchanged: Blender render → `G` → save `synthetic.png` / `realistic.png` / `side_by_side.png`.
- Natural ablations (each maps to a component, satisfying §6 of the submission guide): remove L1; remove perceptual; remove GAN (pure regression → blurry, shows GAN's role); CUT-init vs scratch; with/without orientation fix; +NCE side-loss on PGN data.
- Quantitative metrics: **FID** (realism) + a **piece-occupancy error** — run a simple square-occupancy check (the board is grid-aligned, so threshold/learned per-cell occupancy) on outputs vs FEN. The phantom-piece rate is the headline number and should collapse toward ~0.

---

## 5. Fallback options

- **Fallback 1 — Misalignment-robust hybrid (if L1 blurs pieces).** Keep CUT's **PatchNCE** (alignment-invariant content) and *add* only a **paired LPIPS/perceptual + feature-matching** term (no pixel-L1). Fine-tune from the CUT checkpoint with `nce_idt=True`, higher λ_NCE. Lower risk to the existing style, still injects occupancy supervision; slightly weaker hard-occupancy enforcement than full pix2pix.

- **Fallback 2 — Stay unpaired, attack the root statistically (if pairs prove too noisy/misaligned).** (a) Shrink the domain gap: render **occlusion-free, higher-realism** synthetic (real textures, soft shadows, varied piece sets) so the discriminator has less reason to "add clutter"; (b) raise **λ_NCE** and add an explicit identity loss; (c) add an **FEN-occupancy auxiliary loss** — since empty squares are known and grid-aligned, penalize piece-like content on known-empty cells. Reduces but won't fully eliminate phantom pieces.

- **Fallback 3 — Higher ceiling, out of budget (mention only).** Conditional diffusion / ControlNet conditioned on the synthetic image gives top realism + control, but is not trainable to convergence from scratch on a single 1080 Ti within this deadline. Not recommended now; good "future work."

**Priority alignment:** the primary plan attacks correctness first (paired L1/perceptual ⇒ no phantom/missing pieces, positions match the input), then realism (GAN), then explicitly preserves the working dense-position style (CUT-init) — exactly the requested order, inside the time budget, with no change to the inference deliverable.
