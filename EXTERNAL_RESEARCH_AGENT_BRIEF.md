# External Research Brief: Synthetic-to-Real Chessboard Image Translation

You do not have access to the codebase or data. Please advise based only on this
brief. The goal is not to debug implementation details, but to reason about the
best research direction and propose a strong next solution.

## 1. Assignment Goal and Constraints

We need to train an image-to-image generation model for synthetic-to-real
chessboard rendering.

At inference, the required function is conceptually:

```text
generate_chessboard_image(fen, viewpoint)
```

It should:

1. render a synthetic chessboard image from a FEN position and viewpoint,
2. feed that single static synthetic image into a trained image-translation model,
3. save:
   - `synthetic.png`
   - `realistic.png`
   - `side_by_side.png`

The output must look like a real photo while preserving:

- exact board layout,
- exact piece locations,
- piece color,
- piece identity,
- camera/view geometry.

Important constraints from the assignment:

- The final generation model must take a single static synthetic image as input.
- The model cannot use temporal information at inference.
- Temporal/video/PGN information may be used only to create labels or training data.
- Classical or learned labeling methods may be used for training-data construction.
- We are allowed to alter the Blender generator, camera, lighting, materials, and assets.
- We are allowed to download/use additional Blender chess assets.
- The final solution must be a deep-learning/image-translation solution, not a manual postprocessing or compositing hack.
- The project explicitly allows GANs, CycleGAN/image-to-image translation methods, diffusion-based methods, etc.

Priority order:

1. geometric correctness and preservation of piece identity,
2. realism,
3. style/visual polish.

## 2. Available Data

We have real video-derived chessboard frames with labels.

Each real labeled board state includes:

- game id,
- frame id,
- viewpoint: white or black,
- FEN string containing piece-square positions.

For every distinct board state in the labeled games, there is a corresponding real frame.

The real images are board-rectified: a perspective transformation is applied so that the chessboard fills a square image, roughly 512x512. However, after rectification, the physical pieces still show 3D height, oblique silhouettes, shadows, and bodies extending across square boundaries.

We also have synthetic chessboard renders:

- existing synthetic images with left/middle/right variants,
- Blender asset `chess-set.blend`,
- original Blender code that can generate board images from FEN.

The existing synthetic images are visually cleaner than our first attempted custom Blender renders, but they are still much more top-down/icon-like than the real images.

Approximate dataset sizes in the current paired setup:

- train: 736 real board states/views.
- v4 expanded train: 2208 paired samples because each real target was paired with 3 synthetic variants: left, middle, right.
- test: 140 real board states/views, using middle synthetic only.

This means the true amount of real paired supervision is small: hundreds, not tens of thousands.

## 3. Real vs Synthetic Domain Gap

This is the most important issue.

The real images:

- show wooden pieces with visible height and body shape,
- are oblique in the sense that piece bodies, bases, shadows, and vertical geometry are visible even after board rectification,
- have warm board colors, wood grain, shadows, and natural lighting,
- include piece silhouettes that can extend into neighboring squares after rectification.

The current synthetic inputs:

- are close to top-down,
- often represent pieces as icons/discs/symbols rather than full 3D pieces,
- have weak or absent shadows,
- have less texture detail,
- contain much less information about piece height and silhouette.

So the model is effectively asked to learn:

```text
top-down synthetic/icon-like board -> oblique realistic wooden chessboard photo
```

This may be underconstrained for a small paired GAN.

## 4. Previous Approaches and Outcomes

### 4.1 Unpaired Translation

An earlier unpaired CUT/CycleGAN-style direction failed.

Main failures:

- phantom pieces on empty squares,
- missing pieces,
- piece-like texture hallucinated on empty cells,
- poor exact board fidelity.

Diagnosis:

Unpaired image translation is not suitable by itself because the task requires exact structural preservation. Patch-level or cycle-consistency losses do not strongly enforce that a specific square must remain empty or contain a specific piece.

This shifted us toward paired supervision.

### 4.2 Paired / pix2pix-style Training

Paired supervision improved geometry dramatically.

We built paired samples of the form:

```text
[synthetic image | real target image]
```

The model was trained synthetic -> real.

This mostly solved occupancy and phantom-piece failures, but the generated pieces still look bad.

### 4.3 Current Best Direction: V4 Geometry-Conditioned Paired Model

The current best model uses:

- synthetic RGB image,
- one-hot semantic segmentation from FEN,
- geometry/control maps.

Input channels:

```text
RGB synthetic render
+ FEN semantic one-hot map
+ geometry maps
```

The geometry maps include:

1. estimated foreground/silhouette from the synthetic image,
2. hand-coded piece height prior from FEN piece type,
3. synthetic edge map inside occupied cells.

Architecture:

- pix2pixHD-like conditional GAN,
- ResNet-9 generator,
- multi-scale discriminator,
- image size 512x512.

Losses:

- GAN loss,
- discriminator feature-matching loss,
- VGG perceptual loss,
- masked L1 loss,
- optional frozen square-classifier loss for piece identity.

Important detail:

The current model deliberately downweights pixel L1 inside occupied piece cells:

```text
l1_piece_weight = 0.1
```

Reason:

The real piece body is not pixel-aligned to the synthetic icon/silhouette. Strong L1 on piece regions causes averaging/blur.

Downside:

Piece interiors then receive weak direct supervision, so they are left mostly to GAN/VGG/classifier losses.

## 5. Current Best Quantitative Results

Current model: 5-epoch smoke run, not full training.

Score on 140 test boards:

```text
n_boards: 140
square_acc: 97.32%
occupancy_acc: 99.93%
phantom_rate(empty -> piece): 0.07%
missing_rate(piece -> empty): 0.06%
type_acc(both occupied): 92.73%
color_acc(both occupied): 99.88%
whole_board_occupancy_exact: 95.71%
whole_board_full_exact: 24.29%
```

Interpretation:

- Occupancy is nearly solved.
- Piece color is nearly solved.
- Piece identity is still not robust enough.
- Full-board exact correctness is poor: most generated boards have at least one piece-type error.

Per-class recall reveals the hard classes:

```text
white bishop recall: 81.0%
white king recall: 84.3%
black rook recall: 66.1%
white queen recall: 86.8%
```

These are shape-sensitive classes, suggesting that the model lacks sharp, distinctive 3D piece form.

## 6. Current Visual Failures

A harsh visual QA pass found:

1. Pieces are often melted, smeared, or blurry.
2. Many pieces look like beige/brown blobs rather than wooden chess pieces.
3. Non-pawn piece identity is visually ambiguous.
4. Piece bases and tops lack crisp 3D form.
5. Shadows are weak, smeared, or physically inconsistent.
6. White pieces sometimes blend into light squares.
7. Black pieces sometimes merge into dark blobs in dense rows.
8. Board texture is better than before but still sometimes artificial or flat.
9. The classifier score is overly forgiving: it rewards “something classifiable in the right square” even if it looks fake to a human.

Sharpness analysis:

```text
fake occupied-cell sharpness / real occupied-cell sharpness ≈ 0.79
synthetic occupied-cell sharpness / real occupied-cell sharpness ≈ 0.51
```

This means the model sharpens the synthetic source, but generated pieces are still much less detailed than real ones.

## 7. Suspected Root Causes

### Root Cause A: Input representation lacks 3D information

The synthetic input does not contain the correct piece height, oblique silhouette, base shape, or shadow cues.

The model must hallucinate those from weak top-down/icon-like evidence.

This is probably the biggest issue.

### Root Cause B: Board alignment does not imply piece alignment

After perspective rectification, board squares align, but tall pieces do not behave like flat square labels.

Real piece bodies and shadows can extend across cell boundaries.

Therefore pixel-level losses can be contradictory:

- strong L1 on piece areas encourages blur,
- weak L1 allows vague blobs.

### Root Cause C: Current synthetic-view augmentation may be contradictory

The v4 train set maps:

```text
left synthetic -> same real target
middle synthetic -> same real target
right synthetic -> same real target
```

This increases sample count but may confuse the model because different synthetic silhouettes/edge maps are supervised to produce one identical target.

The test set uses only middle.

This may hurt geometry learning.

### Root Cause D: Renderer/camera mismatch

The original Blender asset actually contains 3D chess meshes with meaningful piece heights.

However, the current/old generator camera is effectively top-down. Its comments describe the cameras as looking “straight down.”

This likely prevents the synthetic image from carrying the same 3D cues as the real rectified target.

### Root Cause E: Model architecture may not be ideal

A ResNet-9 pix2pix-style generator may be too weak or not structured correctly for semantic/geometry-controlled object synthesis from small data.

The task may be better framed as semantic image synthesis or controlled generation, not simple RGB-to-RGB translation.

## 8. Current Working Hypothesis

The model has learned “where pieces should be” but not “how each piece should look.”

The most likely reason is not insufficient training time alone. It is that the source image and objective do not provide enough aligned, high-frequency, 3D piece supervision.

Therefore, a real solution probably requires changing the synthetic rendering and conditioning setup, not just adding a small regularizer.

## 9. Candidate Directions We Are Considering

### Direction 1: Camera-matched 3D Blender synthetic + paired conditional GAN

Use the original 3D Blender asset properly:

- oblique camera,
- visible piece height,
- realistic shadows,
- board-filling crop after perspective transform,
- only middle view initially,
- improved materials/lighting.

Export:

- RGB synthetic render,
- semantic piece map,
- depth map,
- silhouette/object mask,
- edge map,
- maybe normals.

Train:

- pix2pixHD-style model first,
- maybe SPADE-style semantic synthesis if semantics are being washed out,
- maybe local piece discriminator after source geometry improves.

### Direction 2: Semantic image synthesis instead of RGB translation

Treat the task as:

```text
semantic layout + depth/silhouette/control maps -> realistic board image
```

Rather than:

```text
synthetic RGB -> real RGB
```

This may better match methods like SPADE, pix2pixHD semantic label-map synthesis, or related conditional synthesis architectures.

Potential issue:

The assignment says the model input is a single static synthetic image. If we derive maps from that synthetic render at inference, or render them as additional channels from the same FEN/synthetic generation process, this may still be acceptable, but this constraint should be considered carefully.

### Direction 3: Conditional diffusion / ControlNet-like approach

Use a pretrained image prior and condition on synthetic/control maps.

Potential benefits:

- better realism from pretrained image priors,
- potentially better texture and piece appearance.

Risks:

- small domain-specific dataset,
- training complexity,
- inference cost,
- exact piece identity may be hard unless conditioning is very strong,
- course environment/hardware may make this impractical.

### Direction 4: Hybrid 3D-aware rendering + learned style transfer

Make Blender renders geometrically close enough to real photos, then the model only needs to learn material/texture/color/lighting transfer.

This seems pragmatically attractive: reduce the problem from “invent 3D piece shape” to “make a plausible 3D render look like the real camera domain.”

## 10. What We Need You To Advise

Please act as an external research advisor. You have full academic freedom under the assignment constraints.

Please do not limit yourself to small tweaks. We want a real plan that can materially improve the result.

Please answer:

1. What is the most likely root cause of the bad piece appearance?
2. Is the current V5 hypothesis correct, or is there a better framing?
3. Should this be treated as paired image translation, semantic image synthesis, conditional diffusion, 3D-aware rendering plus style transfer, or something else?
4. What architecture would you recommend for a small dataset of hundreds of paired real frames?
5. What should the synthetic renderer generate next?
6. Should we use 3D Blender RGB as input, semantic/depth/control maps, or both?
7. How should we avoid blur caused by imperfect piece-level alignment?
8. How should exact piece identity be enforced differentiably?
9. Would a local/crop discriminator solve the right problem, or would it be a band-aid?
10. Would SPADE-style semantic normalization help more than concatenating semantic channels?
11. Is ControlNet/diffusion realistic under these constraints, or too risky?
12. What ablations would make the final report scientifically convincing?
13. What is the shortest experiment schedule that can falsify weak ideas quickly?

Please produce:

- root-cause diagnosis,
- ranked solution plan,
- concrete architecture recommendation,
- concrete data-generation recommendation,
- loss design,
- training-cost estimate,
- risk assessment,
- minimal experiment schedule,
- go/no-go metrics.

Reminder:

The final model must take one static synthetic board image, or information deterministically rendered from the same synthetic board state, and output one realistic image. It must not rely on real target images, video context, manual patching, or temporal information at inference.
