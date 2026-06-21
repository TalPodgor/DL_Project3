# Deep Research Request: Legible Chess-Piece Rendering in Synthetic→Real Chessboard Image Translation

You are a senior computer-vision / generative-modeling research advisor. You do **not**
have access to our code or data — reason from this brief and from the published
literature. Your job is to recommend, with concrete justification and references, the
**method most likely to produce crisp, type-distinguishable 3D chess pieces** under the
hard constraints below. We have already spent five iterations getting "pieces in the
right square" — that part is solved. The *unsolved, non-negotiable* requirement is that a
human (and a strict evaluator) can tell a rook from a bishop from a knight from a queen
in the generated image. We will not accept smeared/blob pieces.

---

## 1. The deliverable and its hard constraints

We must implement `generate_chessboard_image(fen, viewpoint)` that outputs
`synthetic.png`, `realistic.png`, `side_by_side.png`. Priority order (fixed by the
assignment): **(1) geometric correctness + exact piece identity > (2) realism > (3)
style.**

Inference contract (must hold):
- The translation model's input is a **single static synthetic image** rendered from the
  FEN + viewpoint (plus control maps — seg/depth/edge — that are **deterministically
  rendered from the same FEN/synthetic scene**, which is allowed).
- **No real photo, no temporal/video, no PGN** at inference. (Temporal/PGN may be used
  only to build training labels/data.)
- Must be a learned image-translation / generative solution (GAN, diffusion, etc.), not
  manual compositing.
- We **may** modify the Blender renderer/camera/lighting/materials and download
  additional Blender chess assets.

Compute / environment constraints (these are real and have blocked options before):
- Cluster: SLURM, **1 GPU job per user at a time**, GTX 1080 Ti (11 GB) or RTX 3090
  (24 GB). Conda env is **PyTorch 1.10 / CUDA 11.2** (old — modern `diffusers`/
  ControlNet generally need torch ≥1.13/2.x; a new env may be possible but is unverified
  and the course cluster may restrict it). **No Blender and no OpenCV on the cluster.**
- Local machine: Blender 4.5, PIL/NumPy, **no GPU torch**. Workflow = render dataset
  locally, rsync to cluster, train on cluster.
- Each full training run ≈ **5 hours**. Iteration budget is therefore small (a handful of
  runs, not dozens).

## 2. The data (this is the crux)

- **Real targets are only 480×480**, already board-rectified (a homography warps the
  photographed board to a canonical fronto-parallel square). They are then upsampled to
  512 for training. **There is no higher-resolution real source.** Pieces occupy roughly
  one board cell, i.e. **~40–60 px tall**. Even an oracle classifier on *real* 64-px
  square crops tops out around **0.72 piece-type accuracy** — the real photos themselves
  are near the legibility limit at this resolution.
- After board rectification the board squares are axis-aligned (a perfect 8×8 / 64-px
  grid lands on them at 512²), but the **3D pieces are NOT flattened**: their tall bodies
  are sheared/leaning outward and their silhouettes spill into neighboring cells. Board
  alignment ≠ piece alignment.
- Paired dataset: **736 train + 140 test** board states (middle viewpoint only), built as
  `[synthetic | real]` side-by-side. We have FEN ground truth for every frame. Sessions =
  a handful of distinct games (different physical camera setups per game).
- We have the **original 3D chess meshes** (`chess-set.blend`) — so the *true* per-type 3D
  geometry is available to the renderer.

## 3. Domain gap

- **Real:** warm wooden set, visible piece height/body, oblique silhouettes, soft
  shadows, natural lighting, pieces spilling across cell borders — but low-res/soft.
- **Synthetic (current):** rendered from the real meshes but **near top-down, low sample
  count, generic blobby silhouettes**; weak shadows; less texture; carries little
  distinctive per-type 3D cue. Effectively the model is asked to map a near-top-down
  icon-ish board to an oblique soft wooden photo.

## 4. Full history of what we tried and exactly how it failed

1. **Unpaired CUT/CycleGAN.** Failed: phantom/missing pieces, hallucinated texture on
   empty squares, poor exact fidelity. Conclusion: unpaired loss can't enforce
   "this square must contain exactly this piece." → moved to paired.
2. **Paired pix2pix / pix2pixHD (RGB→RGB).** Fixed occupancy/phantoms; pieces still blobby.
3. **+ FEN full-cell semantic one-hot + geometry channels (depth + silhouette + edge),
   ResNet-9 generator, multiscale PatchGAN, losses = GAN + discriminator feature-matching
   + VGG perceptual + mask-weighted L1 (+ frozen square-classifier loss).** This is the
   current best ("V5"). **Occupancy ≈ 0.99, color ≈ 0.99, but piece-type accuracy ≈ 0.89
   and whole-board-all-pieces-exact ≈ 0.16.** L1 inside piece cells is deliberately
   down-weighted (≈0.35) because piece-level misalignment makes strong L1 blur the body.
4. **Latest experiment — a class-conditional LOCAL piece-crop discriminator** (PatchGAN on
   occupied-square crops, conditioned on piece class) **+ crop feature-matching + crop VGG
   + class-balanced/hard-class crop sampling + lowered global VGG.** Warm-started from #3,
   trained 60 epochs. **Result: NO improvement** — type accuracy 0.89→0.87,
   whole-board-exact 0.16→0.16; black-rook and white-bishop recall ticked up a hair, but
   **white queen 0.80→0.55 and white king 0.89→0.69 REGRESSED**, and visually the pieces
   were marginally sharper in texture but **still not type-distinct**. Two likely reasons:
   (a) the fixed 112-px crop window **clips the crowns off the tallest pieces** (queen/
   king), so the local objective never supervised the very feature that distinguishes
   them; (b) more fundamentally, **the objective cannot invent a rook-top / crown that the
   blobby, misaligned conditioning never describes.**

Per-class recall of the current best (lower = harder): black rook 0.64 (→ confused with
king), white bishop 0.76 (→ queen), white queen 0.80, white king 0.89, white rook 0.84.
Low *precision* on knight (~0.48) and queen (~0.49): many pieces collapse into
knight/queen-like blobs.

## 5. Our refined root-cause hypothesis (please critique / confirm / replace)

We now believe the dominant cause is **piece-level geometric misalignment between the
synthetic conditioning and the real target**, compounded by **blobby low-detail synthetic
geometry**:

- The synthetic camera is a **single global pose for all sessions** (it was never
  calibrated per game to the real camera). After both are rectified, the **piece shear/
  lean differs between synthetic and real**, worst for tall pieces — so reconstruction
  losses (L1/VGG/FM) see misaligned bodies and **average them into blobs**, while
  down-weighting L1 to avoid that blur removes the only direct shape supervision.
- The synthetic silhouette/depth themselves are generic "snowman" blobs (low samples,
  top-down), so even the conditioning carries little per-type shape.
- Net: the model has no aligned, high-frequency, type-distinct shape signal to imitate.

Implication we want vetted: **if we make the synthetic render crisp, per-type distinct,
AND geometrically aligned to each session's real camera, the task reduces from "invent 3D
piece shape" to "transfer wooden material/lighting onto an already-correct crisp shape"**
— at which point even modest adversarial sharpening should yield legible pieces, and we
could even render *sharper than* the soft 480-px target.

## 6. Specific questions we need answered

1. Is our root-cause diagnosis correct, or is the real bottleneck something else (e.g.
   capacity, the 480-px ceiling, the paired-L1 framing)?
2. **How do we recover a per-session synthetic camera that matches the real one** when we
   only have the *already-rectified* 480² real images (and FEN, and the board is a known
   planar grid)? Concretely: can we estimate the original oblique camera pose / the
   piece-shear from the rectified images (e.g. via the vanishing/lean of known pieces, a
   planar-homography decomposition, or fitting render elevation to observed shear), and
   then reproduce that exact shear in Blender + the same rectification? Cite practical
   techniques.
3. Given alignment is never perfect, **which losses give crisp, type-distinct pieces
   without blur** under residual misalignment? (e.g. perceptual/contextual losses robust
   to small shifts — Contextual Loss, LPIPS, patch-NCE; correlation/feature losses; aligned
   vs unaligned; whether to *increase* L1 once aligned; focal/hard-class weighting.)
4. Is a **local/crop discriminator the right tool**, and if so how should it be designed
   to actually help the *tallest* pieces (crop window, class conditioning via projection
   vs concat, multi-scale, anti-collapse for rare classes — we saw queen/king regress)?
   Or is a local discriminator a band-aid vs. fixing conditioning?
5. **SPADE / semantic-image-synthesis** vs. concatenating semantic channels: would
   spatially-adaptive normalization meaningfully help type identity here, on ~700 pairs?
6. **Diffusion / ControlNet** (depth/edge/segmentation-conditioned, on a pretrained image
   prior): is this realistic to *preserve exact per-square identity* on ~700 pairs while
   beating the 480-px softness — and is it feasible given **torch 1.10 / single 1080 Ti or
   3090 / 5-h runs**? If you recommend it, specify the smallest viable recipe (base model,
   which ControlNet conditions, LoRA vs full finetune, how identity is guaranteed, VRAM,
   and the env upgrade needed) and the failure modes (identity drift, hallucinated pieces).
7. Can we **beat the 480-px ceiling** legitimately — e.g. a piece-prior super-resolution /
   "render-and-refine" where a generative prior adds plausible crisp piece detail
   conditioned on type — without violating identity or the single-synthetic-input
   constraint? Is this worthwhile or a distraction?
8. **Better Blender data**: exact recommendations to maximize per-type discriminability of
   the synthetic + control maps (camera elevation/lens, samples, materials, shadows,
   resolution, additional channels like normals/curvature/part-segmentation, downloadable
   assets that match the real Staunton set), and how much this alone is expected to buy.
9. What is the **shortest experiment schedule** (given ~5 h/run, ≤ ~6 runs) that
   falsifies weak ideas fast and converges on legible pieces? Give a ranked plan A/B/C with
   go/no-go metrics that are **not** occupancy (we will track per-class recall, whole-board
   exact, and human crop montages by piece type).
10. What **ablations** would make this scientifically convincing in a course report?

## 7. Please produce

- A crisp root-cause verdict (confirm or correct §5).
- A **ranked, concrete solution plan** (method, architecture, conditioning, losses, data
  generation), each with expected payoff, risk, and compute fit to the constraints above.
- Specific, citable techniques for **per-session camera recovery from rectified images**
  and for **misalignment-robust sharp supervision**.
- A clear verdict on diffusion/ControlNet feasibility *under torch 1.10 + 1-GPU + exact-
  identity*, with the minimal recipe if recommended.
- A minimal experiment schedule with go/no-go metrics, and the ablations for the report.

Be concrete and opinionated. Prefer methods proven on **small paired datasets with strict
structural/identity preservation**. Assume we can change the renderer and the training
objective but not the 480-px real targets.
