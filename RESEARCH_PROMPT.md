# Research Prompt — Synthetic→Real Chessboard Image Translation

You are a research agent with full academic freedom. Your job is to find the **best
possible approach** to the problem below and to justify it. You are **not** constrained
to continue, extend, or reuse anything that has been tried before. Every part of the
pipeline — input representation, data generation, model family, conditioning signal,
training objective, inference procedure — is open to redesign. Treat the prior work and
the diagnosis below as *evidence about what fails and why*, not as a foundation you must
build on. Where the diagnosis reports measurements, those are facts you can rely on (and
re-verify); where it interprets them, you are free to disagree.

---

## 1. The deliverable (fixed — this is the product)

A function:

```
generate_chessboard_image(fen, viewpoint)
    -> saves synthetic.png, realistic.png, side_by_side.png
```

- `fen`: a standard FEN placement string (exact board state — which piece type, which
  colour, on which square).
- `viewpoint`: one of two camera sides, `white` or `black` (camera behind White or
  behind Black).
- `realistic.png` must look like a **photograph of a real wooden chess set** in the given
  position and viewpoint.

**Priorities, in strict order (set by the course, non-negotiable):**
1. **Correctness** — the photo must show the *right pieces on the right squares*: no
   phantom pieces on empty squares, no missing pieces, and ideally the correct piece
   *type* on each occupied square. This dominates everything else.
2. **Realism** — convincing as a real photo.
3. **Style fidelity** — matching the look of the reference photos.

This is a university deep-learning course project (final deliverable + written report
with an ablation study). An honestly-evaluated, well-justified method beats an
over-claimed one. (The previous iteration's report claimed piece identity was "solved";
direct image inspection showed it was not — so be rigorous and self-skeptical.)

---

## 2. The data you have

A **paired** dataset (every real photo has a synthetic render of the *identical* FEN and
viewpoint):

- **736 training pairs**, **140 held-out test pairs**.
- Each pair is one `1024×512` PNG: **left half = synthetic input (512×512)**, **right
  half = real target (512×512)**, same position and viewpoint.
- **Training games:** game4 (278), game5 (214), game6 (152), game7 (92) — i.e. **four
  physical recording sessions / board-and-lighting setups**. **Test game:** game2 (140),
  a session never seen in training.
- **Two viewpoints**, balanced (368 white / 368 black in train).
- **Ground-truth FEN is known for every frame** (the dataset was *built* from FEN +
  viewpoint labels). The FEN is therefore available at **both training and inference time**
  — at inference the function is literally *given* the FEN and viewpoint.

### What the two halves actually are (verified by direct inspection + measurement)

- **Synthetic (input):** a Blender 3D render with the camera at a **high elevation, near
  top-down** (the renderer used `≈12°` off vertical with a zoomed 45 mm lens). You look
  *down onto the tops* of the pieces.
- **Real (target):** a **photograph** with the camera at a **low, oblique elevation**:
  you see the pieces largely **from the side**, with visible height, 3D body,
  self-shadowing, and classic Staunton side-silhouettes. The board shows clear
  perspective keystoning.

So the input and target are **two different camera views (different elevation/pose) of the
same 3D scene**, not two "styles" of one pixel layout.

**Measured registration between the aligned halves (40 test pairs):**
- **The board grid is well registered** — vertical offset of the empty mid-board band
  between synthetic and real is **0 px (median)**.
- **The pieces are not registered** — the back-rank piece band is shifted **~+15 px
  (median) ≈ 0.24 of a square** vertically (real content sits lower / the 3D bodies lean
  toward the camera), **and** the real pieces occupy a much **taller vertical footprint**
  (oblique side-view body) than the **compact top-down disc footprint** in the synthetic.
  A 50/50 overlay of the two halves shows the pieces clearly *doubled*; the board does not.

**Where the piece-type information lives:** in the synthetic top-down view, piece type is
encoded in each piece's **top profile** (a rook's crenellated ring, a bishop's notched
dome, the king's cross-from-above, the knight's asymmetry) and **is human-distinguishable
at the working resolution**. In the real oblique view, the same type is encoded in the
**side silhouette**. So the type information is **present in the input — but in a different
visual representation/view than the target.** (It is not "missing"; it is in the wrong
view.)

### Assets / tooling that exist (stated as facts, not as a suggested solution)

- A **parameterised Blender renderer** that produces the synthetic image from a FEN +
  camera. Camera elevation/angle, lens, samples, resolution, and viewpoint are all
  controllable (the deployed dataset used ≈12° off vertical; another script variant uses
  25°; multiple azimuths — left/middle/right — were rendered historically). Blender is
  required wherever the synthetic image is produced.
- Reliable FEN→8×8-grid parsing and FEN-anchored board-orientation logic already exists.
- Existing trained ResNet-9 generator checkpoints from prior attempts, if useful as
  initialisation — or not.

You may use, ignore, re-render, re-pair, augment, or regenerate any of this.

---

## 3. What has been tried, and exactly how it failed

Four successive attempts, **all variants of pixel-aligned supervised image-to-image
translation** (a pix2pix / CUT-derived stack: a ResNet-9 generator, PatchGAN
discriminator(s), with combinations of L1, LSGAN, VGG19 perceptual, and feature-matching
losses). In every case the model was fed the synthetic image and asked to output the real
image, supervised against the paired real target.

- **A — L1 + PatchGAN, 256 px:** removed gross phantom pieces (the L1 anchor keeps empty
  squares empty), but output was blurry with poor colour.
- **B — + VGG19 perceptual, 256 px:** VGG is largely colour-insensitive and came to
  dominate; the board drifted **colder/greyer**. Pieces only marginally sharper.
- **C — current best: multi-scale PatchGAN + feature-matching + VGG + L1, 512 px, 120
  epochs, on a "colour-canonicalised" dataset.** They found the real targets' white
  balance is a *per-session camera constant* uncorrelated with the input (so predicting it
  deterministically is ill-posed) and "fixed" it by transferring every real target's
  chroma to a single fixed warm-wood reference. After this:
  - **Board colour:** globally roughly correct (warm cream squares), though **dark squares
    still drift cold/purple-grey in some positions**, and the colour now matches an
    *artificial fixed reference*, not any real camera's white balance.
  - **Occupancy / position:** largely correct — empty squares stay empty, no gross phantom
    pieces (confirmed across a 15-frame spread of the test set). **This is the one genuine
    success.**
  - **Pieces — the central failure:** the pieces are **smeared, type-less blobs.** You
    **cannot tell a pawn from a knight from a queen.** Confirmed to be **universal** across
    the test set, present in **sparse** positions (so not a crowding problem), and present
    on **in-distribution training frames** the model was trained on (so **not** an
    under-training or generalisation problem — it is a method ceiling).

### Why this family of methods fails here — interpretation of the measurements

The symptom pattern is **"board fine, pieces blobs,"** and the measurements explain it
exactly:

1. **Pixel-aligned losses (L1, VGG, feature-matching) implicitly assume the input and the
   target are spatially registered.** That assumption *holds for the board* (registered to
   0 px) → the board is reconstructed acceptably. It *fails for the pieces* (a ~0.24-square
   vertical shift plus a top-view→side-view shape/extent change) → the loss cannot be
   satisfied by any local appearance remap.
2. **Under misregistration, the regression optimum is the mean.** For a given local input
   the correct output silhouette lands on pixels that vary (per height, per neighbour
   occlusion, per exact pose) across the 736 examples; L1's optimum over that variability
   is the **average → a blob.** The adversarial and feature-matching terms enforce only
   *marginal* realism ("piece-shaped lump"), not the *correct per-instance silhouette*.
3. **The required transform is a 3D viewpoint change, not a 2D restyle.** Mapping a piece's
   *top profile* to its *side silhouette* is a global per-object re-projection. A 2D
   convolutional generator with local receptive fields and no explicit geometry cannot
   represent it; pixel-aligned supervision provides no gradient that would teach it.

In short: the problem is **(re)synthesising a specific photograph of a known 3D board
state from a viewpoint different from the input's**, but it has been treated as **2D
appearance translation of a registered image.** That mismatch — not tuning, scale, or
training budget — is the root cause of the blob pieces. (Colour, separately, was made
"well-posed" only by canonicalising the target, which also caps achievable realism.)

---

## 4. Hard constraints (must be respected)

- **Compute:** a single **NVIDIA GTX 1080 Ti (11 GB VRAM)** for training. SLURM jobs cap
  at **~23.5 h** wall-time; a realistic budget per experiment is **~8–16 GPU-hours** (prior
  full runs were ~7–8 h). Plan for *several* such runs total, not hundreds.
- **Software:** **PyTorch 1.10, CUDA 11.2**, a fixed conda env. Assume you **cannot**
  freely upgrade the framework. If a proposal needs a newer stack or heavy extra
  dependencies, that cost must be called out and justified, and ideally a torch-1.10-viable
  path offered.
- **Data:** **736 paired training images from 4 physical setups**; 140 held-out test from
  a 5th. This is *small*. Any approach must be viable at this scale (and may propose how to
  obtain more data — e.g. via the renderer — if it argues that is necessary).
- **Inference contract:** `generate_chessboard_image(fen, viewpoint)` must run on the same
  class of hardware and produce the three PNGs. It receives the **FEN and viewpoint**.
  Blender is available where synthetic rendering is needed. Inference need not be
  real-time.
- **Correctness is the top priority:** no hallucinated pieces on empty squares, no dropped
  pieces; correct piece *type* per square is the next priority. Realism is below both.

---

## 5. What I want from you

1. **Propose the approach you believe is genuinely best** for producing correct, realistic
   photos under the constraints above. Full freedom — design the whole pipeline. Do not
   anchor on anything in §3.
2. **Justify it against the alternatives you considered and rejected.** I want the real
   decision tree: which other approaches you weighed, and the *specific* reason each was
   set aside (feasibility on 11 GB / torch 1.10, the 736-image / 4-session scale, the
   viewpoint/registration structure, the correctness-first priority, evaluation
   difficulty, inference cost, …). A proposal without its rejected alternatives is
   incomplete.
3. **Make it concrete and buildable:** the input/conditioning representation, the model,
   the training objective(s), the data strategy, and the **exact inference procedure** for
   `generate_chessboard_image(fen, viewpoint)`. Map it to the **GPU-hour and VRAM budget**
   (roughly how many runs, how long each, does it fit in 11 GB).
4. **State how you would evaluate it** — especially how to **measure correctness
   objectively** (right piece, right square, no phantoms), not just eyeball realism — and
   design at least one ablation suitable for the report.
5. **Be honest about the risks and failure modes of your own proposal**, and what you would
   fall back to if the primary approach underperforms.

Ground every claim in the actual problem structure (the measured board-registered /
pieces-misregistered split, the near-overhead-vs-oblique viewpoint gap, the
type-info-present-but-in-the-wrong-view fact, the known FEN at train and inference, the
small multi-session dataset, the 11 GB / torch-1.10 budget, correctness-first). **Do not**
assume any particular method is the answer before you have argued for it against the
alternatives.
