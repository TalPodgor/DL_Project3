# Honest Failure Diagnosis — Wave 4 Results
# For the next session / research agent

## Project in one sentence
Translate Blender-rendered synthetic chessboard images into photorealistic ones while
preserving exact piece positions. Deliverable: generate_chessboard_image(fen, viewpoint)
→ saves synthetic.png / realistic.png / side_by_side.png.

---

## What the current best model actually produces

Look at the images in results/for_next_session/:
- wave4_output_*.png  = our model's output (Wave 4, 120 epochs, 512px)
- real_target_*.png   = what the output should look like
- synthetic_input_example.png = what goes IN to the model
- wave4_train_epoch55_*.png = in-distribution training sample (the model's "best case")

### Honest verdict: the output is not good enough

The pieces are:
1. **Smeared / painted-looking** — they look like watercolor blobs, not 3D chess pieces
2. **Indistinguishable by type** — you cannot tell a rook from a bishop from a queen
3. **Indistinguishable by color in dense positions** — white/black separation is weak
4. **Lack 3D form** — no height, no shadows, no volume

The board:
5. **Dark squares are grayish-purple**, not warm wood brown (though much better than
   before; the light squares are roughly correct)
6. **Lacks wood grain texture** — looks flat/painted

The overall image:
7. **Does not look like a photograph** — it looks like a stylized illustration
8. **Loses sharpness on pieces** even at 512px

---

## Root cause analysis — what is ACTUALLY wrong

### Problem 1: Fundamental geometric domain gap (primary cause of blur/smear)

The synthetic input is rendered near **top-down** (camera ~directly above, ~15-20°
from vertical). The real photos are taken from an **oblique angle** (~45° from
horizontal), showing the full 3D height of pieces.

This means:
- A pawn in the synthetic image is a ~20px circle (top of piece)
- A pawn in the real image is a ~50px tall 3D silhouette with shadow

**The generator must simultaneously:**
- Invent piece height from a flat disc token
- Add correct 3D shadows and lighting
- Match the oblique perspective
- Do all of this per-pixel with L1 supervision against misaligned targets

**L1 loss averages over the height offset** → the "average" pawn is a blurry smear.
No amount of training or loss tuning fixes this if the supervision signal is
geometrically misaligned.

This is NOT just a style transfer problem. It is a 2D→3D reconstruction + style
transfer problem. The current ResNet-9 pix2pix-style approach treats it as pure
style transfer.

### Problem 2: Dataset size is tiny (736 training pairs)

736 pairs for a task requiring:
- 32 piece types (6 types × 2 colors + empty)
- Variable density (1-32 pieces)
- 2 viewpoints (white/black perspective)
- Board positions that never repeat

For comparison: pix2pix used Cityscapes (3000 images of similar scenes).
pix2pixHD used 2975 images. Our task is arguably harder with 22% of that data.

The generator cannot reliably learn piece identity from 736 examples where each
piece type appears in only ~100-200 positions with the piece at varying sizes/lighting.

### Problem 3: The synthetic renders are too simple

The Blender renders appear to use:
- Near-top-down camera (unusual viewpoint, very different from real photos)
- Simple flat-disc piece markers OR low-detail 3D models
- Uniform gray board (no texture variation)
- Single uniform lighting

The real photos have:
- Oblique camera (~45°)
- Detailed carved wooden pieces with grain and polish
- Warm-toned walnut/maple board with grain
- Mixed natural + ambient lighting, soft shadows

The domain gap is so large that even a perfect image-to-image translator cannot
bridge it from 736 samples.

### Problem 4: ResNet-9 generator capacity

ResNet-9 (11M params) was designed for simple style transfer (horse→zebra).
For a task requiring:
- Significant geometric transformation
- Fine detail synthesis (piece carving texture)
- Multi-scale structure (board grid + piece shapes)

...it may simply lack the capacity/architecture to produce sharp, detailed outputs.
pix2pixHD uses a coarse-to-fine generator with local enhancer networks specifically
for this reason.

### Problem 5: The test set (game2) has different recording conditions

Game2 (test) was recorded in different conditions than games 4-7 (train):
- Different camera position/angle
- Different lighting
- Different (or same?) physical board/pieces

Even after color canonicalization, there may be geometric differences (slight camera
angle difference) that hurt test performance specifically.

### What DID improve (to be fair to Wave 4)

- Color: b* error 7.78 → 0.07 (board color now warm and correct globally)
- No phantom pieces on any of 140 test frames
- Sharpness: no longer noisy/artifact-ridden like Wave 3
- Position accuracy: pieces on correct squares

### What is still bad

- Piece identity: indistinguishable types
- Piece rendering: smeared, watercolor, no 3D
- Overall realism: clearly not a photograph
- Dark square texture: missing wood grain

---

## Hard constraints for any solution

- Hardware: GTX 1080 Ti, 11 GB VRAM (or the BGU cluster which sometimes gives
  RTX 3090 24GB under the `course` QoS partition)
- Training budget: ~8-24 GPU hours
- Dataset: 876 paired images (736 train / 140 test), cannot easily get more
  (requires Blender re-rendering and matched real footage)
- Must run inference on a single synthetic image at test time
  (no temporal information, no extra real images at test time)
- Deliverable must remain: generate_chessboard_image(fen, viewpoint)
- Course constraint: must be a "deep learning" solution (not pure CV)

## Dataset structure

- datasets/chess_paired_v2/train/  — 736 [synthetic|real] side-by-side 1024x512 PNGs
- datasets/chess_paired_v2/test/   — 140 [synthetic|real] side-by-side 1024x512 PNGs
- Each PNG: left half = synthetic (512x512), right half = real target (512x512)
- Synthetic: flat gray board, top-down view, Blender rendered, black/white disc tokens
- Real: oblique ~45° view, warm wood board, 3D carved wooden pieces, natural lighting
- All synthetic images are geometrically oriented to match real (FEN-anchored fix applied)

## Cluster access

SSH: bgu (already configured, VPN needed)
Repo: ~/chess_cut_project/contrastive-unpaired-translation/
Conda env: pytorch (torch 1.10, CUDA 11.2)
SLURM: sbatch --partition=gtx1080 --qos=course --gres=gpu:1
Wall limit: 24h per job
Current best weights: checkpoints/chess_hd/latest_net_G.pth (ResNet-9, 11M params)
Dataset on cluster: datasets/chess_paired_v2/{train,test}/

---

## RESEARCH AGENT PROMPT (English, for the next session)
## See: RESEARCH_AGENT_PROMPT.md
