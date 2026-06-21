# HANDOFF — Synthetic→Real Chessboard Translation (state & problem only)

This document describes the **current state** of the project and the **current
problem**. It deliberately does **not** propose a solution.

---

## 1. The task / deliverable (unchanged)

Build `generate_chessboard_image(fen, viewpoint)` which saves `synthetic.png`,
`realistic.png`, `side_by_side.png`. The realistic image must look like a photo of a
real wooden chess set in the given position/viewpoint. Priority order is fixed:
**correctness (right pieces on right squares, no phantom/missing pieces) > realism >
style**. University DL course project (BGU, Fall 2025); a written report + ablation is
also expected.

---

## 2. How to connect to the BGU cluster and run

```bash
ssh bgu            # SSH config + VPN already set up on this machine; user = packler
```

- Repo on cluster: `~/chess_cut_project/contrastive-unpaired-translation/`
- Conda env: `pytorch` (**torch 1.10, CUDA 11.2**). Run python as:
  ```bash
  conda run -n pytorch python <script>.py ...
  ```
- **No `cv2`** and **no Blender** are installed on the cluster.
- GPU jobs use SLURM:
  ```bash
  sbatch <script>.sbatch
  squeue -u packler
  ```
  Working SBATCH header:
  ```bash
  #SBATCH --partition=gtx1080
  #SBATCH --qos=course
  #SBATCH --gres=gpu:1
  #SBATCH --time=0-23:30:00
  #SBATCH --output=/home/packler/chess_cut_project/log_%j.txt
  ```
- **QOS limit: 1 GPU job per user at a time** (`QOSMaxGRESPerUser`). A second submitted
  job stays `PD` (pending) until the first finishes. Plan jobs serially.
- Nodes assigned vary (GTX 1080 Ti 11 GB, or RTX 3090 24 GB).

### Local machine (this Mac)
- Project root: `/Users/rnpqlr/Desktop/empty/dl project/`
- Has **Blender 4.5.5** and Python with **PIL/numpy**, but **no torch** and **no cv2**.
- Workflow used so far: render the dataset locally with Blender, `rsync` it to the
  cluster, train on the cluster. `rsync`/`scp` over `ssh bgu` work.

---

## 3. What was built this session (geometry-first "v3")

Rationale on entry (from the prior diagnosis): the older models produced "blob" pieces
because the synthetic input was rendered ~top-down (~12° off vertical) while the real
photos are oblique; board was registered but pieces were not, so pixel losses averaged
piece bodies into blobs. This session attempted a geometry-first redo. Key facts:

- The **original Blender `.blend` of the real set is lost**, and the cluster has no
  Blender. A substitute **MIT-licensed low-poly chess set** was downloaded and is used:
  `v3_pipeline/assets/ChessScene.blend` (separate meshes per piece type).
- The **real photos are already board-rectified** by the original data pipeline: the
  board is a canonical frontal square (verified — an exact 8×8 / 64px grid lands on the
  squares). The 3D pieces in the real photos are **sheared radially outward** from board
  center (their tall bodies lean toward the image edges).
- **FEN ground truth is known for every frame** (`game{2,4,5,6,7}.csv`) and is available
  at both training and inference time.

### Local scripts (`v3_pipeline/`)
- `render_aligned.py` — Blender headless. Places pieces per FEN on the board, renders an
  **oblique** synthetic RGB + a flat **semantic mask** (per-class colors), and writes the
  board's 4 projected corners. Batch mode via a jobs JSON.
- `rectify_and_pack.py` — numpy homography (no cv2) that warps the oblique render so the
  board maps to the canonical 512² square (matching the real images). RGB bilinear, mask
  nearest.
- `build_paired_dataset_v3.py` — builds the paired dataset (below). Renders are
  **deduplicated by (FEN, viewpoint)** and use a **single global camera** (`elev=35°,
  dist=9, lens=35`) for **all** sessions.
- `calib_sessions.py` — small helper that overlays renders at several elevations on the
  real photos (used to eyeball the global 35°).
- Camera/asset facts: board surface 4.8 units (0.6/square); piece heights (board units):
  pawn 0.46, rook 0.46, knight 0.50, bishop 0.78, queen 0.77, king 0.89.

### Dataset (built locally, uploaded to cluster)
`datasets/chess_paired_v3/` — **736 train (games 4,5,6,7) + 140 test (game2)**. Per frame:
- `{name}.png` = side-by-side `[A synthetic | B real]`, 1024×512.
  - A = view-aligned synthetic RGB (rectified). B = real photo (raw color, resized 512,
    **not** chroma-canonicalized).
- `{name}_seg.png` = class-id mask, single channel, ids 0–14
  (1=empty-light, 2=empty-dark, 3–8 white P/N/B/R/Q/K, 9–14 black p/n/b/r/q/k).
- `labels.json` = `{name: {fen, viewpoint, game, split}}`.

### Model + training (cluster)
- `models/paired_seg_hd_model.py` (`--model paired_seg_hd`) — pix2pixHD-style. Generator
  input = `concat(synthetic RGB 3ch, one-hot mask 15ch) = 18ch` → `resnet_9blocks` G.
  Discriminator = multi-scale PatchGAN with feature matching; plus VGG19 perceptual; plus
  **mask-weighted L1** (`--l1_piece_w 0.3`, i.e. L1 down-weighted inside piece pixels). G
  warm-started from the older `checkpoints/chess_paired/latest_net_G.pth` (first conv
  re-init for the +15 channels).
- `data/seg_aligned_dataset.py` (`--dataset_mode seg_aligned`) — loads A, B, and the seg
  mask with the same crop/flip.
- `train_seg_hd.sbatch` — launches training. A full run **completed 120 epochs**
  (60 flat + 60 decay, lr 2e-4, batch 2, 512px, ~4.5 min/epoch on a 1080 Ti).
  Checkpoints: `checkpoints/chess_segv3/{5,10,...,120,latest}_net_{G,D}.pth`.
- `test_seg_hd.sbatch` — runs `test.py` on the test set → `results/chess_segv3/
  test_latest/images/{real_A,fake_B,real_B}/` (140 each).

### Independent correctness evaluator (cluster)
- `square_eval.py` (+ `square_eval.sbatch`) — a ResNet-18 trained on **real** 64-px square
  crops (labels from FEN) to classify each square into 13 classes (empty + 12 piece
  types). `checkpoints/square_eval.pth`.
  - Ceiling on **real** game2 crops: **occupancy acc ≈ 0.985, piece-type acc ≈ 0.725**
    (type is hard at this crop size even on real images).
  - `--mode score` scores a folder of generated images vs the FEN.
- `score_all.sbatch` produced `report_{new,old,real}.json` on the cluster.

### Where to look at outputs
- Cluster: `results/chess_segv3/test_latest/images/`.
- Local (pulled): `results/segv3_test/{real_A,fake_B,real_B}/`.
- Local review folder (all 140, not cherry-picked): `results/REVIEW_segv3/`
  - `CONTACT_SHEET_generated.png` (all generated, sparse→dense)
  - `side_by_side/<frame>.png` (INPUT | GENERATED | REAL)
  - `generated_only/<frame>.png`

---

## 4. The current problem (honest, no spin)

- **Board: acceptable.** No phantom pieces; occupancy is correct; colors are mostly the
  right warm wood (dark squares sometimes drift cool/gray in some positions).
- **Pieces: not acceptable.** The non-pawn ("special") pieces consistently render as
  **smeared / melted blobs with multiple/duplicated "heads"** — they are **not clearly
  identifiable by type** by eye. This artifact appears across essentially **all** outputs,
  sparse and dense. Pawns (short pieces) look comparatively better; the taller the piece,
  the worse the artifact.
- Quantitative caveat: the per-square evaluator reports *higher* accuracy on the
  **generated** images than on the **real** images, because the generated images are
  cleaner/more canonical and thus easier to classify. Its absolute numbers therefore
  **overstate** the visual quality. **Visual inspection of `results/REVIEW_segv3/` is the
  honest judge, and visually the pieces are bad.**

### Relevant state-facts about why the pieces may be ill-determined (observations, not fixes)
- The synthetic uses a **substitute low-poly set** whose piece **shapes and sizes differ
  from the real physical set** (e.g., real pieces are taller than the rendered ones).
- The dataset was rendered with a **single global camera elevation (35°) for all four
  training sessions**; the real per-session camera angles were **not** individually
  calibrated, so the synthetic/mask piece **shear** does not necessarily match each real
  session's shear.
- Consequence at the data level: in the aligned pairs, piece **bases** line up well
  (board is rectified), but piece **tops/bodies** (synthetic vs real) are only
  approximately aligned. The conditioning therefore does not precisely specify where a
  tall piece's head should be.

### History
- This is the **5th attempt** (4 earlier "waves" + this geometry-first v3). **All five
  produced smeared / indistinct pieces.** The board/occupancy problems were already solved
  in earlier waves; the unsolved part throughout is **realistic, type-distinguishable
  rendering of the (especially tall) pieces.**

---

## 5. Exact commands to reproduce the current outputs

```bash
ssh bgu
cd ~/chess_cut_project/contrastive-unpaired-translation

# regenerate test outputs from the trained generator (epoch 120 / latest):
sbatch test_seg_hd.sbatch latest chess_segv3
# -> results/chess_segv3/test_latest/images/{real_A,fake_B,real_B}/

# score generated vs real with the evaluator:
sbatch score_all.sbatch         # -> report_new.json / report_old.json / report_real.json
```

Pull outputs to the Mac for visual review:
```bash
rsync -a bgu:chess_cut_project/contrastive-unpaired-translation/results/chess_segv3/test_latest/images/ \
  "/Users/rnpqlr/Desktop/empty/dl project/results/segv3_test/"
```

The local builder (Blender) that produced the dataset:
```bash
cd "/Users/rnpqlr/Desktop/empty/dl project/v3_pipeline"
python3 build_paired_dataset_v3.py            # full rebuild (renders via local Blender)
# single render for inspection:
blender -b assets/ChessScene.blend --python render_aligned.py -- \
  --fen "<FEN>" --viewpoint white --out renders/x --elev 35 --dist 9 --lens 35
python3 rectify_and_pack.py --prefix renders/x --viewpoint white
```
