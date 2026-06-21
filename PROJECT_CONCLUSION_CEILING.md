# Synthetic→Real Chessboard Translation — Final Conclusion

**Date:** 2026-06-17
**Verdict:** The project has reached its **practical quality ceiling under the current data and supervision.** Clean piece *topology* and realistic *style* cannot be achieved simultaneously with the available 480px data and the automatic supervision we can generate. This is supported by convergent evidence from every approach tried, not by any single metric.

---

## 1. Objective
Translate a synthetic Blender chessboard render → a realistic wooden-board photo.
Priority: **correctness (right piece on right square) > realism > style.**
Hard data ceiling: real target photos are 480px; pieces are ~40–60px, soft, low-contrast, leaning/overlapping.

## 2. The core defect we could not remove
**Structural piece artifacts** — double heads, side lobes, blob-like pawns, vanishing pieces. Occupancy (is a square occupied) is solved (~99%); piece *legibility and shape* is the wall.

## 3. What was tried, and how each failed (convergent evidence)
| approach | structure | realism | outcome |
|---|---|---|---|
| unpaired GAN baseline `bright_silAB` | double-heads ~27% | realistic | most real-looking, but artifacts remain |
| GAN loss tweaks (`pcomp_*`, `pcx_*`) | ~27–31% (unchanged) | realistic | **no measurable effect on double-heads** |
| geometry-lock `glock_t1p5_train12` | double-heads **11%**, vanish 12.7% | **washed / synthetic / milky** | best structure, but style rejected |
| piece-inserter `piece_inserter_v1` | double-heads 10.4% | washed, detail collapse | removes artifacts by **vanishing** pieces (35.6%) |
| temporal real-sprite compositing (GrabCut) | — | hard patch / cut artifacts | unusable without clean masks |
| **SAM masks** (ViT-H, FEN-prompted; this session) | — | — | masks **loose**: grab the square, not the piece |
| **SAM + color-trim** (this session) | — | — | square removed but pieces **fragmented/holey** |

**Pattern:** every method that achieves clean topology destroys realism; every method that preserves realism reintroduces the artifacts. That trade-off is the ceiling.

## 4. The mask-supervision attempt (this session, in detail)
Hypothesis: the missing ingredient is **real per-piece masks** → enabling a mask-conditioned refiner or a clean real-sprite bank.

- **Feasibility:** cluster login node has internet; installed SAM (`segment-anything`, user-local); downloaded ViT-H (2.5 GB). The FEN labels give exact occupied squares, so SAM can be prompted with **no detector**.
- **Mask bank (ViT-H, 736 boards, GPU):** 14,384 pieces → 8,998 masks accepted = **62.6% coverage** (white 61% / black 64%). Box-prompted from the synthetic silhouette + negative neighbor points + a contrast gate.
- **Quality problem:** accepted masks are **loose** — they include the square tile under the piece (square-contamination 31.4%). Sprites carry background → would reproduce pasted/cut artifacts.
- **Decisive test — color-trim:** keep only pixels differing from the local empty-square color. Contamination 31.4% → 0% and 98.7% of masks "survived" numerically, **but visually the trim carved holes into the pieces** → fragmented/partial cutouts, not solid sprites.
- **Root cause:** at 40–60px, soft and low-contrast, a **lit white-piece pixel is the same color as the light square**, so no color/contrast rule (SAM, GrabCut, or thresholding) can separate them. Only **dark pieces on light squares** segment cleanly.

Artifacts: `v5_work/sprite_sheet.png`, `v5_work/mbv1_viz.png` (loose), `v5_work/trim_compare.png`, `v5_work/trim_viz.png` (trimmed). Bank: cluster `~/mask_bank_v1/`, `~/mask_trim_v1/`.

## 5. Final deliverable model
Neither model fully meets a strict "looks real, no defects" bar — that is the ceiling finding. Choose by which failure mode is acceptable (figure: `v5_work/FINAL_model_comparison.png`):

- **`chess_v5_bright_silAB`** — **best realism** (warm wood, real-ish pieces); residual double-heads. Recommended if *style* matters most. Checkpoint: cluster `checkpoints/chess_v5_bright_silAB/latest_net_G.pth`.
- **`glock_t1p5_train12`** — **best structure** (double-heads 11% vs 27%); looks washed/synthetic. Recommended if the stated *correctness > realism > style* priority governs.

## 6. Honest paths forward (all require NEW supervision/data — none is more loss-tuning)
1. **Manual mask annotation** of a few-hundred-piece subset → train a dedicated piece-matting network → apply to all. The *only* lever not yet exhausted; a human can separate piece-from-square where color cannot. Real labeling effort.
2. **Higher-resolution / higher-contrast capture or re-render** — moves the physical data ceiling. A data-collection effort.
3. **Stop and ship** a least-bad model (§5) and report the ceiling as a result.

**Do NOT** invest further in GAN-loss variants or mask-threshold tweaks — diminishing returns are confirmed across ~10 variants.

## 7. Reproducibility / key assets
- Variant outputs (local, 140 each, shared filenames): `v5_work/eval_*/fake_B`; REAL targets `v5_work/audit_baseline_pieceD/real_B`.
- Metrics: `v5_work/detect_head_artifacts.py` (double-head/vanish vs real), `audit_defects.py` (transparency/halo/detail). **Do not use the `square_eval` classifier / `type_acc` as an arbiter — it rewards merged blobs.**
- Mask pipeline (this session): `v5_work/{sam_probe,mask_bank,mask_trim}.py`; SAM ckpts cluster `~/sam_ckpt/`.
- Datasets: `datasets/chess_v5_oblique_aligned_bright` (and `..._aligned`).
