# CLAUDE.md — agent onboarding guide

Read this first. It orients a new agent (or human) to this repository quickly and
flags the traps that will otherwise waste your time. For the *timeline / why we made
each change* (the project's story), see [HISTORY.md](HISTORY.md).

## TL;DR

- **Project:** BGU "Intro to Deep Learning" **Project 3 — Synthetic-to-Real
  Chessboard Image Generation from FEN**.
- **The one required deliverable:** a function
  `generate_chessboard_image(fen: str, viewpoint: str) -> None` that saves
  `./results/synthetic.png`, `./results/realistic.png`, `./results/side_by_side.png`.
- **Final model:** `chess_v5_bright_silABC` — a **paired, geometry-conditioned
  pix2pixHD-style** translator (codename `paired_geom_hd`). **NOT CycleGAN, NOT CUT.**
- **Authors:** Tal Podgor, Ilay Vanunu, Ran Packler.

## ⚠️ Branch layout — READ BEFORE TRUSTING ANYTHING

This repo has a deliberate split:

| Branch | State |
|---|---|
| **`project3-final-submission`** | The real, finished submission. **You want this branch** (you are probably on it — this file only exists here). |
| **`main`** | **Reverted to the pre-submission state on purpose.** It still shows the *old, abandoned CUT write-up* and does NOT contain the API, the new README, the report, or the deck. Do not treat `main` as current. |

If you cloned and landed on `main` and it looks like a half-finished CUT project —
that's expected. Switch: `git checkout project3-final-submission`.

## ⚠️ The biggest trap: CUT is legacy, not the final model

The project's first iteration used **CUT** (unpaired translation). It was abandoned.
Two leftovers will try to mislead you:

- `trained_model/` — held the **legacy CUT weights** (a different, abandoned model).
  Those `.pth` files were **removed**; only the CUT `train_opt.txt` / `loss_log.txt`
  remain as history. `generate_chessboard_image` never used them.
- `main`'s README describes the project as "CUT". Ignore it; use this branch's README.

The **final** model is `chess_v5_bright_silABC` = `paired_geom_hd`:
- antialiased ResNet-9-block generator, instance norm;
- **21-channel input** = 3 RGB (synthetic render) + 15 one-hot semantic
  (`fen_silhouette`) + 3 geometry (depth, silhouette mask, silhouette edge) → 3 RGB out;
- exact config: `v5_work/final_config/bright_silABC_train_opt.txt`;
- model/dataset source: `v5_work/v5_cluster_src/paired_geom_hd_model.py`,
  `v5_work/v5_cluster_src/v5_oblique_dataset.py`.

## Environment realities (why things "don't run" locally)

The machine that prepared this submission has **Blender + Pillow + numpy**, but:

- ❌ **No PyTorch** on the machine that prepared this → the realistic (GAN) stage
  couldn't be executed here (the code path is complete; run it on any torch env).
- ✅ **The final checkpoint IS in the repo** (force-added, ~44 MB) at
  `checkpoints/chess_v5_bright_silABC/latest_net_G.pth` — no download needed.
  (Verified: it has a 21-ch input conv → the real geometry-conditioned model, not CUT.)
- ✅ **Blender is installed** → the synthetic stage runs and is verified.

So locally: `synthetic.png` renders for real; the realistic stage raises a clean,
explicit error (it never fabricates output). To run end-to-end you need a torch
environment + the checkpoint placed at the path above.

## How to run

```bash
pip install -r requirements.txt          # torch, torchvision, Pillow, numpy, ...
# install Blender, put `blender` on PATH or set $BLENDER
# place checkpoints/chess_v5_bright_silABC/latest_net_G.pth (from Drive)

python run_project3_demo.py --fen "<FEN>" --viewpoint white
# -> ./results/synthetic.png, realistic.png, side_by_side.png
```

Default FEN = starting position; `viewpoint ∈ {white, black}`. Deterministic per
`(fen, viewpoint)`; creates `./results/`; writes only there.

## Pipeline (how the function works)

1. **Render** synthetic + seg + depth with Blender:
   `v5_pipeline/render_oblique_blender.py` (oblique 44°, "bright"; reuses
   `synthetic_chess_generator.py`), then `v5_pipeline/pil_perspective_crop.py`
   rectifies to 512×512. → `./results/synthetic.png`.
2. **Build conditioning** (`chess_v5_infer/preprocess.py`): a framework-free port
   of `v5_oblique_dataset.py` → RGB + one-hot `fen_silhouette` seg + geometry.
3. **Translate** (`chess_v5_infer/networks.py`): a self-contained copy of the CUT/
   pix2pixHD antialiased generator so `latest_net_G.pth` loads without the cloned
   framework. `netG(concat(rgb, onehot, geom))` → realistic RGB.
4. **Save** realistic + side-by-side.

## Where things live (map)

```
generate_chessboard_image.py     ← REQUIRED API (entry point)
run_project3_demo.py             ← CLI wrapper (--fen / --viewpoint)
chess_v5_infer/                  ← framework-free inference helpers (preprocess, networks)
chess-set.blend                  ← Blender asset (needed for the synthetic render)
synthetic_chess_generator.py     ← FEN→Blender primitives (used by the v5 renderer)
v5_pipeline/                     ← FINAL synthetic-render pipeline (render + crop + dataset build)
v5_work/v5_cluster_src/          ← FINAL model + dataset training code (paired_geom_hd, v5_oblique)
v5_work/final_config/            ← bright_silABC_train_opt.txt (exact final config)
v5_work/make_final_report_pdf.py ← regenerates the report PDF
output/pdf/project3_final_report.pdf  ← final report (6 pp)
submission/                      ← manifests + guidelines summary + samples + the .pptx deck
checkpoints/chess_v5_bright_silABC/   ← place latest_net_G.pth here (from Drive; git-ignored)
trained_model/                   ← CUT train config + loss log (dead weights removed)
```

**Noise to ignore** (old experiments / large dumps, mostly git-ignored): `v3_pipeline/`,
`v4_pipeline/`, `cyclegan_try1/`, `datasets/`, `cluster_results/`, `test_output/`,
`results/`, `synthetic_game*/`, `v5_work/eval_*`, `v5_work/*audit*`, and the many
top-level `*.md` research notes. They are research history, not the submission.

## Decisions you should NOT "fix"

- **Final model = `silABC`, on purpose.** A 2026-06-14 re-eval suggested a sibling
  `chess_v5_bright_silAB` (dropping the global Contextual Loss "C") may score
  slightly better on blind realism. The submission deliberately keeps **`silABC`**
  for consistency with the written report + config; `silAB` is noted as future work.
  Do not rename to `silAB` without redoing the report + config.
- **`generate_chessboard_image` vendors a generator** (`chess_v5_infer/networks.py`)
  instead of importing the cloned CUT framework — this is intentional (self-contained
  loader). It loads `state_dict` strictly; if it mismatches, it errors loudly rather
  than silently mis-loading. The exact load was **not verifiable on this machine**
  (no torch/checkpoint) — verify on a torch GPU env.
- **The synthetic image is real; the realistic image is never faked.** If torch/the
  checkpoint are missing, the function raises — keep that behavior.

## Training code & research notes (for the full picture)

- **Custom training modules** live in `v5_work/v5_cluster_src/` (and a v4 copy in
  `v4_pipeline/cluster_src/`): `paired_geom_hd_model.py` (the final model + the two
  discriminators wiring), `paired_hd_model.py` (defines `MultiscaleDiscriminator`,
  the local `NLayerDiscriminatorFeat` piece-crop discriminator, and `VGGPerceptualLoss`),
  `v5_oblique_dataset.py`, `square_eval.py`, plus the `train_/test_/score_*.sbatch`.
  To actually train: `git clone` the CUT framework (taesungp/contrastive-unpaired-translation)
  and drop these into its `models/` and `data/` dirs — `networks.py` and `base_model.py`
  come from that framework (not vendored here). Inference does NOT need any of this.
- **Discriminators:** (1) a global multi-scale PatchGAN (`MultiscaleDiscriminator`,
  num_D=2 scales), (2) a local class-conditional piece-crop discriminator
  (`NLayerDiscriminatorFeat`, 96px crops + one-hot piece label). A frozen ResNet-18
  square classifier (`square_eval.pth`) adds a loss but is NOT a discriminator.
- **The journey / why:** the dated notes in the repo root tell the full story —
  `REFACTOR_LOG.md` (CUT→paired pivot), `HANDOFF.md`, `V5_ROOT_CAUSE_AND_DECISION.md`,
  `V5_PROBE_RESULTS_AND_DECISION.md`, `PROJECT_CONCLUSION_CEILING.md`, plus the
  per-experiment `v5_work/audit_*.json` (the raw ablation numbers) and
  `v5_work/*.py` (the eval/ablation scripts). See also [HISTORY.md](HISTORY.md).

## Submission docs (authoritative detail)

- `submission/SUBMISSION_MANIFEST.md` — deliverable paths + PASS/FAIL checklist + remaining steps.
- `submission/project3_guidelines_summary.md` — the official requirements distilled.
- `submission/CHECKPOINT_MANIFEST.md` — checkpoint name/size/path/link + load contract.
- `submission/project3_presentation.pptx` — the 9-slide oral-presentation deck.

## Open TODOs (as of 2026-06-20)

1. ~~Checkpoint download link~~ — **DONE**: `latest_net_G.pth` is bundled in the
   repo (force-added at `checkpoints/chess_v5_bright_silABC/`).
2. **Run the realistic stage end-to-end** on a torch GPU env with the checkpoint, and
   confirm all three `./results/*.png` (and that `load_state_dict` succeeds).
3. Verify the slide-1 degree program ("B.Sc. Computer Science") and visually proof the deck.
4. Decide whether to open a PR `project3-final-submission` → `main`.
