# HISTORY.md — project timeline & changelog

Narrative history so a new agent can pick up *as if it lived through the project*.
Pairs with [CLAUDE.md](CLAUDE.md) (orientation) — this file is the **why and when**.
Dates are from git + the dated research notes in the repo root; `~` = approximate.

## One-paragraph story

We started with **unpaired CUT** translation (synthetic chessboard → real photo).
It made wood look real but **moved pieces / invented texture on empty squares** —
fatal for a task where the board *state* must stay exact. So we pivoted to **paired,
geometry-conditioned pix2pixHD** (`paired_geom_hd`) through a series of pipelines
(v3 → v4 → v5), discovered that **matching the synthetic camera geometry to the real
photos mattered more than any loss**, and converged on the final model
**`chess_v5_bright_silABC`**. A late blind re-eval hinted a sibling (`silAB`) was
marginally better, but we shipped `silABC` for report/config consistency. The honest
conclusion: clean piece topology vs. photographic style is a **persistent trade-off**
that more loss-tuning won't fix — it needs better supervision/data. On 2026-06-20 the
repo was packaged for submission, then (per request) the work was moved onto a side
branch and `main` was reverted.

## Timeline

| When | Milestone | Evidence |
|---|---|---|
| **2026-05-29** | Initial repo: code + **trained CUT weights** + configs. Unpaired baseline. | git `6149852`, `6a8c01e`; `trained_model/` |
| **~2026-06-03** | CUT diagnosed: realistic texture but **state not preserved**. CycleGAN trial. | `DIAGNOSIS_AND_IMPROVEMENT_PLAN.md`; `cyclegan_try1/`, `trained_model/` |
| **2026-06-05** | **Pivot: unpaired → paired supervised translation.** Paired dataset builder + FEN-anchored orientation fix ("Wave 1"). Wave-4 failure analysis. | `REFACTOR_LOG.md`, `DIAGNOSIS_WAVE4_FAILURES.md`, `build_paired_dataset*.py` |
| **2026-06-06** | **Geometry-first "v3"** pipeline (segmentation-mask conditioning). | `v3_pipeline/`, `HANDOFF.md` |
| **2026-06-07** | **v4 + v5 probes.** Root-cause decision: silhouette-only semantics → high *phantom* rate (rejected); **full-cell FEN + v5 geometry** adopted. | `v4_pipeline/`, `V5_PROBE_RESULTS_AND_DECISION.md`, `V5_ROOT_CAUSE_AND_DECISION.md` |
| **~2026-06-08** | v6-direction research framing; deeper external-research briefs. | `DEEP_RESEARCH_PROMPT_V6.md`, `RESEARCH_AGENT_V5_BRIEF.md` |
| **~2026-06-10→13** | **v5 oblique "aligned bright" dataset** built; `paired_geom_hd` trained on the BGU cluster; ablations run (silAB, silABC, noPieceD, geometry-lock, parallax8). | `v5_pipeline/`, `v5_work/v5_cluster_src/`, `v5_work/final_config/bright_silABC_train_opt.txt` (06-13) |
| **2026-06-14** | **Blind re-eval (bright-no-C):** 3 judges + merge/halo metrics rank **`silAB` (drop loss C) above `silABC`** → silABC "dethroned". (We still ship silABC — see Decisions.) | internal eval; `v5_work/eval_*` |
| **2026-06-17** | **Final conclusion:** the core defect (clean topology vs. realistic style) is a *ceiling* — not removable by more loss-tuning; needs new supervision/data. `bright_silABC` named the deliverable. | `PROJECT_CONCLUSION_CEILING.md` |
| **2026-06-20** | **Submission packaging** (see changelog below). | git `dc1549f`…`197fee1` |

## Changelog — 2026-06-20 submission session

What this session actually changed (all on branch `project3-final-submission`):

1. **Implemented the required API** `generate_chessboard_image(fen, viewpoint)`
   (`generate_chessboard_image.py`) + framework-free helpers `chess_v5_infer/`
   (`preprocess.py` ports the v5 dataset conditioning; `networks.py` vendors the
   antialiased CUT/pix2pixHD generator so `latest_net_G.pth` loads standalone).
   Added CLI `run_project3_demo.py`.
2. **Verified locally:** synthetic render works (real 512×512 PNG, both viewpoints);
   realistic stage fails *cleanly* (no torch + no checkpoint here) — never fabricates.
3. **Rewrote `README.md`** to describe the real v5 system (the old README still said
   "CUT"). Updated `requirements.txt` (+torch/torchvision/Pillow/reportlab) and
   `.gitignore` (exclude GB-scale dumps, keep final source).
4. **Added `submission/`**: guidelines summary, checkpoint manifest, submission
   manifest + PASS/FAIL checklist, real sample synthetic renders.
5. **Author names** (Tal Podgor, Ilay Vanunu, Ran Packler) → README + report;
   **regenerated** `output/pdf/project3_final_report.pdf` (6 pp, all required sections).
6. **Built the 9-slide presentation** `submission/project3_presentation.pptx`.
7. **Added `CLAUDE.md`** agent-onboarding guide, and this `HISTORY.md`.

### Git operations (2026-06-20)

| Commit | What | Branch |
|---|---|---|
| `dc1549f` | Prepare Project 3 final submission (API, README, report, manifests) | originally pushed to `main` |
| `8a9460b` | Add Project 3 presentation deck (9 slides) | originally pushed to `main` |
| `1170412`, `707146e` | **Revert** of the two commits above | `main` |
| `8a9460b` | full submission preserved here | branch `project3-final-submission` |
| `197fee1` | Add CLAUDE.md (+ this HISTORY.md follows) | branch `project3-final-submission` |

**Branch outcome (per user request):** `main` was **reverted** to the pre-submission
state (`707146e`, content identical to `6a8c01e`) — it does **not** contain the
submission. All work lives on **`project3-final-submission`**. No force-push / no
history rewrite was used.

## Key decisions & rationale

- **Unpaired (CUT/CycleGAN) → paired (pix2pixHD).** Unpaired had no direct penalty
  for changing the position; chess needs exact state. *(2026-06-05)*
- **Condition on geometry, not just RGB.** 21-ch input (RGB + `fen_silhouette`
  semantics + depth/silhouette/edge) — derived from the FEN alone at inference. *(v5)*
- **Camera-geometry alignment was the biggest lever.** Per-game elevations ~40.7°
  (vs an earlier 54.9° global) shrank the synth↔real gap more than loss-tuning.
- **Ship `silABC`, not `silAB` — on purpose.** Despite the 2026-06-14 blind re-eval
  favoring `silAB`, the report + config are built around `silABC`; switching would mean
  redoing both. `silAB` is documented as future work. **Do not "fix" this.**
- **Self-contained inference loader.** Vendor the generator rather than depend on the
  cloned CUT framework; load `state_dict` strictly (error loudly, never mis-load).
- **Never fabricate the realistic image.** If torch/checkpoint are missing, raise —
  only the (genuinely rendered) synthetic image is written.

## Current state (2026-06-20) & what's open

- ✅ API + demo + helpers; README; requirements; report (6 pp, with authors);
  manifests; sample renders; 9-slide deck; CLAUDE.md; HISTORY.md — all on
  `project3-final-submission`.
- ✅ Pushed to GitHub. `main` reverted clean.
- ⏳ **Open:** add the checkpoint **download link**; **run the realistic stage
  end-to-end** on a torch GPU env with the checkpoint placed at
  `checkpoints/chess_v5_bright_silABC/latest_net_G.pth`; verify slide-1 degree program
  and visually proof the deck; decide whether to PR `project3-final-submission → main`.

For deeper detail see the dated research notes in the repo root (`REFACTOR_LOG.md`,
`V5_*` , `PROJECT_CONCLUSION_CEILING.md`, `HANDOFF.md`) and `submission/` docs.
