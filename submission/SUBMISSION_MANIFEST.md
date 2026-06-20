# Project 3 — Submission Manifest

Final model: **`chess_v5_bright_silABC`** (`paired_geom_hd`, geometry-conditioned
paired synthetic→real translation). Prepared for the BGU *Intro to Deep Learning*
Project 3 submission.

## Deliverable paths

| Item | Path |
|---|---|
| Required API | [`generate_chessboard_image.py`](../generate_chessboard_image.py) → `generate_chessboard_image(fen, viewpoint)` |
| Inference helpers | [`chess_v5_infer/`](../chess_v5_infer/) (`preprocess.py`, `networks.py`) |
| Demo CLI | [`run_project3_demo.py`](../run_project3_demo.py) |
| README (reproduction) | [`README.md`](../README.md) |
| `requirements.txt` | [`requirements.txt`](../requirements.txt) |
| Final report (PDF, 6 pp) | [`output/pdf/project3_final_report.pdf`](../output/pdf/project3_final_report.pdf) |
| Presentation deck (9 slides) | [`submission/project3_presentation.pptx`](project3_presentation.pptx) |
| Final model source | [`v5_work/v5_cluster_src/`](../v5_work/v5_cluster_src/) (`paired_geom_hd_model.py`, `v5_oblique_dataset.py`) |
| Final config | [`v5_work/final_config/bright_silABC_train_opt.txt`](../v5_work/final_config/bright_silABC_train_opt.txt) |
| Synthetic render pipeline | [`v5_pipeline/`](../v5_pipeline/) + [`chess-set.blend`](../chess-set.blend) |
| Report generator | [`v5_work/make_final_report_pdf.py`](../v5_work/make_final_report_pdf.py) |
| Guidelines summary | [`submission/project3_guidelines_summary.md`](project3_guidelines_summary.md) |
| Checkpoint manifest | [`submission/CHECKPOINT_MANIFEST.md`](CHECKPOINT_MANIFEST.md) |
| Sample outputs (real) | [`submission/sample_outputs/`](sample_outputs/) |

## Checkpoint & data (large; not in Git)

- **Final generator:** `checkpoints/chess_v5_bright_silABC/latest_net_G.pth`
  (~43 MB) — download from the shared drive and place at that path. See
  [`CHECKPOINT_MANIFEST.md`](CHECKPOINT_MANIFEST.md).
- **Dataset:** `chess_v5_oblique_aligned_bright` on the shared drive
  (https://drive.google.com/drive/folders/1oJpGFstyY0AmyeIcL4Cjk1B03UtL3_cN).

## How to run

```bash
pip install -r requirements.txt        # + install Blender, see README
# place checkpoints/chess_v5_bright_silABC/latest_net_G.pth (from Drive)
python run_project3_demo.py --fen "<FEN>" --viewpoint white
# -> ./results/synthetic.png, ./results/realistic.png, ./results/side_by_side.png
```

## Verification performed (this machine)

```bash
python3 -c "from generate_chessboard_image import generate_chessboard_image"   # import OK
python3 run_project3_demo.py --help                                            # CLI OK
python3 run_project3_demo.py --viewpoint white                                 # synthetic OK; clean fail on realistic
python3 -c "from PIL import Image; print(Image.open('results/synthetic.png').size)"  # (512, 512)
mdls -name kMDItemNumberOfPages output/pdf/project3_final_report.pdf           # 6
```

Environment note: this machine has **Blender + Pillow + numpy** but **not
PyTorch** and **not** the final checkpoint, so the realistic stage was exercised
only up to its (clean) failure path. The synthetic stage runs fully and the two
sample renders in `sample_outputs/` are genuine.

## PASS / FAIL — Project 3 requirements

| Requirement | Status | Note |
|---|---|---|
| `generate_chessboard_image(fen, viewpoint) -> None` exists | ✅ PASS | `generate_chessboard_image.py` |
| Saves `./results/synthetic.png` | ✅ PASS | verified (512×512 RGB) |
| Saves `./results/realistic.png` | ⏳ NEEDS CKPT | code path complete; needs torch + checkpoint |
| Saves `./results/side_by_side.png` (synth\|real) | ⏳ NEEDS CKPT | produced once realistic runs |
| Creates `./results/` if missing | ✅ PASS | `_results_dir()` |
| `viewpoint` ∈ {white, black}, both work | ✅ PASS | both rendered (see sample_outputs) |
| Deterministic for same (fen, viewpoint) | ✅ PASS | fixed render params; `eval()`, no dropout/randomness |
| No interactive input / no return value / writes only `./results/` | ✅ PASS | by construction |
| PNG, RGB, compatible dims | ✅ PASS | 512×512 / 1024×512 RGB |
| No fabricated realistic output when checkpoint missing | ✅ PASS | raises precise error instead |
| GitHub repo: all source + `requirements.txt` + `README.md` | ✅ PASS | committed |
| README: setup / training / inference instructions | ✅ PASS | `README.md` |
| Report PDF, ≤ 20 pages, scientific structure | ✅ PASS | 6 pp |
| Ablation study in report | ✅ PASS | report §6 (Ablation Study) |
| Dataset on shared drive | ✅ PASS | Drive link in README |
| Author names in report/README | ✅ PASS | Tal Podgor, Ilay Vanunu, Ran Packler |

## Remaining manual steps

1. ~~Student names~~ — **DONE** (Tal Podgor, Ilay Vanunu, Ran Packler) in the
   report PDF and README.
2. **Checkpoint link** — add the real download URL in
   `CHECKPOINT_MANIFEST.md` and `generate_chessboard_image.py::CHECKPOINT_URL`,
   and confirm `latest_net_G.pth` is on the shared drive.
3. **End-to-end realistic test** — on a torch GPU env with the checkpoint placed,
   run `python run_project3_demo.py` and confirm all three `./results/*.png`.
4. **Push** — `git push origin main` (see below).
5. **Drive dataset format** — ensure the uploaded chess dataset includes a
   `gt.csv` (`image_name, FEN, view`) per the guidelines' Chess dataset format.
6. **Deadline** — the PDF lists Jan 24 2026 (already past); confirm resubmission/extension.
