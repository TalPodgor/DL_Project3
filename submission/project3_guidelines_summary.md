# Project 3 — Submission requirements (summary of the official guidelines)

Source: `Projects Final Submission Guidelines .pdf`. This file distills the parts
relevant to **Project 3 — Chessboard Image Generation from FEN** and the
all-projects rules, and is the checklist the submission is built against.

## Project 3 — required evaluation API

```python
def generate_chessboard_image(fen: str, viewpoint: str) -> None:
    """Generate synthetic and realistic chessboard images from a given FEN."""
```

- **`fen`** — `str`, same FEN format/conventions as the Projects 1 & 2 ground-truth CSVs.
- **`viewpoint`** — `str`, one of `"white"` / `"black"` = which color is **closest to the camera**.
- The function **saves images to disk** under `./results/` (it must **not** return them):
  - `./results/synthetic.png` — clean synthetic rendering of the board (no photorealism required); must reflect the FEN + viewpoint.
  - `./results/realistic.png` — realistic image derived from the synthetic one; must preserve the board state + viewpoint.
  - `./results/side_by_side.png` — single image, **left = synthetic, right = realistic**, aligned and clearly visible.
- **Image format:** PNG, RGB. Resolution not fixed, but the three must have compatible dimensions.
- **Constraints:** do not return images; no interactive input; do not overwrite files outside `./results/`; **deterministic** for the same `(fen, viewpoint)`; **must create `./results/` if it does not exist**.

## Final report (all projects)

- PDF, **≤ 20 pages**, 12 pt (or standard LaTeX defaults), English. Do not pad length.
- Scientific-paper structure: Abstract, Introduction, Related Work, Method,
  Experiments, **Ablation Study (required)**, What Did Not Work (optional),
  Discussion/Limitations (optional), References.
- Ablation must show each component is beneficial/necessary (remove-component tables + text).

## Code submission (required)

- A **GitHub repository** containing all source code, `requirements.txt`, and a
  `README.md` that explains how to reproduce results.
- Clear instructions for: environment setup (from `git clone` to installs),
  training, and running inference / evaluation.

## Datasets (all projects)

- Any dataset used or generated must be uploaded to a **shared drive** (university
  drive up to 2 TB, or Google Drive). This includes synthetic datasets and labeled PGNs.
- **Chess dataset format:** `dataset_root/{images/, gt.csv}` where `gt.csv` has 3
  columns: `image_name` (.png/.jpg), the FEN string, and the view specification.

## Optional / other

- **Webpage** (GitHub Pages) — optional but strongly recommended.
- **Presentation** — 7–10 min oral (separate from this repo packaging).
- Do **not** use AI tools to artificially inflate text; quality/clarity/honesty over length.

## Deadlines (from the PDF)

- Presentations: Jan 20–21, 2026. **Final submission: Jan 24, 2026.**
- ⚠️ This date is in the past relative to the working environment's current date
  (2026‑06‑20). Confirm with the course staff whether this is a resubmission /
  extension before submitting.

## Mapping to this repository

| Requirement | Where it is satisfied |
|---|---|
| `generate_chessboard_image(fen, viewpoint)` | `generate_chessboard_image.py` (+ `chess_v5_infer/`) |
| `./results/{synthetic,realistic,side_by_side}.png` | written by that function; demo via `run_project3_demo.py` |
| `requirements.txt` | repo root |
| `README.md` reproduction guide | repo root |
| Final report PDF (≤20 pp, ablation) | `output/pdf/project3_final_report.pdf` (6 pp) |
| Dataset on shared drive | Google Drive (link in `README.md`) |
| Checkpoint handling | `submission/CHECKPOINT_MANIFEST.md` |
