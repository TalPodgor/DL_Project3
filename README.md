# Project 3 — Synthetic-to-Real Image Translation for Chessboard Rendering

BGU Intro to Deep Learning, Fall 2025 — Final Project 3.

The model translates Blender-rendered synthetic chessboards into photorealistic
images while preserving piece positions and camera viewpoint. The translation
model is **CUT** (Contrastive Unpaired Translation, Park et al. ECCV 2020).

- Repository: https://github.com/TalPodgor/DL_Project3
- Datasets and large artifacts: https://drive.google.com/drive/folders/1oJpGFstyY0AmyeIcL4Cjk1B03UtL3_cN

## Current status

- **Synthetic generation pipeline:** complete (Blender + PyBlender, see `synthetic_chess_generator.py`).
- **Real-image preprocessing:** complete (`preprocess_real_images.py`, `preprocess_synthetic_images.py`, `organize_cut_dataset.py`).
- **CUT training:** complete — all 400 epochs (Apr 5–10, 2026). Weights in `trained_model/`.
- **Inference function `generate_chessboard_image(fen, viewpoint)`:** not yet implemented (see *Next steps*).
- **Report and ablation:** not yet written.

Full training configuration is in `trained_model/train_opt.txt` and summarised in
`PROJECT_CONTEXT_FOR_CLAUDE.txt`.

## Repository layout

```
.
├── chess-set.blend                  Blender asset (3D chess pieces + board)
├── synthetic_chess_generator.py     Render synthetic boards from FEN
├── chess_position_api_v2.py         Alternative renderer (earlier version)
├── preprocess_real_images.py        Build dataset/real/ from gameX_per_frame/
├── preprocess_synthetic_images.py   Build dataset/synthetic/
├── organize_cut_dataset.py          Build dataset/{trainA,trainB,testA,testB}
├── postprocess_crop.py              Crop renders to board only
├── run_generation.py                Batch synthetic generation
├── create_comparisons.py            Build side-by-side comparison grids
├── cluster_setup.sh                 One-time BGU cluster setup
├── train_cut.sh                     Launch CUT training
├── resume_training.sh               Resume after disconnect
├── test_cut.sh                      Inference on test set
├── requirements.txt
├── game{2,4,5,6,7}.csv              Frame-to-FEN labels
└── trained_model/
    ├── latest_net_G.pth             Trained generator (43 MB)
    ├── latest_net_F.pth             NCE projection head
    ├── latest_net_D.pth             Discriminator (only for resuming)
    ├── train_opt.txt                Full hyperparameter dump
    ├── test_opt.txt
    └── loss_log.txt                 Per-iteration losses, all 400 epochs
```

Data folders (`dataset/`, `real/`, `synthetic/`, `game*_per_frame/`,
`synthetic_game*/`, `test_output/`, `training_progress/`) live on Google Drive
per the course requirement.

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/TalPodgor/DL_Project3.git
   cd DL_Project3
   ```
2. Download the dataset folder from the Drive link above. Place its contents
   directly in the project root so paths match the layout in the next section.
3. Install Blender (https://www.blender.org/download/).
4. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. To re-run training or inference with the CUT codebase, clone it into the
   project root (it is intentionally git-ignored):
   ```bash
   git clone https://github.com/taesungp/contrastive-unpaired-translation.git
   ```

After step 2, your local layout should be:

```
DL_Project3/
├── dataset/{trainA,trainB,testA,testB}/
├── real/
├── synthetic/
├── game{2,4,5,6,7}_per_frame/
├── synthetic_game{2,4,5,6,7}/
├── training_progress/
├── test_output/
├── trained_model/                   (already in repo)
├── chess-set.blend                  (already in repo)
└── ...all .py and .sh scripts       (already in repo)
```

## Training (already done — reproduction only)

Run on a CUDA GPU with at least 11 GB VRAM (GTX 1080 Ti or better).

```bash
bash cluster_setup.sh        # one-time, clones CUT and installs deps
bash train_cut.sh            # 400 epochs, batch_size=4
```

Key hyperparameters (see `trained_model/train_opt.txt`):

| Setting | Value |
|---|---|
| Framework | CUT (`CUT_mode CUT`) |
| Generator | ResNet 9-blocks, ngf=64, instance norm |
| Discriminator | PatchGAN, 3 layers, ndf=64 |
| NCE layers | 0, 4, 8, 12, 16 |
| GAN loss | LSGAN |
| `lambda_GAN`, `lambda_NCE` | 1.0, 1.0 |
| `nce_idt` | True |
| Optimizer | Adam, lr=2e-4, β₁=0.5 |
| Schedule | 200 epochs flat + 200 epochs linear decay |
| Total epochs | 400 |
| Batch size | 4 |
| Image pipeline | load 286 → random crop 256, random flip |

## Inference (TODO — required for submission)

The course requires this function:

```python
def generate_chessboard_image(fen: str, viewpoint: str) -> None:
    """
    viewpoint ∈ {"white", "black"} (which color sits closest to the camera).
    Must save three files in ./results/:
        ./results/synthetic.png       (Blender render)
        ./results/realistic.png       (CUT-translated)
        ./results/side_by_side.png    (synthetic | realistic)
    """
```

Pipeline:
1. Call `synthetic_chess_generator.py` via Blender with the given FEN and viewpoint.
2. Load `trained_model/latest_net_G.pth` (ResNet 9-blocks generator, 3-channel
   input/output, instance norm — see `train_opt.txt`).
3. Forward-pass the synthetic render through the generator.
4. Save the three PNGs.

## Notes for collaborators

- The training run is finished — do not retrain unless ablating.
- Hyperparameter source of truth is `trained_model/train_opt.txt`.
- `loss_log.txt` covers all 400 epochs — useful for the report's loss curves.
- Earlier project notes are in `PROJECT_CONTEXT_FOR_CLAUDE.txt`,
  `BGU_CLUSTER_TRAINING_GUIDE.txt`, and `FULL_TRAINING_INSTRUCTIONS.txt`.

## Submission checklist (from the course PDF)

- [x] Synthetic-to-real translation model trained
- [ ] `generate_chessboard_image(fen, viewpoint)` implemented
- [ ] Final report (PDF)
- [ ] Ablation study
- [x] GitHub repo
- [x] Dataset uploaded to shared drive
- [ ] README updated with reproduction instructions (this file)
