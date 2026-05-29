================================================================================
DL Project 3 — Shared Data Folder
================================================================================

Code repository: https://github.com/TalPodgor/DL_Project3

The repo intentionally excludes the contents below per the course rule that all
datasets (used or generated) must live on a shared drive.

Download this entire folder, then place each subfolder directly inside the
cloned repo root so paths match what the scripts expect.

================================================================================
FOLDER CONTENTS
================================================================================

dataset/
    CUT training dataset (final, organized).
    ├── trainA/     2,664 synthetic images   (games 4, 5, 6, 7; both viewpoints)
    ├── trainB/       736 real images        (games 4, 5, 6, 7; both viewpoints)
    ├── testA/        462 synthetic images   (game 2)
    └── testB/        140 real images        (game 2)

real/
    Raw real-frame images collected from gameX_per_frame/tagged_images/, with
    180° rotated copies for viewpoint augmentation. Produced by
    preprocess_real_images.py.

synthetic/
    Raw Blender-rendered images collected from synthetic_gameX/cropped/, with
    180° rotated copies. Produced by preprocess_synthetic_images.py.

game{2,4,5,6,7}_per_frame/
    Per-game extracted frames from the original game videos. Each contains a
    tagged_images/ subdir with the labeled frames used as real data.

synthetic_game{2,4,5,6,7}/
    Per-game Blender renders generated from the FEN labels in gameX.csv.
    Contains raw/ and cropped/ subdirs.

training_progress/
    51 epochs of CUT training sample images (real_A, fake_B, real_B, idt_B per
    epoch). Useful for the report's training-progress figures.

test_output/
    CUT inference results on the testA set (game 2).
    ├── images/         translated outputs
    ├── comparisons/    synthetic | translated side-by-sides
    └── grid/           grid view

test_output.zip
    Zipped copy of test_output/ (217 MB). Exceeds GitHub's per-file limit, so
    it lives here only.

================================================================================
WHAT IS *NOT* HERE (already in the GitHub repo)
================================================================================

Already in the repo, no need to download:
    chess-set.blend                  Blender asset
    trained_model/                   trained CUT weights + configs + loss log
    *.py and *.sh                    all code and shell scripts
    game{2,4,5,6,7}.csv              frame-to-FEN labels

================================================================================
TARGET LOCAL LAYOUT (after cloning the repo and downloading this folder)
================================================================================

DL_Project3/
├── (everything from GitHub)
├── dataset/
├── real/
├── synthetic/
├── game2_per_frame/  ... game7_per_frame/
├── synthetic_game2/  ... synthetic_game7/
├── training_progress/
└── test_output/
