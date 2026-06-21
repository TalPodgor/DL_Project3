#!/bin/bash
# Wave 2 — Paired (supervised) fine-tune of the chess synth->real translator.
# Runs on the BGU cluster CUT checkout (GTX 1080 Ti).
#
# WHAT THIS DOES
#   Fine-tunes the ALREADY-TRAINED generator/discriminator with a supervised
#   objective (LSGAN + L1) on the aligned dataset built in Wave 1. The L1 term ties
#   each output to the paired real target of the same FEN -> kills phantom pieces.
#
# PREREQUISITES (do these once before running)
#   1. Copy the two drop-in patches into the repo:
#        cp models/paired_cut_model.py ~/chess_cut_project/contrastive-unpaired-translation/models/
#        cp data/aligned_dataset.py    ~/chess_cut_project/contrastive-unpaired-translation/data/
#      (the CUT repo ships only an UNALIGNED dataset, so the aligned one must be added).
#   2. Upload the Wave 1 dataset to the repo:
#        ~/chess_cut_project/contrastive-unpaired-translation/datasets/chess_paired/{train,test}/
#   3. Make sure the trained weights latest_net_{G,D}.pth are reachable (see INIT_SRC below).

set -e

REPO=~/chess_cut_project/contrastive-unpaired-translation
cd "$REPO"

# Activate conda env if available
if command -v conda &> /dev/null; then
    conda activate cut_env 2>/dev/null || true
fi

NAME=chess_paired
CKPT_DIR=./checkpoints
DEST="$CKPT_DIR/$NAME"

# ---------------------------------------------------------------------------
# Seed the fine-tune with the trained weights so --continue_train loads them.
# Looks for latest_net_{G,D}.pth in (in order): the prior chess_cut run, then a
# ./trained_model/ upload. Override by exporting INIT_SRC=/path/to/weights.
# ---------------------------------------------------------------------------
mkdir -p "$DEST"
if [ ! -f "$DEST/latest_net_G.pth" ] || [ ! -f "$DEST/latest_net_D.pth" ]; then
    if [ -z "$INIT_SRC" ]; then
        if [ -f "$CKPT_DIR/chess_cut/latest_net_G.pth" ]; then
            INIT_SRC="$CKPT_DIR/chess_cut"
        elif [ -f "./trained_model/latest_net_G.pth" ]; then
            INIT_SRC="./trained_model"
        fi
    fi
    if [ -z "$INIT_SRC" ] || [ ! -f "$INIT_SRC/latest_net_G.pth" ]; then
        echo "ERROR: could not find latest_net_G.pth / latest_net_D.pth to initialize from."
        echo "       Set INIT_SRC=/path/to/folder containing latest_net_{G,D}.pth and re-run."
        exit 1
    fi
    echo "Seeding fine-tune from: $INIT_SRC"
    cp "$INIT_SRC/latest_net_G.pth" "$DEST/latest_net_G.pth"
    cp "$INIT_SRC/latest_net_D.pth" "$DEST/latest_net_D.pth"
fi

echo "=========================================="
echo "Wave 2: Paired supervised fine-tune (LSGAN + L1)"
echo "  dataset : ./datasets/chess_paired (aligned [synthetic | real])"
echo "  init    : $DEST/latest_net_{G,D}.pth"
echo "  lr      : 2e-5,  schedule: 30 flat + 30 decay,  lambda_L1: 10"
echo "=========================================="

# --continue_train + --epoch latest -> load latest_net_{G,D}.pth from $DEST.
# Fresh optimizers (CUT does not checkpoint optimizer state) start at --lr.
# load_size 286 / crop_size 256 keeps light augmentation; the aligned dataset
# applies the SAME crop+flip to both halves, so A and B stay registered.
python train.py \
    --dataroot ./datasets/chess_paired \
    --name "$NAME" \
    --model paired_cut \
    --dataset_mode aligned \
    --direction AtoB \
    --lambda_GAN 1.0 \
    --lambda_L1 10.0 \
    --batch_size 4 \
    --load_size 286 \
    --crop_size 256 \
    --lr 0.00002 \
    --n_epochs 30 \
    --n_epochs_decay 30 \
    --epoch_count 1 \
    --gpu_ids 0 \
    --display_id -1 \
    --save_epoch_freq 10 \
    --print_freq 100 \
    --checkpoints_dir "$CKPT_DIR" \
    --continue_train \
    --epoch latest

echo ""
echo "=========================================="
echo "Wave 2 fine-tune complete."
echo "Weights: $DEST/latest_net_G.pth"
echo "Next: run inference on held-out SPARSE test frames and check for phantom pieces."
echo "=========================================="
