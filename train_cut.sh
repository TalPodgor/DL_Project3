#!/bin/bash
# CUT Training Script for BGU Cluster (GTX 1080 Ti)
# Run this after setup and uploading dataset

cd ~/chess_cut_project/contrastive-unpaired-translation

# Activate conda environment if using conda
if command -v conda &> /dev/null; then
    conda activate cut_env
fi

echo "=========================================="
echo "Starting CUT Training"
echo "=========================================="
echo "Dataset: chess (synthetic -> real)"
echo "GPU: GTX 1080 Ti"
echo ""

# Training command
# --dataroot: path to dataset with trainA, trainB folders
# --name: experiment name (checkpoints saved to checkpoints/chess_cut)
# --CUT_mode: CUT (not FastCUT)
# --batch_size: 4 works well on GTX 1080 Ti with 256x256 images
# --load_size/crop_size: resize to 286 then crop to 256 (standard augmentation)
# --n_epochs: epochs with initial learning rate
# --n_epochs_decay: epochs with linearly decaying learning rate

python train.py \
    --dataroot ./datasets/chess \
    --name chess_cut \
    --CUT_mode CUT \
    --model cut \
    --batch_size 4 \
    --load_size 286 \
    --crop_size 256 \
    --n_epochs 200 \
    --n_epochs_decay 200 \
    --gpu_ids 0 \
    --display_id -1 \
    --save_epoch_freq 20 \
    --print_freq 100 \
    --checkpoints_dir ./checkpoints

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Checkpoints saved to: ./checkpoints/chess_cut/"
