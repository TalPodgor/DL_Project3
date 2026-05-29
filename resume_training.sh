#!/bin/bash
# Resume CUT Training from Last Checkpoint
# Use this when your session expired and you want to continue

cd ~/chess_cut_project/contrastive-unpaired-translation

# Activate conda environment if using conda
if command -v conda &> /dev/null; then
    conda activate cut_env 2>/dev/null
fi

echo "=========================================="
echo "Resuming CUT Training from Last Checkpoint"
echo "=========================================="

# Check if checkpoint exists
if [ ! -f "./checkpoints/chess_cut/latest_net_G.pth" ]; then
    echo "ERROR: No checkpoint found! Run train_cut.sh first."
    exit 1
fi

echo "Found checkpoint, resuming training..."

# Find the last saved epoch by looking at checkpoint files
LAST_EPOCH=$(ls ./checkpoints/chess_cut/ | grep -oP '^\d+(?=_net_G\.pth)' | sort -n | tail -1)

if [ -z "$LAST_EPOCH" ]; then
    LAST_EPOCH=1
    echo "Could not detect last epoch, starting from epoch 1 with loaded weights"
else
    echo "Detected last saved epoch: $LAST_EPOCH"
    # Continue from the next epoch
    LAST_EPOCH=$((LAST_EPOCH + 1))
    echo "Resuming from epoch: $LAST_EPOCH"
fi

# Resume training with --continue_train flag and correct epoch
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
    --epoch_count $LAST_EPOCH \
    --gpu_ids 0 \
    --display_id -1 \
    --save_epoch_freq 20 \
    --print_freq 100 \
    --checkpoints_dir ./checkpoints \
    --continue_train

echo ""
echo "Training Complete!"
