#!/bin/bash
# CUT Testing Script - Generate realistic images from synthetic test set
# Run this after training is complete

cd ~/chess_cut_project/contrastive-unpaired-translation

# Activate conda environment if using conda
if command -v conda &> /dev/null; then
    conda activate cut_env
fi

echo "=========================================="
echo "Running CUT Inference on Test Set"
echo "=========================================="

# Test command
# Uses testA (synthetic test images) and generates realistic versions
# Results saved to results/chess_cut/test_latest/

python test.py \
    --dataroot ./datasets/chess \
    --name chess_cut \
    --CUT_mode CUT \
    --model cut \
    --load_size 256 \
    --crop_size 256 \
    --phase test \
    --gpu_ids 0 \
    --num_test 500 \
    --results_dir ./results

echo ""
echo "=========================================="
echo "Testing Complete!"
echo "=========================================="
echo "Results saved to: ./results/chess_cut/test_latest/"
