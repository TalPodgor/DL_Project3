#!/bin/bash
# Wave 2 verification — run the fine-tuned paired model on the held-out test set.
# Produces, for each test pair: real_A (synthetic input), fake_B (translated),
# real_B (real target) -> open them side by side and check that empty squares stay
# empty (no phantom pieces), especially on the SPARSE positions.

cd ~/chess_cut_project/contrastive-unpaired-translation

if command -v conda &> /dev/null; then
    conda activate cut_env 2>/dev/null || true
fi

echo "=========================================="
echo "Wave 2: inference with the paired model"
echo "=========================================="

# --dataset_mode aligned + --phase test reads datasets/chess_paired/test/*.png
# (each [synthetic | real]); the model outputs fake_B alongside the real target,
# so you can eyeball correctness directly.
python test.py \
    --dataroot ./datasets/chess_paired \
    --name chess_paired \
    --model paired_cut \
    --dataset_mode aligned \
    --direction AtoB \
    --load_size 256 \
    --crop_size 256 \
    --phase test \
    --gpu_ids 0 \
    --num_test 200 \
    --results_dir ./results

echo ""
echo "Results saved to: ./results/chess_paired/test_latest/"
echo "Compare fake_B vs real_B; verify no pieces on empty squares."
