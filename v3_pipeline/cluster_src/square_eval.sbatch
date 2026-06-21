#!/bin/bash
#SBATCH --job-name=sq_eval
#SBATCH --partition=gtx1080
#SBATCH --qos=course
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00:00
#SBATCH --output=/home/packler/chess_cut_project/sqeval_log_%j.txt

# M3: train the independent per-square correctness evaluator on REAL crops.
cd ~/chess_cut_project/contrastive-unpaired-translation
echo '=== GPU ===' && nvidia-smi --query-gpu=name,memory.free --format=csv && date
conda run -n pytorch python square_eval.py --mode train \
    --data ./datasets/chess_paired_v3 --out ./checkpoints/square_eval.pth --epochs 18
echo '=== EVAL TRAIN DONE ===' && date
