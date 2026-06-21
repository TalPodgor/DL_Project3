#!/bin/bash
#SBATCH --job-name=segv3_test
#SBATCH --partition=gtx1080
#SBATCH --qos=course
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:20:00
#SBATCH --output=/home/packler/chess_cut_project/segv3_test_log_%j.txt

# Wave 5 evaluation: run the trained seg-conditioned model on the held-out game2 test
# set. Emits real_A (synthetic) / fake_B (generated) / real_B (real) triplets, which
# square_eval.py then scores for correctness.
# Usage: sbatch test_seg_hd.sbatch [EPOCH] [NAME]

EPOCH=${1:-latest}; NAME=${2:-chess_segv3}
cd ~/chess_cut_project/contrastive-unpaired-translation

conda run -n pytorch python test.py \
    --dataroot ./datasets/chess_paired_v3 \
    --name "$NAME" \
    --model paired_seg_hd \
    --dataset_mode seg_aligned \
    --direction AtoB \
    --input_nc 3 --output_nc 3 --num_seg_classes 15 \
    --netG resnet_9blocks --ngf 64 --normG instance --no_dropout \
    --load_size 512 --crop_size 512 \
    --phase test --epoch "$EPOCH" \
    --gpu_ids 0 --num_test 140 \
    --results_dir ./results
echo "done epoch=$EPOCH name=$NAME"
