#!/bin/bash
#SBATCH --job-name=chess_test_hd
#SBATCH --partition=gtx1080
#SBATCH --qos=course
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:15:00
#SBATCH --output=/home/packler/chess_cut_project/test_hd_log_%j.txt

# Wave 4 evaluation: run the trained pix2pixHD-style model on the held-out game2 test
# set (and optionally a train subset). Emits real_A / fake_B / real_B triplets.
# Usage: sbatch test_hd.sbatch [EPOCH]   (default: latest)

EPOCH=${1:-latest}
cd ~/chess_cut_project/contrastive-unpaired-translation

conda run -n pytorch python test.py \
    --dataroot ./datasets/chess_paired_v2 \
    --name chess_hd \
    --model paired_hd \
    --dataset_mode aligned \
    --direction AtoB \
    --input_nc 3 --output_nc 3 \
    --netG resnet_9blocks --ngf 64 --normG instance --no_dropout \
    --load_size 512 --crop_size 512 \
    --phase test --epoch "$EPOCH" \
    --gpu_ids 0 --num_test 140 \
    --results_dir ./results
echo "done epoch=$EPOCH"
