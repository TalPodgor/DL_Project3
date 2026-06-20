#!/bin/bash
#SBATCH --job-name=v5obl_test
#SBATCH --partition=gtx1080
#SBATCH --qos=course
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:25:00
#SBATCH --output=/home/packler/chess_cut_project/v5_oblique_test_log_%j.txt

set -euo pipefail

# Usage: sbatch test_v5_oblique_hd.sbatch [EPOCH] [NAME]

EPOCH=${1:-latest}
NAME=${2:-chess_v5_oblique_a}
DATAROOT=${DATAROOT:-./datasets/chess_v5_oblique}
NUM_TEST=${NUM_TEST:-140}

cd ~/chess_cut_project/contrastive-unpaired-translation

conda run -n pytorch python test.py \
  --dataroot "$DATAROOT" \
  --name "$NAME" \
  --model paired_geom_hd \
  --dataset_mode v5_oblique \
  --direction AtoB \
  --input_nc 3 --output_nc 3 --num_seg_classes 15 --num_geom_channels 3 \
  --netG resnet_9blocks --ngf 64 --normG instance --no_dropout \
  --load_size 512 --crop_size 512 \
  --phase test --epoch "$EPOCH" \
  --gpu_ids 0 --num_test "$NUM_TEST" \
  --results_dir ./results

echo "done epoch=$EPOCH name=$NAME dataroot=$DATAROOT"
