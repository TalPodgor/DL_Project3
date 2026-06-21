#!/bin/bash
#SBATCH --job-name=geomv4_test
#SBATCH --partition=gtx1080
#SBATCH --qos=course
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:20:00
#SBATCH --output=/home/packler/chess_cut_project/geomv4_test_log_%j.txt

# Usage: sbatch test_geom_hd.sbatch [EPOCH] [NAME]

EPOCH=${1:-latest}
NAME=${2:-chess_geomv4_smoke}

cd ~/chess_cut_project/contrastive-unpaired-translation

conda run -n pytorch python test.py \
  --dataroot ./datasets/chess_existing_geom_v4 \
  --name "$NAME" \
  --model paired_geom_hd \
  --dataset_mode geom_aligned \
  --direction AtoB \
  --input_nc 3 --output_nc 3 --num_seg_classes 15 --num_geom_channels 3 \
  --netG resnet_9blocks --ngf 64 --normG instance --no_dropout \
  --load_size 512 --crop_size 512 \
  --phase test --epoch "$EPOCH" \
  --gpu_ids 0 --num_test 140 \
  --results_dir ./results

echo "done epoch=$EPOCH name=$NAME"
