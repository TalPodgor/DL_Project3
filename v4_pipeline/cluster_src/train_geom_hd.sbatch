#!/bin/bash
#SBATCH --job-name=chess_geomv4
#SBATCH --partition=gtx1080
#SBATCH --qos=course
#SBATCH --gres=gpu:1
#SBATCH --time=0-23:30:00
#SBATCH --output=/home/packler/chess_cut_project/geomv4_log_%j.txt

set -euo pipefail

# v4 smoke/full training.
# Input = existing synthetic RGB + one-hot FEN mask + synthetic-derived geometry maps.
# Defaults are intentionally short enough for a first visual gate.
#
# Env overrides:
#   NAME=chess_geomv4_smoke BS=2 EP=20 EPD=0 PIECEW=0.1 PCLS=0.5 RESUME=1

cd ~/chess_cut_project/contrastive-unpaired-translation
echo '=== GPU ===' && nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv && date

BS=${BS:-2}
EP=${EP:-20}
EPD=${EPD:-0}
NAME=${NAME:-chess_geomv4_smoke}
PIECEW=${PIECEW:-0.1}
PCLS=${PCLS:-0.5}
INIT=${INIT:-./checkpoints/chess_segv3/latest_net_G.pth}

if [ ! -f "$INIT" ]; then
  INIT=./checkpoints/chess_paired/latest_net_G.pth
fi

INIT_ARGS="--g_init_path $INIT --epoch_count 1"
if [ "${RESUME:-0}" = "1" ]; then
  INIT_ARGS="--continue_train --epoch latest"
  echo "RESUMING from checkpoints/$NAME/latest_net_{G,D}.pth"
fi

PCLS_ARGS="--lambda_piece_cls $PCLS --piece_cls_path ./checkpoints/square_eval.pth"
if [ "$PCLS" != "0" ] && [ "$PCLS" != "0.0" ] && [ ! -f ./checkpoints/square_eval.pth ]; then
  echo "Missing ./checkpoints/square_eval.pth but PCLS=$PCLS"
  exit 2
fi

conda run -n pytorch python train.py \
  --dataroot ./datasets/chess_existing_geom_v4 \
  --name "$NAME" \
  --model paired_geom_hd \
  --dataset_mode geom_aligned \
  --direction AtoB \
  --input_nc 3 --output_nc 3 --num_seg_classes 15 --num_geom_channels 3 \
  --load_size 512 --crop_size 512 \
  --batch_size "$BS" \
  --netG resnet_9blocks --ngf 64 --normG instance --no_dropout \
  --n_layers_D 3 --ndf 64 --num_D 2 --gan_mode lsgan \
  --lambda_GAN 1.0 --lambda_feat 10.0 --lambda_VGG 5.0 --lambda_L1 10.0 \
  --l1_piece_w "$PIECEW" --vgg_max 256 \
  $PCLS_ARGS \
  --lr 0.0002 --beta1 0.5 \
  --n_epochs "$EP" --n_epochs_decay "$EPD" \
  --gpu_ids 0 --display_id -1 \
  --save_epoch_freq 5 --save_latest_freq 5000 --print_freq 100 \
  --checkpoints_dir ./checkpoints \
  $INIT_ARGS

echo '=== TRAIN DONE ===' && date
