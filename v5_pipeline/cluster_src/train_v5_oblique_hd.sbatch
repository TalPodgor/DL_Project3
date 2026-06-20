#!/bin/bash
#SBATCH --job-name=chess_v5obl
#SBATCH --partition=gtx1080
#SBATCH --qos=course
#SBATCH --gres=gpu:1
#SBATCH --time=0-23:30:00
#SBATCH --output=/home/packler/chess_cut_project/v5_oblique_log_%j.txt

set -euo pipefail

# V5A training ablation.
# Input = oblique Blender RGB + recovered semantic silhouettes + depth/silhouette/edge geometry.
# Dataset must be middle-only; do not train left/right samples against the same real target.
#
# Env overrides:
#   DATAROOT=./datasets/chess_v5_oblique NAME=chess_v5_oblique_a BS=2 EP=30 EPD=20
#   PIECEW=0.35 PCLS=1.0 INIT=./checkpoints/chess_geomv4_smoke5/latest_net_G.pth RESUME=1

cd ~/chess_cut_project/contrastive-unpaired-translation
echo '=== GPU ===' && nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv && date

DATAROOT=${DATAROOT:-./datasets/chess_v5_oblique}
NAME=${NAME:-chess_v5_oblique_a}
BS=${BS:-2}
EP=${EP:-30}
EPD=${EPD:-20}
PIECEW=${PIECEW:-0.35}
PCLS=${PCLS:-1.0}
INIT=${INIT:-./checkpoints/chess_geomv4_smoke5/latest_net_G.pth}

if [ ! -d "$DATAROOT/train" ]; then
  echo "Missing train split: $DATAROOT/train"
  exit 2
fi

if [ ! -f "$INIT" ]; then
  INIT=./checkpoints/chess_geomv4_smoke/latest_net_G.pth
fi
if [ ! -f "$INIT" ]; then
  INIT=./checkpoints/chess_paired/latest_net_G.pth
fi

INIT_ARGS="--epoch_count 1"
if [ -f "$INIT" ]; then
  INIT_ARGS="--g_init_path $INIT --epoch_count 1"
  echo "Warm-starting G from $INIT"
else
  echo "No compatible warm-start checkpoint found; training G from scratch"
fi

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
  --dataroot "$DATAROOT" \
  --name "$NAME" \
  --model paired_geom_hd \
  --dataset_mode v5_oblique \
  --direction AtoB \
  --input_nc 3 --output_nc 3 --num_seg_classes 15 --num_geom_channels 3 \
  --load_size 512 --crop_size 512 \
  --batch_size "$BS" \
  --netG resnet_9blocks --ngf 64 --normG instance --no_dropout \
  --n_layers_D 3 --ndf 64 --num_D 2 --gan_mode lsgan \
  --lambda_GAN 1.0 --lambda_feat 10.0 --lambda_VGG 5.0 --lambda_L1 7.5 \
  --l1_piece_w "$PIECEW" --vgg_max 256 \
  $PCLS_ARGS \
  --lr 0.0002 --beta1 0.5 \
  --n_epochs "$EP" --n_epochs_decay "$EPD" \
  --gpu_ids 0 --display_id -1 \
  --save_epoch_freq 5 --save_latest_freq 5000 --print_freq 100 \
  --checkpoints_dir ./checkpoints \
  $INIT_ARGS

echo '=== TRAIN DONE ===' && date
