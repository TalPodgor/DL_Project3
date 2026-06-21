#!/bin/bash
#SBATCH --job-name=chess_segv3
#SBATCH --partition=gtx1080
#SBATCH --qos=course
#SBATCH --gres=gpu:1
#SBATCH --time=0-23:30:00
#SBATCH --output=/home/packler/chess_cut_project/segv3_log_%j.txt

# Wave 5 (geometry-first v3): mask-conditioned pix2pixHD on the view-aligned dataset.
# Input  = concat(view-aligned synthetic RGB, one-hot FEN semantic mask).
# Target = real photo (board-rectified canonical 512, RAW colour).
#   loss_G = GAN + 10*FM + 5*VGG + 10*maskL1(piece_w=0.3)
# G warm-starts from the Wave 2 generator (first conv re-inits for the +K mask channels).
#
# Env overrides: BS, EP, EPD, RESUME=1, NAME, PIECEW

cd ~/chess_cut_project/contrastive-unpaired-translation
echo '=== GPU ===' && nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv && date

BS=${BS:-2}; EP=${EP:-60}; EPD=${EPD:-60}; NAME=${NAME:-chess_segv3}; PIECEW=${PIECEW:-0.3}

INIT_ARGS="--g_init_path ./checkpoints/chess_paired/latest_net_G.pth --epoch_count 1"
if [ "${RESUME:-0}" = "1" ]; then
    INIT_ARGS="--continue_train --epoch latest"
    echo "RESUMING from checkpoints/$NAME/latest_net_{G,D}.pth"
fi

conda run -n pytorch python train.py \
    --dataroot ./datasets/chess_paired_v3 \
    --name "$NAME" \
    --model paired_seg_hd \
    --dataset_mode seg_aligned \
    --direction AtoB \
    --input_nc 3 --output_nc 3 --num_seg_classes 15 \
    --load_size 512 --crop_size 512 \
    --batch_size "$BS" \
    --netG resnet_9blocks --ngf 64 --normG instance --no_dropout \
    --n_layers_D 3 --ndf 64 --num_D 2 --gan_mode lsgan \
    --lambda_GAN 1.0 --lambda_feat 10.0 --lambda_VGG 5.0 --lambda_L1 10.0 \
    --l1_piece_w "$PIECEW" --vgg_max 256 \
    --lr 0.0002 --beta1 0.5 \
    --n_epochs "$EP" --n_epochs_decay "$EPD" \
    --gpu_ids 0 --display_id -1 \
    --save_epoch_freq 5 --save_latest_freq 5000 --print_freq 100 \
    --checkpoints_dir ./checkpoints \
    $INIT_ARGS

echo '=== TRAIN DONE ===' && date
