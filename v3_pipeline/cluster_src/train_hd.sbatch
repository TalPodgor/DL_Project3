#!/bin/bash
#SBATCH --job-name=chess_hd
#SBATCH --partition=gtx1080
#SBATCH --qos=course
#SBATCH --gres=gpu:1
#SBATCH --time=0-23:30:00
#SBATCH --output=/home/packler/chess_cut_project/hd_log_%j.txt

# Wave 4 FULL training: pix2pixHD-style paired translation at 512px on the
# colour-canonicalised v2 dataset.  G warm-starts from the Wave 2 generator;
# the multi-scale discriminator trains from scratch.
#
#   loss_G = GAN + 10*FM(multi-scale D feats) + 5*VGG19 + 10*L1
#
# Env overrides:  BS (batch), EP (flat epochs), EPD (decay epochs), RESUME=1
# Defaults are tuned from the smoke test.

cd ~/chess_cut_project/contrastive-unpaired-translation
echo '=== GPU ===' && nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv && date

BS=${BS:-2}
EP=${EP:-60}
EPD=${EPD:-60}

INIT_ARGS="--g_init_path ./checkpoints/chess_paired/latest_net_G.pth --epoch_count 1"
if [ "${RESUME:-0}" = "1" ]; then
    INIT_ARGS="--continue_train --epoch latest"
    echo "RESUMING from checkpoints/chess_hd/latest_net_{G,D}.pth"
fi

conda run -n pytorch python train.py \
    --dataroot ./datasets/chess_paired_v2 \
    --name chess_hd \
    --model paired_hd \
    --dataset_mode aligned \
    --direction AtoB \
    --input_nc 3 --output_nc 3 \
    --load_size 512 --crop_size 512 \
    --batch_size "$BS" \
    --netG resnet_9blocks --ngf 64 --normG instance --no_dropout \
    --n_layers_D 3 --ndf 64 --num_D 2 --gan_mode lsgan \
    --lambda_GAN 1.0 --lambda_feat 10.0 --lambda_VGG 5.0 --lambda_L1 10.0 --vgg_max 256 \
    --lr 0.0002 --beta1 0.5 \
    --n_epochs "$EP" --n_epochs_decay "$EPD" \
    --gpu_ids 0 --display_id -1 \
    --save_epoch_freq 5 --save_latest_freq 5000 --print_freq 100 \
    --checkpoints_dir ./checkpoints \
    $INIT_ARGS

echo '=== TRAIN DONE ===' && date
