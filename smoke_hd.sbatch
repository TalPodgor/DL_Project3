#!/bin/bash
#SBATCH --job-name=chess_hd_smoke
#SBATCH --partition=gtx1080
#SBATCH --qos=course
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:30:00
#SBATCH --output=/home/packler/chess_cut_project/smoke_hd_log_%j.txt

# Wave 4 SMOKE TEST: verify the pix2pixHD-style model runs end-to-end at 512px on a
# 1080 Ti (G warm-start loads, multi-scale D builds, memory fits, losses finite),
# before committing to the full multi-hour run.  Tiny subset + 1 epoch.

cd ~/chess_cut_project/contrastive-unpaired-translation
echo '=== GPU ===' && nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv && date

conda run -n pytorch python train.py \
    --dataroot ./datasets/chess_paired_v2 \
    --name chess_hd_smoke \
    --model paired_hd \
    --dataset_mode aligned \
    --direction AtoB \
    --input_nc 3 --output_nc 3 \
    --load_size 512 --crop_size 512 \
    --batch_size 2 \
    --netG resnet_9blocks --ngf 64 --normG instance --no_dropout \
    --n_layers_D 3 --ndf 64 --num_D 2 --gan_mode lsgan \
    --lambda_GAN 1.0 --lambda_feat 10.0 --lambda_VGG 5.0 --lambda_L1 10.0 --vgg_max 256 \
    --g_init_path ./checkpoints/chess_paired/latest_net_G.pth \
    --lr 0.0002 --beta1 0.5 \
    --n_epochs 1 --n_epochs_decay 0 --epoch_count 1 \
    --max_dataset_size 64 \
    --gpu_ids 0 --display_id -1 --no_html \
    --save_epoch_freq 1 --print_freq 10 \
    --checkpoints_dir ./checkpoints

echo '=== peak GPU mem after run ===' && nvidia-smi --query-gpu=memory.used --format=csv
echo '=== SMOKE DONE ===' && date
