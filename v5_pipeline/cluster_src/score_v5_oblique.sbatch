#!/bin/bash
#SBATCH --job-name=v5obl_score
#SBATCH --partition=gtx1080
#SBATCH --qos=course
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:20:00
#SBATCH --output=/home/packler/chess_cut_project/v5_oblique_score_log_%j.txt

set -euo pipefail

# Usage: sbatch score_v5_oblique.sbatch [EPOCH] [NAME]

EPOCH=${1:-latest}
NAME=${2:-chess_v5_oblique_a}
DATAROOT=${DATAROOT:-./datasets/chess_v5_oblique}

cd ~/chess_cut_project/contrastive-unpaired-translation

IMG_DIR="./results/${NAME}/test_${EPOCH}/images/fake_B"
OUT="./report_${NAME}_${EPOCH}.json"

conda run -n pytorch python square_eval.py \
  --mode score \
  --data "$DATAROOT" \
  --ckpt ./checkpoints/square_eval.pth \
  --images "$IMG_DIR" \
  --out "$OUT"

echo "scored $IMG_DIR -> $OUT"
