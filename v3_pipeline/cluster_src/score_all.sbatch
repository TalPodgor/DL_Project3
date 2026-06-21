#!/bin/bash
#SBATCH --job-name=segv3_score
#SBATCH --partition=gtx1080
#SBATCH --qos=course
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:20:00
#SBATCH --output=/home/packler/chess_cut_project/score_log_%j.txt
cd ~/chess_cut_project/contrastive-unpaired-translation
D=./datasets/chess_paired_v3; C=./checkpoints/square_eval.pth
echo "###### NEW seg-conditioned (fake_B) ######"
conda run -n pytorch python square_eval.py --mode score --data $D --ckpt $C \
  --images ./results/chess_segv3/test_latest/images/fake_B --out report_new.json
echo "###### REAL target (sanity) ######"
conda run -n pytorch python square_eval.py --mode score --data $D --ckpt $C \
  --images ./results/chess_segv3/test_latest/images/real_B --out report_real.json
echo "###### OLD Wave-4 (blobs) ######"
conda run -n pytorch python square_eval.py --mode score --data $D --ckpt $C \
  --images ./results/chess_hd/test_latest/images/fake_B --out report_old.json
echo DONE
